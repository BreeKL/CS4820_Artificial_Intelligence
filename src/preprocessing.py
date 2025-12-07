"""
Light Curve Data Preprocessing Module

Handles loading, cleaning, and preprocessing of Kepler/TESS light curve data.
Implements sigma-clipping, trend removal, normalization, and segmentation.
"""

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class LightCurvePreprocessor:
    """Preprocessor for astronomical light curve data."""
    
    def __init__(
        self,
        sigma_threshold: float = 3.0,
        rolling_window: int = 50,
        savgol_window: int = 101,
        savgol_poly: int = 3,
        max_gap_days: float = 2.0,
        segment_duration_days: float = 90.0,
        cadence_minutes: float = 30.0
    ):
        """
        Initialize preprocessor with configuration parameters.
        
        Args:
            sigma_threshold: Standard deviations for outlier removal
            rolling_window: Window size for rolling statistics
            savgol_window: Window length for Savitzky-Golay filter (must be odd)
            savgol_poly: Polynomial order for Savitzky-Golay filter
            max_gap_days: Maximum gap duration for linear interpolation
            segment_duration_days: Duration of each light curve segment
            cadence_minutes: Observation cadence in minutes
        """
        self.sigma_threshold = sigma_threshold
        self.rolling_window = rolling_window
        self.savgol_window = savgol_window if savgol_window % 2 == 1 else savgol_window + 1
        self.savgol_poly = savgol_poly
        self.max_gap_days = max_gap_days
        self.segment_duration_days = segment_duration_days
        self.cadence_minutes = cadence_minutes
        self.cadence_days = cadence_minutes / (24 * 60)
        
    def load_fits(self, filepath: str) -> pd.DataFrame:
        """
        Load light curve data from FITS file.
        
        Args:
            filepath: Path to FITS file
            
        Returns:
            DataFrame with 'time' and 'flux' columns
        """
        with fits.open(filepath) as hdul:
            data = hdul[1].data
            
            # Handle different column naming conventions
            time_col = None
            flux_col = None
            
            for col in data.columns.names:
                if 'TIME' in col.upper():
                    time_col = col
                if 'FLUX' in col.upper() and 'SAP' in col.upper():
                    flux_col = col
                elif 'PDCSAP_FLUX' in col.upper():
                    flux_col = col
            
            if time_col is None or flux_col is None:
                raise ValueError("Could not find TIME and FLUX columns in FITS file")
            
            df = pd.DataFrame({
                'time': data[time_col],
                'flux': data[flux_col]
            })
            
            # Remove NaN values
            df = df.dropna().reset_index(drop=True)
            
        return df
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load light curve data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with 'time' and 'flux' columns
        """
        df = pd.read_csv(filepath)
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        # Handle different naming conventions
        time_cols = ['time', 'btjd', 'bjd', 'mjd']
        flux_cols = ['flux', 'sap_flux', 'pdcsap_flux']
        
        time_col = next((col for col in time_cols if col in df.columns), None)
        flux_col = next((col for col in flux_cols if col in df.columns), None)
        
        if time_col is None or flux_col is None:
            raise ValueError("Could not find time and flux columns in CSV file")
        
        df = df[[time_col, flux_col]].rename(
            columns={time_col: 'time', flux_col: 'flux'}
        )
        
        # Remove NaN values
        df = df.dropna().reset_index(drop=True)
        
        return df
    
    def sigma_clip(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers using sigma clipping with rolling statistics.
        
        Args:
            df: DataFrame with 'time' and 'flux' columns
            
        Returns:
            DataFrame with outliers removed
        """
        flux = df['flux'].values
        
        # Calculate rolling mean and std
        rolling_mean = pd.Series(flux).rolling(
            window=self.rolling_window, 
            center=True, 
            min_periods=1
        ).mean().values
        
        rolling_std = pd.Series(flux).rolling(
            window=self.rolling_window, 
            center=True, 
            min_periods=1
        ).std().values
        
        # Identify outliers
        deviation = np.abs(flux - rolling_mean)
        threshold = self.sigma_threshold * rolling_std
        mask = deviation <= threshold
        
        return df[mask].reset_index(drop=True)
    
    def remove_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove low-frequency trends using Savitzky-Golay filter.
        
        Args:
            df: DataFrame with 'time' and 'flux' columns
            
        Returns:
            DataFrame with detrended flux
        """
        flux = df['flux'].values
        
        if len(flux) < self.savgol_window:
            # If data is too short, use simpler detrending
            trend = np.median(flux)
        else:
            trend = savgol_filter(flux, self.savgol_window, self.savgol_poly)
        
        df = df.copy()
        df['flux'] = flux / trend
        
        return df
    
    def interpolate_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill gaps using linear interpolation for short gaps.
        
        Args:
            df: DataFrame with 'time' and 'flux' columns
            
        Returns:
            DataFrame with interpolated values
        """
        time = df['time'].values
        flux = df['flux'].values
        
        # Calculate time differences
        time_diffs = np.diff(time)
        median_cadence = np.median(time_diffs)
        
        # Find gaps larger than expected cadence
        gap_indices = np.where(time_diffs > 2 * median_cadence)[0]
        
        if len(gap_indices) == 0:
            return df
        
        # Process each gap
        new_times = [time]
        new_fluxes = [flux]
        
        for gap_idx in gap_indices:
            gap_duration = time_diffs[gap_idx]
            
            # Only interpolate short gaps
            if gap_duration <= self.max_gap_days:
                n_points = int(gap_duration / median_cadence)
                if n_points > 1:
                    interp_times = np.linspace(
                        time[gap_idx], 
                        time[gap_idx + 1], 
                        n_points + 2
                    )[1:-1]
                    
                    # Linear interpolation
                    interp_flux = np.interp(
                        interp_times, 
                        [time[gap_idx], time[gap_idx + 1]], 
                        [flux[gap_idx], flux[gap_idx + 1]]
                    )
                    
                    new_times.append(interp_times)
                    new_fluxes.append(interp_flux)
        
        # Combine all segments
        if len(new_times) > 1:
            time = np.concatenate(new_times)
            flux = np.concatenate(new_fluxes)
            
            # Sort by time
            sort_idx = np.argsort(time)
            time = time[sort_idx]
            flux = flux[sort_idx]
            
            df = pd.DataFrame({'time': time, 'flux': flux})
        
        return df
    
    # Z-score normalization to preserve relative amplitudes
    def normalize(self, flux: np.ndarray) -> np.ndarray:
        """
        Normalize flux values while preserving relative amplitudes.
        
        Args:
            flux: Array of flux values
            
        Returns:
            Normalized flux array
        """
        # Standardization (zero mean, unit variance)
        mean = np.mean(flux)
        std = np.std(flux)
        
        if std > 0:
            normalized = (flux - mean) / std
        else:
            normalized = flux - mean
        
        return normalized

    # Min-max normalization with outlier clipping
    # def normalize(self, flux: np.ndarray) -> np.ndarray:
    #     """Simple normalization for divided flux."""
    #     # Clip outliers
    #     flux = np.clip(flux, np.percentile(flux, 1), np.percentile(flux, 99))
        
    #     # Min-max to [0, 1] or [-1, 1]
    #     flux_min = flux.min()
    #     flux_max = flux.max()
    #     normalized = (flux - flux_min) / (flux_max - flux_min)  # [0, 1]
    #     # Or: normalized = 2 * (flux - flux_min) / (flux_max - flux_min) - 1  # [-1, 1]
    
    #     return normalized

    
    def segment_light_curve(
        self, 
        df: pd.DataFrame, 
        overlap: float = 0.0
    ) -> List[np.ndarray]:
        """
        Segment light curve into fixed-duration windows.
        
        Args:
            df: DataFrame with 'time' and 'flux' columns
            overlap: Overlap fraction between segments (0.0 to 0.5)
            
        Returns:
            List of flux arrays for each segment
        """
        time = df['time'].values
        flux = df['flux'].values
        
        segments = []
        start_time = time[0]
        end_time = time[-1]
        
        step_size = self.segment_duration_days * (1 - overlap)
        current_start = start_time
        
        while current_start + self.segment_duration_days <= end_time:
            current_end = current_start + self.segment_duration_days
            
            # Extract segment
            mask = (time >= current_start) & (time < current_end)
            segment_flux = flux[mask]
            
            if len(segment_flux) > 0:
                segments.append(segment_flux)
            
            current_start += step_size
        
        return segments
    
    def preprocess(
        self, 
        filepath: str, 
        segment: bool = True,
        overlap: float = 0.0
    ) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            filepath: Path to light curve file (FITS or CSV)
            segment: Whether to segment the light curve
            overlap: Overlap fraction for segmentation
            
        Returns:
            Tuple of (segments, times) if segment=True, else (full_flux, times)
        """
        # Load data
        if filepath.endswith('.fits'):
            df = self.load_fits(filepath)
        elif filepath.endswith('.csv'):
            df = self.load_csv(filepath)
        else:
            raise ValueError("File must be FITS or CSV format")
        
        print(f"Loaded {len(df)} data points")
        
        # Sigma clipping
        df = self.sigma_clip(df)
        print(f"After sigma clipping: {len(df)} data points")
        
        # Remove trend
        df = self.remove_trend(df)
        print("Trend removed")
        
        # Interpolate gaps
        df = self.interpolate_gaps(df)
        print(f"After interpolation: {len(df)} data points")
        
        # Normalize
        df['flux'] = self.normalize(df['flux'].values)
        print("Flux normalized")
        
        if segment:
            # Segment light curve
            segments = self.segment_light_curve(df, overlap=overlap)
            print(f"Created {len(segments)} segments")
            return segments, df['time'].values
        else:
            return df['flux'].values, df['time'].values


class DataAugmenter:
    """Data augmentation for light curve time series."""
    
    @staticmethod
    def add_noise(flux: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to flux values."""
        noise = np.random.normal(0, noise_level, len(flux))
        return flux + noise
    
    @staticmethod
    def time_shift(flux: np.ndarray, max_shift: int = 50) -> np.ndarray:
        """Randomly shift the time series."""
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(flux, shift)
    
    @staticmethod
    def amplitude_scale(flux: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """Scale the amplitude of the signal."""
        scale = np.random.uniform(*scale_range)
        return flux * scale
    
    @staticmethod
    def time_warp(flux: np.ndarray, warp_factor: float = 0.1) -> np.ndarray:
        """Apply random time warping."""
        n = len(flux)
        original_indices = np.arange(n)
        warp = np.random.normal(1.0, warp_factor, n)
        warp = np.cumsum(warp)
        warp = (warp - warp[0]) / (warp[-1] - warp[0]) * (n - 1)
        
        # Interpolate
        warped_flux = np.interp(original_indices, warp, flux)
        return warped_flux
    
    def augment(
        self, 
        flux: np.ndarray, 
        methods: List[str] = ['noise', 'shift', 'scale','warp']
    ) -> np.ndarray:
        """
        Apply multiple augmentation methods.
        
        Args:
            flux: Input flux array
            methods: List of augmentation methods to apply
            
        Returns:
            Augmented flux array
        """
        augmented = flux.copy()
        
        for method in methods:
            if method == 'noise':
                augmented = self.add_noise(augmented)
            elif method == 'shift':
                augmented = self.time_shift(augmented)
            elif method == 'scale':
                augmented = self.amplitude_scale(augmented)
            elif method == 'warp':
                augmented = self.time_warp(augmented)
        
        return augmented