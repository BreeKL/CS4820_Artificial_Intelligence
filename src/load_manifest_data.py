"""
Custom Data Loader for Manifest-Based Light Curve Dataset

This script loads light curves from a manifest.csv file and prepares
them for binary classification (planet vs not planet).
"""


import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import sys
sys.path.append('src')

from preprocessing import LightCurvePreprocessor, DataAugmenter


class ManifestDataLoader:
    """Load and preprocess light curves from a manifest file for binary classification."""
    
    def __init__(
        self,
        manifest_path: str,
        data_root: str = 'data',
        preprocessor_config: Dict = None
    ):
        """
        Initialize the manifest data loader.
        
        Args:
            manifest_path: Path to manifest.csv file
            data_root: Root directory containing the data
            preprocessor_config: Configuration for preprocessing
        """
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        
        # Load manifest
        self.manifest = pd.read_csv(manifest_path)
        print(f"Loaded manifest with {len(self.manifest)} light curves")
        
        # Initialize preprocessor
        if preprocessor_config is None:
            preprocessor_config = {
                'sigma_threshold': 3.0,
                'rolling_window': 50,
                'savgol_window': 101,
                'savgol_poly': 3,
                'max_gap_days': 2.0,
                'segment_duration_days': 90.0,
                'cadence_minutes': 30.0
            }
        
        self.preprocessor = LightCurvePreprocessor(**preprocessor_config)
        
        # Binary classification: planet = 1, everything else = 0
        self.label_map = {'planet': 1, 'not_planet': 0}
        print(f"Binary classification: planet=1, not_planet=0")
    
    def get_label_names(self) -> List[str]:
        """Get list of label names in order."""
        return ['not_planet', 'planet']
    
    def convert_to_binary_label(self, original_label: str) -> int:
        """
        Convert original labels to binary (planet vs not planet).
        
        Args:
            original_label: Original label (EB, BEB, planet, star, etc.)
            
        Returns:
            1 if planet, 0 otherwise
        """
        return 1 if original_label.lower() == 'planet' else 0
    
    def load_single_light_curve(self, curve_path: str) -> pd.DataFrame:
        """
        Load a single light curve CSV file.
        
        Args:
            curve_path: Path to light curve CSV (relative to data_root)
            
        Returns:
            DataFrame with 'time' and 'flux' columns
        """
        # Construct full path
        full_path = self.data_root / curve_path
        
        # Load CSV
        df = pd.read_csv(full_path)
        
        # Ensure columns are named correctly
        df.columns = df.columns.str.lower().str.strip()
        
        if 'time' not in df.columns or 'flux' not in df.columns:
            raise ValueError(f"CSV must have 'time' and 'flux' columns. Found: {df.columns.tolist()}")
        
        # Remove NaN values
        df = df.dropna().reset_index(drop=True)
        
        return df[['time', 'flux']]

    def calculate_optimal_segment_duration(self, sample_size: int = 10) -> float:
        """
        Calculate optimal segment duration based on actual light curve lengths.
        
        Args:
            sample_size: Number of light curves to sample
            
        Returns:
            Recommended segment duration in days
        """
        print("\nAnalyzing light curve durations...")
        durations = []
        
        for idx, row in self.manifest.head(sample_size).iterrows():
            try:
                df = self.load_single_light_curve(row['curve_path'])
                time_span = df['time'].max() - df['time'].min()
                durations.append(time_span)
            except Exception as e:
                continue
        
        if durations:
            avg_duration = np.mean(durations)
            min_duration = np.min(durations)
            max_duration = np.max(durations)
            
            print(f"Light curve duration statistics (days):")
            print(f"  Average: {avg_duration:.2f}")
            print(f"  Min: {min_duration:.2f}")
            print(f"  Max: {max_duration:.2f}")
            
            # Recommend segment duration as 80% of minimum to ensure at least 1 segment per curve
            recommended = min_duration * 0.8
            print(f"  Recommended segment duration: {recommended:.2f} days")
            
            return recommended
        else:
            print("Could not analyze durations, using default")
            return 90.0
    
    def process_all_light_curves(
        self,
        segment: bool = True,
        overlap: float = 0.0,
        max_curves: int = None,
        auto_adjust_segment_duration: bool = True
    ) -> Tuple[List[np.ndarray], List[int], List[str]]:
        """
        Process all light curves in the manifest.
        
        Args:
            segment: Whether to segment light curves
            overlap: Overlap fraction for segmentation
            max_curves: Maximum number of curves to process (None for all)
            auto_adjust_segment_duration: Automatically adjust segment duration based on data
            
        Returns:
            Tuple of (flux_segments, binary_labels, tic_ids)
        """
        # Auto-adjust segment duration if requested
        if auto_adjust_segment_duration and segment:
            optimal_duration = self.calculate_optimal_segment_duration(sample_size=20)
            self.preprocessor.segment_duration_days = optimal_duration
            print(f"\nUsing segment duration: {optimal_duration:.2f} days")
        
        all_segments = []
        all_labels = []
        all_tic_ids = []
        
        # Determine how many to process
        n_curves = len(self.manifest) if max_curves is None else min(max_curves, len(self.manifest))
        
        print(f"\nProcessing {n_curves} light curves...")
        
        skipped_count = 0
        
        for idx, row in self.manifest.head(n_curves).iterrows():
            tic_id = str(row['tic_id'])
            label_str = row['label']
            curve_path = row['curve_path']
            
            # Convert to binary label
            binary_label = self.convert_to_binary_label(label_str)
            
            try:
                print(f"[{idx+1}/{n_curves}] Processing TIC {tic_id} (original: {label_str}, binary: {'planet' if binary_label == 1 else 'not_planet'})...")
                
                # Load light curve
                df = self.load_single_light_curve(curve_path)
                
                if len(df) < 10:
                    print(f"  Skipping: Too few data points ({len(df)})")
                    skipped_count += 1
                    continue
                
                # Apply preprocessing steps manually
                # Sigma clipping
                df = self.preprocessor.sigma_clip(df)
                
                if len(df) < 10:
                    print(f"  Skipping: Too few points after sigma clipping ({len(df)})")
                    skipped_count += 1
                    continue
                
                # Remove trend
                df = self.preprocessor.remove_trend(df)
                
                # Interpolate gaps
                df = self.preprocessor.interpolate_gaps(df)
                
                # Normalize
                flux_normalized = self.preprocessor.normalize(df['flux'].values)
                df['flux'] = flux_normalized
                
                if segment:
                    # Segment the light curve
                    segments = self.preprocessor.segment_light_curve(df, overlap=overlap)
                    
                    if len(segments) == 0:
                        print(f"  Warning: No segments created, using full light curve instead")
                        # Use full light curve if no segments created
                        all_segments.append(flux_normalized)
                        all_labels.append(binary_label)
                        all_tic_ids.append(tic_id)
                        print(f"  Using full curve ({len(flux_normalized)} points)")
                    else:
                        # Add each segment
                        for seg in segments:
                            all_segments.append(seg)
                            all_labels.append(binary_label)
                            all_tic_ids.append(tic_id)
                        print(f"  Created {len(segments)} segments")
                else:
                    # Use entire light curve
                    all_segments.append(flux_normalized)
                    all_labels.append(binary_label)
                    all_tic_ids.append(tic_id)
                    print(f"  Processed full light curve ({len(flux_normalized)} points)")
                
            except Exception as e:
                print(f"  Error processing TIC {tic_id}: {str(e)}")
                skipped_count += 1
                continue
        
        print(f"\n{'='*60}")
        print(f"Processing complete!")
        print(f"{'='*60}")
        print(f"Total segments/curves created: {len(all_segments)}")
        print(f"Skipped: {skipped_count}")
        print(f"\nBinary label distribution:")
        unique, counts = np.unique(all_labels, return_counts=True)
        for label_idx, count in zip(unique, counts):
            label_name = 'not_planet' if label_idx == 0 else 'planet'
            print(f"  {label_name}: {count} ({100*count/len(all_labels):.1f}%)")
        
        # Print segment length statistics
        segment_lengths = [len(seg) for seg in all_segments]
        print(f"\nSegment length statistics:")
        print(f"  Mean: {np.mean(segment_lengths):.1f} points")
        print(f"  Min: {np.min(segment_lengths)} points")
        print(f"  Max: {np.max(segment_lengths)} points")
        print(f"  Median: {np.median(segment_lengths):.1f} points")
        
        return all_segments, all_labels, all_tic_ids
    
    def create_train_val_test_splits(
        self,
        flux_segments: List[np.ndarray],
        labels: List[int],
        tic_ids: List[str],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        max_len: int = 4320,
        save_dir: str = 'data/processed',
        seed: int = 42
    ):
        """
        Create and save train/val/test splits.
        
        Args:
            flux_segments: List of flux arrays
            labels: List of binary label integers (0 or 1)
            tic_ids: List of TIC IDs
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            max_len: Maximum sequence length (pad/truncate)
            save_dir: Directory to save processed data
            seed: Random seed
        """
        np.random.seed(seed)
        
        # Pad/truncate all segments to same length
        def pad_segment(seg, length):
            if len(seg) >= length:
                return seg[:length]
            else:
                return np.concatenate([seg, np.zeros(length - len(seg))])
        
        print(f"\nPadding/truncating segments to length {max_len}...")
        flux_array = np.array([pad_segment(seg, max_len) for seg in flux_segments])
        labels_array = np.array(labels)
        tic_ids_array = np.array(tic_ids)
        
        # Shuffle
        indices = np.arange(len(flux_array))
        np.random.shuffle(indices)
        
        # Split
        n_train = int(len(indices) * train_ratio)
        n_val = int(len(indices) * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save splits
        np.savez(
            save_path / 'train_data.npz',
            flux=flux_array[train_idx],
            labels=labels_array[train_idx],
            tic_ids=tic_ids_array[train_idx]
        )
        
        np.savez(
            save_path / 'val_data.npz',
            flux=flux_array[val_idx],
            labels=labels_array[val_idx],
            tic_ids=tic_ids_array[val_idx]
        )
        
        np.savez(
            save_path / 'test_data.npz',
            flux=flux_array[test_idx],
            labels=labels_array[test_idx],
            tic_ids=tic_ids_array[test_idx]
        )
        
        print(f"\nSaved processed data to {save_path}")
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Val: {len(val_idx)} samples")
        print(f"  Test: {len(test_idx)} samples")
        
        # Print class distribution per split
        for split_name, split_idx in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
            split_labels = labels_array[split_idx]
            n_not_planet = np.sum(split_labels == 0)
            n_planet = np.sum(split_labels == 1)
            print(f"\n  {split_name} distribution:")
            print(f"    not_planet: {n_not_planet} ({100*n_not_planet/len(split_labels):.1f}%)")
            print(f"    planet: {n_planet} ({100*n_planet/len(split_labels):.1f}%)")
        
        # Save label mapping
        label_names = self.get_label_names()
        label_info = {
            'label_map': self.label_map,
            'label_names': label_names,
            'n_classes': 2
        }
        
        import json
        with open(save_path / 'label_info.json', 'w') as f:
            json.dump(label_info, f, indent=4)
        
        print(f"\nSaved label information to {save_path / 'label_info.json'}")


def main():
    """Main function to load and process manifest data."""
    
    # Configuration
    manifest_path = 'data/manifest.csv'
    data_root = 'data'
    
    # Preprocessing config (will be auto-adjusted)
    preprocessor_config = {
        'sigma_threshold': 3.0,
        'rolling_window': 50,
        'savgol_window': 101,
        'savgol_poly': 3,
        'max_gap_days': 2.0,
        'segment_duration_days': 90.0,  # Will be auto-adjusted
        'cadence_minutes': 30.0
    }
    
    # Initialize loader
    loader = ManifestDataLoader(
        manifest_path=manifest_path,
        data_root=data_root,
        preprocessor_config=preprocessor_config
    )
    
    # Process all light curves (with auto-adjustment of segment duration)
    flux_segments, labels, tic_ids = loader.process_all_light_curves(
        segment=True,
        overlap=0.0,
        max_curves=None,  # Process all curves
        auto_adjust_segment_duration=True  # Automatically adjust based on data
    )
    
    if len(flux_segments) == 0:
        print("\nERROR: No segments were created!")
        print("Please check your light curve files and preprocessing settings.")
        return
    
    # Create splits and save
    loader.create_train_val_test_splits(
        flux_segments=flux_segments,
        labels=labels,
        tic_ids=tic_ids,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        max_len=4320,
        save_dir='data/processed',
        seed=42
    )
    
    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)
    print("\nBinary classification setup:")
    print("  Class 0: not_planet (EB, BEB, star, etc.)")
    print("  Class 1: planet")
    print("\nYou can now train the model with:")
    print("  python src/main.py --config configs/config.yaml")


if __name__ == "__main__":
    main()
