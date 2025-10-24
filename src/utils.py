"""
Utility Functions for Light Curve Analysis

Includes data loading helpers, visualization tools, and inference pipelines.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import yaml
import json
from astropy.io import fits


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save configuration
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def plot_light_curve(
    time: np.ndarray,
    flux: np.ndarray,
    title: str = "Light Curve",
    xlabel: str = "Time (BTJD)",
    ylabel: str = "Normalized Flux",
    save_path: Optional[str] = None
):
    """
    Plot a light curve.
    
    Args:
        time: Time array
        flux: Flux array
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save figure
    """
    plt.figure(figsize=(12, 4))
    plt.plot(time, flux, 'k.', markersize=1, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_multiple_light_curves(
    data_list: List[Tuple[np.ndarray, np.ndarray, str]],
    save_path: Optional[str] = None
):
    """
    Plot multiple light curves in a grid.
    
    Args:
        data_list: List of (time, flux, label) tuples
        save_path: Path to save figure
    """
    n_curves = len(data_list)
    n_cols = min(3, n_curves)
    n_rows = (n_curves + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, (time, flux, label) in enumerate(data_list):
        ax = axes[idx]
        ax.plot(time, flux, 'k.', markersize=0.5, alpha=0.5)
        ax.set_title(label)
        ax.set_xlabel('Time')
        ax.set_ylabel('Flux')
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_curves, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', etc.
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot loss
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def calculate_segment_length(cadence_minutes: float, duration_days: float) -> int:
    """
    Calculate the number of data points in a segment.
    
    Args:
        cadence_minutes: Observation cadence in minutes
        duration_days: Segment duration in days
        
    Returns:
        Number of data points
    """
    points_per_day = (24 * 60) / cadence_minutes
    return int(points_per_day * duration_days)


def pad_or_truncate(sequence: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad or truncate a sequence to target length.
    
    Args:
        sequence: Input sequence
        target_length: Desired length
        
    Returns:
        Processed sequence
    """
    if len(sequence) > target_length:
        return sequence[:target_length]
    elif len(sequence) < target_length:
        pad_length = target_length - len(sequence)
        return np.concatenate([sequence, np.zeros(pad_length)])
    return sequence


class InferencePipeline:
    """Pipeline for making predictions on new light curves."""
    
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: str = None
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Optional path to config file
            device: Device to run inference on
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint.get('config', {})
        
        # Override with external config if provided
        if config_path:
            external_config = load_config(config_path)
            self.config.update(external_config)
        
        # Load model
        from model import create_model
        self.model = create_model(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load preprocessor
        from preprocessing import LightCurvePreprocessor
        self.preprocessor = LightCurvePreprocessor(
            segment_duration_days=self.config.get('segment_duration_days', 90.0),
            cadence_minutes=self.config.get('cadence_minutes', 30.0)
        )
        
        print(f"Loaded model from {model_path}")
        print(f"Device: {self.device}")
    
    def preprocess_file(self, filepath: str) -> List[np.ndarray]:
        """
        Preprocess a light curve file.
        
        Args:
            filepath: Path to light curve file
            
        Returns:
            List of preprocessed segments
        """
        segments, _ = self.preprocessor.preprocess(filepath, segment=True)
        return segments
    
    def predict_segment(
        self,
        flux: np.ndarray,
        timestamps: Optional[np.ndarray] = None
    ) -> Tuple[int, np.ndarray]:
        """
        Predict class for a single segment.
        
        Args:
            flux: Flux array
            timestamps: Optional timestamps
            
        Returns:
            Tuple of (predicted_class, class_probabilities)
        """
        # Pad/truncate to expected length
        max_len = self.config.get('max_len', 4320)
        flux = pad_or_truncate(flux, max_len)
        
        # Convert to tensor
        flux_tensor = torch.FloatTensor(flux).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        flux_tensor = flux_tensor.to(self.device)
        
        timestamp_tensor = None
        if timestamps is not None:
            timestamps = pad_or_truncate(timestamps, max_len)
            timestamp_tensor = torch.FloatTensor(timestamps).unsqueeze(0)  # (1, seq_len)
            timestamp_tensor = timestamp_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(flux_tensor, timestamp_tensor)
            probs = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            class_probs = probs[0].cpu().numpy()
        
        return predicted_class, class_probs
    
    def predict_file(
        self,
        filepath: str,
        aggregate: bool = True
    ) -> Dict:
        """
        Predict classes for all segments in a file.
        
        Args:
            filepath: Path to light curve file
            aggregate: Whether to aggregate predictions across segments
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        segments = self.preprocess_file(filepath)
        
        if len(segments) == 0:
            return {
                'error': 'No valid segments found',
                'file': filepath
            }
        
        # Predict each segment
        predictions = []
        probabilities = []
        
        for segment in segments:
            pred_class, class_probs = self.predict_segment(segment)
            predictions.append(pred_class)
            probabilities.append(class_probs)
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        result = {
            'file': filepath,
            'n_segments': len(segments),
            'segment_predictions': predictions.tolist(),
            'segment_probabilities': probabilities.tolist()
        }
        
        if aggregate:
            # Aggregate by majority vote
            unique, counts = np.unique(predictions, return_counts=True)
            majority_class = unique[np.argmax(counts)]
            
            # Average probabilities
            avg_probs = np.mean(probabilities, axis=0)
            
            result['aggregated_prediction'] = int(majority_class)
            result['aggregated_probabilities'] = avg_probs.tolist()
            result['confidence'] = float(np.max(avg_probs))
        
        return result
    
    def predict_batch(
        self,
        filepaths: List[str],
        save_results: Optional[str] = None
    ) -> List[Dict]:
        """
        Predict classes for multiple files.
        
        Args:
            filepaths: List of file paths
            save_results: Optional path to save results JSON
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for filepath in filepaths:
            print(f"Processing {filepath}...")
            try:
                result = self.predict_file(filepath)
                results.append(result)
            except Exception as e:
                print(f"Error processing {filepath}: {str(e)}")
                results.append({
                    'file': filepath,
                    'error': str(e)
                })
        
        if save_results:
            with open(save_results, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Results saved to {save_results}")
        
        return results


def create_data_splits(
    data_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split data files into train/val/test sets.
    
    Args:
        data_dir: Directory containing data files
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
        
    Returns:
        Tuple of (train_files, val_files, test_files)
    """
    np.random.seed(seed)
    
    # Get all files
    data_path = Path(data_dir)
    files = list(data_path.glob('*.fits')) + list(data_path.glob('*.csv'))
    files = [str(f) for f in files]
    
    # Shuffle
    np.random.shuffle(files)
    
    # Calculate split indices
    n_total = len(files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]
    
    print(f"Total files: {n_total}")
    print(f"Train: {len(train_files)}")
    print(f"Validation: {len(val_files)}")
    print(f"Test: {len(test_files)}")
    
    return train_files, val_files, test_files


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    return {
        'trainable': trainable,
        'total': total,
        'non_trainable': total - trainable
    }


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_kepler_file_info(filepath: str) -> Dict:
    """
    Extract metadata from Kepler FITS file.
    
    Args:
        filepath: Path to FITS file
        
    Returns:
        Dictionary with file metadata
    """
    with fits.open(filepath) as hdul:
        header = hdul[0].header
        
        info = {
            'kepler_id': header.get('KEPLERID', 'Unknown'),
            'object': header.get('OBJECT', 'Unknown'),
            'ra': header.get('RA_OBJ', None),
            'dec': header.get('DEC_OBJ', None),
            'kepmag': header.get('KEPMAG', None),
            'quarter': header.get('QUARTER', None),
            'obsmode': header.get('OBSMODE', 'Unknown'),
            'exposure': header.get('EXPOSURE', None)
        }
    
    return info


if __name__ == "__main__":
    # Example usage
    
    # Load config
    config = {
        'model': {
            'd_model': 128,
            'n_heads': 4,
            'n_layers': 3
        },
        'training': {
            'learning_rate': 1e-4,
            'batch_size': 32
        }
    }
    
    # Save config
    save_config(config, 'configs/config.yaml')
    
    # Load it back
    loaded_config = load_config('configs/config.yaml')
    print("Loaded config:", loaded_config)
    
    # Test inference pipeline
    try:
        pipeline = InferencePipeline('checkpoints/best_model.pth')
        
        # Predict single file
        result = pipeline.predict_file('data/raw/example_lightcurve.fits')
        print("\nPrediction result:")
        print(json.dumps(result, indent=2))
        
    except FileNotFoundError:
        print("Model checkpoint not found. Train a model first.")