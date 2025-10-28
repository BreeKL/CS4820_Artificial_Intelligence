"""
Evaluation Module for Light Curve Classifier

Implements comprehensive evaluation metrics including accuracy, precision,
recall, F1-score, ROC-AUC, and confusion matrix visualization.
"""

# Set matplotlib backend before importing pyplot (prevents blocking)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import yaml

from model import LightCurveTransformer


class ModelEvaluator:
    """Evaluator for trained light curve classifier."""
    
    def __init__(
        self,
        model: LightCurveTransformer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained LightCurveTransformer model
            device: Device to use for evaluation
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    def predict(
        self,
        data_loader,
        return_probabilities: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for a dataset.
        
        Args:
            data_loader: DataLoader with test data
            return_probabilities: Return class probabilities instead of labels
            
        Returns:
            Tuple of (predictions, true_labels)
        """
        all_predictions = []
        all_labels = []
        all_probs = []
        
        # Disable tqdm for very small datasets
        use_tqdm = len(data_loader) > 10
        iterator = tqdm(data_loader, desc='Predicting', disable=not use_tqdm)
        
        self.model.eval()  # Ensure eval mode
        
        with torch.no_grad():
            for batch in iterator:
                # Handle different batch formats
                if len(batch) == 3:
                    flux, labels, timestamps = batch
                    # Only move to device if timestamps exist and aren't None
                    if timestamps is not None and timestamps[0] is not None:
                        timestamps = timestamps.to(self.device)
                    else:
                        timestamps = None
                else:
                    flux, labels = batch
                    timestamps = None
                
                # Move data to device
                flux = flux.to(self.device)
                
                # Forward pass
                outputs = self.model(flux, timestamps)
                probs = torch.softmax(outputs, dim=1)
                
                # Get predictions (use argmax for efficiency)
                predicted = torch.argmax(outputs, dim=1)
                
                # Move to CPU and convert to numpy in one step
                all_predictions.append(predicted.cpu().numpy())
                all_labels.append(labels.numpy())
                all_probs.append(probs.cpu().numpy())
        
        # Concatenate all batches at once (faster than extend)
        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)
        probs = np.concatenate(all_probs)
        
        if return_probabilities:
            return probs, labels
        else:
            return predictions, labels
    
    def calculate_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            predictions: Predicted class labels
            labels: True class labels
            probabilities: Class probabilities (for ROC-AUC)
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(labels, predictions)
        
        # For binary classification
        if len(np.unique(labels)) == 2:
            metrics['precision'] = precision_score(labels, predictions, zero_division=0)
            metrics['recall'] = recall_score(labels, predictions, zero_division=0)
            metrics['f1_score'] = f1_score(labels, predictions, zero_division=0)
            
            if probabilities is not None:
                # ROC-AUC for positive class
                try:
                    metrics['roc_auc'] = roc_auc_score(labels, probabilities[:, 1])
                except ValueError:
                    metrics['roc_auc'] = 0.0
        else:
            # Multi-class metrics
            metrics['precision_macro'] = precision_score(labels, predictions, average='macro', zero_division=0)
            metrics['recall_macro'] = recall_score(labels, predictions, average='macro', zero_division=0)
            metrics['f1_score_macro'] = f1_score(labels, predictions, average='macro', zero_division=0)
            
            metrics['precision_weighted'] = precision_score(labels, predictions, average='weighted', zero_division=0)
            metrics['recall_weighted'] = recall_score(labels, predictions, average='weighted', zero_division=0)
            metrics['f1_score_weighted'] = f1_score(labels, predictions, average='weighted', zero_division=0)
            
            if probabilities is not None:
                try:
                    metrics['roc_auc_ovr'] = roc_auc_score(
                        labels, probabilities, multi_class='ovr'
                    )
                except ValueError:
                    metrics['roc_auc_ovr'] = 0.0
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.
        
        Args:
            predictions: Predicted class labels
            labels: True class labels
            class_names: Names of classes for labels
            save_path: Path to save figure
        """
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names or range(len(cm)),
            yticklabels=class_names or range(len(cm))
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to {save_path}")
            plt.close()  # Close figure to free memory
        else:
            plt.show()
    
    def plot_roc_curve(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot ROC curve for binary classification.
        
        Args:
            labels: True class labels
            probabilities: Class probabilities
            save_path: Path to save figure
        """
        try:
            fpr, tpr, thresholds = roc_curve(labels, probabilities[:, 1])
            auc = roc_auc_score(labels, probabilities[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved ROC curve to {save_path}")
                plt.close()  # Close figure to free memory
            else:
                plt.show()
        except ValueError as e:
            print(f"Could not plot ROC curve: {e}")
    
    def print_classification_report(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        class_names: Optional[List[str]] = None
    ):
        """
        Print detailed classification report.
        
        Args:
            predictions: Predicted class labels
            labels: True class labels
            class_names: Names of classes
        """
        print("\nClassification Report:")
        print("=" * 60)
        report = classification_report(
            labels, 
            predictions, 
            target_names=class_names,
            digits=4,
            zero_division=0
        )
        print(report)
    
    def evaluate(
        self,
        data_loader,
        class_names: Optional[List[str]] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Complete evaluation pipeline.
        
        Args:
            data_loader: DataLoader with test data
            class_names: Names of classes
            save_dir: Directory to save plots
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*70)
        print("STARTING EVALUATION")
        print("="*70)
        
        # Get predictions
        probabilities, labels = self.predict(data_loader, return_probabilities=True)
        predictions = np.argmax(probabilities, axis=1)
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, labels, probabilities)
        
        # Print metrics
        print("\nEvaluation Metrics:")
        print("=" * 60)
        for metric_name, value in metrics.items():
            print(f"{metric_name:20s}: {value:.4f}")
        
        # Print classification report
        self.print_classification_report(predictions, labels, class_names)
        
        # Create save directory
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nSaving visualizations to {save_dir}")
        
        # Plot confusion matrix
        cm_path = save_dir / 'confusion_matrix.png' if save_dir else None
        self.plot_confusion_matrix(predictions, labels, class_names, cm_path)
        
        # Plot ROC curve for binary classification
        if len(np.unique(labels)) == 2:
            roc_path = save_dir / 'roc_curve.png' if save_dir else None
            self.plot_roc_curve(labels, probabilities, roc_path)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        
        return metrics
    
    def visualize_attention(
        self,
        flux: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        layer_idx: int = -1,
        head_idx: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Visualize attention weights for a single sample.
        
        Args:
            flux: Input flux tensor (1, seq_len, 1)
            timestamps: Optional timestamps (1, seq_len)
            layer_idx: Which layer to visualize (-1 for last)
            head_idx: Which attention head to visualize
            save_path: Path to save figure
        """
        self.model.eval()
        
        with torch.no_grad():
            flux = flux.to(self.device)
            if timestamps is not None:
                timestamps = timestamps.to(self.device)
            
            # Forward pass with attention tracking
            _ = self.model(flux, timestamps, return_attention=True)
            attention_weights = self.model.get_attention_weights()
        
        if len(attention_weights) == 0:
            print("No attention weights captured")
            return
        
        # Get attention for specific layer and head
        attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot flux
        flux_data = flux[0, :, 0].cpu().numpy()
        ax1.plot(flux_data, linewidth=0.5)
        ax1.set_title('Input Light Curve')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Normalized Flux')
        ax1.grid(alpha=0.3)
        
        # Plot attention weights
        im = ax2.imshow(attn, aspect='auto', cmap='viridis', interpolation='nearest')
        ax2.set_title(f'Attention Weights (Layer {layer_idx}, Head {head_idx})')
        ax2.set_xlabel('Key Position')
        ax2.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention visualization to {save_path}")
            plt.close()
        else:
            plt.show()


def load_model_for_evaluation(checkpoint_path: str, device: str = None) -> LightCurveTransformer:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint_path = Path(checkpoint_path)
    
    # Load full config from yaml file in checkpoint directory
    config_file = checkpoint_path.parent / 'config.yaml'
    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found at {config_file}. "
            f"Make sure you're pointing to the correct checkpoint directory."
        )
    
    with open(config_file, 'r') as f:
        full_config = yaml.safe_load(f)
        model_config = full_config['model']
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create and load model
    from model import create_model
    model = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"Model config: d_model={model_config['d_model']}, "
          f"n_layers={model_config['n_layers']}, "
          f"n_heads={model_config['n_heads']}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    return model


if __name__ == "__main__":
    # Example evaluation
    from torch.utils.data import DataLoader
    from train import LightCurveDataset, collate_fn
    
    # Load model
    model = load_model_for_evaluation('checkpoints/best_model.pth')
    
    # Load test data
    test_data = np.load('data/processed/test_data.npz')
    test_flux = test_data['flux']
    test_labels = test_data['labels']
    
    test_dataset = LightCurveDataset(test_flux, test_labels)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Evaluate
    evaluator = ModelEvaluator(model)
    metrics = evaluator.evaluate(
        test_loader,
        class_names=['Non-Transit', 'Transit'],
        save_dir='evaluation_results'
    )
    
    print(f"\nFinal Test Accuracy: {metrics['accuracy']:.2%}")