"""
Enhanced Training Module with Warmup, Label Smoothing, and Class Balancing
Optimized for small datasets with CPU training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
from tqdm import tqdm
import json

from model import LightCurveTransformer


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss to prevent overconfidence."""
    
    def __init__(self, n_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.n_classes = n_classes
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class WarmupScheduler:
    """Linear warmup followed by cosine annealing."""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
    def step(self):
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class LightCurveDataset(Dataset):
    """PyTorch dataset for light curve data."""
    
    def __init__(
        self, 
        flux_data: np.ndarray, 
        labels: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        max_len: int = 4320
    ):
        self.flux_data = flux_data
        self.labels = labels
        self.timestamps = timestamps
        self.max_len = max_len
        
        # Pad or truncate sequences
        self.flux_data = self._process_sequences(self.flux_data)
        if self.timestamps is not None:
            self.timestamps = self._process_sequences(self.timestamps)
    
    def _process_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """Pad or truncate sequences to max_len."""
        processed = []
        for seq in sequences:
            if len(seq) > self.max_len:
                seq = seq[:self.max_len]
            elif len(seq) < self.max_len:
                pad_len = self.max_len - len(seq)
                seq = np.concatenate([seq, np.zeros(pad_len)])
            processed.append(seq)
        return np.array(processed)
    
    def __len__(self) -> int:
        return len(self.flux_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        flux = torch.FloatTensor(self.flux_data[idx]).unsqueeze(-1)
        label = torch.LongTensor([self.labels[idx]])[0]
        
        if self.timestamps is not None:
            timestamp = torch.FloatTensor(self.timestamps[idx])
            return flux, label, timestamp
        else:
            return flux, label, None


class Trainer:
    """Enhanced trainer with warmup, label smoothing, and class balancing."""
    
    def __init__(
        self,
        model: LightCurveTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Calculate class weights if dataset is imbalanced
        self.class_weights = self._calculate_class_weights()
        print(f"Class weights: {self.class_weights}")
        
        # Loss function with class weights and optional label smoothing
        label_smoothing = config.get('label_smoothing', 0.0)
        if label_smoothing > 0:
            print(f"Using label smoothing: {label_smoothing}")
            self.criterion = LabelSmoothingLoss(
                n_classes=config.get('n_classes', 2),
                smoothing=label_smoothing
            )
        else:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(device))
        
        # Optimizer
        lr = config.get('learning_rate', 1e-4)
        weight_decay = config.get('weight_decay', 0.01)
        
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Warmup scheduler
        num_epochs = config.get('num_epochs', 100)
        warmup_epochs = config.get('warmup_epochs', 5)
        
        if warmup_epochs > 0:
            print(f"Using warmup scheduler: {warmup_epochs} warmup epochs")
            self.scheduler = WarmupScheduler(
                self.optimizer,
                warmup_epochs=warmup_epochs,
                total_epochs=num_epochs,
                base_lr=lr,
                min_lr=lr * 0.01
            )
        else:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=config.get('scheduler_patience', 5)
            )
        
        # Gradient clipping
        self.gradient_clip = config.get('gradient_clip', None)
        if self.gradient_clip:
            print(f"Using gradient clipping: {self.gradient_clip}")
        
        # Early stopping
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights based on inverse frequency."""
        all_labels = []
        for batch in self.train_loader:
            if len(batch) == 3:
                _, labels, _ = batch
            else:
                _, labels = batch
            all_labels.extend(labels.numpy())
        
        all_labels = np.array(all_labels)
        unique, counts = np.unique(all_labels, return_counts=True)
        
        # Inverse frequency weighting
        total = len(all_labels)
        weights = total / (len(unique) * counts)
        
        # Normalize weights to sum to number of classes
        weights = weights / weights.sum() * len(unique)
        
        weight_tensor = torch.FloatTensor(weights)
        
        print(f"\nClass distribution:")
        for cls, count, weight in zip(unique, counts, weights):
            print(f"  Class {cls}: {count} samples ({100*count/total:.1f}%), weight: {weight:.3f}")
        
        return weight_tensor
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 3:
                flux, labels, timestamps = batch
                timestamps = timestamps.to(self.device) if timestamps is not None else None
            else:
                flux, labels = batch
                timestamps = None
            
            flux = flux.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(flux, timestamps)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)
            for batch in pbar:
                if len(batch) == 3:
                    flux, labels, timestamps = batch
                    timestamps = timestamps.to(self.device) if timestamps is not None else None
                else:
                    flux, labels = batch
                    timestamps = None
                
                flux = flux.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(flux, timestamps)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix({
                    'loss': total_loss / (len(pbar)),
                    'acc': 100. * correct / total
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def train(self, num_epochs: int):
        """Main training loop."""
        print(f"\n{'='*70}")
        print(f"Training Configuration")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"Learning rate: {self.config.get('learning_rate')}")
        print(f"Batch size: {self.config.get('batch_size')}")
        print(f"Weight decay: {self.config.get('weight_decay')}")
        print(f"Dropout: {self.model.dropout.p if hasattr(self.model, 'dropout') else 'N/A'}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 70)
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # Update learning rate
            if isinstance(self.scheduler, WarmupScheduler):
                current_lr = self.scheduler.step()
            else:
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Print summary
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"LR:         {current_lr:.2e}")
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                print(f"✓ Validation improved by {improvement:.4f}")
            else:
                self.epochs_without_improvement += 1
                print(f"✗ No improvement for {self.epochs_without_improvement} epochs")
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\n{'='*70}")
                print(f"Early stopping triggered after {epoch + 1} epochs")
                print(f"{'='*70}")
                break
        
        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training completed in {elapsed_time / 60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")
        
        # Save training history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    if len(batch[0]) == 3:
        flux, labels, timestamps = zip(*batch)
        flux = torch.stack(flux)
        labels = torch.stack(labels)
        if timestamps[0] is not None:
            timestamps = torch.stack(timestamps)
        else:
            timestamps = None
        return flux, labels, timestamps
    else:
        flux, labels = zip(*batch)
        return torch.stack(flux), torch.stack(labels)