"""
Training Module for Light Curve Transformer

Implements training loop with learning rate scheduling, early stopping,
mixed precision training, and checkpoint saving.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import time
from tqdm import tqdm
import json

from model import LightCurveTransformer


class LightCurveDataset(Dataset):
    """PyTorch dataset for light curve data."""
    
    def __init__(
        self, 
        flux_data: np.ndarray, 
        labels: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        max_len: int = 4320
    ):
        """
        Initialize dataset.
        
        Args:
            flux_data: Array of flux segments, shape (n_samples, seq_len)
            labels: Array of labels, shape (n_samples,)
            timestamps: Optional array of timestamps, shape (n_samples, seq_len)
            max_len: Maximum sequence length (pad/truncate)
        """
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
                # Truncate
                seq = seq[:self.max_len]
            elif len(seq) < self.max_len:
                # Pad with zeros
                pad_len = self.max_len - len(seq)
                seq = np.concatenate([seq, np.zeros(pad_len)])
            processed.append(seq)
        return np.array(processed)
    
    def __len__(self) -> int:
        return len(self.flux_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        flux = torch.FloatTensor(self.flux_data[idx]).unsqueeze(-1)  # (seq_len, 1)
        label = torch.LongTensor([self.labels[idx]])[0]
        
        if self.timestamps is not None:
            timestamp = torch.FloatTensor(self.timestamps[idx])
            return flux, label, timestamp
        else:
            return flux, label, None


class Trainer:
    """Trainer class for light curve transformer."""
    
    def __init__(
        self,
        model: LightCurveTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize trainer.
        
        Args:
            model: LightCurveTransformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device to use for training
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        optimizer_name = config.get('optimizer', 'adamw')
        lr = config.get('learning_rate', 1e-4)
        weight_decay = config.get('weight_decay', 0.01)
        
        if optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optim.Adam(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config.get('scheduler_patience', 5),
            verbose=True
        )
        
        # Mixed precision training
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
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
            'val_acc': []
        }
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 3:
                flux, labels, timestamps = batch
                timestamps = timestamps.to(self.device) if timestamps[0] is not None else None
            else:
                flux, labels = batch
                timestamps = None
            
            flux = flux.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(flux, timestamps)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(flux, timestamps)
                loss = self.criterion(outputs, labels)
                loss.backward()
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
        """
        Validate model.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for batch in pbar:
                if len(batch) == 3:
                    flux, labels, timestamps = batch
                    timestamps = timestamps.to(self.device) if timestamps[0] is not None else None
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
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Saved best model with validation loss: {self.best_val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def train(self, num_epochs: int):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1, is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\nTraining completed in {elapsed_time / 60:.2f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
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


if __name__ == "__main__":
    # Example training setup
    from model import create_model
    
    # Configuration
    config = {
        'input_dim': 1,
        'd_model': 128,
        'n_heads': 4,
        'n_layers': 3,
        'd_ff': 512,
        'n_classes': 2,
        'dropout': 0.1,
        'max_len': 4320,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'batch_size': 32,
        'num_epochs': 100,
        'early_stopping_patience': 15,
        'scheduler_patience': 5,
        'use_amp': True,
        'checkpoint_dir': 'checkpoints'
    }
    
    # Create dummy data
    n_samples = 1000
    seq_len = 4320
    
    train_flux = np.random.randn(n_samples, seq_len)
    train_labels = np.random.randint(0, 2, n_samples)
    
    val_flux = np.random.randn(200, seq_len)
    val_labels = np.random.randint(0, 2, 200)
    
    # Create datasets
    train_dataset = LightCurveDataset(train_flux, train_labels, max_len=seq_len)
    val_dataset = LightCurveDataset(val_flux, val_labels, max_len=seq_len)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Create model
    model = create_model(config)
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Train
    trainer.train(config['num_epochs'])