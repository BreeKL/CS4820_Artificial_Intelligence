"""
Hyperparameter Tuning with Optuna for Light Curve Transformer
Simplified single-process version for CPU training.
"""

import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import yaml
import json
from typing import Dict
import sys
from tqdm import tqdm

# Limit threading to prevent resource issues
torch.set_num_threads(8)
import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'

from model import create_model
from train import Trainer, LightCurveDataset, collate_fn
from utils import set_seed, load_config


class OptunaTrainer:
    """Simplified single-process trainer for Optuna optimization."""
    
    def __init__(
        self,
        base_config: Dict,
        train_data: Dict,
        val_data: Dict,
        n_trials: int = 20,
        study_name: str = "lightcurve_optimization"
    ):
        """
        Initialize Optuna trainer.
        
        Args:
            base_config: Base configuration dictionary
            train_data: Training data dictionary with 'flux', 'labels'
            val_data: Validation data dictionary
            n_trials: Number of optimization trials
            study_name: Name for the Optuna study
        """
        self.base_config = base_config
        self.train_data = train_data
        self.val_data = val_data
        self.n_trials = n_trials
        self.study_name = study_name
        
        # Setup device
        if torch.cuda.is_available():
            self.device = 'cuda'
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = 'cpu'
            print("Using CPU for training")
        
        # Create results directory
        self.results_dir = Path('optuna_results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nDataset info:")
        print(f"  Train samples: {len(train_data['flux'])}")
        print(f"  Val samples: {len(val_data['flux'])}")
        
    def create_model_config(self, trial: Trial) -> Dict:
        """Create model configuration from Optuna trial."""
        # Suggest hyperparameters
        d_model = trial.suggest_categorical('d_model', [64, 128, 256])
        n_heads = trial.suggest_categorical('n_heads', [4, 8])
        n_layers = trial.suggest_int('n_layers', 2, 4)
        
        # Ensure d_model is divisible by n_heads
        while d_model % n_heads != 0:
            n_heads = 4 if n_heads == 8 else 8
        
        d_ff = trial.suggest_categorical('d_ff', [256, 512, 1024])
        dropout = trial.suggest_float('dropout', 0.1, 0.4)
        
        model_config = {
            'input_dim': 1,
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'd_ff': d_ff,
            'n_classes': 2,
            'dropout': dropout,
            'max_len': self.base_config['model']['max_len'],
            'use_temporal_encoding': True
        }
        
        return model_config
    
    def create_training_config(self, trial: Trial) -> Dict:
        """Create training configuration from Optuna trial."""
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32])
        
        label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.1)
        gradient_clip = trial.suggest_float('gradient_clip', 0.5, 2.0)
        
        training_config = {
            'batch_size': batch_size,
            'num_epochs': 15,  # Reduced for faster trials
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'optimizer': 'adamw',
            'scheduler_patience': 5,
            'early_stopping_patience': 8,
            'use_amp': False,  # Disabled for CPU
            'num_workers': 0,  # Single process
            'seed': 42,
            'warmup_epochs': 2,
            'label_smoothing': label_smoothing,
            'gradient_clip': gradient_clip,
            'checkpoint_dir': str(self.results_dir / f'trial_{trial.number}'),
            'n_classes': 2
        }
        
        return training_config
    
    def objective(self, trial: Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation loss (to be minimized)
        """
        # Set seed
        set_seed(42)
        
        # Create configurations
        model_config = self.create_model_config(trial)
        training_config = self.create_training_config(trial)
        
        print(f"\n{'='*70}")
        print(f"Trial {trial.number}")
        print(f"{'='*70}")
        print(f"Model: d_model={model_config['d_model']}, "
              f"n_heads={model_config['n_heads']}, "
              f"n_layers={model_config['n_layers']}")
        print(f"Training: lr={training_config['learning_rate']:.2e}, "
              f"batch={training_config['batch_size']}, "
              f"wd={training_config['weight_decay']:.2e}")
        
        # Create datasets (no augmentation for speed)
        train_dataset = LightCurveDataset(
            self.train_data['flux'],
            self.train_data['labels'],
            self.train_data.get('timestamps'),
            max_len=model_config['max_len']
        )
        
        val_dataset = LightCurveDataset(
            self.val_data['flux'],
            self.val_data['labels'],
            self.val_data.get('timestamps'),
            max_len=model_config['max_len']
        )
        
        # Create data loaders (single process, no pin_memory)
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_fn
        )
        
        # Create model
        model = create_model(model_config)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
        
        # Prune if model is too large
        if n_params > 3_000_000:
            print(f"Model too large ({n_params:,} > 3M), pruning trial")
            raise optuna.TrialPruned()
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            device=self.device
        )
        
        # Training loop with pruning
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(training_config['num_epochs']):
            try:
                train_loss, train_acc = trainer.train_epoch()
                val_loss, val_acc = trainer.validate()
                
                # Update best validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Report for pruning
                trial.report(val_loss, epoch)
                
                # Check if should prune
                if trial.should_prune():
                    print(f"Trial pruned at epoch {epoch+1}")
                    raise optuna.TrialPruned()
                
                print(f"Epoch {epoch+1}/{training_config['num_epochs']}: "
                      f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                      f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%")
                
                # Early stopping within trial
                if patience_counter >= 5:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                    
            except Exception as e:
                print(f"Error in epoch {epoch+1}: {e}")
                return float('inf')
        
        # Log results
        trial.set_user_attr('best_val_loss', best_val_loss)
        trial.set_user_attr('final_val_acc', val_acc)
        trial.set_user_attr('n_parameters', n_params)
        
        return best_val_loss
    
    def run_optimization(self):
        """Run Optuna optimization."""
        print(f"\n{'='*70}")
        print(f"Starting Optuna Optimization: {self.n_trials} trials")
        print(f"{'='*70}\n")
        
        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            direction='minimize',
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5)
        )
        
        # Run optimization (single process)
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            catch=(Exception,)
        )
        
        # Print results
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        
        print(f"\nNumber of finished trials: {len(study.trials)}")
        
        completed_trials = [t for t in study.trials 
                          if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials 
                        if t.state == optuna.trial.TrialState.PRUNED]
        
        print(f"Completed trials: {len(completed_trials)}")
        print(f"Pruned trials: {len(pruned_trials)}")
        
        if len(completed_trials) == 0:
            print("\n❌ No trials completed successfully!")
            return None
        
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value (val_loss): {trial.value:.4f}")
        print(f"  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        # Save results
        results = {
            'best_params': trial.params,
            'best_value': trial.value,
            'best_trial_number': trial.number,
            'n_trials': len(study.trials),
            'n_completed': len(completed_trials),
            'study_name': self.study_name
        }
        
        results_path = self.results_dir / 'optimization_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to {results_path}")
        
        # Create optimized config
        optimized_config = self.base_config.copy()
        optimized_config['model'].update({
            'd_model': trial.params['d_model'],
            'n_heads': trial.params['n_heads'],
            'n_layers': trial.params['n_layers'],
            'd_ff': trial.params['d_ff'],
            'dropout': trial.params['dropout']
        })
        optimized_config['training'].update({
            'learning_rate': trial.params['learning_rate'],
            'weight_decay': trial.params['weight_decay'],
            'batch_size': trial.params['batch_size'],
            'label_smoothing': trial.params['label_smoothing'],
            'gradient_clip': trial.params['gradient_clip']
        })
        
        config_path = self.results_dir / 'optimized_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(optimized_config, f, default_flow_style=False)
        
        print(f"Optimized config saved to {config_path}")
        
        return study


def main():
    """Main function to run optimization."""
    
    # Load base configuration
    print("Loading configuration...")
    base_config = load_config('configs/config.yaml')
    
    # Load data
    print("Loading data...")
    train_data_file = np.load('data/processed/train_data.npz')
    val_data_file = np.load('data/processed/val_data.npz')
    
    train_data = {
        'flux': train_data_file['flux'],
        'labels': train_data_file['labels'],
        'timestamps': train_data_file.get('timestamps', None)
    }
    
    val_data = {
        'flux': val_data_file['flux'],
        'labels': val_data_file['labels'],
        'timestamps': val_data_file.get('timestamps', None)
    }
    
    print(f"\nTrain samples: {len(train_data['flux'])}")
    print(f"Val samples: {len(val_data['flux'])}")
    
    # Class distribution
    unique, counts = np.unique(train_data['labels'], return_counts=True)
    print("\nClass distribution (train):")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} ({100*count/len(train_data['labels']):.1f}%)")
    
    # Create optimizer
    optimizer = OptunaTrainer(
        base_config=base_config,
        train_data=train_data,
        val_data=val_data,
        n_trials=20,  # Reduced for reasonable CPU runtime
        study_name='lightcurve_optimization_cpu'
    )
    
    # Run optimization
    study = optimizer.run_optimization()
    
    if study:
        print("\n" + "="*70)
        print("To train with optimized config, run:")
        print("  python src/main.py --config optuna_results/optimized_config.yaml")
        print("="*70)


if __name__ == "__main__":
    main()
