"""
Hyperparameter Tuning with Optuna for Light Curve Transformer
OPTIMIZED FOR GOOGLE COLAB GPU
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
import gc

# GPU optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Auto-tune kernels
    torch.backends.cuda.matmul.allow_tf32 = True  # Faster matmul
    torch.backends.cudnn.allow_tf32 = True

from model import create_model
from train import Trainer, LightCurveDataset, collate_fn
from utils import set_seed, load_config


class OptunaTrainerGPU:
    """GPU-optimized trainer for Optuna on Colab."""
    
    def __init__(
        self,
        base_config: Dict,
        train_data: Dict,
        val_data: Dict,
        n_trials: int = 50,  # More trials since GPU is faster
        study_name: str = "lightcurve_optimization_gpu"
    ):
        """
        Initialize Optuna trainer for GPU.
        
        Args:
            base_config: Base configuration dictionary
            train_data: Training data dictionary with 'flux', 'labels'
            val_data: Validation data dictionary
            n_trials: Number of optimization trials (50 recommended for GPU)
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
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            self.device = 'cpu'
            print("⚠️  GPU not available, using CPU (will be slower)")
        
        # Create results directory
        self.results_dir = Path('optuna_results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nDataset info:")
        print(f"  Train: {len(train_data['flux'])} samples")
        print(f"  Val: {len(val_data['flux'])} samples")
        
    def create_model_config(self, trial: Trial) -> Dict:
        """Create model configuration from Optuna trial."""
        # Larger search space for GPU
        d_model = trial.suggest_categorical('d_model', [64, 128, 256])
        n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
        n_layers = trial.suggest_int('n_layers', 2, 4)
        
        # Ensure d_model is divisible by n_heads
        while d_model % n_heads != 0:
            n_heads = 4 if n_heads == 8 else 8
        
        d_ff = trial.suggest_categorical('d_ff', [256, 512, 1024])
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        
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
        
        label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.2)
        gradient_clip = trial.suggest_float('gradient_clip', 0.5, 2.0)
        
        training_config = {
            'batch_size': batch_size,
            'num_epochs': 15,  
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'optimizer': 'adamw',
            'scheduler_patience': 5,
            'early_stopping_patience': 8,
            'use_amp': True,  # Mixed precision for faster GPU training
            'num_workers': 2,  # Colab can handle 2 workers
            'seed': 42,
            'warmup_epochs': 3,
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
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Set seed
        set_seed(42 + trial.number)  # Different seed per trial
        
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
        
        try:
            # Create datasets
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
            
            # Create data loaders (pin_memory for faster GPU transfer)
            train_loader = DataLoader(
                train_dataset,
                batch_size=training_config['batch_size'],
                shuffle=True,
                num_workers=training_config['num_workers'],
                pin_memory=True if torch.cuda.is_available() else False,
                collate_fn=collate_fn,
                persistent_workers=True if training_config['num_workers'] > 0 else False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=training_config['batch_size'],
                shuffle=False,
                num_workers=training_config['num_workers'],
                pin_memory=True if torch.cuda.is_available() else False,
                collate_fn=collate_fn,
                persistent_workers=True if training_config['num_workers'] > 0 else False
            )
            
            # Create model
            model = create_model(model_config)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {n_params:,}")
            
            # Prune if model is too large for GPU
            if torch.cuda.is_available():
                gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if n_params > 10_000_000 and gpu_mem_gb < 16:  # 10M limit for <16GB GPU
                    print(f"Model too large for GPU memory, pruning trial")
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
            best_val_acc = 0.0
            patience_counter = 0
            
            for epoch in range(training_config['num_epochs']):
                train_loss, train_acc = trainer.train_epoch()
                val_loss, val_acc = trainer.validate()
                
                # Update best validation metrics
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Report for pruning
                trial.report(val_loss, epoch)
                
                # Check if should prune
                if trial.should_prune():
                    print(f"Trial pruned at epoch {epoch+1}")
                    raise optuna.TrialPruned()
                
                # Print progress
                print(f"Epoch {epoch+1:2d}/{training_config['num_epochs']}: "
                      f"Loss={train_loss:.4f}/{val_loss:.4f}, "
                      f"Acc={train_acc:.1f}%/{val_acc:.1f}%")
                
                # Early stopping within trial
                if patience_counter >= 5:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                # Clear cache periodically
                if epoch % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Log results
            trial.set_user_attr('best_val_loss', best_val_loss)
            trial.set_user_attr('best_val_acc', best_val_acc)
            trial.set_user_attr('n_parameters', n_params)
            
            # Cleanup
            del model, trainer, train_loader, val_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            return best_val_loss
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️  GPU OOM, pruning trial")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                raise optuna.TrialPruned()
            else:
                raise e
    
    def run_optimization(self):
        """Run Optuna optimization."""
        print(f"\n{'='*70}")
        print(f"Starting GPU-Optimized Optuna: {self.n_trials} trials")
        print(f"{'='*70}\n")
        
        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            direction='minimize',
            sampler=TPESampler(seed=42, n_startup_trials=10),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True,
            catch=(Exception,)
        )
        
        # Print results
        self._print_results(study)
        self._save_results(study)
        
        return study
    
    def _print_results(self, study):
        """Print optimization results."""
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE")
        print("="*70)
        
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        
        print(f"\nFinished: {len(study.trials)} trials")
        print(f"  Completed: {len(completed)}")
        print(f"  Pruned: {len(pruned)}")
        
        if len(completed) == 0:
            print("\n❌ No trials completed!")
            return
        
        trial = study.best_trial
        print(f"\n🏆 Best Trial #{trial.number}:")
        print(f"  Val Loss: {trial.value:.4f}")
        print(f"  Val Acc: {trial.user_attrs.get('best_val_acc', 0):.2f}%")
        print(f"\n  Parameters:")
        for key, value in trial.params.items():
            print(f"    {key:20s}: {value}")
    
    def _save_results(self, study):
        """Save optimization results."""
        if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) == 0:
            return
        
        trial = study.best_trial
        
        # Save JSON results
        results = {
            'best_params': trial.params,
            'best_value': trial.value,
            'best_val_acc': trial.user_attrs.get('best_val_acc', 0),
            'best_trial_number': trial.number,
            'n_trials': len(study.trials),
            'n_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'study_name': self.study_name
        }
        
        with open(self.results_dir / 'optimization_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save optimized config
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
        
        with open(self.results_dir / 'optimized_config.yaml', 'w') as f:
            yaml.dump(optimized_config, f, default_flow_style=False)
        
        print(f"\n✓ Results saved to {self.results_dir}/")


def main():
    """Main function for Colab."""
    
    # Load configuration
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
    
    # Create optimizer
    optimizer = OptunaTrainerGPU(
        base_config=base_config,
        train_data=train_data,
        val_data=val_data,
        n_trials=50,  # 50 trials practical on GPU
        study_name='lightcurve_colab_gpu_optimization'
    )
    
    # Run optimization
    study = optimizer.run_optimization()
    
    print("\n" + "="*70)
    print("✓ Training complete! Download results:")
    print("  - optuna_results/optimized_config.yaml")
    print("  - optuna_results/optimization_results.json")
    print("="*70)


if __name__ == "__main__":
    main()
