"""
Main Training Script for Light Curve Transformer

Run this script to train the model with configuration from config.yaml
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import sys


from preprocessing import LightCurvePreprocessor, DataAugmenter
from model import create_model
from train import Trainer, LightCurveDataset, collate_fn
from utils import load_config, save_config, set_seed, count_parameters


def prepare_data(config):
    """
    Prepare training, validation, and test datasets.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    print("Preparing datasets...")
    
    # Initialize preprocessor
    preprocessor = LightCurvePreprocessor(
        sigma_threshold=config['preprocessing']['sigma_threshold'],
        rolling_window=config['preprocessing']['rolling_window'],
        savgol_window=config['preprocessing']['savgol_window'],
        savgol_poly=config['preprocessing']['savgol_poly'],
        max_gap_days=config['preprocessing']['max_gap_days'],
        segment_duration_days=config['preprocessing']['segment_duration_days'],
        cadence_minutes=config['preprocessing']['cadence_minutes']
    )
    
    # Load preprocessed data or create from raw files
    processed_dir = Path(config['data']['processed_dir'])
    
    if (processed_dir / 'train_data.npz').exists():
        # Load preprocessed data
        print("Loading preprocessed data...")
        train_data = np.load(processed_dir / 'train_data.npz')
        val_data = np.load(processed_dir / 'val_data.npz')
        test_data = np.load(processed_dir / 'test_data.npz')
        
        train_flux = train_data['flux']
        train_labels = train_data['labels']
        train_timestamps = train_data.get('timestamps', None)
        
        val_flux = val_data['flux']
        val_labels = val_data['labels']
        val_timestamps = val_data.get('timestamps', None)
        
        test_flux = test_data['flux']
        test_labels = test_data['labels']
        test_timestamps = test_data.get('timestamps', None)
        
    else:
        print("Preprocessed data not found!")
        print("Please preprocess your raw data first.")
        print("See notebooks/exploration.ipynb for examples.")
        sys.exit(1)
    
    # Apply data augmentation to training set if enabled
    if config['augmentation']['enabled']:
        print("Applying data augmentation...")
        augmenter = DataAugmenter()
        augmented_flux = []
        augmented_labels = []
        
        for flux, label in zip(train_flux, train_labels):
            # Original sample
            augmented_flux.append(flux)
            augmented_labels.append(label)
            
            # Augmented samples (1 per original)
            aug_flux = augmenter.augment(
                flux,
                methods=config['augmentation']['methods']
            )
            augmented_flux.append(aug_flux)
            augmented_labels.append(label)
        
        train_flux = np.array(augmented_flux)
        train_labels = np.array(augmented_labels)
        print(f"Augmented training set: {len(train_flux)} samples")
    
    # Create datasets
    max_len = config['model']['max_len']
    
    train_dataset = LightCurveDataset(
        train_flux, train_labels, train_timestamps, max_len
    )
    val_dataset = LightCurveDataset(
        val_flux, val_labels, val_timestamps, max_len
    )
    test_dataset = LightCurveDataset(
        test_flux, test_labels, test_timestamps, max_len
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def main(args):
    """Main training function."""
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config['training']['seed'])
    
    # Create output directories
    Path(config['paths']['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['results_dir']).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_config(config, Path(config['paths']['checkpoint_dir']) / 'config.yaml')
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(config)
    
    # Create model
    print("\nCreating model...")
    model = create_model(config['model'])
    
    # Print model info
    param_counts = count_parameters(model)
    print(f"Model parameters:")
    print(f"  Trainable: {param_counts['trainable']:,}")
    print(f"  Total: {param_counts['total']:,}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training']
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    

    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    trainer.train(num_epochs=config['training']['num_epochs'])
    
    # Evaluate on test set
    if not args.skip_test:
        print("\n" + "="*60)
        print("Evaluating on test set...")
        print("="*60)
        
        from evaluate import ModelEvaluator
        
        # Load best model
        best_model_path = Path(config['paths']['checkpoint_dir']) / 'best_model.pth'
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate(
            test_loader,
            class_names=config.get('classes', None),
            save_dir=config['paths']['results_dir']
        )
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)
        print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
        print(f"Test accuracy: {metrics['accuracy']:.4f}")
        print(f"Results saved to: {config['paths']['results_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Light Curve Transformer Model"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../configs/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Skip evaluation on test set after training'
    )
    
    args = parser.parse_args()
    main(args)