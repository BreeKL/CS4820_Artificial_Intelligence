"""
Parallel Balanced Dataset Preparation (3-5x faster than sequential version)

Uses multiprocessing to parallelize I/O-heavy preprocessing operations.

CLI Arguments:
    --manifest PATH        Path to manifest.csv (default: data/manifest.csv)
    --data-root PATH       Root directory containing light curve CSVs (default: .)
    --n-planets INT        Number of planet light curves to sample (default: 1500)
    --n-non-planets INT    Number of non-planet light curves to sample (default: 1500)
    --test-only            Create test_data.npz only (no train/val splits)
    --processes INT        Number of parallel processes (default: auto)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

sys.path.append('src')
from preprocessing import LightCurvePreprocessor


def process_single_light_curve_worker(args):
    """
    Worker function to process a single light curve.
    
    This function is designed to be called by multiprocessing Pool.
    Each argument is passed as a tuple to avoid pickling issues.
    
    Args:
        args: Tuple of (row_dict, data_root, preprocessor_config, segment, overlap)
        
    Returns:
        Dict with processed segments or None if failed
    """
    row_dict, data_root, preprocessor_config, segment, overlap = args
    
    try:
        # Create preprocessor in worker process
        preprocessor = LightCurvePreprocessor(**preprocessor_config)
        
        # Load light curve
        curve_path = Path(data_root) / row_dict['curve_path']
        df = pd.read_csv(curve_path)
        df.columns = df.columns.str.lower().str.strip()
        
        if 'time' not in df.columns or 'flux' not in df.columns:
            return None
        
        df = df[['time', 'flux']].dropna().reset_index(drop=True)
        
        if len(df) < 100:
            return None
        
        # Preprocessing steps
        df = preprocessor.sigma_clip(df)
        if len(df) < 100:
            return None
        
        df = preprocessor.remove_trend(df)
        df = preprocessor.interpolate_gaps(df)
        flux_normalized = preprocessor.normalize(df['flux'].values)
        
        # Binary label
        binary_label = 1 if row_dict['label'] == 'planet' else 0
        
        # Segmentation
        if segment:
            df['flux'] = flux_normalized
            segments = preprocessor.segment_light_curve(df, overlap=overlap)
            
            if len(segments) == 0:
                segments = [flux_normalized]
        else:
            segments = [flux_normalized]
        
        return {
            'segments': segments,
            'label': binary_label,
            'tic_id': str(row_dict['tic_id']),
            'original_label': row_dict['label'],
            'success': True
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'tic_id': str(row_dict.get('tic_id', 'unknown'))
        }


class ParallelBalancedDatasetCreator:
    """Create balanced dataset with parallel preprocessing."""
    
    def __init__(
        self,
        manifest_path: str,
        data_root: str = 'data',
        n_planet: int = 1500,
        n_non_planet: int = 1500,
        preprocessor_config: Dict = None,
        n_processes: int = None
    ):
        """
        Initialize parallel dataset creator.
        
        Args:
            manifest_path: Path to manifest.csv
            data_root: Root directory containing data
            n_planet: Number of planet light curves
            n_non_planet: Number of non-planet light curves
            preprocessor_config: Preprocessing configuration
            n_processes: Number of processes (default: min(8, cpu_count()))
        """
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        self.n_planet = n_planet
        self.n_non_planet = n_non_planet
        
        # Load manifest
        print(f"Loading manifest from {manifest_path}...")
        self.manifest = pd.read_csv(manifest_path)
        print(f"Total light curves in manifest: {len(self.manifest)}")
        
        # Preprocessing config
        if preprocessor_config is None:
            preprocessor_config = {
                'sigma_threshold': 3.0,
                'rolling_window': 50,
                'savgol_window': 101,
                'savgol_poly': 3,
                'max_gap_days': 2.0,
                'segment_duration_days': 27.0,
                'cadence_minutes': 30.0
            }
        self.preprocessor_config = preprocessor_config
        
        # Number of processes
        if n_processes is None:
            # Use at most 8 processes, leave some CPU for system
            n_processes = min(8, max(1, cpu_count() - 2))
        self.n_processes = n_processes
        
        print(f"Using {self.n_processes} processes for parallel preprocessing")
        
        # Analyze dataset
        self.analyze_manifest()
    
    def analyze_manifest(self):
        """Analyze the manifest and print statistics."""
        print("\n" + "="*70)
        print("MANIFEST ANALYSIS")
        print("="*70)
        
        label_counts = self.manifest['label'].value_counts()
        print("\nLabel distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")
        
        planet_count = len(self.manifest[self.manifest['label'] == 'planet'])
        non_planet_count = len(self.manifest[self.manifest['label'] != 'planet'])
        
        print(f"\nBinary classification:")
        print(f"  Planet: {planet_count}")
        print(f"  Non-planet: {non_planet_count}")
        
        print(f"\nTarget counts:")
        print(f"  Planet needed: {self.n_planet}")
        print(f"  Non-planet needed: {self.n_non_planet}")
        
        if planet_count < self.n_planet:
            print(f"\n⚠️  WARNING: Only {planet_count} planet samples available")
        if non_planet_count < self.n_non_planet:
            print(f"⚠️  WARNING: Only {non_planet_count} non-planet samples available")
    
    def select_balanced_subset(self, seed: int = 42) -> pd.DataFrame:
        """Select balanced subset of light curves."""
        np.random.seed(seed)
        
        planet_df = self.manifest[self.manifest['label'] == 'planet'].copy()
        non_planet_df = self.manifest[self.manifest['label'] != 'planet'].copy()
        
        n_planet_sample = min(self.n_planet, len(planet_df))
        n_non_planet_sample = min(self.n_non_planet, len(non_planet_df))
        
        print(f"\nSampling:")
        print(f"  Planet: {n_planet_sample} / {len(planet_df)}")
        print(f"  Non-planet: {n_non_planet_sample} / {len(non_planet_df)}")
        
        planet_sample = planet_df.sample(n=n_planet_sample, random_state=seed)
        non_planet_sample = non_planet_df.sample(n=n_non_planet_sample, random_state=seed)
        
        balanced_df = pd.concat([planet_sample, non_planet_sample], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        
        print(f"\nBalanced dataset: {len(balanced_df)} total light curves")
        
        return balanced_df
    
    def process_light_curves_parallel(
        self,
        selected_df: pd.DataFrame,
        segment: bool = True,
        overlap: float = 0.0
    ) -> Tuple[List[np.ndarray], List[int], List[str], List[str]]:
        """
        Process light curves in parallel using multiprocessing.
        
        Args:
            selected_df: DataFrame with selected light curves
            segment: Whether to segment light curves
            overlap: Overlap fraction for segmentation
            
        Returns:
            Tuple of (flux_segments, binary_labels, tic_ids, original_labels)
        """
        n_curves = len(selected_df)
        
        print(f"\n{'='*70}")
        print(f"PARALLEL PROCESSING: {n_curves} light curves with {self.n_processes} processes")
        print("="*70)
        
        # Prepare arguments for worker processes
        # Convert DataFrame rows to dicts to avoid pickling issues
        args_list = [
            (
                row.to_dict(),
                str(self.data_root),
                self.preprocessor_config,
                segment,
                overlap
            )
            for _, row in selected_df.iterrows()
        ]
        
        # Process in parallel with progress bar
        print(f"\nProcessing light curves...")
        with Pool(processes=self.n_processes) as pool:
            results = list(tqdm(
                pool.imap(process_single_light_curve_worker, args_list),
                total=len(args_list),
                desc="Processing",
                unit="files"
            ))
        
        # Collect results
        all_segments = []
        all_labels = []
        all_tic_ids = []
        all_original_labels = []
        
        successful = 0
        failed = 0
        
        for result in results:
            if result and result.get('success', False):
                successful += 1
                for segment in result['segments']:
                    all_segments.append(segment)
                    all_labels.append(result['label'])
                    all_tic_ids.append(result['tic_id'])
                    all_original_labels.append(result['original_label'])
            else:
                failed += 1
        
        print(f"\n{'='*70}")
        print("PROCESSING COMPLETE")
        print("="*70)
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total segments created: {len(all_segments)}")
        
        # Statistics
        if len(all_segments) > 0:
            segment_lengths = [len(seg) for seg in all_segments]
            print(f"\nSegment length statistics:")
            print(f"  Mean: {np.mean(segment_lengths):.1f} points")
            print(f"  Median: {np.median(segment_lengths):.1f} points")
            print(f"  Min: {np.min(segment_lengths)} points")
            print(f"  Max: {np.max(segment_lengths)} points")
            
            unique, counts = np.unique(all_labels, return_counts=True)
            print(f"\nBinary label distribution (segments):")
            for label_idx, count in zip(unique, counts):
                label_name = 'planet' if label_idx == 1 else 'non-planet'
                print(f"  {label_name}: {count} ({100*count/len(all_labels):.1f}%)")
        
        return all_segments, all_labels, all_tic_ids, all_original_labels
    
    def create_splits(
        self,
        flux_segments: List[np.ndarray],
        labels: List[int],
        tic_ids: List[str],
        original_labels: List[str],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        max_len: int = 4320,
        save_dir: str = 'data/processed',
        seed: int = 42
    ):
        """Create and save train/val/test splits."""
        np.random.seed(seed)
        
        print(f"\n{'='*70}")
        print("CREATING TRAIN/VAL/TEST SPLITS")
        print("="*70)
        
        def pad_segment(seg, length):
            if len(seg) >= length:
                return seg[:length]
            else:
                return np.concatenate([seg, np.zeros(length - len(seg))])
        
        print(f"Padding/truncating to {max_len} points...")
        flux_array = np.array([pad_segment(seg, max_len) for seg in flux_segments])
        labels_array = np.array(labels)
        tic_ids_array = np.array(tic_ids)
        original_labels_array = np.array(original_labels)
        
        indices = np.arange(len(flux_array))
        np.random.shuffle(indices)
        
        n_train = int(len(indices) * train_ratio)
        n_val = int(len(indices) * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving to {save_path}...")
        
        np.savez(
            save_path / 'train_data.npz',
            flux=flux_array[train_idx],
            labels=labels_array[train_idx],
            tic_ids=tic_ids_array[train_idx],
            original_labels=original_labels_array[train_idx]
        )
        
        np.savez(
            save_path / 'val_data.npz',
            flux=flux_array[val_idx],
            labels=labels_array[val_idx],
            tic_ids=tic_ids_array[val_idx],
            original_labels=original_labels_array[val_idx]
        )
        
        np.savez(
            save_path / 'test_data.npz',
            flux=flux_array[test_idx],
            labels=labels_array[test_idx],
            tic_ids=tic_ids_array[test_idx],
            original_labels=original_labels_array[test_idx]
        )
        
        print(f"✓ Saved splits:")
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Val: {len(val_idx)} samples")
        print(f"  Test: {len(test_idx)} samples")
        
        for split_name, split_idx in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
            split_labels = labels_array[split_idx]
            n_non_planet = np.sum(split_labels == 0)
            n_planet = np.sum(split_labels == 1)
            total = len(split_labels)
            
            print(f"\n  {split_name} distribution:")
            print(f"    non-planet: {n_non_planet} ({100*n_non_planet/total:.1f}%)")
            print(f"    planet: {n_planet} ({100*n_planet/total:.1f}%)")
        
        metadata = {
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'n_test': len(test_idx),
            'max_len': max_len,
            'n_processes_used': self.n_processes,
            'preprocessing_config': self.preprocessor_config
        }
        
        with open(save_path / 'dataset_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\n✓ Saved metadata to {save_path / 'dataset_metadata.json'}")
        print("="*70)


def main():
    """Main function with CLI arguments."""
    import time
    
    parser = argparse.ArgumentParser(description="Parallel light curve preprocessing")
    parser.add_argument('--manifest', default='data/manifest.csv', help='Path to manifest.csv')
    parser.add_argument('--data-root', default='.', help='Root directory for light curves')
    parser.add_argument('--n-planets', type=int, default=1500, help='Number of planet samples')
    parser.add_argument('--n-non-planets', type=int, default=1500, help='Number of non-planet samples')
    parser.add_argument('--test-only', action='store_true', help='Create test_data.npz only (no splits)')
    parser.add_argument('--processes', type=int, default=None, help='Number of processes (auto=default)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print(f"\nConfiguration:")
    print(f"  Manifest: {args.manifest}")
    print(f"  Data root: {args.data_root}")
    print(f"  Planets: {args.n_planets}")
    print(f"  Non-planets: {args.n_non_planets}")
    print(f"  Test only: {args.test_only}")
    
    n_cpus = cpu_count()
    n_processes = args.processes or min(8, max(1, n_cpus - 2))
    
    print(f"\nSystem info:")
    print(f"  CPU cores: {n_cpus}")
    print(f"  Processes: {n_processes}")
    
    # Create dataset creator
    creator = ParallelBalancedDatasetCreator(
        manifest_path=args.manifest,
        data_root=args.data_root,
        n_planet=args.n_planets,
        n_non_planet=args.n_non_planets,
        n_processes=n_processes
    )
    
    # Select balanced subset
    selected_df = creator.select_balanced_subset(seed=42)
    selected_df.to_csv('data/selected_manifest.csv', index=False)
    
    # Process light curves
    flux_segments, labels, tic_ids, original_labels = creator.process_light_curves_parallel(
        selected_df,
        segment=True,
        overlap=0.0
    )
    
    if len(flux_segments) == 0:
        print("\n❌ ERROR: No segments were created!")
        return
    
    save_path = Path('data/processed')
    save_path.mkdir(parents=True, exist_ok=True)
    
    max_len = 4320
    
    if args.test_only:
        # TEST-ONLY MODE: Save everything as test_data.npz
        print(f"\n{'='*70}")
        print("TEST-ONLY MODE: Saving all data as test_data.npz")
        print("="*70)
        
        def pad_segment(seg, length):
            if len(seg) >= length:
                return seg[:length]
            else:
                return np.concatenate([seg, np.zeros(length - len(seg))])
        
        flux_array = np.array([pad_segment(seg, max_len) for seg in flux_segments])
        labels_array = np.array(labels)
        tic_ids_array = np.array(tic_ids)
        original_labels_array = np.array(original_labels)
        
        np.savez(
            save_path / 'test_data.npz',
            flux=flux_array,
            labels=labels_array,
            tic_ids=tic_ids_array,
            original_labels=original_labels_array
        )
        
        metadata = {
            'n_test': len(flux_array),
            'max_len': max_len,
            'n_processes_used': n_processes,
            'preprocessing_config': creator.preprocessor_config,
            'test_only': True
        }
        with open(save_path / 'test_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"✓ Saved {len(flux_array)} test samples to {save_path}/test_data.npz")
        
    else:
        # NORMAL MODE: Create train/val/test splits
        creator.create_splits(
            flux_segments=flux_segments,
            labels=labels,
            tic_ids=tic_ids,
            original_labels=original_labels,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            max_len=max_len,
            save_dir='data/processed',
            seed=42
        )
    
    elapsed_time = time.time() - start_time
    print(f"\n✓ Complete in {elapsed_time/60:.1f} minutes")
    print(f"Speed: {len(selected_df)/elapsed_time:.1f} files/sec")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()