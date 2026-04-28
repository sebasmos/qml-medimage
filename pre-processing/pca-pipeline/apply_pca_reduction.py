#!/usr/bin/env python3
"""
Apply PCA Dimensionality Reduction to Medical Imaging Embeddings
=================================================================

Reduces embeddings from 88,064 features to a configurable number of components.

This is Stage 1 of the two stage PCA pipeline:
  Stage 1: 88,064 -> N components (this script)
  Stage 2: N -> 16 components (done later for quantum circuits)

Key Features:
  - Processes a SINGLE specified pkl file (by number)
  - Ultra memory efficient using scipy.sparse.linalg.svds
  - Works even with very limited RAM

Usage:
    # Process data_type1_insurance.pkl (default)
    python apply_pca_reduction.py --n_components 1999
    
    # Process a specific file by number
    python apply_pca_reduction.py --n_components 500 --file_num 2
    
    # Via SLURM:
    sbatch run_pca_reduction.sh 1999 1

Output:
    Creates the PCA reduced pkl file in:
    /orcd/pool/006/lceli_shared/DATASET/data-cleaned-pca-{n_components}/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, Generator
import gc

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import joblib


def get_data_file(input_dir: str, file_num: int) -> Path:
    """Get the specific pickle data file by number."""
    input_path = Path(input_dir)
    data_file = input_path / f"data_type{file_num}_insurance.pkl"
    
    if not data_file.exists():
        raise FileNotFoundError(f"File not found: {data_file}")
    
    return data_file


def process_embeddings_in_batches(
    df: pd.DataFrame,
    batch_size: int
) -> Generator[np.ndarray, None, None]:
    """Yield batches of flattened embeddings from dataframe."""
    n_samples = len(df)
    embeddings = df["embedding"].values
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_embeddings = embeddings[start_idx:end_idx]
        X_batch = np.stack([np.array(emb, dtype=np.float32).flatten() for emb in batch_embeddings])
        yield X_batch


def compute_global_statistics_streaming(
    df: pd.DataFrame,
    batch_size: int = 200
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compute global mean and std in a streaming fashion."""
    print("\n--- Computing global statistics (streaming) ---")
    
    n_samples = len(df)
    running_sum = None
    running_sum_sq = None
    n_features = None
    samples_processed = 0
    
    for batch in process_embeddings_in_batches(df, batch_size):
        if running_sum is None:
            n_features = batch.shape[1]
            running_sum = np.zeros(n_features, dtype=np.float64)
            running_sum_sq = np.zeros(n_features, dtype=np.float64)
        
        running_sum += batch.sum(axis=0).astype(np.float64)
        running_sum_sq += (batch ** 2).sum(axis=0).astype(np.float64)
        samples_processed += len(batch)
        
        print(f"  Stats: {samples_processed}/{n_samples} samples", end="\r")
        
        del batch
        gc.collect()
    
    print()
    
    global_mean = (running_sum / n_samples).astype(np.float32)
    global_var = (running_sum_sq / n_samples) - (global_mean.astype(np.float64) ** 2)
    global_std = np.sqrt(np.maximum(global_var, 1e-10)).astype(np.float32)
    
    print(f"  Mean range: [{global_mean.min():.4f}, {global_mean.max():.4f}]")
    print(f"  Std range: [{global_std.min():.4f}, {global_std.max():.4f}]")
    print(f"  Features: {n_features}")
    
    del running_sum, running_sum_sq, global_var
    gc.collect()
    
    return global_mean, global_std, n_features


def fit_svd_memory_efficient(
    df: pd.DataFrame,
    global_mean: np.ndarray,
    global_std: np.ndarray,
    n_components: int,
    batch_size: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit SVD using scipy.sparse.linalg.svds which is very memory efficient.
    
    Returns:
        Tuple of (U, s, Vt) where Vt contains the principal components
    """
    print(f"\n--- Fitting SVD with {n_components} components ---")
    print("  Using scipy.sparse.linalg.svds (memory efficient)")
    
    n_samples = len(df)
    n_features = len(global_mean)
    
    # Adjust components if needed
    max_components = min(n_samples - 1, n_features - 1)
    actual_components = min(n_components, max_components)
    
    if actual_components < n_components:
        print(f"  WARNING: Reducing n_components from {n_components} to {actual_components}")
    
    # Build standardized matrix chunk by chunk
    # Use float32 to save memory
    print("  Building standardized matrix...")
    
    X_std = np.zeros((n_samples, n_features), dtype=np.float32)
    samples_processed = 0
    
    for batch in process_embeddings_in_batches(df, batch_size):
        batch_size_actual = len(batch)
        start_idx = samples_processed
        end_idx = samples_processed + batch_size_actual
        
        # Standardize in place
        X_std[start_idx:end_idx] = ((batch - global_mean) / global_std).astype(np.float32)
        
        samples_processed += batch_size_actual
        print(f"  Standardized: {samples_processed}/{n_samples}", end="\r")
        
        del batch
        gc.collect()
    
    print()
    print(f"  Matrix shape: {X_std.shape}")
    print(f"  Matrix memory: {X_std.nbytes / 1e9:.2f} GB")
    
    # Use scipy svds which is memory efficient
    # It uses ARPACK which doesn't need to form full matrices
    print(f"  Computing SVD with {actual_components} components...")
    print("  (This may take a few minutes...)")
    
    # svds returns components in ascending order of singular values
    # We want descending, so we'll reverse later
    U, s, Vt = svds(X_std, k=actual_components)
    
    # Reverse to get descending order (largest singular values first)
    U = U[:, ::-1]
    s = s[::-1]
    Vt = Vt[::-1, :]
    
    # Compute explained variance ratio
    # Total variance is sum of squared singular values / (n_samples - 1)
    total_var = np.sum(X_std ** 2) / (n_samples - 1)
    explained_var = (s ** 2) / (n_samples - 1)
    explained_var_ratio = explained_var / total_var
    
    print(f"  Variance captured: {np.sum(explained_var_ratio)*100:.2f}%")
    print(f"  Components: {actual_components}")
    
    del X_std
    gc.collect()
    
    return U, s, Vt, explained_var_ratio


def transform_and_save(
    df: pd.DataFrame,
    output_file: Path,
    Vt: np.ndarray,
    global_mean: np.ndarray,
    global_std: np.ndarray,
    explained_var_ratio: np.ndarray,
    batch_size: int = 200
) -> None:
    """
    Transform the data using Vt and save to output file.
    
    Transform is: X_reduced = X_std @ Vt.T
    """
    print(f"\n{'='*60}")
    print("PHASE 2: Transforming and saving dataset")
    print(f"{'='*60}")
    
    original_shape = df["embedding"].values[0].shape
    n_samples = len(df)
    n_components = Vt.shape[0]
    
    print(f"\n  Samples: {n_samples}")
    print(f"  Original embedding shape: {original_shape}")
    print(f"  Target components: {n_components}")
    
    # Transform in batches: X_reduced = X_std @ Vt.T
    X_reduced_list = []
    samples_transformed = 0
    
    for batch in process_embeddings_in_batches(df, batch_size):
        batch_std = ((batch - global_mean) / global_std).astype(np.float32)
        batch_reduced = batch_std @ Vt.T
        X_reduced_list.append(batch_reduced)
        
        samples_transformed += len(batch)
        print(f"  Transformed: {samples_transformed}/{n_samples}", end="\r")
        
        del batch, batch_std, batch_reduced
        gc.collect()
    
    print()
    
    X_reduced = np.vstack(X_reduced_list)
    del X_reduced_list
    gc.collect()
    
    print(f"  Reduced embedding shape: ({X_reduced.shape[1]},)")
    
    # Update dataframe
    df["embedding"] = [X_reduced[i] for i in range(len(X_reduced))]
    
    df.attrs["pca_n_components"] = n_components
    df.attrs["pca_variance_explained"] = float(np.sum(explained_var_ratio))
    df.attrs["original_embedding_shape"] = original_shape
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(output_file)
    print(f"  Saved to: {output_file}")
    
    del X_reduced
    gc.collect()
    
    # Save model components
    models_dir = output_file.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    file_num = output_file.stem.split('type')[1].split('_')[0]
    
    model_path = models_dir / f"svd_components_{n_components}_type{file_num}.npz"
    stats_path = models_dir / f"global_stats_{n_components}_type{file_num}.npz"
    
    np.savez_compressed(model_path, Vt=Vt, explained_var_ratio=explained_var_ratio)
    np.savez(stats_path, mean=global_mean, std=global_std)
    
    print(f"\nSaved SVD components to: {model_path}")
    print(f"Saved global statistics to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply PCA dimensionality reduction to a single medical imaging embeddings file"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/orcd/pool/006/lceli_shared/DATASET/data-cleaned",
        help="Input directory containing pickle files"
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="/orcd/pool/006/lceli_shared/DATASET",
        help="Base output directory (will create data-cleaned-pca-{n} subdirectory)"
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=1999,
        help="Number of PCA components (max is n_samples-1, so 1999 for 2000 samples)"
    )
    parser.add_argument(
        "--file_num",
        type=int,
        default=1,
        help="File number to process (e.g., 1 for data_type1_insurance.pkl)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=200,
        help="Batch size for processing (smaller = less memory)"
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default=None,
        help="Custom output directory suffix. Default: pca-{n_components}"
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: Input directory not found: {args.input_dir}")
        sys.exit(1)

    try:
        data_file = get_data_file(args.input_dir, args.file_num)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    suffix = args.output_suffix or f"pca-{args.n_components}"
    output_dir = Path(args.output_base) / f"data-cleaned-{suffix}"
    output_file = output_dir / data_file.name

    print("=" * 60)
    print("PCA DIMENSIONALITY REDUCTION (Memory Efficient)")
    print("=" * 60)
    print(f"\nInput file:        {data_file}")
    print(f"Output file:       {output_file}")
    print(f"Target components: {args.n_components}")
    print(f"Batch size:        {args.batch_size}")
    print(f"\nPipeline: 88,064 -> {args.n_components} features")

    # Load dataframe
    print(f"\nLoading {data_file.name}...")
    df = pd.read_pickle(data_file)
    n_samples = len(df)
    print(f"  Loaded DataFrame with {n_samples} samples")

    print(f"\n{'='*60}")
    print("PHASE 1: Fitting SVD")
    print(f"{'='*60}")

    # Compute statistics (streaming)
    global_mean, global_std, n_features = compute_global_statistics_streaming(
        df, batch_size=args.batch_size
    )

    # Fit SVD
    U, s, Vt, explained_var_ratio = fit_svd_memory_efficient(
        df, global_mean, global_std, args.n_components, batch_size=args.batch_size
    )
    
    # We only need Vt for transformation
    del U, s
    gc.collect()

    # Transform and save
    transform_and_save(
        df,
        output_file,
        Vt,
        global_mean,
        global_std,
        explained_var_ratio,
        batch_size=args.batch_size
    )

    actual_components = Vt.shape[0]
    total_variance = np.sum(explained_var_ratio)
    
    del df, Vt
    gc.collect()

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nReduced dataset saved to: {output_file}")
    print(f"Variance retained: {total_variance*100:.2f}%")
    print(f"Original features: {n_features}")
    print(f"Reduced features:  {actual_components}")

    return 0


if __name__ == "__main__":
    sys.exit(main())