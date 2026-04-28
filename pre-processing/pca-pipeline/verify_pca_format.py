#!/usr/bin/env python3
"""
Verify PCA Reduced Dataset Format
==================================

Compares original and PCA reduced pkl files to ensure:
1. Same columns exist
2. Same number of samples
3. Insurance labels are preserved
4. Only embedding shape has changed
5. Metadata attributes are present

Usage:
    python verify_pca_format.py --file_num 1
    python verify_pca_format.py --file_num 2 --n_components 1999
    python verify_pca_format.py --all  # Check all files

Output:
    Prints comparison report showing any differences
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def get_file_paths(input_dir: str, output_dir: str, file_num: int):
    """Get original and PCA file paths."""
    original = Path(input_dir) / f"data_type{file_num}_insurance.pkl"
    pca_reduced = Path(output_dir) / f"data_type{file_num}_insurance.pkl"
    return original, pca_reduced


def compare_dataframes(original_path: Path, pca_path: Path, verbose: bool = True) -> dict:
    """
    Compare original and PCA reduced dataframes.
    
    Returns dict with comparison results.
    """
    results = {
        "original_path": str(original_path),
        "pca_path": str(pca_path),
        "original_exists": original_path.exists(),
        "pca_exists": pca_path.exists(),
        "errors": [],
        "warnings": [],
        "passed": True
    }
    
    if not original_path.exists():
        results["errors"].append(f"Original file not found: {original_path}")
        results["passed"] = False
        return results
    
    if not pca_path.exists():
        results["errors"].append(f"PCA file not found: {pca_path}")
        results["passed"] = False
        return results
    
    # Load dataframes
    if verbose:
        print(f"\nLoading original: {original_path.name}")
    df_orig = pd.read_pickle(original_path)
    
    if verbose:
        print(f"Loading PCA reduced: {pca_path.name}")
    df_pca = pd.read_pickle(pca_path)
    
    results["original_samples"] = len(df_orig)
    results["pca_samples"] = len(df_pca)
    results["original_columns"] = list(df_orig.columns)
    results["pca_columns"] = list(df_pca.columns)
    
    # Check 1: Same number of samples
    if len(df_orig) != len(df_pca):
        results["errors"].append(
            f"Sample count mismatch: original={len(df_orig)}, pca={len(df_pca)}"
        )
        results["passed"] = False
    
    # Check 2: Same columns
    orig_cols = set(df_orig.columns)
    pca_cols = set(df_pca.columns)
    
    missing_in_pca = orig_cols - pca_cols
    extra_in_pca = pca_cols - orig_cols
    
    if missing_in_pca:
        results["errors"].append(f"Columns missing in PCA file: {missing_in_pca}")
        results["passed"] = False
    
    if extra_in_pca:
        results["warnings"].append(f"Extra columns in PCA file: {extra_in_pca}")
    
    # Check 3: Insurance labels preserved (if column exists)
    insurance_cols = [c for c in df_orig.columns if 'insurance' in c.lower()]
    
    for col in insurance_cols:
        if col in df_pca.columns:
            orig_values = df_orig[col].values
            pca_values = df_pca[col].values
            
            if not np.array_equal(orig_values, pca_values):
                # Check for NaN handling
                orig_nan = pd.isna(orig_values)
                pca_nan = pd.isna(pca_values)
                
                if np.array_equal(orig_nan, pca_nan):
                    # NaN positions match, check non NaN values
                    orig_valid = orig_values[~orig_nan]
                    pca_valid = pca_values[~pca_nan]
                    
                    if not np.array_equal(orig_valid, pca_valid):
                        results["errors"].append(f"Insurance column '{col}' values differ")
                        results["passed"] = False
                else:
                    results["errors"].append(f"Insurance column '{col}' NaN positions differ")
                    results["passed"] = False
            
            results[f"{col}_preserved"] = True
        else:
            results["errors"].append(f"Insurance column '{col}' missing in PCA file")
            results["passed"] = False
    
    # Check 4: Other columns preserved (except embedding)
    common_cols = orig_cols & pca_cols
    non_embedding_cols = [c for c in common_cols if c != 'embedding']
    
    for col in non_embedding_cols:
        orig_values = df_orig[col].values
        pca_values = df_pca[col].values
        
        try:
            # Handle different types
            if orig_values.dtype == object:
                match = all(
                    (pd.isna(o) and pd.isna(p)) or (o == p) 
                    for o, p in zip(orig_values, pca_values)
                )
            else:
                orig_nan = pd.isna(orig_values)
                pca_nan = pd.isna(pca_values)
                if np.array_equal(orig_nan, pca_nan):
                    match = np.allclose(
                        orig_values[~orig_nan], 
                        pca_values[~pca_nan], 
                        equal_nan=True
                    ) if len(orig_values[~orig_nan]) > 0 else True
                else:
                    match = False
            
            if not match:
                results["errors"].append(f"Column '{col}' values differ")
                results["passed"] = False
        except Exception as e:
            results["warnings"].append(f"Could not compare column '{col}': {e}")
    
    # Check 5: Embedding shape changed correctly
    if 'embedding' in df_orig.columns and 'embedding' in df_pca.columns:
        orig_emb_shape = df_orig['embedding'].iloc[0].shape
        pca_emb_shape = df_pca['embedding'].iloc[0].shape
        
        results["original_embedding_shape"] = orig_emb_shape
        results["pca_embedding_shape"] = pca_emb_shape
        
        # PCA should reduce dimensions
        orig_size = np.prod(orig_emb_shape)
        pca_size = np.prod(pca_emb_shape)
        
        if pca_size >= orig_size:
            results["warnings"].append(
                f"PCA embedding not smaller: original={orig_emb_shape}, pca={pca_emb_shape}"
            )
        
        # Check all embeddings have same shape
        pca_shapes = df_pca['embedding'].apply(lambda x: x.shape).unique()
        if len(pca_shapes) > 1:
            results["errors"].append(f"Inconsistent PCA embedding shapes: {pca_shapes}")
            results["passed"] = False
    
    # Check 6: PCA metadata attributes
    expected_attrs = ['pca_n_components', 'pca_variance_explained', 'original_embedding_shape']
    
    for attr in expected_attrs:
        if attr in df_pca.attrs:
            results[f"attr_{attr}"] = df_pca.attrs[attr]
        else:
            results["warnings"].append(f"Missing PCA metadata attribute: {attr}")
    
    # Cleanup
    del df_orig, df_pca
    
    return results


def print_results(results: dict, file_num: int):
    """Print comparison results in a readable format."""
    print("\n" + "=" * 60)
    print(f"VERIFICATION RESULTS: data_type{file_num}_insurance.pkl")
    print("=" * 60)
    
    if results["passed"]:
        print("\n✓ STATUS: PASSED")
    else:
        print("\n✗ STATUS: FAILED")
    
    print(f"\nOriginal: {results.get('original_path', 'N/A')}")
    print(f"PCA:      {results.get('pca_path', 'N/A')}")
    
    if results.get("original_samples"):
        print(f"\nSamples: {results['original_samples']} -> {results['pca_samples']}")
    
    if results.get("original_embedding_shape"):
        print(f"Embedding shape: {results['original_embedding_shape']} -> {results['pca_embedding_shape']}")
    
    if results.get("attr_pca_n_components"):
        print(f"PCA components: {results['attr_pca_n_components']}")
    
    if results.get("attr_pca_variance_explained"):
        print(f"Variance explained: {results['attr_pca_variance_explained']*100:.2f}%")
    
    if results.get("original_columns"):
        print(f"\nColumns preserved: {len(results['original_columns'])}")
        print(f"  {results['original_columns']}")
    
    if results["errors"]:
        print("\n ERRORS:")
        for err in results["errors"]:
            print(f"  ✗ {err}")
    
    if results["warnings"]:
        print("\n⚠ WARNINGS:")
        for warn in results["warnings"]:
            print(f"  ⚠ {warn}")
    
    return results["passed"]


def main():
    parser = argparse.ArgumentParser(
        description="Verify PCA reduced dataset format matches original"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/orcd/pool/006/lceli_shared/DATASET/data-cleaned",
        help="Input directory containing original pickle files"
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="/orcd/pool/006/lceli_shared/DATASET",
        help="Base output directory"
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=1999,
        help="Number of PCA components (used to find output directory)"
    )
    parser.add_argument(
        "--file_num",
        type=int,
        default=None,
        help="File number to verify (e.g., 1 for data_type1_insurance.pkl)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Verify all files (1 to 11)"
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default=None,
        help="Custom output directory suffix. Default: pca-{n_components}"
    )

    args = parser.parse_args()

    # Construct output directory
    suffix = args.output_suffix or f"pca-{args.n_components}"
    output_dir = os.path.join(args.output_base, f"data-cleaned-{suffix}")

    print("=" * 60)
    print("PCA FORMAT VERIFICATION")
    print("=" * 60)
    print(f"\nOriginal directory: {args.input_dir}")
    print(f"PCA directory:      {output_dir}")

    # Determine which files to check
    if args.all:
        file_nums = range(1, 12)  # 1 to 11
    elif args.file_num:
        file_nums = [args.file_num]
    else:
        print("\nERROR: Specify --file_num or --all")
        sys.exit(1)

    all_passed = True
    summary = []

    for file_num in file_nums:
        original_path, pca_path = get_file_paths(args.input_dir, output_dir, file_num)
        
        # Skip if original doesn't exist
        if not original_path.exists():
            print(f"\nSkipping file_num={file_num}: original not found")
            continue
        
        results = compare_dataframes(original_path, pca_path)
        passed = print_results(results, file_num)
        
        summary.append({
            "file_num": file_num,
            "passed": passed,
            "errors": len(results["errors"]),
            "warnings": len(results["warnings"])
        })
        
        if not passed:
            all_passed = False

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for s in summary:
        status = "✓ PASS" if s["passed"] else "✗ FAIL"
        print(f"  data_type{s['file_num']}_insurance.pkl: {status} "
              f"(errors: {s['errors']}, warnings: {s['warnings']})")
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL VERIFICATIONS PASSED")
        return 0
    else:
        print("SOME VERIFICATIONS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())