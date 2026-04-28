#!/usr/bin/env python3
"""
PCA Variance Analysis for Medical Imaging Embeddings
=====================================================

Analyzes whether PCA can reduce 88,064 features directly to 16 dimensions
or whether an intermediate dimensionality reduction step is required.

Usage:
    python pca_variance_analysis.py --data_path /path/to/data.pkl --output_dir /path/to/output

Output:
    - pca_variance_analysis.csv: Detailed variance analysis for target dimensions
    - Final recommendation printed to stdout
"""

import argparse
import os
import sys
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
import gc


def load_embeddings(data_path: str) -> Tuple[np.ndarray, int]:
    """
    Load embeddings from pickle file and flatten to 2D array.

    Returns:
        X: 2D array of shape (n_samples, n_features)
        n_samples: Number of samples loaded
    """
    print(f"Loading data from: {data_path}")
    df = pd.read_pickle(data_path)

    if "embedding" not in df.columns:
        raise ValueError("Column 'embedding' not found in dataframe")

    # Convert embeddings to numpy array and flatten
    embeddings = df["embedding"].values
    X = np.stack([np.array(emb, dtype=np.float32).flatten() for emb in embeddings])

    print(f"  Loaded {X.shape[0]} samples")
    print(f"  Original embedding shape: {embeddings[0].shape}")
    print(f"  Flattened feature dimension: {X.shape[1]}")

    return X, X.shape[0]


def analyze_pca_variance(
    X: np.ndarray,
    target_components: List[int],
    max_pca_components: int = 1500
) -> Tuple[pd.DataFrame, Dict]:
    """
    Fit PCA and analyze cumulative explained variance for target dimensions.

    Uses randomized SVD for memory efficiency with high-dimensional data.
    Only computes up to max_pca_components to save memory.

    Returns:
        results_df: DataFrame with variance analysis
        summary: Dictionary with key findings
    """
    n_samples, n_features = X.shape
    print(f"\nStandardizing features (zero-centering)...")

    # Standardize (zero-center) the features - use float32 to save memory
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    gc.collect()

    # Limit components to save memory
    max_possible = min(n_samples, n_features)
    n_components = min(max_pca_components, max_possible)
    print(f"Fitting PCA with {n_components} components...")
    print(f"  (n_samples={n_samples}, n_features={n_features})")
    print(f"  Using randomized SVD solver for memory efficiency...")

    # Use randomized solver - much more memory efficient
    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)
    pca.fit(X_scaled)

    del X_scaled  # Free memory after fitting
    gc.collect()

    # Get cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Analyze target dimensions
    results = []
    for n_comp in target_components:
        if n_comp <= n_components:
            var_explained = cumulative_variance[n_comp - 1]

            # Determine recommendation flag
            if n_comp == 16:
                if var_explained >= 0.80:
                    flag = "DIRECT_PCA_OK"
                elif var_explained >= 0.50:
                    flag = "MARGINAL"
                else:
                    flag = "INTERMEDIATE_REQUIRED"
            else:
                flag = ""

            results.append({
                "n_components": n_comp,
                "cumulative_variance_explained": float(var_explained),
                "variance_percent": float(var_explained * 100),
                "recommendation_flag": flag
            })
        else:
            print(f"  Warning: {n_comp} components exceeds computed components ({n_components}), skipping")

    results_df = pd.DataFrame(results)

    # Find components needed for various thresholds
    def find_components_for_variance(target_var: float) -> int:
        idx = np.searchsorted(cumulative_variance, target_var)
        if idx < len(cumulative_variance):
            return int(idx + 1)
        else:
            # Variance not reached within computed components
            return None

    summary = {
        "variance_at_16": float(cumulative_variance[15]) if n_components >= 16 else None,
        "variance_at_32": float(cumulative_variance[31]) if n_components >= 32 else None,
        "variance_at_64": float(cumulative_variance[63]) if n_components >= 64 else None,
        "variance_at_100": float(cumulative_variance[99]) if n_components >= 100 else None,
        "variance_at_256": float(cumulative_variance[255]) if n_components >= 256 else None,
        "variance_at_512": float(cumulative_variance[511]) if n_components >= 512 else None,
        "variance_at_1000": float(cumulative_variance[999]) if n_components >= 1000 else None,
        "components_for_95pct": find_components_for_variance(0.95),
        "components_for_99pct": find_components_for_variance(0.99),
        "components_for_999pct": find_components_for_variance(0.999),
        "n_components_computed": int(n_components),
        "max_possible_components": int(max_possible),
        "total_variance_captured": float(cumulative_variance[-1])
    }

    return results_df, summary


def determine_recommendation(summary: Dict, target_variance: float = 0.999) -> Tuple[str, int, float, str]:
    """
    Determine whether one-stage or two-stage PCA is recommended.

    Recommendation is based on target variance retention percentage,
    not fixed component counts.

    Args:
        summary: Dictionary with PCA analysis results
        target_variance: Target variance to retain in intermediate step (default 99.9%)

    Returns:
        strategy: "ONE_STAGE" or "TWO_STAGE"
        intermediate_dim: Recommended intermediate dimension (0 if one-stage)
        intermediate_variance: Variance retained at intermediate dimension
        explanation: Human-readable explanation
    """
    var_16 = summary["variance_at_16"]
    n_computed = summary["n_components_computed"]
    max_possible = summary["max_possible_components"]
    total_var = summary["total_variance_captured"]

    if var_16 is None:
        return "INSUFFICIENT_DATA", 0, 0.0, "Not enough samples to determine variance at 16 components"

    if var_16 >= 0.80:
        return (
            "ONE_STAGE",
            0,
            var_16,
            f"Direct PCA to 16 dimensions is acceptable.\n"
            f"Variance retained at 16 components: {var_16*100:.2f}% (>= 80% threshold)"
        )
    else:
        # Two-stage PCA required
        # Use target variance percentage to determine intermediate dimension
        if target_variance >= 0.999:
            intermediate = summary["components_for_999pct"]
            var_label = "99.9%"
        elif target_variance >= 0.99:
            intermediate = summary["components_for_99pct"]
            var_label = "99%"
        else:
            intermediate = summary["components_for_95pct"]
            var_label = "95%"

        # If we couldn't reach target variance, use all computed components
        if intermediate is None:
            intermediate = n_computed
            actual_var = total_var
            var_note = f"(max reachable with {n_computed} components: {total_var*100:.2f}%)"
        else:
            actual_var = target_variance
            var_note = ""

        # Cap at max_possible (which equals n_samples for our case)
        intermediate = min(intermediate, max_possible)

        comp_95 = summary['components_for_95pct']
        comp_99 = summary['components_for_99pct']
        comp_999 = summary['components_for_999pct']

        threshold_lines = []
        if comp_95:
            threshold_lines.append(f"  - 95% variance: {comp_95} components")
        if comp_99:
            threshold_lines.append(f"  - 99% variance: {comp_99} components")
        if comp_999:
            threshold_lines.append(f"  - 99.9% variance: {comp_999} components")
        else:
            threshold_lines.append(f"  - 99.9% variance: >{n_computed} components (not reached)")

        threshold_str = "\n".join(threshold_lines)

        return (
            "TWO_STAGE",
            intermediate,
            actual_var,
            f"Two-stage PCA is REQUIRED.\n"
            f"Variance at 16 components: {var_16*100:.2f}% (< 80% threshold)\n"
            f"\n"
            f"RECOMMENDED INTERMEDIATE DIMENSION: {intermediate}\n"
            f"  - Target: ~{var_label} of total variance {var_note}\n"
            f"  - Variance captured with {n_computed} components: {total_var*100:.2f}%\n"
            f"\n"
            f"Pipeline: 88,064 -> {intermediate} -> 16 features\n"
            f"\n"
            f"Variance thresholds:\n{threshold_str}"
        )


def print_analysis_report(results_df: pd.DataFrame, summary: Dict, recommendation: Tuple):
    """Print a formatted analysis report to stdout."""
    strategy, intermediate_dim, intermediate_var, explanation = recommendation

    print("\n" + "=" * 80)
    print("PCA VARIANCE ANALYSIS REPORT")
    print("=" * 80)

    print(f"\nComputed {summary['n_components_computed']} of {summary['max_possible_components']} possible components")
    print(f"Total variance captured: {summary['total_variance_captured']*100:.2f}%")

    print("\n--- Variance Explained by Target Dimensions ---")
    print(results_df.to_string(index=False))

    print("\n--- Key Thresholds ---")
    comp_95 = summary['components_for_95pct']
    comp_99 = summary['components_for_99pct']
    comp_999 = summary['components_for_999pct']
    n_computed = summary['n_components_computed']

    print(f"  Components for 95% variance:  {comp_95 if comp_95 else f'>{n_computed}'}")
    print(f"  Components for 99% variance:  {comp_99 if comp_99 else f'>{n_computed}'}")
    print(f"  Components for 99.9% variance: {comp_999 if comp_999 else f'>{n_computed}'}")

    print("\n--- Variance at Key Dimensions ---")
    for dim in [16, 32, 64, 100, 256, 512, 1000]:
        key = f"variance_at_{dim}"
        if summary.get(key) is not None:
            print(f"  {dim:4d} components: {summary[key]*100:6.2f}%")

    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    print(f"\nStrategy: {strategy}")
    if intermediate_dim > 0:
        print(f"Optimal intermediate dimension: {intermediate_dim}")
    print(f"\n{explanation}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="PCA Variance Analysis for Medical Imaging Embeddings"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/orcd/pool/006/lceli_shared/DATASET/data-cleaned/data_type1_insurance.pkl",
        help="Path to pickle file with embeddings"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/sebasmos/orcd/pool/code/QML-MedImage/pre-processing",
        help="Directory to save output CSV"
    )
    parser.add_argument(
        "--target_variance",
        type=float,
        default=0.999,
        help="Target variance to retain in intermediate PCA step (default: 0.999 = 99.9%%)"
    )
    parser.add_argument(
        "--max_components",
        type=int,
        default=1500,
        help="Maximum PCA components to compute (saves memory). Default: 1500"
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.isfile(args.data_path):
        print(f"ERROR: Data file not found: {args.data_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Target dimensions to analyze
    target_components = [16, 32, 64, 100, 256, 512, 1000]

    # Load data
    X, n_samples = load_embeddings(args.data_path)

    # Run PCA analysis with memory-efficient settings
    results_df, summary = analyze_pca_variance(X, target_components, max_pca_components=args.max_components)

    # Determine recommendation based on target variance
    recommendation = determine_recommendation(summary, target_variance=args.target_variance)
    strategy, intermediate_dim, intermediate_var, explanation = recommendation

    # Add recommendation to results
    results_df["final_recommendation"] = ""
    results_df.loc[results_df["n_components"] == 16, "final_recommendation"] = strategy
    if intermediate_dim > 0:
        idx = results_df[results_df["n_components"] == intermediate_dim].index
        if len(idx) > 0:
            results_df.loc[idx[0], "final_recommendation"] = "OPTIMAL_INTERMEDIATE"

    # Save CSV
    output_path = os.path.join(args.output_dir, "pca_variance_analysis.csv")
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print report
    print_analysis_report(results_df, summary, recommendation)

    # Return exit code based on recommendation
    if strategy == "ONE_STAGE":
        return 0
    elif strategy == "TWO_STAGE":
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
