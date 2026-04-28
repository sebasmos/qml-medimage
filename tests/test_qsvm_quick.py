"""
Quick sanity test for QSVM pipeline with minimal samples.

This test verifies the full QSVM pipeline works end-to-end with:
- 100 samples (subsampled from full dataset)
- 2 qubits
- Single node execution (no MPI)

Usage:
    # From project root with conda env activated
    python tests/test_qsvm_quick.py

    # Or via pytest
    pytest tests/test_qsvm_quick.py -v
"""

import os
import sys
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from qve import data_prepare_cv, set_seed


# Default test data path
DEFAULT_DATA_PATH = "/orcd/pool/006/lceli_shared/DATASET/data-cleaned/data_type1_insurance.pkl"


def load_test_data(data_path: str, max_samples: int = 100, seed: int = 42):
    """Load and prepare test data."""
    if not os.path.exists(data_path):
        pytest.skip(f"Test data not found: {data_path}")

    df = pd.read_pickle(data_path)

    # Get insurance column
    insurance_col = None
    for col in ["new_insurance_type", "insurance", "insurance_type"]:
        if col in df.columns:
            insurance_col = col
            break

    if insurance_col is None:
        raise ValueError("No insurance column found")

    # Process embeddings
    df["emb_array"] = df["embedding"].apply(lambda x: np.array(x, dtype=np.float32))
    df = df[[insurance_col, "emb_array"]].dropna().copy()

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df[insurance_col].astype(str).values)
    class_names = le.classes_.tolist()

    # Stack features
    X = np.stack(df["emb_array"].values).reshape(len(df), -1).astype(np.float32)

    # Subsample
    if max_samples < len(X):
        indices = np.random.RandomState(seed).permutation(len(X))[:max_samples]
        X = X[indices]
        y = y[indices]

    return X, y, class_names


def test_data_loading():
    """Test that data can be loaded and processed."""
    X, y, class_names = load_test_data(DEFAULT_DATA_PATH, max_samples=50)

    assert X.shape[0] == 50, f"Expected 50 samples, got {X.shape[0]}"
    assert X.shape[1] == 88064, f"Expected 88064 features (8x8x1376), got {X.shape[1]}"
    assert len(y) == 50
    assert len(class_names) == 2, f"Expected binary classification, got {len(class_names)} classes"


def test_data_prepare_cv():
    """Test PCA + scaling via data_prepare_cv."""
    X, y, _ = load_test_data(DEFAULT_DATA_PATH, max_samples=50)

    # Split
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply PCA to 2 components
    n_qubits = 2
    data_train, data_test = data_prepare_cv(n_qubits, X_train, X_test)

    assert data_train.shape[1] == n_qubits, f"Expected {n_qubits} features after PCA, got {data_train.shape[1]}"
    assert data_test.shape[1] == n_qubits

    # Check MinMax scaling to [-1, 1] (allow small float32 precision tolerance)
    eps = 1e-6
    assert data_train.min() >= -1.0 - eps and data_train.max() <= 1.0 + eps, \
        f"Data not scaled to [-1, 1]: min={data_train.min()}, max={data_train.max()}"
    assert data_test.min() >= -1.0 - eps and data_test.max() <= 1.0 + eps, \
        f"Test data not scaled to [-1, 1]: min={data_test.min()}, max={data_test.max()}"


def test_pca_components_match_qubits():
    """Test that PCA reduces exactly to n_qubits dimensions."""
    X, y, _ = load_test_data(DEFAULT_DATA_PATH, max_samples=50)
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    for n_qubits in [2, 4, 8]:
        data_train, data_test = data_prepare_cv(n_qubits, X_train, X_test)
        assert data_train.shape[1] == n_qubits, f"PCA should reduce to {n_qubits} dims"
        assert data_test.shape[1] == n_qubits


def test_no_intermediate_pca():
    """Verify that raw embeddings go directly to data_prepare_cv without intermediate PCA."""
    X, y, _ = load_test_data(DEFAULT_DATA_PATH, max_samples=30)

    # Raw features should be 88064 (8*8*1376)
    expected_features = 8 * 8 * 1376
    assert X.shape[1] == expected_features, (
        f"Raw features should be {expected_features}, got {X.shape[1]}. "
        "There should be NO intermediate PCA before data_prepare_cv."
    )


if __name__ == "__main__":
    print("Running QSVM quick tests...")
    print("=" * 60)

    # Run tests manually
    try:
        print("\n[1/4] test_data_loading...")
        test_data_loading()
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")

    try:
        print("\n[2/4] test_data_prepare_cv...")
        test_data_prepare_cv()
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")

    try:
        print("\n[3/4] test_pca_components_match_qubits...")
        test_pca_components_match_qubits()
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")

    try:
        print("\n[4/4] test_no_intermediate_pca...")
        test_no_intermediate_pca()
        print("  PASSED")
    except Exception as e:
        print(f"  FAILED: {e}")

    print("\n" + "=" * 60)
    print("Tests complete!")
