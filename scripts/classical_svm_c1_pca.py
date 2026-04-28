#!/usr/bin/env python3
"""
Classical SVM C=1 baseline at PCA-9/10/11/12 for MedSigLIP-448 DT9.

Reproduces the EXACT preprocessing from the QSVM pipeline:
  1. 80/10/10 stratified split with random_state=seed
  2. StandardScaler (fit on train)
  3. PCA(n_components=q) (fit on train)
  4. MinMaxScaler([-1,1]) (fit on train+test combined — legacy behavior)
  5. SVM with C=1.0, kernel={linear, rbf}

This validates the Tier 1 claim: QSVM q=11 acc=0.769 > classical ?

Output: tests/c1_forced_extended/svm/medsiglip-448/data_type9/pca_{q}/metrics_summary.csv
"""

import os
import sys
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
)

# ── Configuration ──────────────────────────────────────────────────────────
# Set QML_DATA_ROOT to the root of your local qml-mimic-cxr-embeddings download,
# e.g. export QML_DATA_ROOT=/path/to/qml-mimic-cxr-embeddings
_data_root = os.environ.get(
    'QML_DATA_ROOT',
    '/orcd/pool/006/lceli_shared/DATASET/qml-mimic-cxr-embeddings'
)
DATA_PATH = os.path.join(
    _data_root,
    'medsiglip-448-embeddings/20-seeds/seed_0/data_type9_n2371.parquet'
)
SEED = 0
PCA_DIMS = [9, 10, 11, 12]
KERNELS = ["linear", "rbf"]
C_VALUE = 1.0

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "tests", "c1_forced_extended", "svm", "medsiglip-448", "data_type9")


def data_prepare_cv(n_dim, sample_train, sample_test):
    """
    Exact replica of qve.process.data_prepare_cv with default flags
    (fix_leakage=False, pi_angles=False).
    """
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test = std_scale.transform(sample_test)

    pca = PCA(n_components=n_dim, svd_solver="auto").fit(sample_train)
    sample_train = pca.transform(sample_train)
    sample_test = pca.transform(sample_test)

    # Legacy behavior: MinMaxScaler fit on BOTH train + test
    samples = np.append(sample_train, sample_test, axis=0)
    minmax_scale = MinMaxScaler(feature_range=(-1, 1)).fit(samples)
    sample_train = minmax_scale.transform(sample_train)
    sample_test = minmax_scale.transform(sample_test)

    return sample_train, sample_test


def load_and_prepare():
    """Load parquet, extract embeddings and labels."""
    print(f"Loading: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    print(f"  Loaded {len(df)} samples")

    df["emb_array"] = df["embedding"].apply(lambda x: np.array(x, dtype=np.float32))

    # Find insurance column
    for col in ["new_insurance_type", "insurance", "insurance_type"]:
        if col in df.columns:
            insurance_col = col
            break
    else:
        raise ValueError("No insurance column found")

    df = df.dropna(subset=[insurance_col, "emb_array"]).copy()
    print(f"  After cleaning: {len(df)} samples")

    le = LabelEncoder()
    y = le.fit_transform(df[insurance_col].astype(str).values)
    class_names = le.classes_.tolist()
    assert len(class_names) == 2, f"Expected binary, got {class_names}"
    print(f"  Classes: {class_names}")
    print(f"  Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    X = np.stack(df["emb_array"].values).reshape(len(df), -1).astype(np.float32)
    print(f"  Feature shape: {X.shape}")

    return X, y, class_names


def split_data(X, y):
    """80/10/10 stratified split — identical to QSVM pipeline."""
    indices = np.arange(len(X))

    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X, y, indices, test_size=0.2, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_svm(X_train, y_train, X_test, y_test, kernel, C):
    """Train SVM and return metrics dict."""
    t0 = time.time()
    clf = SVC(kernel=kernel, C=C, probability=True, random_state=SEED)
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    infer_time = time.time() - t0

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_proba)),
        "train_time_sec": train_time,
        "infer_time_sec": infer_time,
    }


def main():
    print("=" * 80)
    print("CLASSICAL SVM C=1 BASELINE — PCA-9/10/11/12")
    print("=" * 80)
    print(f"Seed: {SEED}")
    print(f"C: {C_VALUE}")
    print(f"Kernels: {KERNELS}")
    print(f"PCA dims: {PCA_DIMS}")
    print()

    X, y, class_names = load_and_prepare()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    all_results = []

    for q in PCA_DIMS:
        print(f"\n--- PCA-{q} ---")

        # Apply preprocessing (same as QSVM)
        # For val evaluation:
        train_pca_val, val_pca = data_prepare_cv(q, X_train, X_val)
        # For test evaluation:
        train_pca_test, test_pca = data_prepare_cv(q, X_train, X_test)

        out_dir = os.path.join(OUTPUT_BASE, f"pca_{q}")
        os.makedirs(out_dir, exist_ok=True)

        rows = []
        for kernel in KERNELS:
            print(f"  kernel={kernel}, C={C_VALUE}")

            # Train metrics
            train_metrics = run_svm(train_pca_test, y_train, train_pca_test, y_train, kernel, C_VALUE)
            # Val metrics
            val_metrics = run_svm(train_pca_val, y_train, val_pca, y_val, kernel, C_VALUE)
            # Test metrics
            test_metrics = run_svm(train_pca_test, y_train, test_pca, y_test, kernel, C_VALUE)

            row = {"kernel": kernel, "C": C_VALUE, "pca_dim": q}
            for prefix, m in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
                for k, v in m.items():
                    row[f"{prefix}_{k}"] = v

            rows.append(row)

            print(f"    test_acc={test_metrics['accuracy']:.4f}  "
                  f"test_f1={test_metrics['f1']:.4f}  "
                  f"test_auc={test_metrics['auc']:.4f}")

            all_results.append(row)

        df_out = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, "metrics_summary.csv")
        df_out.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        # Save dataset info
        info = {
            "experiment": {
                "type": "Classical SVM C=1",
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pca_components": q,
                "C": C_VALUE,
            },
            "data": {
                "source_path": DATA_PATH,
            },
            "seed": SEED,
            "samples": {
                "total": len(X_train) + len(X_val) + len(X_test),
                "train": len(X_train),
                "val": len(X_val),
                "test": len(X_test),
            },
            "classes": class_names,
        }
        with open(os.path.join(out_dir, "dataset_info.json"), "w") as f:
            json.dump(info, f, indent=2)

    # Print summary comparison table
    print("\n" + "=" * 80)
    print("SUMMARY: Classical SVM C=1 vs QSVM")
    print("=" * 80)
    print(f"{'PCA':>4} {'Kernel':>8} {'Test Acc':>10} {'Test F1':>10} {'Test AUC':>10}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['pca_dim']:>4} {r['kernel']:>8} "
              f"{r['test_accuracy']:>10.4f} {r['test_f1']:>10.4f} {r['test_auc']:>10.4f}")

    # Save combined results
    combined_path = os.path.join(OUTPUT_BASE, "all_pca_dims_summary.csv")
    pd.DataFrame(all_results).to_csv(combined_path, index=False)
    print(f"\nCombined results: {combined_path}")


if __name__ == "__main__":
    main()
