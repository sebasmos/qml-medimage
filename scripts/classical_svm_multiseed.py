#!/usr/bin/env python3
"""
Classical SVM C=1 multi-seed evaluation for paper Tier 1 comparison.

Runs linear + rbf SVM at C=1 for all seeds × PCA dims × models.
Matches QSVM preprocessing exactly: StandardScaler → PCA(q) → MinMaxScaler[-1,1].

Usage:
    python scripts/classical_svm_multiseed.py --output_dir tests/multiseed_classical/
"""
import argparse
import os
import csv
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score


def load_data(parquet_path):
    """Load embeddings and labels from parquet file."""
    df = pd.read_parquet(parquet_path)
    X = np.stack(df['embedding'].values).astype(np.float64)
    y = (df['new_insurance_type'] == 'Private').astype(int).values
    return X, y


def run_single(X, y, pca_dim, seed, kernels=('linear', 'rbf')):
    """Run classical SVM C=1 for one seed and PCA dim."""
    # Same split as QSVM: 80/10/10, stratified
    idx = np.arange(len(X))
    idx_train, idx_temp = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=seed, stratify=y[idx_temp])

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]

    # DT9 preprocessing: StandardScaler → PCA(q) → MinMaxScaler[-1,1]
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=pca_dim).fit(X_train_s)
    X_train_p = pca.transform(X_train_s)
    X_val_p = pca.transform(X_val_s)
    X_test_p = pca.transform(X_test_s)

    mms = MinMaxScaler(feature_range=(-1, 1)).fit(X_train_p)
    X_train_m = mms.transform(X_train_p)
    X_val_m = mms.transform(X_val_p)
    X_test_m = mms.transform(X_test_p)

    results = []
    for kern in kernels:
        svc = SVC(kernel=kern, C=1.0, probability=True, random_state=seed)
        svc.fit(X_train_m, y_train)

        y_pred = svc.predict(X_test_m)
        y_proba = svc.predict_proba(X_test_m)[:, 1]

        results.append({
            'kernel': kern,
            'C': 1.0,
            'pca_dim': pca_dim,
            'seed': seed,
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_precision': precision_score(y_test, y_pred, zero_division=0),
            'test_recall': recall_score(y_test, y_pred, zero_division=0),
            'test_f1': f1_score(y_test, y_pred),
            'test_auc': roc_auc_score(y_test, y_proba),
            'train_size': len(y_train),
            'val_size': len(y_val),
            'test_size': len(y_test),
            'pca_explained_variance': pca.explained_variance_ratio_.sum(),
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='tests/multiseed_classical/')
    parser.add_argument('--seeds', default='0,1,2,3,4,5,6,7,8,9')
    parser.add_argument('--pca_dims', default='2,3,4,5,6,8,9,10,11,12,16')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    pca_dims = [int(d) for d in args.pca_dims.split(',')]

    # 3 main + 2 supplementary models (matching paper + appendix)
    # Set QML_DATA_ROOT to the root of your local qml-mimic-cxr-embeddings download,
    # e.g. export QML_DATA_ROOT=/path/to/qml-mimic-cxr-embeddings
    data_root = os.environ.get(
        'QML_DATA_ROOT',
        '/orcd/pool/006/lceli_shared/DATASET/qml-mimic-cxr-embeddings'
    )
    models = {
        'medsiglip-448':    os.path.join(data_root, 'medsiglip-448-embeddings/20-seeds'),
        'rad-dino':         os.path.join(data_root, 'rad-dino-embeddings/20-seeds'),
        'vit-patch32-cls':  os.path.join(data_root, 'vit-base-patch32-224-embeddings/20-seeds'),
        'vit-patch32-gap':  os.path.join(data_root, 'vit-base-patch32-224-embeddings/20-seeds'),
        'vit-patch16-cls':  os.path.join(data_root, 'vit-base-patch16-224-embeddings/20-seeds'),
    }

    data_filters = {
        'medsiglip-448': 'data_type9',
        'rad-dino': 'data_type9',
        'vit-patch32-cls': 'data_type9_*_cls_embedding',
        'vit-patch32-gap': 'data_type9_*_gap_embedding',
        'vit-patch16-cls': 'data_type9_*_cls_embedding',
    }

    all_results = []

    for model_name, base_dir in models.items():
        print(f'\n=== {model_name} ===')
        for seed in seeds:
            seed_dir = os.path.join(base_dir, f'seed_{seed}')
            pattern = os.path.join(seed_dir, data_filters[model_name] + '*.parquet')
            files = glob.glob(pattern)
            if not files:
                # Try simpler pattern
                files = glob.glob(os.path.join(seed_dir, 'data_type9*.parquet'))
                if model_name == 'vit-patch32-gap':
                    files = [f for f in files if 'gap' in f]
                elif model_name in ('vit-patch32-cls', 'vit-patch16-cls'):
                    files = [f for f in files if 'cls' in f]
                else:
                    files = [f for f in files if 'cls' not in f and 'gap' not in f]

            if not files:
                print(f'  seed_{seed}: NO DATA FOUND')
                continue

            X, y = load_data(files[0])

            for pca_dim in pca_dims:
                if pca_dim > X.shape[1]:
                    continue
                results = run_single(X, y, pca_dim, seed)
                for r in results:
                    r['model'] = model_name
                    all_results.append(r)

                    # Save per-config CSV
                    out_dir = os.path.join(args.output_dir, model_name, 'data_type9',
                                          f'pca_{pca_dim}', f'seed_{seed}')
                    os.makedirs(out_dir, exist_ok=True)
                    with open(os.path.join(out_dir, 'metrics_summary.csv'), 'w', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=r.keys())
                        w.writeheader()
                        w.writerow(r)

            print(f'  seed_{seed}: done ({len(pca_dims)} PCA dims)')

    # Save combined summary
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, 'all_results_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_results[0].keys())
        w.writeheader()
        w.writerows(all_results)
    print(f'\nSaved {len(all_results)} results to {summary_path}')


if __name__ == '__main__':
    main()
