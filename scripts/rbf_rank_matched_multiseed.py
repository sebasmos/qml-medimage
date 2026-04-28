#!/usr/bin/env python3
"""
Multi-seed effective-rank-matched RBF kernel experiment.

For each (model, q) config where a saved quantum kernel exists (seed_0 only),
this script uses the seed_0 quantum kernel's effective rank as a fixed TARGET
and evaluates three classifiers across all 10 seeds:

  1. Default RBF (gamma='scale', C=1)
  2. Rank-matched RBF (gamma=gamma* s.t. eff_rank(RBF(gamma*)) == target_rank, C=1)
  3. QSVM (multiseed F1 loaded from master_long.csv)

Key question: on the ~9/10 seeds where classical linear SVM collapses (F1=0),
does the rank-matched RBF also collapse? If yes, the quantum kernel has
structural properties beyond its effective rank. If no, matching rank alone
is sufficient to avoid collapse.

Output:
  tests/analysis/rbf_rank_matched_multiseed/results.csv
  tests/analysis/rbf_rank_matched_multiseed/summary.csv   (collapse rates)
"""
import argparse
import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics.pairwise import rbf_kernel

COLLAPSE_THRESHOLD = 0.05   # F1 < this → "collapsed"
N_SEEDS = 10

# Set QML_DATA_ROOT to the root of your local qml-mimic-cxr-embeddings download,
# e.g. export QML_DATA_ROOT=/path/to/qml-mimic-cxr-embeddings
_data_root = os.environ.get(
    'QML_DATA_ROOT',
    '/orcd/pool/006/lceli_shared/DATASET/qml-mimic-cxr-embeddings'
)
_medsig_dir = os.path.join(_data_root, 'medsiglip-448-embeddings/20-seeds')

CONFIGS = [
    {
        'model': 'medsiglip-448',
        'q': 4,
        'data_dir': _medsig_dir,
        'kernel_path': 'tests/save_kernels/medsiglip-448/data_type9/q4/seed_0/kernels/K_quantum_train_full.npy',
    },
    {
        'model': 'medsiglip-448',
        'q': 6,
        'data_dir': _medsig_dir,
        'kernel_path': 'tests/save_kernels/medsiglip-448/data_type9/q6/seed_0/kernels/K_quantum_train_full.npy',
    },
    {
        'model': 'medsiglip-448',
        'q': 11,
        'data_dir': _medsig_dir,
        'kernel_path': 'tests/save_kernels/medsiglip-448/data_type9/q11/seed_0/kernels/K_quantum_train_full.npy',
    },
    {
        'model': 'medsiglip-448',
        'q': 16,
        'data_dir': _medsig_dir,
        'kernel_path': 'tests/save_kernels/medsiglip-448/data_type9/q16/seed_0/kernels/K_quantum_train_full.npy',
    },
]

MASTER_LONG_CSV = 'tests/analysis/multiseed_aggregate/master_long.csv'


def load_data(base_dir, seed):
    seed_dir = os.path.join(base_dir, f'seed_{seed}')
    files = glob.glob(os.path.join(seed_dir, 'data_type9*.parquet'))
    files = [f for f in files if 'cls' not in f and 'gap' not in f]
    if not files:
        files = glob.glob(os.path.join(seed_dir, 'data_type9*.parquet'))
    df = pd.read_parquet(files[0])
    X = np.stack(df['embedding'].values).astype(np.float64)
    y = (df['new_insurance_type'] == 'Private').astype(int).values
    return X, y


def preprocess(X, y, pca_dim, seed):
    idx = np.arange(len(X))
    idx_train, idx_temp = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=seed, stratify=y[idx_temp])
    X_train, y_train = X[idx_train], y[idx_train]
    X_test,  y_test  = X[idx_test],  y[idx_test]

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    pca = PCA(n_components=pca_dim).fit(X_train_s)
    X_train_p = pca.transform(X_train_s)
    X_test_p  = pca.transform(X_test_s)

    mms = MinMaxScaler(feature_range=(-1, 1)).fit(X_train_p)
    X_train_m = mms.transform(X_train_p)
    X_test_m  = mms.transform(X_test_p)

    return X_train_m, y_train, X_test_m, y_test


def eff_rank(K):
    K_n = K / np.trace(K)
    eigvals = np.linalg.eigvalsh(K_n)
    eigvals = np.maximum(eigvals, 0)
    s = eigvals.sum()
    if s <= 0:
        return 1.0
    p = eigvals / s
    p = p[p > 1e-15]
    return float(np.exp(-np.sum(p * np.log(p))))


def rbf_eff_rank(X_train, gamma):
    K = rbf_kernel(X_train, gamma=gamma)
    return eff_rank(K)


def find_gamma(X_train, target_rank, tol=0.05, max_iter=50):
    gamma_lo, gamma_hi = 1e-6, 1e3
    er_lo = rbf_eff_rank(X_train, gamma_lo)
    er_hi = rbf_eff_rank(X_train, gamma_hi)

    if target_rank <= er_lo:
        return gamma_lo
    if target_rank >= er_hi:
        return gamma_hi

    for _ in range(max_iter):
        gamma_mid = np.sqrt(gamma_lo * gamma_hi)
        er_mid = rbf_eff_rank(X_train, gamma_mid)
        if abs(er_mid - target_rank) / target_rank < tol:
            return gamma_mid
        if er_mid < target_rank:
            gamma_lo = gamma_mid
        else:
            gamma_hi = gamma_mid

    return np.sqrt(gamma_lo * gamma_hi)


def quantum_eff_rank(kernel_path):
    K_Q = np.load(kernel_path)
    return eff_rank(K_Q)


def load_qsvm_per_seed(model, q):
    """Load QSVM per-seed F1 from master_long.csv (n_params column = q)."""
    df = pd.read_csv(MASTER_LONG_CSV)
    sub = df[(df['method'] == 'qsvm') &
             (df['model'] == model) &
             (df['n_params'] == q)].copy()
    if sub.empty:
        return {}
    return dict(zip(sub['seed'].astype(int), sub['test_f1']))


def run_config(cfg, output_dir):
    model, q = cfg['model'], cfg['q']
    print(f"\n{'='*70}")
    print(f"model={model}  q={q}")

    # Fixed target rank from seed_0 quantum kernel
    target_rank = quantum_eff_rank(cfg['kernel_path'])
    print(f"  Target eff_rank (from seed_0 quantum kernel): {target_rank:.3f}")

    # Load QSVM per-seed F1
    qsvm_per_seed = load_qsvm_per_seed(model, q)

    rows = []
    for seed in range(N_SEEDS):
        print(f"  --- seed {seed} ---")
        try:
            X, y = load_data(cfg['data_dir'], seed)
        except Exception as e:
            print(f"    SKIP (load error): {e}")
            continue

        X_train, y_train, X_test, y_test = preprocess(X, y, q, seed)

        # --- Default RBF (gamma='scale') ---
        gamma_scale = 1.0 / (X_train.shape[1] * X_train.var())
        er_default = rbf_eff_rank(X_train, gamma_scale)
        svc_def = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=seed)
        svc_def.fit(X_train, y_train)
        f1_def  = f1_score(y_test, svc_def.predict(X_test), zero_division=0)
        acc_def = accuracy_score(y_test, svc_def.predict(X_test))
        print(f"    default RBF:  eff_rank={er_default:.2f}  F1={f1_def:.4f}  acc={acc_def:.4f}")

        # --- Rank-matched RBF ---
        gamma_star = find_gamma(X_train, target_rank)
        er_matched = rbf_eff_rank(X_train, gamma_star)
        svc_rm = SVC(kernel='rbf', C=1.0, gamma=gamma_star, random_state=seed)
        svc_rm.fit(X_train, y_train)
        f1_rm  = f1_score(y_test, svc_rm.predict(X_test), zero_division=0)
        acc_rm = accuracy_score(y_test, svc_rm.predict(X_test))
        print(f"    rank-matched: eff_rank={er_matched:.2f} (target {target_rank:.2f})  "
              f"F1={f1_rm:.4f}  acc={acc_rm:.4f}  gamma*={gamma_star:.4e}")

        # --- QSVM from master_long ---
        qsvm_f1 = qsvm_per_seed.get(seed, float('nan'))
        print(f"    QSVM (from CSV): F1={qsvm_f1:.4f}")

        rows.append({
            'model': model,
            'q': q,
            'seed': seed,
            'target_eff_rank': target_rank,
            'gamma_scale': gamma_scale,
            'er_default_rbf': er_default,
            'f1_default_rbf': f1_def,
            'acc_default_rbf': acc_def,
            'collapsed_default_rbf': int(f1_def < COLLAPSE_THRESHOLD),
            'gamma_star': gamma_star,
            'er_matched_rbf': er_matched,
            'f1_matched_rbf': f1_rm,
            'acc_matched_rbf': acc_rm,
            'collapsed_matched_rbf': int(f1_rm < COLLAPSE_THRESHOLD),
            'f1_qsvm': qsvm_f1,
            'collapsed_qsvm': int(qsvm_f1 < COLLAPSE_THRESHOLD) if not np.isnan(qsvm_f1) else np.nan,
        })

    return rows


def summarise(df):
    """Compute per-config collapse rates and mean F1."""
    rows = []
    for (model, q), grp in df.groupby(['model', 'q']):
        n = len(grp)
        rows.append({
            'model': model,
            'q': q,
            'n_seeds': n,
            'target_eff_rank': grp['target_eff_rank'].iloc[0],
            # Collapse rates
            'collapse_rate_default_rbf':  grp['collapsed_default_rbf'].sum() / n,
            'collapse_rate_matched_rbf':  grp['collapsed_matched_rbf'].sum() / n,
            'collapse_rate_qsvm':         grp['collapsed_qsvm'].mean(),
            # Mean F1
            'f1_mean_default_rbf':  grp['f1_default_rbf'].mean(),
            'f1_mean_matched_rbf':  grp['f1_matched_rbf'].mean(),
            'f1_mean_qsvm':         grp['f1_qsvm'].mean(),
            'f1_std_default_rbf':   grp['f1_default_rbf'].std(),
            'f1_std_matched_rbf':   grp['f1_matched_rbf'].std(),
            'f1_std_qsvm':          grp['f1_qsvm'].std(),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='tests/analysis/rbf_rank_matched_multiseed')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_rows = []
    for cfg in CONFIGS:
        rows = run_config(cfg, args.output_dir)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(args.output_dir, 'results.csv')
    df.to_csv(out_csv, index=False)

    summary = summarise(df)
    out_sum = os.path.join(args.output_dir, 'summary.csv')
    summary.to_csv(out_sum, index=False)

    print(f"\n{'='*70}")
    print("COLLAPSE RATES (fraction of 10 seeds where F1 < 0.05):")
    print(summary[['model', 'q', 'target_eff_rank',
                   'collapse_rate_default_rbf', 'collapse_rate_matched_rbf',
                   'collapse_rate_qsvm']].to_string(index=False))
    print(f"\nMEAN F1 ACROSS 10 SEEDS:")
    print(summary[['model', 'q',
                   'f1_mean_default_rbf', 'f1_mean_matched_rbf',
                   'f1_mean_qsvm']].to_string(index=False))
    print(f"\nResults saved to {out_csv}")
    print(f"Summary saved to {out_sum}")


if __name__ == '__main__':
    main()
