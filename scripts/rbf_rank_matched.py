#!/usr/bin/env python3
"""
Effective-rank-matched RBF kernel experiment (LEO-7).

For each (model, q) config where a saved quantum kernel exists, this script:
  1. Uses identical preprocessing to classical_svm_multiseed.py
     (StandardScaler -> PCA(q) -> MinMaxScaler[-1,1])
  2. Loads the saved quantum kernel K_Q, trace-normalises it, computes
     its Shannon effective rank (target_rank)
  3. Binary-searches gamma such that eff_rank(RBF_train(gamma)) == target_rank
  4. Trains SVC(C=1, kernel='rbf', gamma=gamma*) and reports minority-class F1
  5. Also reports QSVM F1 from paired_stats.csv for direct comparison

Output: tests/analysis/rbf_rank_matched/results.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import rbf_kernel


# ── configs: only where quantum kernels are saved ─────────────────────────────
CONFIGS = [
    {
        'model': 'medsiglip-448',
        'q': 4,
        'data_dir': '/orcd/pool/006/lceli_shared/DATASET/qml-mimic-cxr-embeddings/medsiglip-448-embeddings/20-seeds',
        'kernel_path': 'tests/save_kernels/medsiglip-448/data_type9/q4/seed_0/kernels/K_quantum_train_full.npy',
    },
    {
        'model': 'medsiglip-448',
        'q': 6,
        'data_dir': '/orcd/pool/006/lceli_shared/DATASET/qml-mimic-cxr-embeddings/medsiglip-448-embeddings/20-seeds',
        'kernel_path': 'tests/save_kernels/medsiglip-448/data_type9/q6/seed_0/kernels/K_quantum_train_full.npy',
    },
    {
        'model': 'medsiglip-448',
        'q': 11,
        'data_dir': '/orcd/pool/006/lceli_shared/DATASET/qml-mimic-cxr-embeddings/medsiglip-448-embeddings/20-seeds',
        'kernel_path': 'tests/save_kernels/medsiglip-448/data_type9/q11/seed_0/kernels/K_quantum_train_full.npy',
    },
    {
        'model': 'medsiglip-448',
        'q': 16,
        'data_dir': '/orcd/pool/006/lceli_shared/DATASET/qml-mimic-cxr-embeddings/medsiglip-448-embeddings/20-seeds',
        'kernel_path': 'tests/save_kernels/medsiglip-448/data_type9/q16/seed_0/kernels/K_quantum_train_full.npy',
    },
]

# QSVM seed_0 F1 reference values from paired_stats.csv (mean over 10 seeds, but
# seed_0 single result is in qubit_sweep results; use multiseed mean as reference)
QSVM_F1_REF = {
    ('medsiglip-448', 4):  0.212,
    ('medsiglip-448', 6):  0.286,
    ('medsiglip-448', 11): 0.343,
    ('medsiglip-448', 16): 0.377,
}


def load_data(base_dir, seed=0):
    """Load embeddings — same as classical_svm_multiseed.py."""
    import glob
    seed_dir = os.path.join(base_dir, f'seed_{seed}')
    files = glob.glob(os.path.join(seed_dir, 'data_type9*.parquet'))
    # Exclude cls/gap variants for medsiglip and rad-dino
    files = [f for f in files if 'cls' not in f and 'gap' not in f]
    if not files:
        files = glob.glob(os.path.join(seed_dir, 'data_type9*.parquet'))
    df = pd.read_parquet(files[0])
    X = np.stack(df['embedding'].values).astype(np.float64)
    y = (df['new_insurance_type'] == 'Private').astype(int).values
    return X, y


def preprocess(X, y, pca_dim, seed):
    """Identical to classical_svm_multiseed.py: StandardScaler→PCA→MinMaxScaler[-1,1]."""
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
    """Shannon effective rank of kernel matrix (trace-normalised internally)."""
    K_n = K / np.trace(K)
    eigvals = np.linalg.eigvalsh(K_n)
    eigvals = np.maximum(eigvals, 0)
    s = eigvals.sum()
    if s <= 0:
        return 1.0
    p = eigvals / s
    p = p[p > 1e-15]
    return float(np.exp(-np.sum(p * np.log(p))))


def quantum_eff_rank(kernel_path):
    """Load saved quantum kernel and compute its effective rank."""
    K_Q = np.load(kernel_path)
    return eff_rank(K_Q)


def rbf_eff_rank(X_train, gamma):
    """Compute RBF kernel effective rank for given gamma."""
    K = rbf_kernel(X_train, gamma=gamma)
    return eff_rank(K)


def find_gamma(X_train, target_rank, n_train=None, tol=0.05, max_iter=50):
    """Binary search for gamma giving RBF effective rank == target_rank."""
    N = len(X_train)
    # Bracket: gamma_lo gives eff_rank near 1, gamma_hi gives eff_rank near N
    gamma_lo, gamma_hi = 1e-6, 1e3

    # Verify bracket
    er_lo = rbf_eff_rank(X_train, gamma_lo)
    er_hi = rbf_eff_rank(X_train, gamma_hi)
    print(f"    bracket: gamma={gamma_lo:.1e} → eff_rank={er_lo:.2f}, "
          f"gamma={gamma_hi:.1e} → eff_rank={er_hi:.2f}, target={target_rank:.2f}")

    if target_rank <= er_lo:
        print(f"    target below bracket low — using gamma_lo={gamma_lo}")
        return gamma_lo
    if target_rank >= er_hi:
        print(f"    target above bracket high — using gamma_hi={gamma_hi}")
        return gamma_hi

    for i in range(max_iter):
        gamma_mid = np.sqrt(gamma_lo * gamma_hi)  # geometric midpoint
        er_mid = rbf_eff_rank(X_train, gamma_mid)
        print(f"    iter {i+1:2d}: gamma={gamma_mid:.4e} → eff_rank={er_mid:.3f} (target={target_rank:.3f})")
        if abs(er_mid - target_rank) / target_rank < tol:
            print(f"    converged at gamma={gamma_mid:.4e}")
            return gamma_mid
        if er_mid < target_rank:
            gamma_lo = gamma_mid
        else:
            gamma_hi = gamma_mid

    gamma_final = np.sqrt(gamma_lo * gamma_hi)
    print(f"    max_iter reached; using gamma={gamma_final:.4e}")
    return gamma_final


def run_config(cfg, seed=0):
    print(f"\n{'='*60}")
    print(f"model={cfg['model']}  q={cfg['q']}  seed={seed}")

    X, y = load_data(cfg['data_dir'], seed=seed)
    X_train, y_train, X_test, y_test = preprocess(X, y, cfg['q'], seed=seed)

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Quantum effective rank from saved kernel
    target_rank = quantum_eff_rank(cfg['kernel_path'])
    print(f"  Quantum eff rank (target): {target_rank:.3f}")

    # Default RBF C=1 (gamma='scale') as reference
    svc_default = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=seed)
    svc_default.fit(X_train, y_train)
    f1_default  = f1_score(y_test, svc_default.predict(X_test), zero_division=0)
    acc_default = accuracy_score(y_test, svc_default.predict(X_test))
    gamma_scale = 1.0 / (X_train.shape[1] * X_train.var())
    er_default  = rbf_eff_rank(X_train, gamma_scale)
    print(f"  Default RBF (gamma=scale={gamma_scale:.4e}): eff_rank={er_default:.3f}, F1={f1_default:.4f}, acc={acc_default:.4f}")

    # Rank-matched RBF
    print("  Searching for rank-matched gamma...")
    gamma_star = find_gamma(X_train, target_rank)
    er_matched  = rbf_eff_rank(X_train, gamma_star)

    svc_matched = SVC(kernel='rbf', C=1.0, gamma=gamma_star, random_state=seed)
    svc_matched.fit(X_train, y_train)
    f1_matched  = f1_score(y_test, svc_matched.predict(X_test), zero_division=0)
    acc_matched = accuracy_score(y_test, svc_matched.predict(X_test))
    print(f"  Rank-matched RBF (gamma={gamma_star:.4e}): eff_rank={er_matched:.3f}, F1={f1_matched:.4f}, acc={acc_matched:.4f}")

    qsvm_f1 = QSVM_F1_REF.get((cfg['model'], cfg['q']), float('nan'))
    print(f"  QSVM F1 (multiseed mean ref): {qsvm_f1:.4f}")
    print(f"  QSVM vs rank-matched RBF delta F1: {qsvm_f1 - f1_matched:+.4f}")

    return {
        'model': cfg['model'],
        'q': cfg['q'],
        'seed': seed,
        'quantum_eff_rank': target_rank,
        'gamma_scale': gamma_scale,
        'er_default_rbf': er_default,
        'f1_default_rbf': f1_default,
        'acc_default_rbf': acc_default,
        'gamma_star': gamma_star,
        'er_matched_rbf': er_matched,
        'f1_matched_rbf': f1_matched,
        'acc_matched_rbf': acc_matched,
        'qsvm_f1_ref': qsvm_f1,
        'delta_f1_qsvm_vs_matched': qsvm_f1 - f1_matched,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='tests/analysis/rbf_rank_matched')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    for cfg in CONFIGS:
        row = run_config(cfg, seed=args.seed)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.output_dir, 'results.csv')
    df.to_csv(out_csv, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to {out_csv}")
    print(df[['model','q','quantum_eff_rank','er_matched_rbf','f1_matched_rbf','qsvm_f1_ref','delta_f1_qsvm_vs_matched']].to_string(index=False))


if __name__ == '__main__':
    main()
