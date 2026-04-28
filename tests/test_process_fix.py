"""
test_process_fix.py — compare OLD vs FIXED preprocessing on medsiglip-448 seed_0 data_type5

Tests two issues in qve/process.py:
  #6 (data leakage):   MinMaxScaler was fit on train+test combined
  #7 (angle compression): feature_range=(-1, 1) instead of (-pi, pi)

Runs a lightweight kernel evaluation (no GPU — uses CPU cosine kernel proxy
to measure concentration) plus a full SVM fit to detect accuracy differences.

Usage:
  python tests/test_process_fix.py

Expected: both fixes together should reduce kernel concentration (off-diagonal
entries moving away from 1.0) and ideally maintain or improve accuracy/AUC.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

# ── Data path ─────────────────────────────────────────────────────────────────
DATA_FILE = (
    "/orcd/pool/006/lceli_shared/DATASET/qml-mimic-cxr-embeddings/"
    "medsiglip-448-embeddings/20-seeds/seed_0/data_type5_n1999.parquet"
)
N_DIM = 8   # qubits = 8 (as in best hybrid result)
SEED  = 42


# ── Preprocessing variants ────────────────────────────────────────────────────

def preprocess_old(n_dim, sample_train, sample_test):
    """Original code: scaler fit on train+test, range [-1, 1]"""
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test  = std_scale.transform(sample_test)
    pca = PCA(n_components=n_dim, svd_solver="auto").fit(sample_train)
    sample_train = pca.transform(sample_train)
    sample_test  = pca.transform(sample_test)
    samples = np.append(sample_train, sample_test, axis=0)
    mm = MinMaxScaler(feature_range=(-1, 1)).fit(samples)       # leaky + compressed
    return mm.transform(sample_train), mm.transform(sample_test)


def preprocess_new(n_dim, sample_train, sample_test):
    """Fixed code: scaler fit on train only, range [-pi, pi]"""
    std_scale = StandardScaler().fit(sample_train)
    sample_train = std_scale.transform(sample_train)
    sample_test  = std_scale.transform(sample_test)
    pca = PCA(n_components=n_dim, svd_solver="auto").fit(sample_train)
    sample_train = pca.transform(sample_train)
    sample_test  = pca.transform(sample_test)
    mm = MinMaxScaler(feature_range=(-np.pi, np.pi)).fit(sample_train)  # fixed
    return mm.transform(sample_train), mm.transform(sample_test)


# ── Kernel concentration metric ───────────────────────────────────────────────

def cosine_kernel(X, Y=None):
    """Inner-product kernel as CPU stand-in for the quantum fidelity kernel."""
    if Y is None:
        Y = X
    # normalise rows → dot product is cosine similarity ∈ [0,1] after squaring
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    return np.abs(Xn @ Yn.T) ** 2  # |<x|y>|^2 mimics fidelity kernel


def kernel_concentration(K):
    """
    Mean off-diagonal value of the kernel matrix.
    Higher → more concentrated (less discriminative).
    """
    n = K.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return K[mask].mean(), K[mask].std()


# ── Main ──────────────────────────────────────────────────────────────────────

def load_and_split(path, seed):
    df = pd.read_parquet(path)
    for col in ["new_insurance_type", "insurance", "insurance_type"]:
        if col in df.columns:
            label_col = col
            break
    else:
        raise ValueError("No insurance label column found")

    X = np.stack(df["embedding"].values).astype(np.float32)
    y = (df[label_col] == df[label_col].unique()[0]).astype(int).values

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def evaluate(name, X_tr, X_te, y_tr, y_te):
    # SVM with RBF (best classical proxy when quantum GPU not available)
    clf = SVC(kernel="rbf", probability=True, random_state=SEED)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)
    f1  = f1_score(y_te, y_pred, zero_division=0)

    K = cosine_kernel(X_tr)
    conc_mean, conc_std = kernel_concentration(K)

    return {"name": name, "acc": acc, "auc": auc, "f1": f1,
            "kernel_conc_mean": conc_mean, "kernel_conc_std": conc_std}


def main():
    print(f"Loading: {DATA_FILE}")
    X_tr, X_val, X_te, y_tr, y_val, y_te = load_and_split(DATA_FILE, SEED)
    print(f"  Train={len(X_tr)}, Val={len(X_val)}, Test={len(X_te)}")
    print(f"  Class balance (train): {y_tr.mean():.2f}")
    print()

    # Preprocess both ways
    Xtr_old, Xte_old = preprocess_old(N_DIM, X_tr.copy(), X_te.copy())
    Xtr_new, Xte_new = preprocess_new(N_DIM, X_tr.copy(), X_te.copy())

    # Feature range check
    print(f"Feature range OLD train: [{Xtr_old.min():.4f}, {Xtr_old.max():.4f}]")
    print(f"Feature range NEW train: [{Xtr_new.min():.4f}, {Xtr_new.max():.4f}]")
    print(f"Feature range OLD test:  [{Xte_old.min():.4f}, {Xte_old.max():.4f}]")
    print(f"Feature range NEW test:  [{Xte_new.min():.4f}, {Xte_new.max():.4f}]")
    print()

    # Leakage check: test data range should stay within train min/max for new
    for dim in range(N_DIM):
        tr_min, tr_max = Xtr_new[:, dim].min(), Xtr_new[:, dim].max()
        te_min, te_max = Xte_new[:, dim].min(), Xte_new[:, dim].max()
        if te_min < tr_min - 0.01 or te_max > tr_max + 0.01:
            print(f"  [INFO] dim {dim}: test range [{te_min:.3f},{te_max:.3f}] "
                  f"slightly outside train [{tr_min:.3f},{tr_max:.3f}] (expected for fixed scaler)")

    print()

    # Evaluate
    r_old = evaluate("OLD (leaky, [-1,1])", Xtr_old, Xte_old, y_tr, y_te)
    r_new = evaluate("NEW (train-only, [-π,π])", Xtr_new, Xte_new, y_tr, y_te)

    results = [r_old, r_new]
    print(f"{'Variant':<30} {'Acc':>6} {'AUC':>6} {'F1':>6} {'KConc_mean':>12} {'KConc_std':>10}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<30} {r['acc']:>6.4f} {r['auc']:>6.4f} {r['f1']:>6.4f} "
              f"{r['kernel_conc_mean']:>12.6f} {r['kernel_conc_std']:>10.6f}")

    print()
    delta_acc  = r_new["acc"]  - r_old["acc"]
    delta_auc  = r_new["auc"]  - r_old["auc"]
    delta_f1   = r_new["f1"]   - r_old["f1"]
    delta_conc = r_new["kernel_conc_mean"] - r_old["kernel_conc_mean"]

    print(f"Delta (new - old):  acc={delta_acc:+.4f}  auc={delta_auc:+.4f}  "
          f"f1={delta_f1:+.4f}  kernel_conc={delta_conc:+.6f}")

    if delta_conc < -0.01:
        print("  → Kernel LESS concentrated after fix (good — more discriminative signal)")
    elif delta_conc > 0.01:
        print("  → Kernel MORE concentrated after fix (unexpected — worth investigating)")
    else:
        print("  → Kernel concentration roughly unchanged (proxy kernel insensitive to angle range)")

    print()
    print("NOTE: This test uses a CPU cosine proxy for the quantum fidelity kernel.")
    print("      The actual cuQuantum kernel may show larger concentration differences.")
    print("      Run with --dry-run on SLURM to verify before full submission.")


if __name__ == "__main__":
    main()
