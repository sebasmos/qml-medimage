#!/usr/bin/env python3
"""Paired bootstrap test: QSVM q=11 vs Classical linear SVM PCA-11.

Resamples the N=238 test set 2000 times, computes (QSVM − Classical) deltas
for accuracy and F1, and reports the 95% CI of each delta.

Output: tests/analysis/paired_bootstrap_q11.csv
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERN_DIR = os.path.join(
    PROJECT,
    "tests/save_kernels/medsiglip-448/data_type9/q11/seed_0"
)
OUTPUT_CSV = os.path.join(PROJECT, "tests/analysis/paired_bootstrap_q11.csv")

N_BOOTSTRAP = 2000
CI = 0.95
SEED = 42
PCA_DIM = 11

# ── 1. Load QSVM predictions ─────────────────────────────────────────────
print("Loading QSVM kernel model + predictions …")
K_test = np.load(os.path.join(KERN_DIR, "kernels/K_quantum_test.npy"))
y_test = np.load(os.path.join(KERN_DIR, "kernels/y_test.npy"))
model_q = joblib.load(os.path.join(KERN_DIR, "qsvm_model_11qubits.pkl"))
y_pred_qsvm = model_q.predict(K_test)

acc_qsvm = accuracy_score(y_test, y_pred_qsvm)
f1_qsvm = f1_score(y_test, y_pred_qsvm)
print(f"  QSVM  — acc={acc_qsvm:.4f}, F1={f1_qsvm:.4f}, N={len(y_test)}")

# ── 2. Refit classical linear SVM on PCA-11 features ─────────────────────
print("Refitting classical SVM (linear, C=1) on PCA-11 features …")
samples_train = joblib.load(os.path.join(KERN_DIR, "samples_train.pkl"))
samples_test = joblib.load(os.path.join(KERN_DIR, "samples_test.pkl"))

X_train_raw = samples_train["X"]
y_train = samples_train["y"]
X_test_raw = samples_test["X"]
# y_test already loaded from kernels — verify consistency
assert np.array_equal(y_test, samples_test["y"]), "y_test mismatch!"

# Preprocessing identical to QSVM pipeline (legacy: MinMax fit on train+test)
scaler = StandardScaler().fit(X_train_raw)
X_train_sc = scaler.transform(X_train_raw)
X_test_sc = scaler.transform(X_test_raw)

pca = PCA(n_components=PCA_DIM, svd_solver="auto").fit(X_train_sc)
X_train_pca = pca.transform(X_train_sc)
X_test_pca = pca.transform(X_test_sc)

combined = np.vstack([X_train_pca, X_test_pca])
mm = MinMaxScaler(feature_range=(-1, 1)).fit(combined)
X_train_mm = mm.transform(X_train_pca)
X_test_mm = mm.transform(X_test_pca)

clf_classical = SVC(kernel="linear", C=1.0, random_state=0)
clf_classical.fit(X_train_mm, y_train)
y_pred_classical = clf_classical.predict(X_test_mm)

acc_classical = accuracy_score(y_test, y_pred_classical)
f1_classical = f1_score(y_test, y_pred_classical)
print(f"  Classical — acc={acc_classical:.4f}, F1={f1_classical:.4f}")

# ── 3. Point deltas ──────────────────────────────────────────────────────
delta_acc = acc_qsvm - acc_classical
delta_f1 = f1_qsvm - f1_classical
print(f"\n  Point delta acc  = {delta_acc:+.4f}")
print(f"  Point delta F1   = {delta_f1:+.4f}")

# ── 4. Paired bootstrap ──────────────────────────────────────────────────
print(f"\nRunning paired bootstrap ({N_BOOTSTRAP} resamples) …")
rng = np.random.RandomState(SEED)
n = len(y_test)
alpha = (1 - CI) / 2

boot_delta_acc = []
boot_delta_f1 = []

for b in range(N_BOOTSTRAP):
    idx = rng.randint(0, n, size=n)
    yt = y_test[idx]

    # skip degenerate resamples (single class)
    if len(np.unique(yt)) < 2:
        continue

    yp_q = y_pred_qsvm[idx]
    yp_c = y_pred_classical[idx]

    a_q = accuracy_score(yt, yp_q)
    a_c = accuracy_score(yt, yp_c)
    f_q = f1_score(yt, yp_q, zero_division=0)
    f_c = f1_score(yt, yp_c, zero_division=0)

    boot_delta_acc.append(a_q - a_c)
    boot_delta_f1.append(f_q - f_c)

boot_delta_acc = np.array(boot_delta_acc)
boot_delta_f1 = np.array(boot_delta_f1)

ci_acc_lo = np.percentile(boot_delta_acc, alpha * 100)
ci_acc_hi = np.percentile(boot_delta_acc, (1 - alpha) * 100)
ci_f1_lo = np.percentile(boot_delta_f1, alpha * 100)
ci_f1_hi = np.percentile(boot_delta_f1, (1 - alpha) * 100)

print(f"\n{'='*60}")
print(f"  Paired bootstrap 95% CI  (QSVM − Classical)")
print(f"{'='*60}")
print(f"  Δ accuracy : {delta_acc:+.4f}  95% CI [{ci_acc_lo:+.4f}, {ci_acc_hi:+.4f}]")
print(f"  Δ F1       : {delta_f1:+.4f}  95% CI [{ci_f1_lo:+.4f}, {ci_f1_hi:+.4f}]")
print(f"{'='*60}")

acc_ci_includes_zero = (ci_acc_lo <= 0 <= ci_acc_hi)
f1_ci_includes_zero = (ci_f1_lo <= 0 <= ci_f1_hi)
print(f"  Acc CI includes zero? {acc_ci_includes_zero}")
print(f"  F1  CI includes zero? {f1_ci_includes_zero}")

# ── 5. Save results ──────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

rows = [
    {
        "metric": "accuracy",
        "qsvm_point": acc_qsvm,
        "classical_point": acc_classical,
        "delta_point": delta_acc,
        "ci_lo": ci_acc_lo,
        "ci_hi": ci_acc_hi,
        "ci_includes_zero": acc_ci_includes_zero,
        "n_bootstrap": len(boot_delta_acc),
        "n_test": n,
    },
    {
        "metric": "f1",
        "qsvm_point": f1_qsvm,
        "classical_point": f1_classical,
        "delta_point": delta_f1,
        "ci_lo": ci_f1_lo,
        "ci_hi": ci_f1_hi,
        "ci_includes_zero": f1_ci_includes_zero,
        "n_bootstrap": len(boot_delta_f1),
        "n_test": n,
    },
]
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved → {OUTPUT_CSV}")
print(df.to_string(index=False))
