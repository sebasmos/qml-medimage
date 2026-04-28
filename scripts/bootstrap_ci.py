#!/usr/bin/env python3
"""Bootstrap confidence intervals for QSVM / SVM results.

Two modes of operation:

1. Kernel mode (--run_dir points to a save_kernels run with kernels/):
   Loads the saved SVC model + precomputed kernel matrices, re-derives
   predictions on the test set, and bootstraps over test samples.

2. Results-only mode (--run_dir points to any QSVM run without kernels/):
   Loads the SVC model, the test kernel from the model's support vectors,
   and the test samples.  Falls back to confusion-matrix reconstruction
   if kernel re-prediction is not possible.

Usage examples:
    # From a save_kernels run (has kernels/ subdirectory):
    python scripts/bootstrap_ci.py \\
        --run_dir tests/save_kernels/medsiglip-448/data_type9/q4/seed_0

    # With explicit paths:
    python scripts/bootstrap_ci.py \\
        --model_pkl  tests/.../qsvm_model_4qubits.pkl \\
        --kernel_test tests/.../kernels/K_quantum_test.npy \\
        --y_test      tests/.../kernels/y_test.npy

    # Multiple directories (prints table for each):
    python scripts/bootstrap_ci.py \\
        --run_dir tests/save_kernels/medsiglip-448/data_type9/q4/seed_0 \\
                  tests/save_kernels/medsiglip-448/data_type9/q6/seed_0

    # Output to CSV:
    python scripts/bootstrap_ci.py --run_dir ... --output_csv results_ci.csv
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Core bootstrap
# ---------------------------------------------------------------------------

def bootstrap_ci(y_true, y_pred, y_proba=None, n_bootstrap=2000, ci=0.95, seed=42):
    """Compute bootstrap confidence intervals for classification metrics.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)
    y_proba : array-like, shape (n,), optional — predicted probabilities for AUC
    n_bootstrap : int
    ci : float — confidence level (default 0.95)
    seed : int — random seed for reproducibility

    Returns
    -------
    dict[str, dict] — {metric_name: {mean, std, ci_low, ci_high, point}}
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_proba is not None:
        y_proba = np.asarray(y_proba)

    metric_fns = {
        "accuracy":  lambda yt, yp, ypr: accuracy_score(yt, yp),
        "precision": lambda yt, yp, ypr: precision_score(yt, yp, zero_division=0),
        "recall":    lambda yt, yp, ypr: recall_score(yt, yp, zero_division=0),
        "f1":        lambda yt, yp, ypr: f1_score(yt, yp, zero_division=0),
    }
    if y_proba is not None:
        metric_fns["auc"] = lambda yt, yp, ypr: roc_auc_score(yt, ypr)

    # Point estimates on full data
    point_estimates = {}
    for name, fn in metric_fns.items():
        try:
            point_estimates[name] = fn(y_true, y_pred, y_proba)
        except Exception:
            point_estimates[name] = float("nan")

    # Bootstrap
    boot_values = {name: [] for name in metric_fns}
    alpha = (1 - ci) / 2

    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt_b = y_true[idx]
        yp_b = y_pred[idx]
        ypr_b = y_proba[idx] if y_proba is not None else None

        # Skip degenerate resamples (single class)
        if len(np.unique(yt_b)) < 2:
            continue

        for name, fn in metric_fns.items():
            try:
                boot_values[name].append(fn(yt_b, yp_b, ypr_b))
            except Exception:
                pass

    results = {}
    for name in metric_fns:
        vals = np.array(boot_values[name])
        if len(vals) == 0:
            results[name] = {
                "point": point_estimates.get(name, float("nan")),
                "mean": float("nan"), "std": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
            }
        else:
            results[name] = {
                "point": point_estimates[name],
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "ci_low": float(np.percentile(vals, alpha * 100)),
                "ci_high": float(np.percentile(vals, (1 - alpha) * 100)),
            }
    return results


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_from_run_dir(run_dir):
    """Auto-detect and load model + test predictions from a run directory.

    Returns (y_true, y_pred, y_proba_or_None, label).
    """
    run_dir = os.path.abspath(run_dir)
    kern_dir = os.path.join(run_dir, "kernels")

    # --- Mode 1: saved kernels available ---
    if os.path.isdir(kern_dir):
        K_test_path = os.path.join(kern_dir, "K_quantum_test.npy")
        y_test_path = os.path.join(kern_dir, "y_test.npy")

        if not os.path.exists(K_test_path):
            raise FileNotFoundError(f"K_quantum_test.npy not found in {kern_dir}")
        if not os.path.exists(y_test_path):
            raise FileNotFoundError(f"y_test.npy not found in {kern_dir}")

        K_test = np.load(K_test_path)
        y_test = np.load(y_test_path)

        # Find model pkl
        model_pkl = _find_model_pkl(run_dir)
        model = joblib.load(model_pkl)

        y_pred = model.predict(K_test)
        try:
            y_proba = model.predict_proba(K_test)[:, 1]
        except Exception:
            y_proba = model.decision_function(K_test)

        label = os.path.basename(os.path.dirname(run_dir)) + "/" + os.path.basename(run_dir)
        return y_test, y_pred, y_proba, label

    # --- Mode 2: no kernels — try to reconstruct from confusion matrix ---
    cm_test_csv = os.path.join(run_dir, "confusion_matrix_test.csv")
    samples_test_pkl = os.path.join(run_dir, "samples_test.pkl")

    if os.path.exists(cm_test_csv) and os.path.exists(samples_test_pkl):
        cm_df = pd.read_csv(cm_test_csv, index_col=0)
        cm = cm_df.values
        samples = joblib.load(samples_test_pkl)
        y_test = samples["y"]

        # Reconstruct predictions from confusion matrix + true labels
        # CM[i,j] = count of (true=i, pred=j)
        y_pred = _reconstruct_preds_from_cm(y_test, cm)
        label = os.path.basename(os.path.dirname(run_dir)) + "/" + os.path.basename(run_dir)
        print(f"  [INFO] No kernels/ in {run_dir}; reconstructed preds from confusion matrix.")
        return y_test, y_pred, None, label

    raise FileNotFoundError(
        f"Cannot load predictions from {run_dir}. "
        "Need either kernels/ subdirectory or confusion_matrix_test.csv + samples_test.pkl."
    )


def _find_model_pkl(run_dir):
    """Find the qsvm_model_*qubits.pkl file in run_dir."""
    for fname in os.listdir(run_dir):
        if fname.startswith("qsvm_model_") and fname.endswith(".pkl"):
            return os.path.join(run_dir, fname)
    raise FileNotFoundError(f"No qsvm_model_*qubits.pkl found in {run_dir}")


def _reconstruct_preds_from_cm(y_true, cm):
    """Reconstruct per-sample predictions that are consistent with a confusion matrix.

    For each true class i, the CM tells us how many samples were predicted as
    each class j.  We assign predictions accordingly (deterministic order).
    """
    n_classes = cm.shape[0]
    y_pred = np.empty_like(y_true)

    for true_class in range(n_classes):
        mask = (y_true == true_class)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            continue

        # Build prediction vector for this true class
        preds_for_class = []
        for pred_class in range(n_classes):
            count = int(cm[true_class, pred_class])
            preds_for_class.extend([pred_class] * count)

        # Handle potential rounding mismatches
        if len(preds_for_class) < len(indices):
            # Pad with most-common prediction
            most_common = int(np.argmax(cm[true_class]))
            preds_for_class.extend([most_common] * (len(indices) - len(preds_for_class)))
        elif len(preds_for_class) > len(indices):
            preds_for_class = preds_for_class[:len(indices)]

        y_pred[indices] = preds_for_class

    return y_pred


def load_from_explicit_paths(model_pkl, kernel_test, y_test_path):
    """Load from explicit file paths."""
    model = joblib.load(model_pkl)
    K_test = np.load(kernel_test)
    y_test = np.load(y_test_path)

    y_pred = model.predict(K_test)
    try:
        y_proba = model.predict_proba(K_test)[:, 1]
    except Exception:
        y_proba = model.decision_function(K_test)

    label = os.path.basename(os.path.dirname(model_pkl))
    return y_test, y_pred, y_proba, label


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_results(results, label="", ci=0.95):
    pct = int(ci * 100)
    print(f"\n{'=' * 62}")
    if label:
        print(f"  {label}")
        print(f"{'=' * 62}")
    header = f"{'Metric':>11s} | {'Point':>7s} | {'Mean':>7s} | {'Std':>7s} | {pct}% CI"
    print(header)
    print("-" * 62)
    for name in ["accuracy", "precision", "recall", "f1", "auc"]:
        if name not in results:
            continue
        v = results[name]
        print(f"{name:>11s} | {v['point']:.4f} | {v['mean']:.4f} | {v['std']:.4f} | "
              f"[{v['ci_low']:.4f}, {v['ci_high']:.4f}]")
    print()


def results_to_df(all_results):
    """Convert list of (label, results_dict) to a flat DataFrame."""
    rows = []
    for label, res in all_results:
        for metric, vals in res.items():
            row = {"run": label, "metric": metric}
            row.update(vals)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for QSVM / SVM results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run_dir", nargs="+", default=None,
        help="Path(s) to run directory (e.g. tests/save_kernels/.../seed_0). "
             "Auto-detects kernel mode vs confusion-matrix mode.",
    )
    parser.add_argument("--model_pkl", default=None, help="Explicit path to SVC model pkl")
    parser.add_argument("--kernel_test", default=None, help="Explicit path to K_quantum_test.npy")
    parser.add_argument("--y_test", default=None, help="Explicit path to y_test.npy")
    parser.add_argument("--n_bootstrap", type=int, default=2000, help="Number of bootstrap resamples (default: 2000)")
    parser.add_argument("--ci", type=float, default=0.95, help="Confidence level (default: 0.95)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_csv", default=None, help="Save results to CSV file")
    args = parser.parse_args()

    all_results = []

    # Explicit-paths mode
    if args.model_pkl and args.kernel_test and args.y_test:
        y_true, y_pred, y_proba, label = load_from_explicit_paths(
            args.model_pkl, args.kernel_test, args.y_test
        )
        res = bootstrap_ci(y_true, y_pred, y_proba, args.n_bootstrap, args.ci, args.seed)
        all_results.append((label, res))
        print_results(res, label, args.ci)

    # Run-dir mode
    elif args.run_dir:
        for rd in args.run_dir:
            try:
                y_true, y_pred, y_proba, label = load_from_run_dir(rd)
                res = bootstrap_ci(y_true, y_pred, y_proba, args.n_bootstrap, args.ci, args.seed)
                all_results.append((label, res))
                print_results(res, label, args.ci)
            except Exception as e:
                print(f"\n[ERROR] {rd}: {e}", file=sys.stderr)
    else:
        parser.error("Provide --run_dir or --model_pkl + --kernel_test + --y_test")

    # Save CSV
    if args.output_csv and all_results:
        df = results_to_df(all_results)
        df.to_csv(args.output_csv, index=False)
        print(f"Saved to {args.output_csv}")


if __name__ == "__main__":
    main()
