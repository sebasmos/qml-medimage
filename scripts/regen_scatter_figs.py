"""
Regenerate 6 PCA scatter figures (3 models × q=4,6).

Differences from analysis_collapse.py:
- No ax.set_title() — title removed entirely
- Yellow annotation box replaced with plain dark-gray text, no box
- Saves to BOTH temporal/figures/ AND figures/

Usage:
    python scripts/regen_scatter_figs.py
"""

import os
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OUT_DIRS = [
    os.path.join(REPO_ROOT, "temporal", "figures"),
    os.path.join(REPO_ROOT, "figures"),
]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

# Model label → (pkl path template, display name)
MODELS = {
    "medsiglip-448": (
        os.path.join(REPO_ROOT, "tests/qubit_sweep/medsiglip-448/data_type9/q{q}/seed_0/samples_train.pkl"),
        "MedSigLIP-448",
    ),
    "rad-dino": (
        os.path.join(REPO_ROOT, "tests/qubit_sweep/rad-dino/data_type9/q{q}/seed_0/samples_train.pkl"),
        "RAD-DINO",
    ),
    "vit-patch32-cls": (
        os.path.join(REPO_ROOT, "tests/qubit_sweep/vit-patch32-cls/data_type9+cls_embedding/q{q}/seed_0/samples_train.pkl"),
        "ViT-Patch32-CLS",
    ),
}

QUBITS = [4, 6]

# Color-blind friendly palette (Okabe-Ito)
CLASS_COLORS = ["#0072B2", "#D55E00"]   # blue, orange
CLASS_NAMES  = ["Medicaid/Medicare", "Private"]


# ---------------------------------------------------------------------------
# PCA pipeline (mirrors qve/process.py:data_prepare_cv)
# StandardScaler -> PCA(n_components=q) -> MinMaxScaler[-1,1]
# ---------------------------------------------------------------------------
def apply_pca_pipeline(X_raw, n_components):
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)

    pca = PCA(n_components=n_components, svd_solver="auto", random_state=42)
    X_pca = pca.fit_transform(X_std)

    mms = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = mms.fit_transform(X_pca)

    return X_scaled, pca.explained_variance_ratio_


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def make_scatter_plots():
    saved = []
    for model_key, (pkl_tmpl, display_name) in MODELS.items():
        for q in QUBITS:
            pkl_path = pkl_tmpl.format(q=q)
            if not os.path.exists(pkl_path):
                print(f"  SKIP {model_key} q{q}: file not found ({pkl_path})")
                continue

            d = joblib.load(pkl_path)
            X_raw = d["X"]
            y     = d["y"]

            X_pca, explained = apply_pca_pipeline(X_raw, n_components=q)
            X_2d = X_pca[:, :2]

            n_classes   = len(np.unique(y))
            colors      = CLASS_COLORS[:n_classes]
            class_labels = CLASS_NAMES[:n_classes]

            fig, ax = plt.subplots(figsize=(6, 5))

            for cls_idx, cls_id in enumerate(np.unique(y)):
                mask = y == cls_id
                ax.scatter(
                    X_2d[mask, 0], X_2d[mask, 1],
                    c=colors[cls_idx],
                    label=f"{class_labels[cls_idx]} (n={mask.sum()})",
                    alpha=0.45,
                    s=12,
                    edgecolors='none',
                    rasterized=True,
                )

            ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)", fontsize=11)
            ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)", fontsize=11)
            # NO ax.set_title(...)
            ax.legend(fontsize=9, markerscale=2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Plain text annotation — no box, dark gray
            ax.text(0.03, 0.03, "Lin. SVM: collapsed (F1=0)",
                    transform=ax.transAxes, fontsize=8, color='#444444',
                    verticalalignment='bottom')

            plt.tight_layout()

            fname = f"scatter_{model_key}_q{q}.png"
            for out_dir in OUT_DIRS:
                out_path = os.path.join(out_dir, fname)
                plt.savefig(out_path, dpi=150, bbox_inches='tight')
                saved.append(out_path)
                print(f"  Saved: {out_path}")

            plt.close()
            print(f"    PC1=[{X_2d[:,0].min():.3f},{X_2d[:,0].max():.3f}]"
                  f"  PC2=[{X_2d[:,1].min():.3f},{X_2d[:,1].max():.3f}]"
                  f"  var: {explained[0]*100:.1f}% / {explained[1]*100:.1f}%")

    return saved


if __name__ == "__main__":
    print("Regenerating scatter figures ...")
    saved = make_scatter_plots()
    print(f"\nDone. {len(saved)} file(s) written.")
    for p in saved:
        size_kb = os.path.getsize(p) / 1024
        print(f"  {p}  ({size_kb:.0f} KB)")
