import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = 'tests/analysis/qubit_scaling_data.csv'
OUT = 'temporal/figures/qubit_scaling_curve.png'

df = pd.read_csv(CSV)
models = ['medsiglip-448', 'rad-dino', 'vit-patch32-cls']
labels = ['MedSigLIP-448', 'RAD-DINO', 'ViT-p32']
colors = ['#2166ac', '#e66101', '#4dac26']
markers = ['o', 's', '^']

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

for ax, metric, ylabel in zip(
        axes,
        ['test_acc', 'test_f1'],
        ['Test Accuracy', 'Minority-class F1']):

    for model, label, color, marker in zip(models, labels, colors, markers):
        sub = df[df['model'] == model].sort_values('q')
        ax.plot(sub['q'], sub[metric],
                color=color, marker=marker, ms=7, lw=1.8, label=label)

    ax.set_xlabel('Number of Qubits (q)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(sorted(df['q'].unique()))

plt.tight_layout(pad=1.5)
plt.savefig(OUT, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved: {OUT}')
