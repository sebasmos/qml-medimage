import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

K4 = np.load('tests/save_kernels/medsiglip-448/data_type9/q4/seed_0/kernels/K_quantum_train_full.npy')
K6 = np.load('tests/save_kernels/medsiglip-448/data_type9/q6/seed_0/kernels/K_quantum_train_full.npy')
eL4 = np.load('tests/analysis/#35/eigenvalues_medsiglip_448_q4.npy')
eL6 = np.load('tests/analysis/#35/eigenvalues_medsiglip_448_q6.npy')

# Trace-normalize quantum kernels (matching qve/core.py)
K4_n = K4 / np.trace(K4)
K6_n = K6 / np.trace(K6)
eQ4 = np.linalg.eigvalsh(K4_n)[::-1]
eQ6 = np.linalg.eigvalsh(K6_n)[::-1]

# Trace-normalize linear eigenvalues (divide by trace = sum of all eigenvalues)
eL4_n = eL4 / eL4.sum()
eL6_n = eL6 / eL6.sum()


def eff_rank(eigvals):
    ev = np.maximum(eigvals, 0)
    s = ev.sum()
    if s <= 0:
        return 0.0
    p = ev / s
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def mask_zeros(ev, thresh=1e-14):
    return np.where(ev > thresh, ev, np.nan)


def cum_energy(ev):
    ev_pos = np.maximum(ev, 0)
    s = ev_pos.sum()
    return np.cumsum(ev_pos) / (s + 1e-300)


er_Q4 = eff_rank(eQ4); er_Q6 = eff_rank(eQ6)
er_L4 = eff_rank(eL4_n); er_L6 = eff_rank(eL6_n)
print(f"Quantum q=4 eff rank: {er_Q4:.2f}, q=6: {er_Q6:.2f}")
print(f"Linear  q=4 eff rank: {er_L4:.2f}, q=6: {er_L6:.2f}")

N = len(eQ4)
idx = np.arange(1, N + 1)

eQ4_p = mask_zeros(eQ4)
eQ6_p = mask_zeros(eQ6)
eL4_p = mask_zeros(eL4_n)
eL6_p = mask_zeros(eL6_n)

# Last non-nan index for each linear series
L4_last = int(np.where(~np.isnan(eL4_p))[0][-1])
L6_last = int(np.where(~np.isnan(eL6_p))[0][-1])
print(f"Linear q=4 terminates at index {L4_last + 1} (rank={L4_last + 1})")
print(f"Linear q=6 terminates at index {L6_last + 1} (rank={L6_last + 1})")

# Colors: blue/green for quantum, orange/magenta for linear
CQ4 = '#2166ac'; CQ6 = '#4dac26'
CL4 = '#e66101'; CL6 = '#d01c8b'

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── Left: eigenvalue decay ──────────────────────────────────────────────────
ax = axes[0]
ax.semilogy(idx, eQ4_p, color=CQ4, lw=1.8, label=f'Quantum q=4 (eff.rank={er_Q4:.1f})')
ax.semilogy(idx, eQ6_p, color=CQ6, lw=1.8, label=f'Quantum q=6 (eff.rank={er_Q6:.1f})')

# Linear: thicker dashed lines, only plot non-nan region
ax.semilogy(idx[:L4_last + 1], eL4_p[:L4_last + 1],
            color=CL4, lw=2.8, ls='--',
            label=f'Linear q=4 (eff.rank={er_L4:.1f})')
ax.semilogy(idx[:L6_last + 1], eL6_p[:L6_last + 1],
            color=CL6, lw=2.8, ls='--',
            label=f'Linear q=6 (eff.rank={er_L6:.1f})')

# X marker at the point where linear series ends (rank cutoff)
ax.semilogy([L4_last + 1], [eL4_p[L4_last]],
            marker='X', color=CL4, ms=11, zorder=6, clip_on=False)
ax.semilogy([L6_last + 1], [eL6_p[L6_last]],
            marker='X', color=CL6, ms=11, zorder=6, clip_on=False)

# Vertical rank-boundary lines
ax.axvline(L4_last + 1, color=CL4, ls=':', lw=1.3, alpha=0.55, label='Rank boundary q=4')
ax.axvline(L6_last + 1, color=CL6, ls=':', lw=1.3, alpha=0.55, label='Rank boundary q=6')

ax.set_xlabel('Eigenvalue index (sorted descending)', fontsize=11)
ax.set_ylabel('Normalised eigenvalue (log scale)', fontsize=11)

ax.set_xlim(0, 80)
ax.legend(fontsize=8.5, loc='lower left')
ax.grid(True, alpha=0.3)

# ── Right: cumulative energy ────────────────────────────────────────────────
ax2 = axes[1]
cQ4 = cum_energy(eQ4); cQ6 = cum_energy(eQ6)
cL4 = cum_energy(eL4_n); cL6 = cum_energy(eL6_n)

ax2.plot(idx, cQ4, color=CQ4, lw=1.8, label='Quantum q=4')
ax2.plot(idx, cQ6, color=CQ6, lw=1.8, label='Quantum q=6')
# Linear cumulative: flat after rank cutoff (already at 1.0), show up to cutoff+1
ax2.plot(idx[:L4_last + 2], cL4[:L4_last + 2], color=CL4, lw=2.8, ls='--', label='Linear q=4')
ax2.plot(idx[:L6_last + 2], cL6[:L6_last + 2], color=CL6, lw=2.8, ls='--', label='Linear q=6')
ax2.plot([L4_last + 1], [cL4[L4_last]], marker='X', color=CL4, ms=11, zorder=6)
ax2.plot([L6_last + 1], [cL6[L6_last]], marker='X', color=CL6, ms=11, zorder=6)

ax2.axhline(0.99, color='gray', ls=':', lw=0.9, label='99% var')
ax2.axhline(0.90, color='gray', ls='--', lw=0.9, label='90% var')
ax2.set_xlabel('Number of eigenvalues', fontsize=11)
ax2.set_ylabel('Cumulative explained variance', fontsize=11)

ax2.set_xlim(0, 50)
ax2.set_ylim(0, 1.02)
ax2.legend(fontsize=8.5, loc='lower right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = 'temporal/figures/quantum_vs_linear_eigenspectrum_q4q6.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}")
