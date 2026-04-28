# Multi-Seed Commands for QML-MedImage Paper

## Overview

Full 10-seed validation: **5 models x 11 qubit counts x 10 seeds = 550 QSVM configs** + 1,100 classical SVM configs.

All QSVM jobs use: **trace normalization, C=1.0, DT9, single_mode, GPU**.

Split between Sebastian (q2,q4,q8,q9,q10,q11,q12,q16) and Felipe (q3,q5,q6).

**Models:**
- 3 main: MedSigLIP-448, RAD-DINO, ViT-patch32-cls
- 2 supplementary: ViT-patch32-gap, ViT-patch16-cls

---

## Sebastian's QSVM Jobs (already submitted)

```bash
module load miniforge/24.3.0-0 && module load cuda/12.4.0
conda activate qml-medimage
cd /orcd/pool/005/sebasmos/code/QML-MedImage

# === Main models: MedSigLIP-448 (seeds 0-9, q2,4,8,9,10,11,12,16) ===
for q in 2 4 8 9 10 11 12 16; do for s in 0 1 2 3 4 5 6 7 8 9; do
    QUBITS=$q SEED=$s sbatch slurm/multiseed_medsig_dt9.sh
done; done

# === Main models: RAD-DINO (seeds 0-9, q2,4,8,9,10,11,12,16) ===
# Submitted to mit_normal_gpu for faster allocation
for q in 2 4 8 9 10 11 12 16; do for s in 0 1 2 3 4 5 6 7 8 9; do
    QUBITS=$q SEED=$s sbatch slurm/multiseed_raddino_dt9_normal.sh
done; done

# === Main models: ViT-patch32-cls (seeds 0-9, q2,4,8,9,10,11,12,16) ===
for q in 2 4 8 9 10 11 12 16; do for s in 0 1 2 3 4 5 6 7 8 9; do
    QUBITS=$q SEED=$s sbatch slurm/multiseed_vit_dt9.sh
done; done

# === Supplementary: ViT-patch32-gap (seeds 0-9, q2,4,8,9,10,11,12,16) ===
for q in 2 4 8 9 10 11 12 16; do for s in 0 1 2 3 4 5 6 7 8 9; do
    QUBITS=$q SEED=$s sbatch slurm/multiseed_vit32gap_dt9.sh
done; done

# === Supplementary: ViT-patch16-cls (seeds 0-9, q2,4,8,9,10,11,12,16) ===
for q in 2 4 8 9 10 11 12 16; do for s in 0 1 2 3 4 5 6 7 8 9; do
    QUBITS=$q SEED=$s sbatch slurm/multiseed_vit16cls_dt9.sh
done; done
```

**Sebastian total: 400 jobs (8 qubit counts x 10 seeds x 5 models)**

## Sebastian's Classical SVM Jobs (already completed)

```bash
# CPU partition, no GPU needed. All 5 models, all seeds, all PCA dims.
sbatch slurm/classical_svm_multiseed.sh
```

**1,100 configs completed. Results: `tests/multiseed_classical/all_results_summary.csv`**

---

## Felipe's QSVM Jobs (TO RUN)

Felipe runs q3, q5, q6, q9, q10 with all 10 seeds. Sebastian handles q2, q4, q8, q11, q12, q16. This balances the workload (~3-4 days each).

### Setup

```bash
# 1. Clone the repo (or pull latest if already cloned)
cd /orcd/pool/005/felipeos/code/
git clone https://github.com/sebasmos/QML-MedImage-draft.git QML-MedImage
cd QML-MedImage
git checkout main

# 2. Activate environment
module load miniforge/24.3.0-0 && module load cuda/12.4.0
conda activate qml-medimage
```

### Submit jobs (seeds 0-9, q3,q5,q6,q9,q10, all 5 models)

**IMPORTANT: Use the `felipe_` prefixed scripts. These save to `tests/multiseed_felipe_v2/` to avoid conflicts with Sebastian's results.**

```bash
# === Main: MedSigLIP-448 q3,q5,q6,q9,q10 (50 jobs) ===
for q in 3 5 6 9 10; do for s in 0 1 2 3 4 5 6 7 8 9; do
    QUBITS=$q SEED=$s sbatch slurm/felipe_multiseed_medsig_dt9.sh
done; done

# === Main: RAD-DINO q3,q5,q6,q9,q10 (50 jobs) ===
for q in 3 5 6 9 10; do for s in 0 1 2 3 4 5 6 7 8 9; do
    QUBITS=$q SEED=$s sbatch slurm/felipe_multiseed_raddino_dt9.sh
done; done

# === Main: ViT-patch32-cls q3,q5,q6,q9,q10 (50 jobs) ===
for q in 3 5 6 9 10; do for s in 0 1 2 3 4 5 6 7 8 9; do
    QUBITS=$q SEED=$s sbatch slurm/felipe_multiseed_vit_dt9.sh
done; done

# === Supplementary: ViT-patch32-gap q3,q5,q6,q9,q10 (50 jobs) ===
for q in 3 5 6 9 10; do for s in 0 1 2 3 4 5 6 7 8 9; do
    QUBITS=$q SEED=$s sbatch slurm/felipe_multiseed_vit32gap_dt9.sh
done; done

# === Supplementary: ViT-patch16-cls q3,q5,q6,q9,q10 (50 jobs) ===
for q in 3 5 6 9 10; do for s in 0 1 2 3 4 5 6 7 8 9; do
    QUBITS=$q SEED=$s sbatch slurm/felipe_multiseed_vit16cls_dt9.sh
done; done
```

**Felipe total: 250 jobs (5 qubit counts x 10 seeds x 5 models)**

### Expected runtime
- q3,q5,q6: ~30 min each
- q9,q10: ~1-1.5h each
- With 3 GPU quota: ~67h = ~2.8 days

### Output locations
Results saved to a SEPARATE folder (`multiseed_felipe_v2`) to avoid conflicts:
- `tests/multiseed_felipe_v2/medsiglip-448/data_type9/q{3,5,6,9,10}/seed_{0-9}/`
- `tests/multiseed_felipe_v2/rad-dino/data_type9/q{3,5,6,9,10}/seed_{0-9}/`
- `tests/multiseed_felipe_v2/vit-patch32-cls/data_type9/q{3,5,6,9,10}/seed_{0-9}/`
- `tests/multiseed_felipe_v2/vit-patch32-gap/data_type9/q{3,5,6,9,10}/seed_{0-9}/`
- `tests/multiseed_felipe_v2/vit-patch16-cls/data_type9/q{3,5,6,9,10}/seed_{0-9}/`

### Monitor progress
```bash
squeue -u felipeos
find tests/multiseed_medsig tests/multiseed_raddino tests/multiseed_vit tests/multiseed_vit32gap tests/multiseed_vit16cls -name "metrics_summary.csv" | wc -l
```

---

## Coverage Matrix

| q | Sebastian | Felipe | Seeds |
|---|---|---|---|
| 2 | x | | 0-9 |
| 3 | | x | 0-9 |
| 4 | x | | 0-9 |
| 5 | | x | 0-9 |
| 6 | | x | 0-9 |
| 8 | x | | 0-9 |
| 9 | | x | 0-9 |
| 10 | | x | 0-9 |
| 11 | x | | 0-9 |
| 12 | x | | 0-9 |
| 16 | x | | 0-9 |

All 5 models at every qubit count, all 10 seeds.

---

## Full Job Summary

| Model | Role | Sebastian (q2,4,8,11,12,16) | Felipe (q3,5,6,9,10) | Total |
|---|---|---|---|---|
| MedSigLIP-448 | Main | 60 jobs | 50 jobs | 110 |
| RAD-DINO | Main | 60 jobs | 50 jobs | 110 |
| ViT-patch32-cls | Main | 60 jobs | 50 jobs | 110 |
| ViT-patch32-gap | Supplementary | 60 jobs | 50 jobs | 110 |
| ViT-patch16-cls | Supplementary | 60 jobs | 50 jobs | 110 |
| **QSVM Total** | | **300 jobs** | **250 jobs** | **550 jobs** |
| Classical SVM | CPU (Sebastian) | 1,100 configs | | 1,100 |

---

## Key parameters (DO NOT CHANGE)

- **normalize_method**: trace (critical for paper claims)
- **c_values**: 1.0 (Tier 1 purity)
- **data_type_filter**: data_type9 (DT9 = Uncertainty Coreset)
- **single_mode**: yes (1 GPU per job)
- **GPU**: H100 or H200

---

## SLURM Scripts

| Script | Model | Partition |
|---|---|---|
| `slurm/multiseed_medsig_dt9.sh` | MedSigLIP-448 | mit_preemptable |
| `slurm/multiseed_raddino_dt9.sh` | RAD-DINO | mit_preemptable |
| `slurm/multiseed_raddino_dt9_normal.sh` | RAD-DINO | mit_normal_gpu |
| `slurm/multiseed_vit_dt9.sh` | ViT-patch32-cls | mit_preemptable |
| `slurm/multiseed_vit_dt9_normal.sh` | ViT-patch32-cls | mit_normal_gpu |
| `slurm/multiseed_vit32gap_dt9.sh` | ViT-patch32-gap | mit_preemptable |
| `slurm/multiseed_vit16cls_dt9.sh` | ViT-patch16-cls | mit_preemptable |
| `slurm/classical_svm_multiseed.sh` | All 5 (CPU) | mit_normal |

---

## After completion

Copy Felipe's results to Sebastian's repo:
```bash
cp -r /orcd/pool/005/felipeos/code/QML-MedImage/tests/multiseed_felipe_v2/ \
      /orcd/pool/005/sebasmos/code/QML-MedImage/tests/
```

Result folders:
- **Sebastian's**: `tests/multiseed_medsig/`, `tests/multiseed_raddino/`, `tests/multiseed_vit/`, `tests/multiseed_vit32gap/`, `tests/multiseed_vit16cls/`
- **Felipe's**: `tests/multiseed_felipe_v2/` (all models in one folder)

## WARNING

Felipe's OLD results at `/orcd/pool/006/lceli_shared/ResultsFelipe/QML/April/` used
NO trace normalization and are NOT comparable to the paper. Do NOT use them.
All multi-seed runs MUST use `--normalize_method trace`.
