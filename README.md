# QML-MedImage: Quantum Advantage in Medical Insurance Classification

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Tests](https://img.shields.io/badge/Tests-pytest-orange.svg)](tests/)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-qml--mimic--cxr--embeddings-yellow.svg)](https://huggingface.co/datasets/MITCriticalData/qml-mimic-cxr-embeddings)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0%2B-6929C4.svg)](https://qiskit.org/)
[![cuQuantum](https://img.shields.io/badge/NVIDIA-cuQuantum-76B900.svg)](https://developer.nvidia.com/cuquantum-sdk)
[![arXiv](https://img.shields.io/badge/arXiv-2604.24597-b31b1b.svg)](https://arxiv.org/abs/2604.24597)

Quantum Support Vector Machine (QSVM) for binary insurance classification on
MIMIC-CXR chest radiographs using frozen embeddings from medical foundation
models.

Paper: [Quantum Kernel Advantage over Classical Collapse in Medical Foundation Model Embeddings](https://arxiv.org/abs/2604.24597)

## Table of Contents

- [Quick Start](#quick-start)
- [Running Locally (Python)](#running-locally-python)
- [SLURM Cluster Usage](#slurm-cluster-usage)
  - [Grid Launchers](#grid-launchers): [Classical SVM](slurm/0-svm-classical-grid-insurance/README.md) | [QSVM Baseline](slurm/1-qsvm-grid-insurance/README.md) | [Hybrid QSVM](slurm/2-hybrid-model-insurance/README.md) | [VQC](slurm/3-vqc-model-insurance/README.md)
  - [Single-Job Scripts](#single-job-scripts)
- [CLI Arguments](#cli-arguments)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Citation](#citation)

## Quick Start

```bash
# Create environment
conda create -n qml-medimage python=3.11 -y
conda activate qml-medimage

# Install package
cd /path/to/QML-MedImage
pip install -e .

# On HPC: load modules first
module load miniforge/24.3.0-0
module load cuda/12.4.0

# Verify installation
python -c "import sklearn; import qiskit; print('OK')"
```

## Running Locally (Python)

All scripts can be run directly with `python`. Set `QML_DATA_ROOT` to the
directory where you downloaded the embeddings from HuggingFace:

```bash
export QML_DATA_ROOT=/path/to/qml-mimic-cxr-embeddings
```

### Example 1: QSVM on MedSigLIP-448 (q=11, Tier-1 paper result)

```bash
python scripts/qsvm_cuda_embeddings_insurance.py \
    --data_path $QML_DATA_ROOT/medsiglip-448-embeddings/20-seeds/seed_0/data_type9_n2371.parquet \
    --output_dir results/qsvm-medsiglip-q11-seed0 \
    --qubits 11 \
    --normalize_method trace \
    --seed 0
```

### Example 2: Classical SVM baseline (Tier-1 comparison, C=1, all seeds)

```bash
python scripts/classical_svm_multiseed.py \
    --output_dir results/svm-baseline \
    --seeds 0,1,2,3,4,5,6,7,8,9 \
    --pca_dims 2,3,4,5,6,8,9,10,11,12,16
```

### Example 3: Tier-2 RBF rank-matched comparison

```bash
python scripts/rbf_rank_matched_multiseed.py \
    --output_dir results/rbf-rank-matched
```

### Example 4: Aggregate multi-seed results and run bootstrap

```bash
python scripts/aggregate_multiseed.py --run_dir results/qsvm-medsiglip-q11-seed0
python scripts/bootstrap_ci.py --run_dir results/qsvm-medsiglip-q11-seed0
```

### Results

Results are saved to the specified `--output_dir`:
```
output_dir/
├── metrics_summary.csv        # Accuracy, AUC, F1 scores per seed
├── confusion_matrix_test.csv  # Confusion matrix
└── dataset_info.json          # Run configuration
```

## SLURM Cluster Usage

For HPC clusters with SLURM scheduler and GPU nodes. Edit the `#SBATCH`
headers (partition, account) and `DATA_DIR` in each script to match your
cluster before submitting.

### Setup

```bash
module load miniforge/24.3.0-0
module load cuda/12.4.0
conda create -n qml-medimage python=3.11 -y
conda activate qml-medimage

cd /path/to/QML-MedImage
pip install -e .
```

### Grid Launchers

Each experiment type has a dedicated launcher:

- **Classical SVM** (baseline): see [slurm/0-svm-classical-grid-insurance/README.md](slurm/0-svm-classical-grid-insurance/README.md)
- **QSVM Baseline** (pure quantum kernel): see [slurm/1-qsvm-grid-insurance/README.md](slurm/1-qsvm-grid-insurance/README.md)
- **Hybrid QSVM** (quantum-classical kernel): see [slurm/2-hybrid-model-insurance/README.md](slurm/2-hybrid-model-insurance/README.md)
- **VQC** (Variational Quantum Classifier): see [slurm/3-vqc-model-insurance/README.md](slurm/3-vqc-model-insurance/README.md)

### Single-Job Scripts

```bash
sbatch slurm/qsvm_insurance_single-capped.sh           # QSVM quick test (100 samples)
sbatch slurm/qsvm_insurance_single.sh                  # QSVM full dataset
sbatch slurm/qsvm_insurance_multinode-single.sh        # QSVM multi-GPU (single seed)
sbatch slurm/qsvm_insurance_multinode-multiseed.sh     # QSVM multi-GPU (all seeds)
sbatch slurm/svm_insurance.sh                          # Classical SVM baseline
sbatch slurm/multiseed_medsig_dt9.sh                   # Multi-seed MedSigLIP-448
sbatch slurm/multiseed_raddino_dt9.sh                  # Multi-seed RAD-DINO
sbatch slurm/multiseed_vit_dt9.sh                      # Multi-seed ViT-patch32
sbatch slurm/rbf_rank_matched_multiseed.sh             # Tier-2 RBF comparison
```

### Monitor Jobs

```bash
squeue --me                                    # Check job status
sacct -j <JOB_ID> --format=JobID,State,MaxRSS  # Check memory usage
```

## CLI Arguments

### QSVM Scripts

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_path` | Path to .parquet file or directory | Required |
| `--output_dir` | Output directory | Required |
| `--qubits` | Number of qubits (= PCA dims) | 2 |
| `--max_samples` | Max training samples (None=all) | None |
| `--seed` | Random seed | 42 |
| `--normalize_method` | Normalization: `trace`, `minmax`, `none` | `trace` |

### Classical SVM Scripts

| Argument | Description | Default |
|----------|-------------|---------|
| `--output_dir` | Output directory | Required |
| `--seeds` | Comma-separated seed list | `0,1,...,9` |
| `--pca_dims` | Comma-separated PCA dimension list | `2,3,4,5,6,8,9,10,11,12,16` |

## Project Structure

```
QML-MedImage/
├── scripts/                              # Training and analysis scripts
│   ├── qsvm_cuda_embeddings_insurance.py # Main QSVM (GPU, multi-seed)
│   ├── classical_svm_multiseed.py        # Tier-1 classical baseline (C=1)
│   ├── classical_svm_c1_pca.py          # Single-seed C=1 extended sweep
│   ├── rbf_rank_matched_multiseed.py    # Tier-2 RBF comparison
│   ├── aggregate_multiseed.py           # Aggregate seed results
│   ├── bootstrap_ci.py                  # Confidence intervals
│   ├── paired_bootstrap_q11.py          # q=11 significance test
│   ├── regen_eigenspectrum_fig.py       # Figure regeneration
│   ├── regen_qubit_scaling_fig.py
│   └── regen_scatter_figs.py
├── slurm/                               # SLURM job scripts
│   ├── 0-svm-classical-grid-insurance/
│   ├── 1-qsvm-grid-insurance/
│   ├── 2-hybrid-model-insurance/
│   └── 3-vqc-model-insurance/
├── qve/                                 # Quantum kernel module
│   ├── core.py                          # QSVM kernel computation
│   ├── metrics.py                       # Evaluation metrics
│   ├── process.py                       # Data processing
│   └── utils.py
├── pre-processing/                      # Data preparation
│   └── pca-pipeline/                    # PCA reduction scripts
├── tests/                               # Test suite
├── figures/                             # Paper figures
└── docs/
```

## Testing

```bash
# All tests (GPU tests auto-skip if unavailable)
pytest tests/ -v

# Basic tests only (no GPU required)
pytest tests/test_basic.py -v

# GPU tests on HPC
srun --gres=gpu:1 pytest tests/ -v
```

| Test File | GPU Required | Description |
|-----------|--------------|-------------|
| `test_basic.py` | No | Core imports, qve module, sklearn integration |
| `test_imports.py` | Yes | cuQuantum/cupy imports, GPU functionality |
| `test_script_imports.py` | Yes | Full script import chain, circuit conversion |
| `test_qsvm_quick.py` | Yes | End-to-end QSVM with test data |

## Dataset

Pre-computed embeddings (20 seeds × 3 models) are on HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("MITCriticalData/qml-mimic-cxr-embeddings")
```

Or download directly for local use:

```bash
export QML_DATA_ROOT=/path/to/store/embeddings
huggingface-cli download MITCriticalData/qml-mimic-cxr-embeddings \
    --repo-type dataset --local-dir $QML_DATA_ROOT
```

Raw MIMIC-CXR-JPG images require credentialed PhysioNet access:
[physionet.org/content/mimic-cxr-jpg](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

## Requirements

- Python 3.10–3.11
- CUDA 12.x + cuQuantum 24.8 (for QSVM GPU acceleration)
- qiskit >= 1.2.4
- scikit-learn 1.6.1
- pyarrow >= 15.0 (parquet loading)
- mpi4py (multi-GPU, optional)

Install all pinned dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

See `requirements.txt` for the full pinned list.

## Contributing

See [docs/contributing.md](docs/contributing.md) for guidelines.

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{cajas2026qml,
  title   = {Quantum Kernel Advantage over Classical Collapse in Medical
             Foundation Model Embeddings},
  author  = {Cajas Ord\'{o}\~{n}ez, Sebasti\'{a}n A. and Ocampo Osorio, Felipe
             and Koh, Dax Enshan and Al Attrach, Rafi and Marzullo, Aldo
             and Guerra-Adames, Ariel and Andrade, J. Alejandro and Goh, Siong Thye
             and Chen, Chi-Yu and Gorijavolu, Rahul and Yang, Xue
             and Hebdon, Noah Dane and Celi, Leo Anthony},
  journal = {arXiv preprint arXiv:2604.24597},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.24597}
}
```

Based on [QuantumVE](https://github.com/sebasmos/QuantumVE).
