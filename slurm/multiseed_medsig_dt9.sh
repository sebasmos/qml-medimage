#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --job-name=mseed_medsig
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

###############################################################################
# Multi-seed MedSigLIP-448 DT9 rerun  (#55 — fix Felipe's seeds 1-9 failures)
#
# Usage:  QUBITS=4 SEED=1 sbatch slurm/multiseed_medsig_dt9.sh
#
# Parameters match paper claims: trace normalization, C=1.0 (Tier 1 purity).
# One GPU per seed — submit 4 qubit counts x 9 seeds = 36 jobs.
###############################################################################

trap 'echo "Signal received at $(date)"; scontrol requeue $SLURM_JOB_ID; exit 0' USR1

### Environment Setup
cd /home/sebasmos/orcd/pool/code/QML-MedImage
module load miniforge/24.3.0-0
module load cuda/12.4.0
conda activate qml-medimage

### Configuration (set via env or defaults)
QUBITS="${QUBITS:-4}"
SEED="${SEED:-0}"
MODEL="medsiglip-448"
DATA_DIR="/orcd/pool/006/lceli_shared/DATASET/qml-mimic-cxr-embeddings/medsiglip-448-embeddings/20-seeds/seed_${SEED}"
DATA_FILTER="data_type9"
OUTPUT_DIR="tests/multiseed_medsig/${MODEL}/data_type9/q${QUBITS}/seed_${SEED}"
LOG_DIR="tests/multiseed_medsig/logs/${MODEL}"
LOG_FILE="${LOG_DIR}/q${QUBITS}_seed${SEED}_${SLURM_JOB_ID}.log"
COMPLETION_FLAG="${LOG_DIR}/.completed_q${QUBITS}_seed${SEED}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Multi-seed rerun (#55) — ${MODEL} q${QUBITS} seed_${SEED} ==="
echo "  Model:            ${MODEL}"
echo "  Qubits:           ${QUBITS}"
echo "  Seed:             ${SEED}"
echo "  Normalize method: trace"
echo "  C value:          1.0 (forced, Tier 1)"
echo "  Data dir:         ${DATA_DIR}"
echo "  Data filter:      ${DATA_FILTER}"
echo "  Output:           ${OUTPUT_DIR}"
echo "  Job ID:           ${SLURM_JOB_ID}"
echo "  Node:             $(hostname)"
echo "  Attempt:          ${SLURM_RESTART_COUNT:-0}"
date

if [ -f "$COMPLETION_FLAG" ]; then
    echo "Job already completed. Exiting."
    exit 0
fi

python -u scripts/qsvm_cuda_embeddings_insurance.py \
    --data_path        "$DATA_DIR" \
    --data_type_filter "$DATA_FILTER" \
    --output_dir       "$OUTPUT_DIR" \
    --qubits           "$QUBITS" \
    --normalize_method trace \
    --c_values         "1.0" \
    --num_seeds        1 \
    --single_mode

PYTHON_EXIT=$?

if [ $PYTHON_EXIT -eq 0 ]; then
    touch "$COMPLETION_FLAG"
    echo "=== COMPLETED q${QUBITS} seed_${SEED} at $(date) ==="
else
    echo "FAILED q${QUBITS} seed_${SEED} with exit code $PYTHON_EXIT"
    exit 1
fi
