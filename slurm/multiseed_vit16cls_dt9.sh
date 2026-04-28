#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --job-name=mseed_v16cls
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

trap 'echo "Signal received"; scontrol requeue $SLURM_JOB_ID; exit 0' USR1

cd /home/sebasmos/orcd/pool/code/QML-MedImage
module load miniforge/24.3.0-0 && module load cuda/12.4.0
conda activate qml-medimage

QUBITS="${QUBITS:-4}"
SEED="${SEED:-0}"
MODEL="vit-patch16-cls"
DATA_DIR="/orcd/pool/006/lceli_shared/DATASET/qml-mimic-cxr-embeddings/vit-base-patch16-224-embeddings/20-seeds/seed_${SEED}"
DATA_FILTER="data_type9+cls_embedding"
OUTPUT_DIR="tests/multiseed_vit16cls/${MODEL}/data_type9/q${QUBITS}/seed_${SEED}"
LOG_DIR="tests/multiseed_vit16cls/logs/${MODEL}"
LOG_FILE="${LOG_DIR}/q${QUBITS}_seed${SEED}_${SLURM_JOB_ID}.log"
COMPLETION_FLAG="${LOG_DIR}/.completed_q${QUBITS}_seed${SEED}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== Multi-seed (supplementary) — ${MODEL} q${QUBITS} seed_${SEED} ==="
date

if [ -f "$COMPLETION_FLAG" ]; then echo "Already completed."; exit 0; fi

python -u scripts/qsvm_cuda_embeddings_insurance.py \
    --data_path "$DATA_DIR" --data_type_filter "$DATA_FILTER" \
    --output_dir "$OUTPUT_DIR" --qubits "$QUBITS" \
    --normalize_method trace --c_values "1.0" --num_seeds 1 --single_mode

if [ $? -eq 0 ]; then touch "$COMPLETION_FLAG"; echo "COMPLETED at $(date)"; else echo "FAILED"; exit 1; fi
