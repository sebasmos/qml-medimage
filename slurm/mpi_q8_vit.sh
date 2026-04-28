#!/bin/bash
#SBATCH --partition=mit_preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --job-name=qsvm_q8_vit
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

#########################
### SIGNAL HANDLING   ###
#########################
trap 'echo "Signal received at $(date)"; scontrol requeue $SLURM_JOB_ID; exit 0' USR1

### Environment Setup
cd /home/sebasmos/orcd/pool/code/QML-MedImage
module load miniforge/24.3.0-0
module load cuda/12.4.0
conda activate qml-medimage

### Configuration
QUBITS=8
MODEL="vit-patch32-cls"
DATA_DIR="/orcd/pool/006/lceli_shared/DATASET/qml-mimic-cxr-embeddings/vit-base-patch32-224-embeddings/20-seeds"
DATA_FILTER="data_type9"
OUTPUT_DIR="tests/mpi_q8_models/${MODEL}/data_type9/q${QUBITS}"
LOG_DIR="tests/mpi_q8_models/logs/${MODEL}"
LOG_FILE="${LOG_DIR}/q${QUBITS}_${SLURM_JOB_ID}.log"
COMPLETION_FLAG="${LOG_DIR}/.completed_q${QUBITS}"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== q${QUBITS} C-tuning (single GPU, no MPI) — ${MODEL} ==="
echo "  Model:            ${MODEL}"
echo "  Qubits:           ${QUBITS}"
echo "  Normalize method: trace"
echo "  Data type:        ${DATA_FILTER}"
echo "  GPUs:             1 (h100, single_mode)"
echo "  Memory:           200G"
echo "  C grid:           0.01,0.1,1,10,100"
echo "  Goal:             fill q8 gap in qubit scaling curve (issue #45)"
echo "  Data dir:         ${DATA_DIR}"
echo "  Output:           ${OUTPUT_DIR}"
date

if [ -f "$COMPLETION_FLAG" ]; then
    echo "Job already completed. Exiting."
    exit 0
fi

python -u scripts/qsvm_cuda_embeddings_insurance.py \
    --data_path      "$DATA_DIR" \
    --data_type_filter "$DATA_FILTER" \
    --output_dir     "$OUTPUT_DIR" \
    --qubits         "$QUBITS" \
    --normalize_method trace \
    --c_values       "0.01,0.1,1,10,100" \
    --num_seeds      1 \
    --single_mode

PYTHON_EXIT=$?

if [ $PYTHON_EXIT -eq 0 ]; then
    touch "$COMPLETION_FLAG"
    echo "=== COMPLETED q${QUBITS} at $(date) ==="
else
    echo "FAILED with exit code $PYTHON_EXIT"
    exit 1
fi
