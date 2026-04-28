#!/bin/bash
# Submit --save_kernels QSVM job for q11 to capture quantum kernel matrices.
# Purpose: Eigenvalue spectra for paper — extends q4/q6 analysis to q11.
# Job: medsiglip-448 at q11, C=1, seed_0, data_type9.

set -euo pipefail

WORKER="tests/save_kernels_worker.sh"
MEDSIG_DATA="/orcd/pool/006/lceli_shared/DATASET/qml-mimic-cxr-embeddings/medsiglip-448-embeddings/20-seeds/seed_0/data_type9_n2371.parquet"

echo "=== Submitting save_kernels job: medsiglip-448 q11 ==="

# medsiglip-448 q11  (override time to 3h — kernel build at 11 qubits takes ~1.5h)
JOB1=$(QUBITS=11 MODEL_LABEL=medsiglip-448 DATA_DIR="$MEDSIG_DATA" \
    sbatch --parsable --time=03:00:00 "$WORKER")
echo "  medsiglip-448 q11 -> job $JOB1"

echo ""
echo "Submitted job: $JOB1"
echo "Monitor with: squeue -u sebasmos"
echo "Kernels will be saved to: tests/save_kernels/medsiglip-448/data_type9/q11/"
