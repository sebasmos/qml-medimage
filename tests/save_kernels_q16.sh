#!/bin/bash
# Submit --save_kernels QSVM job for q16 to capture quantum kernel matrices.
# Purpose: Eigenvalue spectra for paper — extends q4/q6/q11 analysis to q16.
#          Expected: eff_rank should be LOW → confirms kernel concentration.
#          Completes progression: q4=6.86 → q6=13.94 → q11=43.04 → q16=???
# Job: medsiglip-448 at q16, C=1, seed_0, data_type9.

set -euo pipefail

WORKER="tests/save_kernels_worker.sh"
MEDSIG_DATA="/orcd/pool/006/lceli_shared/DATASET/qml-mimic-cxr-embeddings/medsiglip-448-embeddings/20-seeds/seed_0/data_type9_n2371.parquet"

echo "=== Submitting save_kernels job: medsiglip-448 q16 ==="

# medsiglip-448 q16  (override time to 6h — kernel build at 16 qubits is expensive)
JOB1=$(QUBITS=16 MODEL_LABEL=medsiglip-448 DATA_DIR="$MEDSIG_DATA" \
    sbatch --parsable --time=06:00:00 "$WORKER")
echo "  medsiglip-448 q16 -> job $JOB1"

echo ""
echo "Submitted job: $JOB1"
echo "Monitor with: squeue -u sebasmos"
echo "Kernels will be saved to: tests/save_kernels/medsiglip-448/data_type9/q16/"
