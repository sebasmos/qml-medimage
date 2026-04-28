#!/bin/bash
###############################################################################
# Submit multi-seed MedSigLIP-448 DT9 QSVM jobs (#55)
#
# Submits 4 qubit counts x 9 seeds = 36 jobs.
# Each job uses 1 GPU (h100), ~3h walltime.
#
# Usage:  bash slurm/submit_multiseed_medsig.sh
###############################################################################

set -e

SCRIPT="slurm/multiseed_medsig_dt9.sh"
SUBMITTED=0
JOB_IDS=""

echo "=== Submitting MedSigLIP multi-seed rerun (#55) ==="
echo "  Qubit counts: 2, 4, 8, 16"
echo "  Seeds:        1-9 (seed_0 already done)"
echo "  Total jobs:   36"
echo ""

for q in 2 4 8 16; do
    for s in 1 2 3 4 5 6 7 8 9; do
        JOB_ID=$(QUBITS=$q SEED=$s sbatch --export=ALL,QUBITS=$q,SEED=$s "$SCRIPT" 2>&1 | grep -o '[0-9]*')
        echo "  q=${q} seed=${s} -> job ${JOB_ID}"
        JOB_IDS="${JOB_IDS} ${JOB_ID}"
        SUBMITTED=$((SUBMITTED + 1))
    done
done

echo ""
echo "=== Submitted ${SUBMITTED} jobs ==="
echo "Job IDs:${JOB_IDS}"
echo ""
echo "Monitor with: squeue -u \$USER -n mseed_medsig"
echo "Check logs:   ls tests/multiseed_medsig/logs/medsiglip-448/"
