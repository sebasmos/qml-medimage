#!/bin/bash
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=rbf_multiseed
#SBATCH --output=tests/analysis/rbf_rank_matched_multiseed/logs/rbf_multiseed_%j.log
#SBATCH --error=tests/analysis/rbf_rank_matched_multiseed/logs/rbf_multiseed_%j.err

cd /home/sebasmos/orcd/pool/code/QML-MedImage
module load miniforge/24.3.0-0
conda activate qml-medimage

mkdir -p tests/analysis/rbf_rank_matched_multiseed/logs

echo "=== Multi-seed rank-matched RBF experiment ==="
echo "MedSigLIP-448 at q=4,6,11,16 — 10 seeds each"
echo "Tests: default RBF vs rank-matched RBF vs QSVM (from CSV)"
echo "Key question: does rank-matched RBF collapse on the same seeds as classical?"
echo "Start: $(date)"
echo "Node: $(hostname)"

python -u scripts/rbf_rank_matched_multiseed.py \
    --output_dir tests/analysis/rbf_rank_matched_multiseed

echo "Done: $(date)"
