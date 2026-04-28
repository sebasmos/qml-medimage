#!/bin/bash
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=rbf_rank
#SBATCH --output=tests/analysis/rbf_rank_matched/logs/rbf_rank_%j.log
#SBATCH --error=tests/analysis/rbf_rank_matched/logs/rbf_rank_%j.err

cd /home/sebasmos/orcd/pool/code/QML-MedImage
module load miniforge/24.3.0-0
conda activate qml-medimage

mkdir -p tests/analysis/rbf_rank_matched/logs

echo "=== Rank-matched RBF experiment (LEO-7) ==="
echo "MedSigLIP-448 at q=4,6,11,16 — finds gamma* s.t. eff_rank(RBF)==eff_rank(K_Q)"
echo "Start: $(date)"
echo "Node: $(hostname)"

python -u scripts/rbf_rank_matched.py \
    --output_dir tests/analysis/rbf_rank_matched \
    --seed 0

echo "Done: $(date)"
