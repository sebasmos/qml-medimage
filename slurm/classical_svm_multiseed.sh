#!/bin/bash
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=svm_mseed
#SBATCH --output=tests/multiseed_classical/logs/svm_multiseed_%j.log
#SBATCH --error=tests/multiseed_classical/logs/svm_multiseed_%j.err

cd /home/sebasmos/orcd/pool/code/QML-MedImage
module load miniforge/24.3.0-0
conda activate qml-medimage

mkdir -p tests/multiseed_classical/logs

echo "=== Classical SVM multi-seed (5 models × 11 PCA × 10 seeds) ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

python -u scripts/classical_svm_multiseed.py \
    --output_dir tests/multiseed_classical/ \
    --seeds "0,1,2,3,4,5,6,7,8,9" \
    --pca_dims "2,3,4,5,6,8,9,10,11,12,16"

echo "Done: $(date)"
