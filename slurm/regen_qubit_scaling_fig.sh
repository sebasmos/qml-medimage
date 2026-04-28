#!/bin/bash
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --job-name=regen_qsweep
#SBATCH --output=tests/analysis/multiseed_aggregate/logs/regen_qsweep_%j.log
#SBATCH --error=tests/analysis/multiseed_aggregate/logs/regen_qsweep_%j.err

cd /home/sebasmos/orcd/pool/code/QML-MedImage
module load miniforge/24.3.0-0
conda activate qml-medimage

mkdir -p tests/analysis/multiseed_aggregate/logs

echo "=== Regenerate qubit_scaling_curve.png (no annotation box) ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

python -u scripts/regen_qubit_scaling_fig.py

echo "Done: $(date)"
