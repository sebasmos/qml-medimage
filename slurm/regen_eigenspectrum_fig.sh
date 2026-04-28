#!/bin/bash
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=regen_eigfig
#SBATCH --output=tests/analysis/multiseed_aggregate/logs/regen_eigfig_%j.log
#SBATCH --error=tests/analysis/multiseed_aggregate/logs/regen_eigfig_%j.err

cd /home/sebasmos/orcd/pool/code/QML-MedImage
module load miniforge/24.3.0-0
conda activate qml-medimage

mkdir -p tests/analysis/multiseed_aggregate/logs

echo "=== Regenerate quantum_vs_linear_eigenspectrum_q4q6.png ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

python -u scripts/regen_eigenspectrum_fig.py

echo "Done: $(date)"
