#!/bin/bash
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=agg_mseed
#SBATCH --output=tests/analysis/multiseed_aggregate/logs/aggregate_%j.log
#SBATCH --error=tests/analysis/multiseed_aggregate/logs/aggregate_%j.err

cd /home/sebasmos/orcd/pool/code/QML-MedImage
module load miniforge/24.3.0-0
conda activate qml-medimage

mkdir -p tests/analysis/multiseed_aggregate/logs

echo "=== Aggregate multi-seed (550 QSVM + 1100 classical) ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

python -u scripts/aggregate_multiseed.py \
    --qsvm-roots tests/multiseed_medsig tests/multiseed_raddino \
                 tests/multiseed_vit tests/multiseed_vit32gap \
                 tests/multiseed_vit16cls tests/multiseed_felipe_v2 \
    --classical-csv tests/multiseed_classical/all_results_summary.csv \
    --data-type data_type9 \
    --models medsiglip-448 rad-dino vit-patch32-cls vit-patch32-gap vit-patch16-cls \
    --output-dir tests/analysis/multiseed_aggregate \
    --bootstrap-iters 10000 \
    --bootstrap-seed 42

echo "Done: $(date)"
