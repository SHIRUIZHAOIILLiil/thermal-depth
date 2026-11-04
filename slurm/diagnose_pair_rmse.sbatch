#!/bin/bash
#SBATCH --job-name=iris_rmse_diag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
F="${SCRATCH:?}/from_sc23sz"
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
base="$IRIS_RUNS/outputs/lotus_line_v2"
out="$base/iris_pair_rmse_val_v1/job_${SLURM_JOB_ID:?}"
echo "Output: $out"
python tests/test_pair_rmse_attribution.py
python -u tools/diagnose_pair_rmse.py \
    --local-run "$base/iris_far_pairs_l050_frame4k_v1/job_9163317" \
    --cross-run "$base/iris_cross_region_l050_frame_v1/job_9163371" \
    --diag-root "$IRIS_RUNS/analysis/diag_val" --data-root "$IRIS_MS2_ROOT" \
    --baselines "$SCRATCH/baselines" --out "$out"
