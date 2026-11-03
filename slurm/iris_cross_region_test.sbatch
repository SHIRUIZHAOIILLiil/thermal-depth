#!/bin/bash
#SBATCH --job-name=iris_cross_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
set -euo pipefail
F="${SCRATCH:?}/from_sc23sz"
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
conditions=(day night rain)
base="$IRIS_RUNS/outputs/lotus_line_v2"
python -u tools/test_iris_cross_region_pairs.py --condition "${conditions[$SLURM_ARRAY_TASK_ID]}" \
    --run "$base/iris_cross_region_l050_frame_v1/job_${1:?}" --parent "$base/iris_far_pairs_l050_frame4k_v1/job_${2:?}" \
    --data-root "$IRIS_MS2_ROOT" --manifests "$IRIS_MANIFEST_DIR"
