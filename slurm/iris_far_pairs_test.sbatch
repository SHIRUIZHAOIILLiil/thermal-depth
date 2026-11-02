#!/bin/bash
#SBATCH --job-name=iris_fp_test
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
python -u tools/test_iris_far_pairs.py --run "${PAIR_RUN:?}" --condition "${conditions[$SLURM_ARRAY_TASK_ID]}" \
    --data-root "$IRIS_MS2_ROOT" --manifests "$IRIS_MANIFEST_DIR"
