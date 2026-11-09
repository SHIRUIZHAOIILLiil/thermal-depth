#!/bin/bash
#SBATCH --job-name=iris_self_pseudo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
# MODE=shard as an array (--array=0-3, COUNT=4), then MODE=finalize as one task.
set -euo pipefail
F="${SCRATCH:?}/from_sc23sz"
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
OUT="$IRIS_RUNS/pseudo_gt/self_tmax1_16k"
python tests/test_build_self_pseudo_gt.py
python -u tools/build_self_pseudo_gt.py "${MODE:?MODE=shard|finalize}" --runs "$IRIS_RUNS" --data-root "$IRIS_MS2_ROOT" \
    --out "$OUT" --index "${SLURM_ARRAY_TASK_ID:-0}" --count "${COUNT:-1}"
