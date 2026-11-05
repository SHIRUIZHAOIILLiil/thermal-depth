#!/bin/bash
#SBATCH --job-name=iris_far_restore
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
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
source_run="$base/iris_val_chain_bias4k_v1/job_9174706"
out="$base/iris_far_restore_chain_v1/job_${SLURM_JOB_ID:?}"
python tests/test_iris_far_restore.py
python -u tools/diagnose_iris_far_restore.py --source "$source_run" \
    --data-root "$IRIS_MS2_ROOT" --out "$out"
