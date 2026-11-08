#!/bin/bash
#SBATCH --job-name=iris_tv_gap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
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
python tests/test_train_val_gap.py
python -u tools/diagnose_train_val_gap.py --runs "$IRIS_RUNS" --data-root "$IRIS_MS2_ROOT" \
    --out "$IRIS_RUNS/outputs/lotus_line_v2/iris_train_val_gap_v1/job_${SLURM_JOB_ID:?}"
