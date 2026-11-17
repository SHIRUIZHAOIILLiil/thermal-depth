#!/bin/bash
#SBATCH --job-name=iris_px_gap
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# Pixel-level synth-vs-NeWCRF gap on every 10th val frame (tools/val_pixel_gap.py). Val only.
set -euo pipefail
F="${SCRATCH:?}/from_sc23sz"
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
base="$IRIS_RUNS/outputs/lotus_line_v2"
python tests/test_val_pixel_gap_split.py
# v2 adds the exact per-band AbsRel / delta1 split; job 9489118 is the v1 output.
python -u tools/val_pixel_gap.py --runs "$IRIS_RUNS" --data-root "$IRIS_MS2_ROOT" \
    --newcrf "$base/newcrf_val_ssi_log_v1/job_9488948/per_frame.csv" \
    --previous "$base/iris_pair_rmse_val_v1/job_9167854" \
    --out "$base/val_pixel_gap_v2/job_${SLURM_JOB_ID:?}"
