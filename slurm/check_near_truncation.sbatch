#!/bin/bash
#SBATCH --job-name=near_check
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# Near-side 2% clip of the training target: its AbsRel cost at the LiDAR pixels, 2000 train frames
# (tools/check_near_truncation.py, docs/IRIS_SYNTH_PRETRAIN_V1.md). CPU only, no model, no val/test.
set -euo pipefail
F="${SCRATCH:?}/from_sc23sz"
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
export OMP_NUM_THREADS=4
git log --oneline -1
python tests/test_check_near_truncation.py
python -u tools/check_near_truncation.py --data-root "$IRIS_MS2_ROOT" \
    --train-manifest "$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl" \
    --pseudo-dir "${PSEUDO_DIR:-$F/pseudo_gt/official_train/calibrated_pseudo_depth}" \
    --out "$IRIS_RUNS/outputs/lotus_line_v2/near_truncation_v1/job_${SLURM_JOB_ID:?}"
