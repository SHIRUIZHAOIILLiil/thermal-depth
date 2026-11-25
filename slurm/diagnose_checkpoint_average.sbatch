#!/bin/bash
#SBATCH --job-name=iris_ckpt_avg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
set -euo pipefail
F="${SCRATCH:?}/from_sc23sz"
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
cd "$IRIS_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
python tests/test_checkpoint_average.py
# SOURCE_RUN: tmax1 (default, job 9229303) or the synth arm (docs/SYNTH_POSTHOC_V1.md).
SOURCE_RUN="${SOURCE_RUN:-sd2_cap_pct_logtn_img_l0100_tmax1_s43}"
case "$SOURCE_RUN" in
  sd2_cap_pct_logtn_img_l0100_tmax1_s43) OUT_BASE=iris_checkpoint_average_v1 ;;
  sd2_cap_pct_logtn_img_l0100_tmax1_synth_s43) OUT_BASE=iris_checkpoint_average_synth_v1 ;;
  sd2_nocap_pct_logtn_img_l0100_tmax1_s43) OUT_BASE=iris_checkpoint_average_tmax1_nocap_v1 ;;
  sd2_cap_pct_logtn_img_l0100_s43) OUT_BASE=iris_checkpoint_average_l100_v1 ;;
  sd2_nocap_pct_logtn_img_l0100_s43) OUT_BASE=iris_checkpoint_average_l100_nocap_v1 ;;
  *) echo "!! no control registered for $SOURCE_RUN"; exit 1 ;;
esac
# SAVE_RULE=not_worse: save and use the average when no headline metric gets worse (docs/SD2_POSTHOC_V1.md).
SAVE_RULE="${SAVE_RULE:-registered}"
if [[ "$SAVE_RULE" == not_worse && "$OUT_BASE" == iris_checkpoint_average_v1 ]]; then OUT_BASE=iris_checkpoint_average_tmax1_v2; fi
python -u tools/diagnose_checkpoint_average.py --runs "$IRIS_RUNS" --data-root "$IRIS_MS2_ROOT" \
    --source "$IRIS_RUNS/iris_ms2/$SOURCE_RUN" --save-rule "$SAVE_RULE" \
    --out "$IRIS_RUNS/outputs/lotus_line_v2/$OUT_BASE/job_${SLURM_JOB_ID:?}"
