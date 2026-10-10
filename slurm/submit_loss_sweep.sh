#!/bin/bash
# Submit one loss experiment stage at a time from an AIRE login node.
# Usage:
#   bash ~/Iris/slurm/submit_loss_sweep.sh lambda-smoke
#   bash ~/Iris/slurm/submit_loss_sweep.sh lambda
#   bash ~/Iris/slurm/submit_loss_sweep.sh shape-smoke 0.05
#   bash ~/Iris/slurm/submit_loss_sweep.sh shape 0.05
#   bash ~/Iris/slurm/submit_loss_sweep.sh aligned-smoke 0.05
#   bash ~/Iris/slurm/submit_loss_sweep.sh aligned 0.05
set -euo pipefail

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

STAGE="${1:-}"
BEST_LAMBDA="${2:-}"
SUBMIT_ID="${LOSS_SWEEP_SUBMIT_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
TRAIN_MANIFEST="$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl"
VAL_MANIFEST="$IRIS_MANIFEST_DIR/ms2_val_official3_thermalcap_20260821.jsonl"
[[ -f "$TRAIN_MANIFEST" ]] || { echo "missing $TRAIN_MANIFEST"; exit 1; }
[[ -f "$VAL_MANIFEST" ]] || { echo "missing $VAL_MANIFEST"; exit 1; }

export START=sd2 SEED=43 NO_CAPTIONS=0 RANDOM_FLIP=1 PURE_PSEUDO=0
export NORMTYPE=log_truncnorm PREDICTION_TYPE=sample
export IRIS_THERMAL_STRETCH=percentile IRIS_TEXT_PADDING=legacy
export MAX_TRAIN_STEPS=20000 CKPT_STEP=1000 TRAIN_MANIFEST
export STEPS=16000 FIXED_STEP=16000 VAL_ONLY=1 VAL_STRIDE=1
export SEL_PROMPT=correct TEST_PROMPTS=correct ALIGN_MODE=ssi_log
export ROUTE=b_thermal_unet VAL_MANIFEST

train_exports="ALL,START,SEED,NO_CAPTIONS,RANDOM_FLIP,PURE_PSEUDO,NORMTYPE,PREDICTION_TYPE,IRIS_THERMAL_STRETCH,IRIS_TEXT_PADDING,MAX_TRAIN_STEPS,CKPT_STEP,TRAIN_MANIFEST,RUN_TAG,LAMBDA_IMAGE,LAMBDA_TAIL,TAIL_LOSS,TAIL_ALIGN,TAIL_HUBER_DELTA_M,SMOKE"
eval_exports="ALL,RUN,STEPS,FIXED_STEP,VAL_ONLY,VAL_STRIDE,SEL_PROMPT,TEST_PROMPTS,ALIGN_MODE,ROUTE,VAL_MANIFEST,TRAIN_MANIFEST,IRIS_THERMAL_STRETCH,IRIS_TEXT_PADDING"

lambda_code() {
  case "$1" in
    0.025) echo "0025" ;;
    0.05) echo "0050" ;;
    0.1) echo "0100" ;;
    0.2) echo "0200" ;;
    *) echo "unsupported lambda '$1'" >&2; return 1 ;;
  esac
}

lambda_run() {
  case "$1" in
    0.05) echo "sd2_cap_pct_logtn_img_s43" ;;
    *) echo "sd2_cap_pct_logtn_img_l$(lambda_code "$1")_s43" ;;
  esac
}

submit_eval() {
  local run="$1" dependency="${2:-}" jid
  local args=()
  export RUN="$run"
  [[ -n "$dependency" ]] && args+=(--dependency="afterok:$dependency")
  jid=$(sbatch --parsable -J "v_${run:0:28}" --time=05:00:00 "${args[@]}" \
    --export="$eval_exports" "$IRIS_REPO/slurm/iris_ms2_pipeline.sbatch")
  echo "  val $run -> $jid"
}

submit_smoke() {
  local tag="${1}_${SUBMIT_ID}" jid
  export RUN_TAG="$tag" SMOKE=1
  jid=$(sbatch --parsable -J "sm_${tag:0:27}" --time=00:45:00 \
    --export="$train_exports" "$IRIS_REPO/slurm/iris_ms2.sbatch")
  echo "$jid"
}

submit_train_and_val() {
  local run="$1" train_jid
  export RUN_TAG="$run" SMOKE=0
  train_jid=$(sbatch --parsable -J "tr_${run:0:27}" \
    --export="$train_exports" "$IRIS_REPO/slurm/iris_ms2.sbatch")
  echo "train $run -> $train_jid"
  submit_eval "$run" "$train_jid"
}

case "$STAGE" in
  lambda-smoke)
    export LAMBDA_IMAGE=0.1 LAMBDA_TAIL=0
    export TAIL_LOSS=none TAIL_ALIGN=none TAIL_HUBER_DELTA_M=5
    smoke=$(submit_smoke "loss_lambda_smoke_s43")
    echo "lambda smoke -> $smoke"
    echo "inspect the smoke gates before running the lambda stage"
    ;;
  lambda)
    export LAMBDA_TAIL=0 TAIL_LOSS=none TAIL_ALIGN=none TAIL_HUBER_DELTA_M=5
    submit_eval "sd2_cap_pct_logtn_img_s43"
    for value in 0.025 0.1 0.2; do
      export LAMBDA_IMAGE="$value"
      submit_train_and_val "$(lambda_run "$value")"
    done
    ;;
  shape-smoke|shape)
    [[ -n "$BEST_LAMBDA" ]] || {
      echo "usage: $0 {shape-smoke|shape} <best_lambda>"; exit 1;
    }
    code=$(lambda_code "$BEST_LAMBDA")
    export LAMBDA_IMAGE="$BEST_LAMBDA" LAMBDA_TAIL=0.01
    export TAIL_LOSS=huber TAIL_ALIGN=none TAIL_HUBER_DELTA_M=5
    run="sd2_cap_pct_logtn_img_l${code}_tailh001_s43"
    if [[ "$STAGE" == "shape-smoke" ]]; then
      smoke=$(submit_smoke "loss_shape_smoke_l${code}_s43")
      echo "shape smoke -> $smoke"
      echo "inspect the smoke gates before running the shape stage"
    else
      submit_train_and_val "$run"
      echo "compare against: $(lambda_run "$BEST_LAMBDA")"
    fi
    ;;
  aligned-smoke|aligned)
    [[ -n "$BEST_LAMBDA" ]] || {
      echo "usage: $0 {aligned-smoke|aligned} <best_lambda>"; exit 1;
    }
    code=$(lambda_code "$BEST_LAMBDA")
    export LAMBDA_IMAGE="$BEST_LAMBDA" LAMBDA_TAIL=0.01
    export TAIL_LOSS=huber TAIL_ALIGN=ssi_log TAIL_HUBER_DELTA_M=5
    run="sd2_cap_pct_logtn_img_l${code}_tailh001_ssilog_s43"
    if [[ "$STAGE" == "aligned-smoke" ]]; then
      smoke=$(submit_smoke "loss_aligned_smoke_l${code}_s43")
      echo "aligned smoke -> $smoke"
      echo "inspect the smoke gates before running the aligned stage"
    else
      submit_train_and_val "$run"
      echo "compare against: sd2_cap_pct_logtn_img_l${code}_tailh001_s43"
    fi
    ;;
  *)
    echo "usage: $0 {lambda-smoke|lambda|shape-smoke|shape|aligned-smoke|aligned} [best_lambda]"
    exit 1
    ;;
esac
