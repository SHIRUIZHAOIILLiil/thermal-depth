#!/bin/bash
# Submit the four "where do we lose" diagnostics for one trained arm, from an AIRE
# login node. Diagnostic only; nothing here touches test.
#
#   1-3. slurm/diagnose_val.sbatch: our model and NeWCRF on val, then depth-range /
#        boundary / bad-frame breakdown (tools/diagnose_error_sources.py).
#   4.   gradient probe: resume the arm's training checkpoints (no step taken) and
#        measure how the latent-depth, reconstruction and image-L1 gradients align
#        on the shared U-Net weights, at an early and a late checkpoint.
#
# Usage:
#   bash ~/Iris/slurm/submit_diagnostics.sh sd2_cap_pct_logtn_img_l0100_s43 0.1
#     $1 run name under $IRIS_RUNS/iris_ms2     $2 its lambda_image
#   Optional env: DIAG_STEP (default 16000), PROBE_STEPS (default "4000 16000"),
#                 PROBE_BATCHES (default 50), PROBE_ONLY=1 (skip val inference)
set -euo pipefail

RUN="${1:?usage: $0 <run> <lambda_image>}"
LAMBDA="${2:?usage: $0 <run> <lambda_image>}"
F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
export PSEUDO_DIR="${PSEUDO_DIR:-$F/pseudo_gt/official_train/calibrated_pseudo_depth}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/env.sh"
cd "$IRIS_REPO"

RUN_DIR="$IRIS_RUNS/iris_ms2/$RUN"
DIAG_STEP="${DIAG_STEP:-16000}"
CKPT="$RUN_DIR/converted/step${DIAG_STEP}_weights.pt"
if [[ "${PROBE_ONLY:-0}" != "1" ]]; then
  [[ -s "$CKPT" ]] || { echo "!! 找不到 $CKPT"; exit 1; }
  J=$(sbatch --parsable -J "diag_val_${RUN:0:20}" --export=ALL,CKPT="$CKPT" slurm/diagnose_val.sbatch)
  echo "1-3 val 拆解      -> $J   ($CKPT)"
fi

# Same switches as the arm was trained with (the λ sweep's control config), so the
# probed objective is the one that produced the checkpoint.
export START=sd2 SEED=43 NO_CAPTIONS=0 RANDOM_FLIP=1 PURE_PSEUDO=0
export NORMTYPE=log_truncnorm PREDICTION_TYPE=sample
export IRIS_THERMAL_STRETCH=percentile IRIS_TEXT_PADDING=legacy
export LAMBDA_IMAGE="$LAMBDA" LAMBDA_TAIL=0 TAIL_LOSS=none TAIL_ALIGN=none TAIL_HUBER_DELTA_M=5
export TRAIN_MANIFEST="$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl"
export MAX_TRAIN_STEPS=20000 CKPT_STEP=1000 SMOKE=0
export EXTRA_TRAIN_ARGS="--grad_probe_batches ${PROBE_BATCHES:-50} --grad_probe_seed ${PROBE_SEED:-20261014}"
PROBE_ATTEMPT="${PROBE_ATTEMPT:-$(date +%Y%m%dT%H%M%S)_$$}"
for s in ${PROBE_STEPS:-4000 16000}; do
  [[ -d "$RUN_DIR/checkpoint-$s" ]] || { echo "!! 找不到 $RUN_DIR/checkpoint-$s，跳过"; continue; }
  export RUN_TAG="gradprobe_${RUN}_ckpt${s}_${PROBE_ATTEMPT}" RESUME_FROM="$RUN_DIR/checkpoint-$s"
  J=$(sbatch --parsable -J "gprobe_$s" --time=02:00:00 --export=ALL slurm/iris_ms2.sbatch)
  echo "4  梯度探针 ckpt$s -> $J   (结果: $IRIS_RUNS/iris_ms2/$RUN_TAG/grad_probe_summary.json)"
done
