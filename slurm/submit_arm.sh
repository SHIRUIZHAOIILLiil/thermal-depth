#!/bin/bash
# One new arm on the frozen l100 recipe, changed only by EXTRA_TRAIN_ARGS:
# smoke -> full training -> fixed-step full-val evaluation, chained with afterok so
# nothing has to be watched. afterok only sees exit codes: still read the smoke log.
#
# Usage:
#   bash ~/Iris/slurm/submit_arm.sh <run_tag> "<extra trainer args>"
#   bash ~/Iris/slurm/submit_arm.sh sd2_cap_pct_logtn_img_l0100_tmax1_s43 "--truncnorm_max 1.0"
#
# Compare afterwards against sd2_cap_pct_logtn_img_l0100_s43 at step 16000 with
# tools/select_loss_sweep.py (the command is printed at the end).
set -euo pipefail

RUN="${1:?usage: $0 <run_tag> \"<extra trainer args>\"}"
EXTRA="${2:?give the trainer arguments that make this arm different}"
F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
export PSEUDO_DIR="${PSEUDO_DIR:-$F/pseudo_gt/official_train/calibrated_pseudo_depth}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/env.sh"
cd "$IRIS_REPO"
[[ ! -e "$IRIS_RUNS/iris_ms2/$RUN" ]] || { echo "!! $IRIS_RUNS/iris_ms2/$RUN 已存在，换个名字"; exit 1; }

# The l100 recipe (λ sweep control config with λ_image = 0.1, no tail).
export START=sd2 SEED=43 NO_CAPTIONS=0 RANDOM_FLIP=1 PURE_PSEUDO=0
export NORMTYPE=log_truncnorm PREDICTION_TYPE=sample
export IRIS_THERMAL_STRETCH=percentile IRIS_TEXT_PADDING=legacy
export LAMBDA_IMAGE=0.1 LAMBDA_TAIL=0 TAIL_LOSS=none TAIL_ALIGN=none TAIL_HUBER_DELTA_M=5
export TRAIN_MANIFEST="$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl"
export VAL_MANIFEST="$IRIS_MANIFEST_DIR/ms2_val_official3_thermalcap_20260821.jsonl"
export MAX_TRAIN_STEPS=20000 CKPT_STEP=1000 EXTRA_TRAIN_ARGS="$EXTRA"

export RUN_TAG="${RUN}_smoke" SMOKE=1
SM=$(sbatch --parsable -J "sm_${RUN:0:26}" --time=00:45:00 --export=ALL slurm/iris_ms2.sbatch)
echo "smoke -> $SM"

export RUN_TAG="$RUN" SMOKE=0
TR=$(sbatch --parsable -J "tr_${RUN:0:26}" --dependency=afterok:$SM --export=ALL slurm/iris_ms2.sbatch)
echo "train -> $TR  (等 $SM)"

export RUN STEPS=16000 FIXED_STEP=16000 VAL_ONLY=1 VAL_STRIDE=1
export SEL_PROMPT=correct TEST_PROMPTS=correct ALIGN_MODE=ssi_log ROUTE=b_thermal_unet
VA=$(sbatch --parsable -J "v_${RUN:0:27}" --time=05:00:00 --dependency=afterok:$TR --export=ALL slurm/iris_ms2_pipeline.sbatch)
echo "val   -> $VA  (等 $TR)"

fp=$(sha256sum "$VAL_MANIFEST" | cut -c1-8)
S=eval_eval_affine_invariant_log_space.json
R="$IRIS_RUNS/eval"
echo
echo "冒烟跑完先看：grep -E 'truncnorm|^loss|^thermal' ~/Iris/sm_${RUN:0:26}-$SM.* ；不对就 scancel $TR $VA"
echo "全部完成后比较："
echo "python3 tools/select_loss_sweep.py --control l100=$R/sd2_cap_pct_logtn_img_l0100_s43_val_${fp}_stride1_correct_s16000/$S --candidate new=$R/${RUN}_val_${fp}_stride1_correct_s16000/$S"
