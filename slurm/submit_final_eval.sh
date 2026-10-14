#!/bin/bash
# Final evaluation of one frozen arm from an AIRE login node: val curve (stride 4,
# steps 2000-20000) picks the checkpoint by val AbsRel, then that one checkpoint is
# tested once on day, night and rain. The day job converts, runs the curve and the
# day test; night and rain wait for it and reuse the curve, so all three share one
# selection and no test number can feed back into it.
#
# Usage:
#   bash ~/Iris/slurm/submit_final_eval.sh sd2_cap_pct_logtn_img_l0100_s43
set -euo pipefail

RUN="${1:?usage: $0 <run name under \$IRIS_RUNS/iris_ms2>}"
F=/mnt/scratch/dwdh0363/from_sc23sz
M=$F/manifests/sequence_level_internvl3_8b

export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$M}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/env.sh"

[[ -d "$IRIS_RUNS/iris_ms2/$RUN" ]] || { echo "!! 找不到 $IRIS_RUNS/iris_ms2/$RUN"; exit 1; }

export RUN
export STEPS="2000 4000 6000 8000 10000 12000 14000 16000 18000 20000"
export VAL_STRIDE=4 SEL_PROMPT=correct TEST_PROMPTS=correct ALIGN_MODE=ssi_log
export ROUTE=b_thermal_unet IRIS_THERMAL_STRETCH=percentile IRIS_TEXT_PADDING=legacy
export VAL_MANIFEST=$IRIS_MANIFEST_DIR/ms2_val_official3_thermalcap_20260821.jsonl
export TRAIN_MANIFEST=$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl
for c in day3_common night3 rainy3; do
  [[ -f "$IRIS_MANIFEST_DIR/ms2_test_${c}_thermalcap_20260821.jsonl" ]] \
    || { echo "!! 找不到 test 清单 $c"; exit 1; }
done

cd "$IRIS_REPO"
export TEST_MANIFEST=$IRIS_MANIFEST_DIR/ms2_test_day3_common_thermalcap_20260821.jsonl
JD=$(sbatch --parsable -J "fin_day_${RUN:0:20}" --time=08:00:00 --export=ALL slurm/iris_ms2_pipeline.sbatch)
echo "day   $JD  (转换 + val 曲线 + day test)"
for c in night3 rainy3; do
  export TEST_MANIFEST=$IRIS_MANIFEST_DIR/ms2_test_${c}_thermalcap_20260821.jsonl
  J=$(sbatch --parsable -J "fin_${c}_${RUN:0:20}" --time=03:00:00 --dependency=afterok:$JD \
      --export=ALL slurm/iris_ms2_pipeline.sbatch)
  echo "$c $J  (等 $JD)"
done
