#!/bin/bash
# Submit the test-time ensemble configurations on val for one checkpoint, and print
# the selection command to run once they finish. Nothing here touches test.
#
# Usage:
#   bash ~/Iris/slurm/submit_ensemble_val.sh sd2_cap_pct_logtn_img_l0100_s43 18000
set -euo pipefail

RUN="${1:?usage: $0 <run> <step>}"
STEP="${2:?usage: $0 <run> <step>}"
F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/env.sh"
cd "$IRIS_REPO"

CKPT="$IRIS_RUNS/iris_ms2/$RUN/converted/step${STEP}_weights.pt"
[[ -s "$CKPT" ]] || { echo "!! 找不到 $CKPT"; exit 1; }
export CKPT VAL_STRIDE="${VAL_STRIDE:-4}" REDUCE=mean

# K x flip grid. ENS=1 FLIP=0 is the control and must reproduce the selection curve.
for cfg in "1 0" "4 0" "8 0" "1 1" "4 1" "8 1"; do
  read -r ENS FLIP <<< "$cfg"
  export ENS FLIP
  J=$(sbatch --parsable -J "ens${ENS}$([[ $FLIP == 1 ]] && echo f)" --export=ALL slurm/ensemble_val.sbatch)
  echo "ens=$ENS flip=$FLIP -> $J"
done

fp=$(sha256sum "$IRIS_MANIFEST_DIR/ms2_val_official3_thermalcap_20260821.jsonl" | cut -c1-8)
p="$IRIS_RUNS/eval/${RUN}_val_${fp}_stride${VAL_STRIDE}_correct_s${STEP}_ens"
S=eval_eval_affine_invariant_log_space.json
echo
echo "全部完成后选择（control = 单次预测；门槛与 loss 扫描相同，按 RMSE 排）："
echo "python3 tools/select_loss_sweep.py --control ens1=${p}1/$S --candidate ens4=${p}4/$S ens8=${p}8/$S ens1f=${p}1f/$S ens4f=${p}4f/$S ens8f=${p}8f/$S"
