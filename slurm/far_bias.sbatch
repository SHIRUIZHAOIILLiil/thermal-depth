#!/bin/bash
#SBATCH --job-name=far_bias
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# ⛔ 诊断（val，不碰 test），只读文件、不需要 GPU，所以不写 --partition。
# tools/far_bias_breakdown.py：远处偏近按序列 / 按帧拆开，以及三种逐帧拟合
# （全帧 = 正式口径；只用近处拟合；只用远处拟合）下远处偏差怎么变。
# 用 diagnose_val 已保存的完整 val 逐像素预测：l100（8263904）、tmax1（8285836）、NeWCRF。
#
#   sbatch ~/Iris/slurm/far_bias.sbatch
#   覆盖：A_L100=<attempt 目录> A_TMAX1=<attempt 目录> OUT=<输出目录>
#   CPU 冒烟：sbatch --export=ALL,LIMIT=32 slurm/far_bias.sbatch
#   LIMIT=0（默认）为完整验证集；不训练、不推理。

F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
set -euo pipefail

D="$IRIS_RUNS/analysis/diag_val"
one() {  # exactly one directory matching the pattern, or stop
  local matches=( $1 )
  [[ ${#matches[@]} -eq 1 && -d "${matches[0]}" ]] || { echo "!! 找不到唯一的 $1：${matches[*]}" >&2; exit 1; }
  echo "${matches[0]}"
}
A_L100="${A_L100:-$(one "$D/sd2_cap_pct_logtn_img_l0100_s43_step16000/attempt_8263904_*")}"
A_TMAX1="${A_TMAX1:-$(one "$D/sd2_cap_pct_logtn_img_l0100_tmax1_s43_step16000/attempt_8285836_*")}"
OUT="${OUT:-$IRIS_RUNS/analysis/far_bias/$(date +%Y%m%dT%H%M%S)_${SLURM_JOB_ID:-local}}"
for d in "$A_L100/ours/raw_predictions" "$A_TMAX1/ours/raw_predictions" "$A_L100/newcrf/raw_predictions/newcrf"; do
  [[ -d "$d" ]] || { echo "!! 缺少逐像素预测 $d"; exit 1; }
done
cd "$IRIS_REPO"
git log --oneline -1
echo "l100  = $A_L100"; echo "tmax1 = $A_TMAX1"; echo "out   = $OUT"

python tools/far_bias_breakdown.py \
  --manifest "$IRIS_MANIFEST_DIR/ms2_val_official3_thermalcap_20260821.jsonl" --data-root "$IRIS_MS2_ROOT" \
  --model "l100=$A_L100/ours/raw_predictions:ssi_log" \
  --model "tmax1=$A_TMAX1/ours/raw_predictions:ssi_log" \
  --model "newcrf=$A_L100/newcrf/raw_predictions/newcrf:ssi" \
  --structure-reference newcrf \
  --limit "${LIMIT:-0}" \
  --out "$OUT"
