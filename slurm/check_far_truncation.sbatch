#!/bin/bash
#SBATCH --job-name=far_check
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# ⛔ 诊断（val + train，不碰 test），只读文件、不需要 GPU，所以不写 --partition。
# 远处是不是被训练目标截掉了（tools/check_far_truncation.py）：
#   A. val：用 diagnose_val 保存下来的逐像素预测，按 GT 深度分段看有符号误差（低估还是高估）；
#   B. train：用训练用的数据集类重建补全目标，数一下各深度段的激光像素有多少落在上界之外。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   ATTEMPT  diagnose_val 的 attempt 目录（里面有 ours/ 和 newcrf/ 的 raw_predictions）（必填）
#   TRAIN_FRAMES  B 部分抽多少帧（默认 2000）
#   SKY      可选：天空掩码目录，给了就按天空规则重建目标再数一遍
#
#   ATTEMPT=.../attempt_8263904_xxx sbatch --export=ALL,ATTEMPT ~/Iris/slurm/check_far_truncation.sbatch

F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
set -euo pipefail

# ATTEMPT can be resolved at run time instead: DIAG_DIR (diag_val/<tag>) plus the
# diagnose_val job id DIAG_JOB, whose attempt directory is attempt_<DIAG_JOB>_*.
# That is what lets this job be submitted with --dependency=afterok:<DIAG_JOB>.
if [[ -z "${ATTEMPT:-}" && -n "${DIAG_DIR:-}" && -n "${DIAG_JOB:-}" ]]; then
  matches=( "$DIAG_DIR"/attempt_"${DIAG_JOB}"_* )
  [[ ${#matches[@]} -eq 1 && -d "${matches[0]}" ]] \
    || { echo "!! $DIAG_DIR 下找不到唯一的 attempt_${DIAG_JOB}_*：${matches[*]}"; exit 1; }
  ATTEMPT="${matches[0]}"
  echo "ATTEMPT = $ATTEMPT（由 DIAG_JOB=$DIAG_JOB 解析）"
fi
: "${ATTEMPT:?必须指定 ATTEMPT，或同时给 DIAG_DIR 与 DIAG_JOB}"
[[ -d "$ATTEMPT/ours/raw_predictions" && -d "$ATTEMPT/newcrf/raw_predictions/newcrf" ]] \
  || { echo "!! $ATTEMPT 里没有两份逐像素预测"; exit 1; }
PSEUDO="${PSEUDO_DIR:-$F/pseudo_gt/official_train/calibrated_pseudo_depth}"
cd "$IRIS_REPO"
git log --oneline -1

sky_args=()
[[ -n "${SKY:-}" ]] && sky_args=(--sky-mask-dir "$SKY")
python tools/check_far_truncation.py --data-root "$IRIS_MS2_ROOT" \
  --val-manifest "$IRIS_MANIFEST_DIR/ms2_val_official3_thermalcap_20260821.jsonl" \
  --pred "ours=$ATTEMPT/ours/raw_predictions:ssi_log" \
  --pred "newcrf=$ATTEMPT/newcrf/raw_predictions/newcrf:ssi" \
  --train-manifest "$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl" \
  --pseudo-dir "$PSEUDO" --train-frames "${TRAIN_FRAMES:-2000}" "${sky_args[@]}" \
  --out "$ATTEMPT/far_check"
