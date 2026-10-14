#!/bin/bash
#SBATCH --job-name=ens_val
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 测试时集成（多噪声 ± 水平翻转）在 val 上的一个配置。只用 val 选配置；test 只对选中的那一个跑一次。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   CKPT        converted 权重（必填）
#   ENS         噪声成员数 K（默认 1）
#   FLIP        1 = 每个成员再加一份翻转预测（默认 0）
#   REDUCE      mean | median（默认 mean）
#   VAL_STRIDE  默认 4（与选点曲线同口径）
#
# 输出 $IRIS_RUNS/eval/<run>_val_<fp>_stride<S>_correct_<step>_ens<K>[f][_median]/
# ENS=1 FLIP=0 必须逐位复现选点曲线里同一步的 val 结果（自检）。

F=/mnt/scratch/dwdh0363/from_sc23sz
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$F/data/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$F/manifests/sequence_level_internvl3_8b}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"
set -euo pipefail

: "${CKPT:?必须指定 CKPT}"
[[ -s "$CKPT" ]] || { echo "!! 找不到 $CKPT"; exit 1; }
ENS="${ENS:-1}"; FLIP="${FLIP:-0}"; REDUCE="${REDUCE:-mean}"; VAL_STRIDE="${VAL_STRIDE:-4}"
VAL="$IRIS_MANIFEST_DIR/ms2_val_official3_thermalcap_20260821.jsonl"
TRAIN="$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl"
run=$(basename "$(dirname "$(dirname "$CKPT")")"); step=$(basename "$CKPT" _weights.pt)
fp=$(sha256sum "$VAL" | cut -c1-8)
tag="${run}_val_${fp}_stride${VAL_STRIDE}_correct_s${step#step}_ens${ENS}$([[ "$FLIP" == 1 ]] && echo f)$([[ "$REDUCE" == median ]] && echo _median)"
OUT="$IRIS_RUNS/eval/$tag"
cd "$IRIS_REPO"
git log --oneline -1
echo "ckpt=$CKPT  ens=$ENS flip=$FLIP reduce=$REDUCE stride=$VAL_STRIDE"
echo "out=$OUT"
if [[ -f "$OUT/eval_eval_affine_invariant_log_space.json" ]]; then echo "[skip] 已有结果"; exit 0; fi

extra=(--ensemble-size "$ENS" --ensemble-reduce "$REDUCE")
[[ "$FLIP" == 1 ]] && extra+=(--ensemble-flip)
IRIS_THERMAL_STRETCH=percentile IRIS_TEXT_PADDING=legacy python tools/train_route_suite.py \
  --route b_thermal_unet --output-dir "$OUT" --ms2-root "$IRIS_MS2_ROOT" \
  --train-manifest "$TRAIN" --val-manifest "$VAL" --eval-checkpoint "$CKPT" \
  --val-stride "$VAL_STRIDE" --val-caption-mode correct --align-mode ssi_log \
  --num-inference-steps 1 --condition-latent mode --local-files-only "${extra[@]}"
echo "[完成] $OUT"
