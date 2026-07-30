#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 九分层区域分析。**不申请 GPU** —— 只读 raw_predictions/*.npy 和 GT，纯 CPU 活，
# 走标准分区基本秒进，别拿 GPU 排队。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   PREDS      必填。一个或多个 `目录[:标签]`，**逗号分隔**。
#              analyze_prediction_regions.py 的 --predictions 可重复，
#              多个一起传才能做同图对比 —— 这正是任务 3 要的。
#   MANIFEST   必填，与出预测时同一份
#   TAG        必填，输出目录名
#   ROUTE      可选，查官方对齐表决定 --align；外部模型（AnyThermal）留空
#   ALIGN      可选，默认 auto
#   GT_VIEW    thermal（默认）| rgb —— 按该线实际看到的视角取 GT
#   STRIDE     默认 1；LIMIT 默认 0（全部）；BOOTSTRAP 默认随脚本
#
# ── 用法 ────────────────────────────────────────────────────────────
#   单个模型：
#     PREDS=$SCRATCH/runs/anythermal/Midas_anythermal/raw_predictions:anythermal \
#     MANIFEST=$SCRATCH/manifests/sequence_level_internvl3_8b/ms2_test_16-08-46_rgb_depth_v1_clip75_20260728.jsonl \
#     TAG=regions_at_midas_test \
#       sbatch -J regions_at --export=ALL,PREDS,MANIFEST,TAG ~/Iris/slurm/analyze.sbatch
#
#   任务 3 的三方对比（六线 + 原版 AnyThermal + 零训练直推）：
#     PREDS="$SCRATCH/runs/eval/b_e05_raw/raw_predictions:b_thermal,$SCRATCH/runs/eval/d1_raw/raw_predictions:d1_anythermal,$SCRATCH/runs/anythermal/Midas_anythermal/raw_predictions:anythermal_orig"

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

: "${PREDS:?必须指定 PREDS（逗号分隔的 目录[:标签]）}"
: "${MANIFEST:?必须指定 MANIFEST}"
: "${TAG:?必须指定 TAG}"

[[ -f "$MANIFEST" ]] || { echo "!! 找不到 manifest: $MANIFEST"; exit 1; }

OUT="$IRIS_RUNS/regions/$TAG"
mkdir -p "$OUT"

cd "$IRIS_REPO"

ARGS=(
  --manifest "$MANIFEST"
  --data-root "$IRIS_MS2_ROOT"
  --output-dir "$OUT"
  --stride "${STRIDE:-1}"
  --limit "${LIMIT:-0}"
  --gt-view "${GT_VIEW:-thermal}"
)

# 逗号拆成多个 --predictions，并逐个确认目录里真有 .npy
IFS=',' read -ra ENTRIES <<< "$PREDS"
for entry in "${ENTRIES[@]}"; do
  dir="${entry%%:*}"
  if [[ ! -d "$dir" ]]; then
    echo "!! 找不到预测目录: $dir"
    echo "   train_route_suite.py 的 --eval-checkpoint 默认**不导出** raw 预测，"
    echo "   要在 eval.sbatch 上加 SAVE_RAW=1；run_ms2_anythermal_midas.py 则是无条件写。"
    exit 1
  fi
  count=$(ls "$dir"/*.npy 2>/dev/null | wc -l)
  [[ "$count" -gt 0 ]] || { echo "!! $dir 下没有 .npy"; exit 1; }
  echo "预测集 $entry  ($count 个 .npy)"
  ARGS+=(--predictions "$entry")
done

[[ -n "${ROUTE:-}" ]]     && ARGS+=(--route "$ROUTE")
[[ -n "${ALIGN:-}" ]]     && ARGS+=(--align "$ALIGN")
[[ -n "${BOOTSTRAP:-}" ]] && ARGS+=(--bootstrap "$BOOTSTRAP")

echo "=== python tools/analyze_prediction_regions.py ${ARGS[*]} ==="
python tools/analyze_prediction_regions.py "${ARGS[@]}"
rc=$?

echo "=== 退出码 $rc ==="
ls -la "$OUT"
exit $rc
