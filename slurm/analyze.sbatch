#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 九分层区域分析。**不申请 GPU** —— 它只读 raw_predictions/*.npy 和 GT，
# 纯 CPU 活，走标准分区基本秒进，别拿 GPU 排队。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   PRED_DIR   必填，含 raw_predictions/ 的目录（run_ms2_*_official.py 的输出）
#   MANIFEST   必填，与出预测时同一份
#   TAG        必填，输出目录名
#   STRIDE     默认 1
#   BOOTSTRAP  设了就传给 --bootstrap
#
#   PRED_DIR=$SCRATCH/runs/anythermal/Midas_anythermal \
#   MANIFEST=$SCRATCH/manifests/sequence_level_internvl3_8b/ms2_test_16-08-46_rgb_depth_v1_clip75_20260728.jsonl \
#   TAG=regions_at_midas_test \
#     sbatch -J regions_at --export=ALL,PRED_DIR,MANIFEST,TAG ~/Iris/slurm/analyze.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

: "${PRED_DIR:?必须指定 PRED_DIR（含 raw_predictions/ 的目录）}"
: "${MANIFEST:?必须指定 MANIFEST}"
: "${TAG:?必须指定 TAG}"

[[ -d "$PRED_DIR" ]] || { echo "!! 找不到 $PRED_DIR"; exit 1; }
[[ -d "$PRED_DIR/raw_predictions" ]] || {
  echo "!! $PRED_DIR 下没有 raw_predictions/"
  echo "   注意：train_route_suite.py 的 --eval-checkpoint **不导出 raw 预测**，"
  echo "   要用 tools/run_ms2_lotus_*_official.py --save-raw-pred 才有。"
  exit 1
}
[[ -f "$MANIFEST" ]] || { echo "!! 找不到 manifest: $MANIFEST"; exit 1; }

OUT="$IRIS_RUNS/regions/$TAG"
mkdir -p "$OUT"

cd "$IRIS_REPO"

ARGS=(
  --manifest "$MANIFEST"
  --data-root "$IRIS_MS2_ROOT"
  --output-dir "$OUT"
  --stride "${STRIDE:-1}"
)
[[ -n "${BOOTSTRAP:-}" ]] && ARGS+=(--bootstrap "$BOOTSTRAP")

echo "=== python tools/analyze_prediction_regions.py ${ARGS[*]} ==="
python tools/analyze_prediction_regions.py "${ARGS[@]}"
rc=$?

echo "=== 退出码 $rc ==="
ls -la "$OUT"
exit $rc
