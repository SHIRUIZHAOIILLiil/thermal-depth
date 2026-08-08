#!/bin/bash
#SBATCH --job-name=pseudo_gt_build
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00  # 400 帧实测 27s 推理 → 19,949 帧约 23 分钟，留足余量
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 为**训练集**离线生成 dense pseudo GT，供 dense completion 使用。
#
# AnyThermal 只是一个离线 pseudo-depth generator：它不进 Lotus 的 forward，
# 不做 feature/logits 匹配，也没有任何 model-to-model 的损失项。它的输出按本帧
# 自己的 lidar 标定后，被当作另一份 GT 来源使用，真实回波处永远用真实 GT。
#
# 预检结论（400 帧分层子集，pseudo_gt_audit.sbatch）：
#   真天空 pseudo depth p50 25.79 m / p90 51.70 m，>80m 仅 0.12%，非正值 0%；
#   留出 lidar 点 AbsRel 0.0573（与 in-sample 相同 → 两参数仿射没有过拟合）；
#   近(<15m)拟合 → 远(>30m)打分 0.0895（全量拟合 0.0686，退化 30%，没有崩）。
#
# ⚠️ 只做 training split。val/test 不生成训练用 pseudo GT，也绝不参与训练。
# ⚠️ 磁盘：raw ~16.5 GB（建完可删）+ 标定深度 ~13 GB（float32）/ ~6.5 GB（float16）。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   MANIFEST       默认训练集全量
#   TAG            输出目录名，默认 train_full
#   DTYPE          float32（默认）| float16
#   MIDAS_VARIANT  默认 Midas_anythermal
#
#   sbatch ~/Iris/slurm/pseudo_gt_build.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

VARIANT="${MIDAS_VARIANT:-Midas_anythermal}"
CKPT_DIR="$SCRATCH/models/anythermal_midas/$VARIANT"
BMSD_DIR="$SCRATCH/code/BridgeMultiSpectralDepth"
MANIFEST="${MANIFEST:-$IRIS_MANIFEST_DIR/ms2_train_day2seq_clip75_20260728.jsonl}"
OUT="$IRIS_RUNS/pseudo_gt/${TAG:-train_full}"
RAW="$OUT/raw_predictions_run"

case "$(basename "$MANIFEST")" in
  *test*|*val*) echo "!! 拒绝 $MANIFEST —— pseudo GT 只为 training split 生成"; exit 1 ;;
esac
[[ -f "$MANIFEST" ]] || { echo "!! 找不到 manifest: $MANIFEST"; exit 1; }
[[ -d "$CKPT_DIR" ]] || { echo "!! 找不到 $CKPT_DIR —— 先跑 transfer_checkpoints.sh midas"; exit 1; }
[[ -d "$BMSD_DIR" ]] || { echo "!! 找不到 $BMSD_DIR —— 先跑 transfer_checkpoints.sh bmsd"; exit 1; }
if [[ -d "$RAW" ]] && [[ -n "$(ls -A "$RAW" 2>/dev/null)" ]]; then
  echo "!! 输出目录非空，推理脚本会拒绝: $RAW"; echo "   换 TAG，或先 rm -rf '$OUT'"; exit 1
fi
mkdir -p "$OUT"

CKPT_FILE=$(ls "$CKPT_DIR"/*.ckpt 2>/dev/null | head -1)
[[ -n "$CKPT_FILE" ]] || { echo "!! $CKPT_DIR 下没有 .ckpt"; ls -la "$CKPT_DIR"; exit 1; }
export IRIS_BMSD_ROOT="$BMSD_DIR"
export IRIS_ANYTHERMAL_CKPT="$CKPT_FILE"

cd "$IRIS_REPO"
ROWS=$(grep -c . "$MANIFEST")
echo "=== ① AnyThermal 离线推理（$ROWS 帧，保留 raw：标定必须用未 clamp 的原值）==="
python tools/run_ms2_anythermal_midas.py \
  --manifest "$MANIFEST" --ms2-root "$IRIS_MS2_ROOT" --output-dir "$RAW" \
  --checkpoint "$CKPT_FILE" --max-samples 0 || exit 1

echo
echo "=== ② 深度空间标定（拟合前不 clamp；存深度，range handling 留给训练开关）==="
python tools/build_anythermal_pseudo_gt.py \
  --manifest "$MANIFEST" --ms2-root "$IRIS_MS2_ROOT" \
  --raw-pred-dir "$RAW/raw_predictions" --output-dir "$OUT" \
  --dtype "${DTYPE:-float32}"
rc=$?

echo
echo "=== 退出码 $rc ==="
BUILT=$(ls "$OUT/calibrated_pseudo_depth" 2>/dev/null | wc -l)
echo "标定深度: $BUILT 帧 -> $OUT/calibrated_pseudo_depth"
du -sh "$OUT" 2>/dev/null
if [[ "$BUILT" -eq "$ROWS" ]]; then
  echo
  echo "训练时传：PSEUDO_DIR=$OUT/calibrated_pseudo_depth"
  echo "raw 只用于标定，确认无误后可回收约 16 GB："
  echo "  rm -rf $RAW"
fi
exit $rc
