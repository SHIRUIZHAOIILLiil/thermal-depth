#!/bin/bash
#SBATCH --job-name=pseudo_gt_audit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# dense pseudo-GT completion 的**上机前预检**（不训练，不碰 test）。
#
# MS2 的 lidar 只覆盖约四分之一像素，其余部分在 masked_ssi_l1 里梯度恒为零。
# 这条路线用 AnyThermal 作**离线 pseudo-depth generator**，把没有数值的位置补上：
# 它不进 Lotus 的 forward，不做任何 model-to-model 的项，只是多一份 GT 来源。
#
# 本作业只回答两个能用算术回答的问题：
#   ① 真天空掩码内，标定后的 pseudo depth 落在哪个距离档 —— 决定 range handling，
#      也决定这条路线能不能指望单独修天空；
#   ② 那个仿射够不够得着 —— 上带 lidar 覆盖仅约 1%，天空的 pseudo depth 是从近处
#      外推出去的。近点拟合、远点打分，是同一段外推里 lidar 还能验证的那部分。
#
# ⚠️ 只用 train / val。test 留到最终评估。
# ⚠️ 依赖 anythermal_midas.sbatch 头注里那两次 transfer_checkpoints.sh。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   MANIFEST       抽样池，默认训练集
#   COUNT          子集帧数，默认 400（按序列比例分层 + 偶数间隔）
#   SKY_MASKS      天空掩码目录；不设则跳过真天空统计，只剩上带（是混合，答不了天空）
#   MIN_SKY_PX     一帧至少多少天空像素才进池，默认 2000
#   TAG            输出目录名，默认 pseudo_audit
#   MIDAS_VARIANT  默认 Midas_anythermal
#
#   SKY_MASKS=$SCRATCH/runs/sky_masks/skymask_train_full/masks \
#     sbatch -J pseudo_gt_audit --export=ALL,SKY_MASKS ~/Iris/slurm/pseudo_gt_audit.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

VARIANT="${MIDAS_VARIANT:-Midas_anythermal}"
CKPT_DIR="$SCRATCH/models/anythermal_midas/$VARIANT"
BMSD_DIR="$SCRATCH/code/BridgeMultiSpectralDepth"
MANIFEST="${MANIFEST:-$IRIS_MANIFEST_DIR/ms2_train_day2seq_clip75_20260728.jsonl}"
OUT="$IRIS_RUNS/pseudo_gt/${TAG:-pseudo_audit}"
SUBSET="$OUT/subset_manifest.jsonl"
RAW="$OUT/raw_predictions_run"

case "$(basename "$MANIFEST")" in
  *test*) echo "!! 拒绝 test manifest: $MANIFEST —— 预检只用 train/val"; exit 1 ;;
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

echo "=== ① 分层抽样 ==="
SAMPLE_ARGS=(--manifest "$MANIFEST" --output "$SUBSET" --count "${COUNT:-400}")
# 一帧没有天空就答不了天空问题；这个语料约四分之一是隧道/树荫/高楼
if [[ -n "${SKY_MASKS:-}" ]]; then
  [[ -d "$SKY_MASKS" ]] || { echo "!! 找不到天空掩码目录: $SKY_MASKS"; exit 1; }
  SAMPLE_ARGS+=(--sky-mask-dir "$SKY_MASKS" --min-sky-pixels "${MIN_SKY_PX:-2000}")
fi
python tools/sample_manifest_subset.py "${SAMPLE_ARGS[@]}" || exit 1

echo
echo "=== ② AnyThermal 离线推理（保留 raw，标定要用未 clamp 的原值）==="
python tools/run_ms2_anythermal_midas.py \
  --manifest "$SUBSET" --ms2-root "$IRIS_MS2_ROOT" --output-dir "$RAW" \
  --checkpoint "$CKPT_FILE" --max-samples 0 || exit 1

echo
echo "=== ③ 审计 ==="
AUDIT_ARGS=(
  --manifest "$SUBSET" --ms2-root "$IRIS_MS2_ROOT"
  --raw-pred-dir "$RAW/raw_predictions" --output-dir "$OUT/audit"
)
[[ -n "${SKY_MASKS:-}" ]] && AUDIT_ARGS+=(--sky-mask-dir "$SKY_MASKS")
python tools/audit_pseudo_gt.py "${AUDIT_ARGS[@]}"
rc=$?

echo
echo "=== 退出码 $rc ==="
echo "审计结果: $OUT/audit/audit.json"
exit $rc
