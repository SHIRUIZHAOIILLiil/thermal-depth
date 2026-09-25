#!/bin/bash
#SBATCH --job-name=lotus0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 已发布的 Lotus-G 深度权重，直接推理热像，**不做任何适配**。
#
# 这是对比表里唯一缺的那一类：DA2 有零样本行，我们没有。它回答的是
# 「不训练能得几分」，也就是我们这条线的全部训练贡献了多少。
#
# ⚠️ 措辞要准：我们那条线的起点是 SD2-base（文生图），它根本不出深度，
# 所以「我们的模型不经训练」严格说不存在。这一行是 **同架构同家族的 RGB
# 深度模型零样本用在热像上**，正好对应 DA2 零样本那一行，不能说成
# 「我们的模型未训练版本」。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   COND    day | night | rainy，必填
#   LIMIT   帧数上限，默认 0 ＝ 全量
#           ⚠️ 脚本自身的 --max-samples 默认是 8（调试值），这里必须显式传
#   OUT_TAG 输出目录名，默认 lotus_zeroshot_<COND>
#
#   COND=day sbatch --export=ALL,COND ~/Iris/slurm/lotus_zeroshot.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

: "${COND:?必须指定 COND（day|night|rainy）}"
LIMIT="${LIMIT:-0}"
OUT_TAG="${OUT_TAG:-lotus_zeroshot_$COND}"
OUT="$IRIS_RUNS/analysis/$OUT_TAG"

case "$COND" in
  day)   MANIFEST="$IRIS_MANIFEST_DIR/ms2_test_day3_common_thermalcap_20260821.jsonl"; ENV=test_day ;;
  night) MANIFEST="$IRIS_MANIFEST_DIR/ms2_test_night3_thermalcap_20260821.jsonl";      ENV=test_night ;;
  rainy) MANIFEST="$IRIS_MANIFEST_DIR/ms2_test_rainy3_thermalcap_20260821.jsonl";      ENV=test_rain ;;
  *) echo "!! COND 只支持 day | night | rainy"; exit 1 ;;
esac
[[ -f "$MANIFEST" ]] || { echo "!! 找不到 $MANIFEST"; exit 1; }

# 与我们自己那条线的输入完全一致 —— 否则这一行差的就不只是训练，还有预处理。
export IRIS_THERMAL_STRETCH="${IRIS_THERMAL_STRETCH:-percentile}"
export IRIS_DECODE_CLAMP="${IRIS_DECODE_CLAMP:-0}"

mkdir -p "$OUT"
cd "$IRIS_REPO"
git log --oneline -1
echo "=== 配置 ==="
echo "条件      = $COND   ($ENV)"
echo "权重      = jingheya/lotus-depth-g-v2-1-disparity   **无任何适配**"
echo "caption   = empty   (零样本，不给文本)"
echo "帧数上限  = $LIMIT   (0 ＝ 全量)"
echo "thermal   = $IRIS_THERMAL_STRETCH   (与我方线一致)"
echo "对齐      = ssi_disparity   (该 checkpoint 是 disparity 变体)"
echo "============"

# 不传 --unet-checkpoint ＝ 冻结的预训练 U-Net，这就是零样本。
python tools/run_ms2_lotus_thermal_vae_official.py \
  --manifest "$MANIFEST" \
  --ms2-root "$IRIS_MS2_ROOT" \
  --output-dir "$OUT" \
  --max-samples "$LIMIT" \
  --caption-mode empty \
  --save-raw-pred

echo
echo "=== 官方协议打分 ==="
python tools/run_official_ms2_evaluation.py \
  --manifest "$MANIFEST" \
  --data-root "$IRIS_MS2_ROOT" \
  --prediction-dir "$OUT/raw_predictions" \
  --route iris-lotus \
  --align ssi_disparity \
  --output-dir "$OUT/official"

echo
echo "=== 折算官方子集 ==="
python tools/subset_to_official.py \
  --per-sample "$OUT/official/metrics/per_image.csv" \
  --ms2-root "$IRIS_MS2_ROOT" \
  --test-env "$ENV"

echo
echo "[完成] $OUT"
echo "⚠️ 逐帧 .npy 约 15 GB —— 打完分就可以删：rm -rf $OUT/raw_predictions"
