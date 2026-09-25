#!/bin/bash
#SBATCH --job-name=vaethr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 冻结的 VAE 编码器对热像丢了多少内容。
#
# 起因：b 线的前端就是这只 RGB 上训出来的 VAE 编码器。如果热像在编码那一步
# 就被丢掉了内容，那么下游解不解冻 U-Net、前端换不换 AnyThermal，都补不回来。
#
# 不训练，不用 U-Net，只让图像走一个来回。60 帧几分钟就完。
#
# 三行同一批帧：热像过 VAE / 同帧 RGB 过 VAE / 热像只做 8 倍降采样再升回来。
# 后两行是对照 —— 没有它们，单独一个 PSNR 什么也说明不了。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   COND    day | night | rainy，默认 day
#   N       帧数，默认 60
#   OUT_TAG 输出目录名
#
#   COND=day N=60 sbatch --export=ALL,COND,N ~/Iris/slurm/vae_thermal_probe.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

COND="${COND:-day}"
N="${N:-60}"
OUT_TAG="${OUT_TAG:-vae_thermal_probe_$COND}"
OUT="$IRIS_RUNS/analysis/$OUT_TAG"

case "$COND" in
  day)   MANIFEST="$IRIS_MANIFEST_DIR/ms2_test_day3_common_thermalcap_20260821.jsonl" ;;
  night) MANIFEST="$IRIS_MANIFEST_DIR/ms2_test_night3_thermalcap_20260821.jsonl" ;;
  rainy) MANIFEST="$IRIS_MANIFEST_DIR/ms2_test_rainy3_thermalcap_20260821.jsonl" ;;
  *) echo "!! COND 只支持 day | night | rainy"; exit 1 ;;
esac
[[ -f "$MANIFEST" ]] || { echo "!! 找不到 $MANIFEST"; exit 1; }

# 与训练/推理一致；错了量到的就是预处理差异而不是编码器。
export IRIS_THERMAL_STRETCH="${IRIS_THERMAL_STRETCH:-percentile}"

mkdir -p "$OUT"
cd "$IRIS_REPO"
git log --oneline -1
echo "=== 配置 ==="
echo "条件    = $COND"
echo "帧数    = $N"
echo "VAE     = jingheya/lotus-depth-g-v2-1-disparity 的 vae（冻结，未微调）"
echo "精度    = fp32（量的是编码器丢多少，不掺半精度舍入）"
echo "thermal = $IRIS_THERMAL_STRETCH"
echo "============"

python tools/probe_vae_thermal_roundtrip.py \
  --manifest "$MANIFEST" \
  --ms2-root "$IRIS_MS2_ROOT" \
  --output-dir "$OUT" \
  --limit "$N"

echo
echo "[完成] $OUT"
