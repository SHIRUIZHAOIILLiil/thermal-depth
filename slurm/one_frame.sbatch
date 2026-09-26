#!/bin/bash
#SBATCH --job-name=oneframe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 一帧，多个 prompt，其余全部相同。种子、condition、checkpoint、分辨率全锁死，
# 所以看到的差别只来自那句话。
#
# ⛔ 一帧的数字在噪声里。这个项目在找的效应是 0.001–0.002 量级，帧与帧之间的
#    差异比它大两三个数量级。**这是拿来看图的，不是拿来选 prompt 的。**
#    选 prompt 要在 val 上跑几百帧，而且判据事先写下来。
#
# ⛔ 别拿 test 帧反复换句子试到变好 —— 那是拿 test 选方法。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   FRAME   清单里的 id，必填
#   CKPT    U-Net checkpoint，必填
#   ALIGN   必填，且必须和该 checkpoint 训练时的 norm_type 配套：
#             trunc_disparity → ssi_disparity
#             truncnorm       → ssi
#             log_truncnorm   → ssi_log
#           ⚠️ 三族互不嵌套，选错不会报错，只是数字难看 30 倍。
#   COND    day | night | rainy，默认 day（决定用哪份清单）
#   PROMPTS 每行一句的文本文件，默认 $HOME/one_frame_prompts.txt
#           空文本与清单里那一句会自动加上作对照，不用写进去。
#   OUT_TAG 输出目录名
#
# ── 用法 ────────────────────────────────────────────────────────────
#   先把要试的句子写进文件（一行一句，想试几句写几句）：
#
#     cat > ~/one_frame_prompts.txt <<'EOF'
#     A thermal street scene showing ...
#     EOF
#
#   再投：
#
#     FRAME=2021-08-06-11-23-45_000805 \
#     CKPT=$SCRATCH/runs/iris_ms2/iris_ms2_full8_thermalcap/converted/step20000_weights.pt \
#     ALIGN=ssi_disparity \
#       sbatch --export=ALL,FRAME,CKPT,ALIGN ~/Iris/slurm/one_frame.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

: "${FRAME:?必须指定 FRAME（清单里的 id）}"
: "${CKPT:?必须指定 CKPT}"
: "${ALIGN:?必须指定 ALIGN —— 见脚本头部的对应表，没有默认值是故意的}"
COND="${COND:-day}"
PROMPTS="${PROMPTS:-$HOME/one_frame_prompts.txt}"
OUT_TAG="${OUT_TAG:-one_frame_${FRAME}}"
OUT="$IRIS_RUNS/analysis/$OUT_TAG"

case "$COND" in
  day)   MANIFEST="$IRIS_MANIFEST_DIR/ms2_test_day3_common_thermalcap_20260821.jsonl" ;;
  night) MANIFEST="$IRIS_MANIFEST_DIR/ms2_test_night3_thermalcap_20260821.jsonl" ;;
  rainy) MANIFEST="$IRIS_MANIFEST_DIR/ms2_test_rainy3_thermalcap_20260821.jsonl" ;;
  *) echo "!! COND 只支持 day | night | rainy"; exit 1 ;;
esac
[[ -f "$MANIFEST" ]] || { echo "!! 找不到 $MANIFEST"; exit 1; }
[[ -f "$CKPT" ]]     || { echo "!! 找不到 checkpoint $CKPT"; exit 1; }
if [[ ! -s "$PROMPTS" ]]; then
  echo "!! 找不到或为空：$PROMPTS"
  echo "!! 先写进去，一行一句："
  echo "!!   cat > $PROMPTS <<'EOF'"
  echo "!!   你要试的那一句"
  echo "!!   EOF"
  exit 1
fi

# 与训练/推理一致。错了量到的是预处理差异而不是文本差异。
export IRIS_THERMAL_STRETCH="${IRIS_THERMAL_STRETCH:-percentile}"
export IRIS_DECODE_CLAMP="${IRIS_DECODE_CLAMP:-0}"

mkdir -p "$OUT"
cd "$IRIS_REPO"
git log --oneline -1
echo "=== 配置 ==="
echo "帧      = $FRAME   ($COND)"
echo "权重    = $CKPT"
echo "对齐    = $ALIGN"
echo "thermal = $IRIS_THERMAL_STRETCH"
echo "句子    = $PROMPTS   （$(wc -l < "$PROMPTS") 句，另加空文本与清单原句）"
echo "============"
cat -n "$PROMPTS"
echo

python tools/infer_one_frame.py \
  --manifest "$MANIFEST" \
  --ms2-root "$IRIS_MS2_ROOT" \
  --frame "$FRAME" \
  --unet-checkpoint "$CKPT" \
  --align "$ALIGN" \
  --prompt-file "$PROMPTS" \
  --out-dir "$OUT"

echo
echo "[完成] $OUT"
echo "图：$OUT/one_frame.png　　把它拷回本地看"
