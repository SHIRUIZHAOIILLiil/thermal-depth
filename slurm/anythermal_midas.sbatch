#!/bin/bash
#SBATCH --job-name=at_midas
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 任务 4：原版 AnyThermal（MiDaS 重建器）在同一份考卷上出预测，
# 用于回答「它是不是也错判天空」。
#
# ⚠️ 依赖两样 git 和 HF 预取都拿不到的东西，必须先传：
#     bash slurm/transfer_checkpoints.sh midas   # $SCRATCH/models/anythermal_midas/
#     bash slurm/transfer_checkpoints.sh bmsd    # $SCRATCH/code/BridgeMultiSpectralDepth
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   MIDAS_VARIANT  Midas_anythermal（默认）| Midas_dinov2 | Midas_small
#   MANIFEST       考卷；默认独立 test 集 16-08-46
#   TAG            输出目录名，默认跟 variant 走
#   MAX_SAMPLES    帧数上限。**脚本自身默认只有 8**（调试值），所以这里默认传 0
#                  ＝ 全量；要抽样调试才显式设小值
#
#   MIDAS_VARIANT=Midas_anythermal sbatch -J at_midas \
#     --export=ALL,MIDAS_VARIANT ~/Iris/slurm/anythermal_midas.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

VARIANT="${MIDAS_VARIANT:-Midas_anythermal}"
CKPT_DIR="$SCRATCH/models/anythermal_midas/$VARIANT"
BMSD_DIR="$SCRATCH/code/BridgeMultiSpectralDepth"
MANIFEST="${MANIFEST:-$IRIS_MANIFEST_DIR/ms2_test_16-08-46_rgb_depth_v1_clip75_20260728.jsonl}"
OUT="$IRIS_RUNS/anythermal/${TAG:-$VARIANT}"

[[ -d "$CKPT_DIR" ]] || { echo "!! 找不到 $CKPT_DIR —— 先跑 transfer_checkpoints.sh midas"; exit 1; }
[[ -d "$BMSD_DIR" ]] || { echo "!! 找不到 $BMSD_DIR —— 先跑 transfer_checkpoints.sh bmsd"; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "!! 找不到 manifest: $MANIFEST"; exit 1; }

# run_ms2_anythermal_midas.py 拒绝非空输出目录，重跑时要么换 TAG 要么先清掉
if [[ -d "$OUT" ]] && [[ -n "$(ls -A "$OUT" 2>/dev/null)" ]]; then
  echo "!! 输出目录非空，脚本会拒绝: $OUT"
  echo "   换个 TAG，或先 rm -rf '$OUT'"
  exit 1
fi
mkdir -p "$OUT"
export PYTHONPATH="$BMSD_DIR:${PYTHONPATH:-}"

cd "$IRIS_REPO"

ARGS=(
  --manifest "$MANIFEST"
  --ms2-root "$IRIS_MS2_ROOT"
  --output-dir "$OUT"
  --checkpoint "$CKPT_DIR"
)
# choose_uniform(): count <= 0 或 >= 总数都返回全部。不传的话会用脚本默认的 8。
ARGS+=(--max-samples "${MAX_SAMPLES:-0}")

echo "=== python tools/run_ms2_anythermal_midas.py ${ARGS[*]} ==="
python tools/run_ms2_anythermal_midas.py "${ARGS[@]}"
rc=$?

echo "=== 退出码 $rc ==="
ls -la "$OUT"
exit $rc
