#!/bin/bash
#SBATCH --job-name=ppd0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 已发布的 Pixel-Perfect Depth 权重，直接推热像，**不做任何适配**。
#
# 补的是主表「没做热像适配」那一组里缺的第三条线。Depth Anything V2 和 Lotus-G
# 都有零样本行，PPD 只有适配后的。
#
# 跑全帧，打完分再折算官方子集 —— 和表里其余每一行的做法一致。
# ⚠️ 不要改成「先裁子集再推理」省卡时：那样这一行的帧集来路和别的行不同，
#    表里就混了两种口径。省下的几小时不值这个。
# ⚠️ PPD 约 0.63 秒一帧：day 2.3 万帧约 4.1 小时，rainy 2.5 万帧约 4.4 小时。
#    墙钟给 8 小时。之前 4 小时的 ppdt_day 就是这么超时的。
#
# ⚠️ 对齐用 ssi_log —— PPD 的原生输出是 log 深度（2026-09-24 实测，作业 7941688）。
#    用 ssi_disparity 会把近远弄反，而且不报错。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   COND     day | night | rainy，默认 day
#   PPD_CKPT 已发布权重，默认 $PPD_REPO/checkpoints/ppd.pth
#   OUT_TAG  输出目录名
#
#   COND=day sbatch --export=ALL,COND ~/Iris/slurm/ppd_zeroshot.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

COND="${COND:-day}"
OUT_TAG="${OUT_TAG:-ppd_zeroshot_$COND}"
OUT="$IRIS_RUNS/analysis/$OUT_TAG"

export PPD_REPO="${PPD_REPO:-$SCRATCH/ppd_repo}"
PPD_CKPT="${PPD_CKPT:-$PPD_REPO/checkpoints/ppd.pth}"
PPD_SEM="${PPD_SEM:-$PPD_REPO/checkpoints/depth_anything_v2_vitl.pth}"
PY="${PPD_PY:-$HOME/.conda/envs/ppd/bin/python}"

case "$COND" in
  day)   MANIFEST="$IRIS_MANIFEST_DIR/ms2_test_day3_common_thermalcap_20260821.jsonl"; ENV=test_day ;;
  night) MANIFEST="$IRIS_MANIFEST_DIR/ms2_test_night3_thermalcap_20260821.jsonl";      ENV=test_night ;;
  rainy) MANIFEST="$IRIS_MANIFEST_DIR/ms2_test_rainy3_thermalcap_20260821.jsonl";      ENV=test_rain ;;
  *) echo "!! COND 只支持 day | night | rainy"; exit 1 ;;
esac
[[ -f "$MANIFEST" ]] || { echo "!! 找不到 $MANIFEST"; exit 1; }
[[ -x "$PY" ]]       || { echo "!! 找不到可执行的 $PY"; exit 1; }
[[ -e "$PPD_CKPT" ]] || { echo "!! 找不到已发布权重 $PPD_CKPT"; ls -1 "$PPD_REPO/checkpoints" 2>/dev/null | sed 's/^/     /'; exit 1; }
[[ -e "$PPD_SEM" ]]  || { echo "!! 找不到语义权重 $PPD_SEM"; exit 1; }

# 与适配线、与我方线一致。错了量到的是预处理差异。
export IRIS_THERMAL_STRETCH="${IRIS_THERMAL_STRETCH:-percentile}"

mkdir -p "$OUT"
cd "$IRIS_REPO"
git log --oneline -1
echo "=== 配置 ==="
echo "条件    = $COND   ($ENV)"
echo "权重    = $PPD_CKPT   **已发布，无任何适配**"
echo "语义    = $PPD_SEM"
echo "帧数    = 全量（$(wc -l < "$MANIFEST") 帧），打完分再折算官方子集"
echo "对齐    = ssi_log   (PPD 的原生输出空间，实测)"
echo "thermal = $IRIS_THERMAL_STRETCH"
echo "python  = $PY"
echo "============"

"$PY" tools/run_external_depth_inference.py \
  --model pixel_perfect_depth \
  --manifest "$MANIFEST" \
  --data-root "$IRIS_MS2_ROOT" \
  --out-dir "$OUT/raw_predictions" \
  --stretch "$IRIS_THERMAL_STRETCH" \
  --ppd-checkpoint "$PPD_CKPT" \
  --ppd-semantics "$PPD_SEM"

echo
echo "=== 官方协议打分（全帧）==="
python tools/run_official_ms2_evaluation.py \
  --manifest "$MANIFEST" \
  --data-root "$IRIS_MS2_ROOT" \
  --prediction-dir "$OUT/raw_predictions" \
  --route iris-lotus \
  --align ssi_log \
  --output-dir "$OUT/official"

echo
echo "=== 折算官方子集 ==="
python tools/subset_to_official.py \
  --per-sample "$OUT/official/metrics/per_image.csv" \
  --ms2-root "$IRIS_MS2_ROOT" \
  --test-env "$ENV"

echo
echo "[完成] $OUT"
echo "⚠️ 逐帧 .npy 打完分就可以删：rm -rf $OUT/raw_predictions"
