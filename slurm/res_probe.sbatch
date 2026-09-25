#!/bin/bash
#SBATCH --job-name=resprobe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 同一批帧，两个推理分辨率，打完分放在一起。
#
# 起因：定性图上我们的预测比 Iris 的糊得多。原因不在渲染 —— MS2 是 256x640，
# VAE 下采样 8 倍，latent 就是 32x80，**比 8 像素细的结构在编码那一步就没了**。
# 把热像先放大到 512x1280，latent 变成 64x160，看结构回不回得来、指标动不动。
#
# ⚠️ 这是诊断不是配置。模型训在 256x640 上，改推理分辨率就是分布偏移，
#    完全可能更差。更差也是结果 —— 它把「糊是分辨率造成的」这条排除掉。
#
# ⚠️ 两个分辨率必须跑同一批帧。--max-samples 走 choose_uniform（确定性的均匀抽），
#    同一份 manifest + 同一个 N 必然给出同一批帧，所以这里靠这一点而不是各跑各的。
#    预测最后都降回 256x640 再打分，GT 一律不碰。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   COND    day | night | rainy，默认 day
#   N       抽多少帧，默认 300。0 ＝ 全量（day 是 2.3 万帧，别在探针里这么干）
#   UNET    训练好的 U-Net checkpoint。不给 ＝ 零样本的已发布权重
#   RES     放大后的长边，默认 1280（256x640 → 512x1280）
#   OUT_TAG 输出目录名
#
#   COND=day N=300 UNET=$SCRATCH/runs/... sbatch --export=ALL,COND,N,UNET ~/Iris/slurm/res_probe.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

COND="${COND:-day}"
N="${N:-300}"
RES="${RES:-1280}"
OUT_TAG="${OUT_TAG:-res_probe_${COND}_${RES}}"
OUT="$IRIS_RUNS/analysis/$OUT_TAG"

case "$COND" in
  day)   MANIFEST="$IRIS_MANIFEST_DIR/ms2_test_day3_common_thermalcap_20260821.jsonl"; ENV=test_day ;;
  night) MANIFEST="$IRIS_MANIFEST_DIR/ms2_test_night3_thermalcap_20260821.jsonl";      ENV=test_night ;;
  rainy) MANIFEST="$IRIS_MANIFEST_DIR/ms2_test_rainy3_thermalcap_20260821.jsonl";      ENV=test_rain ;;
  *) echo "!! COND 只支持 day | night | rainy"; exit 1 ;;
esac
[[ -f "$MANIFEST" ]] || { echo "!! 找不到 $MANIFEST"; exit 1; }

EXTRA=()
if [[ -n "${UNET:-}" ]]; then
  [[ -f "$UNET" ]] || { echo "!! 找不到 checkpoint: $UNET"; exit 1; }
  EXTRA+=(--unet-checkpoint "$UNET")
  ALIGN="${ALIGN:-ssi_log}"          # 我们自己训的那条线是 log 空间
else
  ALIGN="${ALIGN:-ssi_disparity}"    # 已发布的 lotus-depth-g 是 disparity 变体
fi

export IRIS_THERMAL_STRETCH="${IRIS_THERMAL_STRETCH:-percentile}"
export IRIS_DECODE_CLAMP="${IRIS_DECODE_CLAMP:-0}"

mkdir -p "$OUT"
cd "$IRIS_REPO"
git log --oneline -1
echo "=== 配置 ==="
echo "条件      = $COND   ($ENV)"
echo "权重      = ${UNET:-已发布 lotus-depth-g-v2-1-disparity（零样本）}"
echo "对齐      = $ALIGN"
echo "帧数      = $N   (0 ＝ 全量)"
echo "分辨率    = 256x640 原生  对  长边 $RES"
echo "============"

for RUN in native scaled; do
  [[ "$RUN" == native ]] && PR=0 || PR="$RES"
  DEST="$OUT/$RUN"
  if [[ -f "$DEST/official/metrics/summary.json" ]]; then
    echo "[skip] $RUN 已有结果"; continue
  fi
  echo; echo "=== $RUN   (--processing-res $PR) ==="
  rm -rf "$DEST"; mkdir -p "$DEST"

  python tools/run_ms2_lotus_thermal_vae_official.py \
    --manifest "$MANIFEST" --ms2-root "$IRIS_MS2_ROOT" --output-dir "$DEST" \
    --max-samples "$N" --caption-mode empty --processing-res "$PR" \
    --save-raw-pred "${EXTRA[@]}"

  python tools/run_official_ms2_evaluation.py \
    --manifest "$MANIFEST" --data-root "$IRIS_MS2_ROOT" \
    --prediction-dir "$DEST/raw_predictions" --route iris-lotus \
    --align "$ALIGN" --output-dir "$DEST/official" --limit "$N"
done

echo
echo "=== 两个分辨率并排 ==="
python - "$OUT" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
rows, counts = {}, {}
for run in ("native", "scaled"):
    p = out / run / "official" / "metrics" / "summary.json"
    if not p.is_file():
        print(f"!! 缺 {p}"); raise SystemExit(1)
    st = json.loads(p.read_text(encoding="utf-8"))["image_wise"]["statistics"]
    rows[run] = {k: st[k]["mean"] for k in ("abs_rel", "rmse", "a1")}
    counts[run] = st["abs_rel"]["count"]

# 帧数不同就没得比 —— 宁可不打印，也不要打印一个对不上的对照。
if counts["native"] != counts["scaled"]:
    print(f"⛔ 两边帧数不同（{counts['native']} vs {counts['scaled']}），这不是一个对照。")
    raise SystemExit(1)

print(f"\n共 {counts['native']} 帧，同一批\n")
print(f"{'分辨率':>12s}{'AbsRel':>11s}{'RMSE (m)':>12s}{'δ1':>11s}")
for run, label in (("native", "256x640"), ("scaled", "放大后")):
    r = rows[run]
    print(f"{label:>12s}{r['abs_rel']:>11.5f}{r['rmse']:>12.3f}{r['a1']:>11.5f}")
print("\n⚠️ 模型训在 256x640 上。放大后变差是预期内的，它回答的是"
      "「糊到底是不是 latent 分辨率造成的」，不是"
      "「我们应该换分辨率推理」。")
print("⚠️ 抽样口径，不能和官方子集的数字混着引。")
PY

echo
echo "[完成] $OUT"
echo "定性图：两边的 raw_predictions 都留着了，可以直接画同一帧的对照。"
