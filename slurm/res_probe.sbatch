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
# VAE 下采样 8 倍，latent 就是 32x80。把热像先放大到 512x1280，latent 变成
# 64x160，看结构回不回得来、指标动不动。
#
# ⚠️ 2026-09-25 补：vae_thermal_probe 已经测出编解码保住了 94% 的梯度幅值
#    （真做 8 倍降采样只剩 37%），所以「比 8 像素细的结构在编码那一步就没了」
#    这个原始假设是错的。这条探针因此从「找答案」降级成「补一个否定证据」：
#    它现在预期得到「放大没用」，而那本身是有价值的结论。
#
# ⚠️ 这是诊断不是配置。模型训在 256x640 上，改推理分辨率就是分布偏移，
#    完全可能更差。
#
# ⚠️ 两个分辨率必须跑同一批帧 —— 而且打分的也得是同一批。
#    第一版栽在这：推理工具的 --max-samples 走 choose_uniform（在全量里均匀抽），
#    打分工具的 --limit 却取**前 N 帧**，两边选出来的根本不是一批，打分时去找一个
#    没推理过的帧就炸了。改成这个仓库里 ceiling_probe 已经在用的做法：
#    **先裁一份子清单，两边都喂它**，两边都不再传 N。
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
echo "帧数      = $N   (0 ＝ 全量；裁成子清单后两边共用)"
echo "分辨率    = 256x640 原生  对  长边 $RES"
echo "============"

# 裁一份子清单，两个分辨率和两次打分全用它。
SUBSET="$OUT/subset.jsonl"
if [[ "$N" == "0" ]]; then
  cp "$MANIFEST" "$SUBSET"
else
  python - "$MANIFEST" "$SUBSET" "$N" <<'PY'
import sys
from pathlib import Path

import numpy as np

src, dst, want = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
rows = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
if want >= len(rows):
    keep = list(range(len(rows)))
else:
    keep = sorted(set(np.linspace(0, len(rows) - 1, want).round().astype(int).tolist()))
dst.write_text("\n".join(rows[i] for i in keep) + "\n", encoding="utf-8")
print(f"[子清单] 从 {len(rows)} 帧里均匀取 {len(keep)} 帧")
PY
fi
FRAMES="$(wc -l < "$SUBSET")"
echo "[子清单] $FRAMES 帧 -> $SUBSET"

for RUN in native scaled; do
  [[ "$RUN" == native ]] && PR=0 || PR="$RES"
  DEST="$OUT/$RUN"
  if [[ -f "$DEST/official/metrics/summary.json" ]]; then
    echo "[skip] $RUN 已有结果"; continue
  fi
  echo; echo "=== $RUN   (--processing-res $PR) ==="
  rm -rf "$DEST"; mkdir -p "$DEST"

  # --max-samples 0 ＝ 子清单里的全部。两边都不再各自抽样。
  # --condition-posterior mode：两个分辨率的 latent 形状不同，用 sample 的话
  # 同一个种子抽出来的随机数也不同，那就又多一个混淆。
  python tools/run_ms2_lotus_thermal_vae_official.py \
    --manifest "$SUBSET" --ms2-root "$IRIS_MS2_ROOT" --output-dir "$DEST" \
    --max-samples 0 --caption-mode empty --processing-res "$PR" \
    --condition-posterior mode --save-raw-pred "${EXTRA[@]}"

  python tools/run_official_ms2_evaluation.py \
    --manifest "$SUBSET" --data-root "$IRIS_MS2_ROOT" \
    --prediction-dir "$DEST/raw_predictions" --route iris-lotus \
    --align "$ALIGN" --output-dir "$DEST/official"
done

echo
echo "=== 两个分辨率并排 ==="
python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows, counts = {}, {}
for run in ("native", "scaled"):
    path = out / run / "official" / "metrics" / "summary.json"
    if not path.is_file():
        print(f"!! 缺 {path}")
        raise SystemExit(1)
    stats = json.loads(path.read_text(encoding="utf-8"))["image_wise"]["statistics"]
    rows[run] = {k: stats[k]["mean"] for k in ("abs_rel", "rmse", "a1")}
    counts[run] = stats["abs_rel"]["count"]

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
