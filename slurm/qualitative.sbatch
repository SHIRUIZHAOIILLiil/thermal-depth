#!/bin/bash
#SBATCH --job-name=qual
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 定性对比图。跑的是 `tools/export_qualitative.py` —— **不要另写出图工具**：
# 2026-08-03 有人写过一个吃 raw npy 的，出来的图与论文插图不是一个观感，已删。
# 差别不在代码质量，在上色约定：这个工具直接 import
# `lotus/utils/image_utils.py::colorize_depth_map`，对**视差**上色且 reverse_color=True
# （红=近 / 蓝=远，与 lotus/infer.py:211 同一约定），稀疏 GT 还会先膨胀再上色。
#
# 它按 ROUTE:CHECKPOINT 现跑模型，所以要 GPU；每帧每模型一次前向，几分钟。
# 顺带每帧输出 sky_band_depths.json（上带中位），与 analyze_sky_band.py 同一个量。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   MODELS     必填。空格分隔的 `ROUTE:CHECKPOINT[:标签]`，整串加引号
#   TAG        必填，输出目录名
#   MANIFEST   考卷；默认独立 test 集 16-08-46
#   FRAME_IDS  帧 id，空格分隔；不设则按 FRAMES 均匀抽
#   FRAMES     默认 5
#   STYLE      both（默认）| official | shared
#              ⚠️ official 风格下每一栏**各自** min-max 归一化，所以六个模型的
#              颜色互相不可比 —— 同样的橙色在不同栏里是不同的深度。横向比较要看
#              shared（强制共用一个归一化区间）。official 那张只用来对齐论文观感。
#   PANEL_W    每栏宽度，默认 640（= 源图原宽）。调大只是渲染更大，不会增加细节。
#   GT_DILATE  稀疏 GT 的膨胀窗口，默认 3；幻灯上看不清就调 5
#   CAPTION    推理时喂的文本：empty（默认）| correct | shuffled。
#              也可以给逗号分隔的一串，一个模型一个，用来把「同一份权重换 prompt」
#              和「换权重」放进同一张图（`--caption-mode` 本来就支持）。
#   PANEL_LABELS on（默认）| off。off = 面板上不画字，给幻灯用；栏目名写在图注里。
#
# ── 用法 ────────────────────────────────────────────────────────────
#   天空修复前后对比：
#     MODELS="b_thermal_unet:$SCRATCH/runs/route_suite/b_thermal_unet/epoch05_weights.pt:b_e5 \
#             b_thermal_unet:$SCRATCH/runs/route_suite/b_skyloss05/epoch05_weights.pt:b_skyloss" \
#     TAG=sky_fix FRAME_IDS="000703 001245 001445" \
#       sbatch -J qual_sky --export=ALL,MODELS,TAG,FRAME_IDS ~/Iris/slurm/qualitative.sbatch
#
#   ⚠️ 挑帧别只挑最好看的。用 sky_band_per_frame.csv 按改善量的分位数取（见
#   docs 里的说明），并在图注里写清楚是哪一档，否则读者会以为那是典型帧。

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

: "${MODELS:?必须指定 MODELS（ROUTE:CHECKPOINT[:标签]，空格分隔）}"
: "${TAG:?必须指定 TAG}"

MANIFEST="${MANIFEST:-$IRIS_MANIFEST_DIR/ms2_test_16-08-46_rgb_depth_v1_clip75_20260728.jsonl}"
OUT="$IRIS_RUNS/vis/$TAG"
[[ -f "$MANIFEST" ]] || { echo "找不到 manifest: $MANIFEST"; exit 1; }
mkdir -p "$OUT"

ARGS=(
  --models $MODELS
  --output-dir "$OUT"
  --val-manifest "$MANIFEST"
  --ms2-root "$IRIS_MS2_ROOT"
  --style "${STYLE:-both}"
  --panel-width "${PANEL_W:-640}"
  --gt-dilate "${GT_DILATE:-3}"
  --caption-mode "${CAPTION:-empty}"
  --panel-labels "${PANEL_LABELS:-on}"
)
if [[ -n "${FRAME_IDS:-}" ]]; then
  ARGS+=(--frame-ids $FRAME_IDS)
else
  ARGS+=(--frames "${FRAMES:-5}")
fi

cd "$IRIS_REPO"
echo "=== python tools/export_qualitative.py ${ARGS[*]} ==="
python tools/export_qualitative.py "${ARGS[@]}"
rc=$?

echo "=== 退出码 $rc ==="
echo "=== 上带中位（每帧每模型） ==="
[[ -f "$OUT/sky_band_depths.json" ]] && python - "$OUT/sky_band_depths.json" <<'PY'
import json, sys
for e in json.load(open(sys.argv[1], encoding="utf-8")):
    gt = f'{e["top_third_gt_median_m"]:.1f}' if e["top_third_gt_median_m"] else "-"
    print(f'  {e["frame"]:>32} {e["model"]:>14}  pred {e["top_third_pred_median_m"]:6.1f} m'
          f'   p90 {e["top_third_pred_p90_m"]:6.1f} m   GT {gt:>5} m')
PY
ls -la "$OUT"
exit $rc
