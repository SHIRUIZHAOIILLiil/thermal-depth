#!/bin/bash
#SBATCH --job-name=vis
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 任务 3 的对比可视化。**不申请 GPU** —— 只读 raw_predictions/*.npy、GT 和输入图，
# 和 analyze.sbatch 一样是纯 CPU 活。
#
# 在集群上出图而不是把数据拉回本地：一次对比要读的 npy 是几个 GB，出来的 PNG 是几 MB。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   PREDS      必填。`目录[:标签][@对齐]`，**逗号分隔**（与 analyze.sbatch 同款写法）。
#              ⚠️ `@对齐` 是这个作业相对 analyze.sbatch 的关键差别：
#              analyze.sbatch 的 --align 一刀切作用于所有目录，所以原版 AnyThermal（ssi）
#              和我方六线（ssi_disparity）**没法进同一个作业**；而图上它们必须并排。
#              不写 @ 就用 ALIGN（默认 ssi_disparity）。
#   MANIFEST   必填，与出预测时同一份
#   TAG        必填，输出目录名
#   PICK       far（默认）| gap | flip | uniform | manual
#   STRATUM    默认 `depth/far >30m`
#   GAP        PICK=gap 时必填，形如 `d2:b`（谁在该分层上领先谁最多）
#   PICK_REF   PICK=far 时按谁的误差排（默认 mean ＝ 全部模型平均）
#   FRAME_IDS  PICK=manual 时的帧 id，空格分隔
#   FRAMES     默认 5；MIN_SEP 默认 30（避免挑到同一瞬间的连续帧）
#   INPUTS     thermal（默认）| rgb | thermal,rgb
#   GT_VIEW    thermal（默认）| rgb —— 一张图只能一个视角，混视角是禁止的
#   FIG_LANG   zh（默认）| en；FONT 指到一个 CJK ttf，没有就自动降级成英文
#              ⚠️ 不能叫 LANG —— 那是 POSIX 的 locale 变量，登录环境里本来就是
#                 en_US.UTF-8 之类，--export=ALL 会把它带进来顶掉默认值，
#                 于是拼出 `--lang en_US.UTF-8`，argparse 直接退出码 2
#   STRIDE     扫描步长，默认 1（挑帧只能从扫过的帧里挑）
#
# ── 用法 ────────────────────────────────────────────────────────────
#   主图：d2 的远端优势（受控归因那一张）
#     PREDS="$SCRATCH/runs/eval/b_e05_raw/raw_predictions:b,\
#$SCRATCH/runs/eval/c2_raw/raw_predictions:c2,\
#$SCRATCH/runs/eval/d2_raw/raw_predictions:d2" \
#     MANIFEST=$SCRATCH/manifests/sequence_level_internvl3_8b/ms2_test_16-08-46_rgb_depth_v1_clip75_20260728.jsonl \
#     TAG=far_gap PICK=gap GAP=d2:b \
#       sbatch -J vis_far --export=ALL,PREDS,MANIFEST,TAG,PICK,GAP ~/Iris/slurm/vis.sbatch
#
#   跟原版 AnyThermal 并排（注意最后一列的 @ssi —— 它跟我方六线不是同一个对齐空间，
#   这也是 analyze.sbatch 做不了这张图的原因）
#     PREDS="$SCRATCH/runs/eval/b_e05_raw/raw_predictions:b_thermal,\
#$SCRATCH/runs/eval/d2_raw/raw_predictions:d2,\
#$SCRATCH/runs/anythermal/Midas_anythermal/raw_predictions:anythermal_orig@ssi" \
#     MANIFEST=... TAG=vs_anythermal PICK=far \
#       sbatch -J vis_at --export=ALL,PREDS,MANIFEST,TAG,PICK ~/Iris/slurm/vis.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

: "${PREDS:?必须指定 PREDS（逗号分隔的 目录[:标签][@对齐]）}"
: "${MANIFEST:?必须指定 MANIFEST}"
: "${TAG:?必须指定 TAG}"

[[ -f "$MANIFEST" ]] || { echo "!! 找不到 manifest: $MANIFEST"; exit 1; }

# 带 choices 的参数在这里先验一遍：直接喂给 argparse 的话，报错埋在 .err 里、
# 退出码只有一个 2，在批处理作业里很难看出是哪个变量被环境顶掉了
case "${FIG_LANG:-zh}" in
  zh|en) ;;
  *) echo "!! FIG_LANG 只能是 zh 或 en，收到 '${FIG_LANG}'"; exit 1 ;;
esac

OUT="$IRIS_RUNS/vis/$TAG"
mkdir -p "$OUT"

cd "$IRIS_REPO"

ARGS=(
  --manifest "$MANIFEST"
  --data-root "$IRIS_MS2_ROOT"
  --output-dir "$OUT"
  --align "${ALIGN:-ssi_disparity}"
  --gt-view "${GT_VIEW:-thermal}"
  --inputs "${INPUTS:-thermal}"
  --pick "${PICK:-far}"
  --stratum "${STRATUM:-depth/far >30m}"
  --frames "${FRAMES:-5}"
  --min-separation "${MIN_SEP:-30}"
  --stride "${STRIDE:-1}"
  --lang "${FIG_LANG:-zh}"
)

# 逗号拆成多个 --predictions，顺便确认目录里真有 .npy。
# 标签后面可能跟 @对齐，数文件时要先剥掉。
IFS=',' read -ra ENTRIES <<< "$PREDS"
for entry in "${ENTRIES[@]}"; do
  dir="${entry%%:*}"
  dir="${dir%%@*}"
  [[ -d "$dir" ]] || {
    echo "!! 找不到预测目录: $dir"
    echo "   六线的 raw 预测要在 eval.sbatch 上加 SAVE_RAW=1 才会写；"
    echo "   零训练直推走 direct.sbatch（它无条件写）。"
    exit 1
  }
  count=$(find "$dir" -maxdepth 1 -name '*.npy' | wc -l || true)
  [[ "$count" -gt 0 ]] || { echo "!! $dir 下没有 .npy"; exit 1; }
  echo "预测集 $entry  ($count 个 .npy)"
  ARGS+=(--predictions "$entry")
done

[[ -n "${GAP:-}" ]] && ARGS+=(--gap "$GAP")
[[ -n "${PICK_REF:-}" ]] && ARGS+=(--pick-ref "$PICK_REF")
[[ -n "${TITLE:-}" ]] && ARGS+=(--title "$TITLE")
[[ -n "${FRAME_IDS:-}" ]] && ARGS+=(--frame-ids $FRAME_IDS)
[[ -n "${RESCAN:-}" ]] && ARGS+=(--rescan)

# 集群的 conda 环境没有中文字体，工具会自动把图上文字降级成英文。
# 要中文就把一个 CJK ttf 传到 $SCRATCH/fonts/ 并设 FONT。
FONT="${FONT:-$SCRATCH/fonts/msyh.ttc}"
[[ -f "$FONT" ]] && ARGS+=(--font "$FONT")

echo "=== python tools/build_comparison_figure.py ${ARGS[*]} ==="
python tools/build_comparison_figure.py "${ARGS[@]}"
rc=$?

echo "=== 退出码 $rc ==="
ls -la "$OUT"
exit $rc
