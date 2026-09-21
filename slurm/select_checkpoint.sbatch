#!/bin/bash
#SBATCH --job-name=selckpt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 在官方 val 上给一条外部线的每个 checkpoint 打分，按我们自己的规则选点。
#
# 为什么要有这个：DA2 与 PPD 两条适配线**都没有 val**（DA2 的训练脚本里没有，
# PPD 的 yaml 里 val_dataset.dataset_opts 是空表）。所以两条线都训完了，却没有
# 合规的办法从一堆 checkpoint 里挑一个 —— 而拿 test 挑是这个项目的红线。
#
# 补的是推理，不是训练：checkpoint 已经在盘上，这里只是给它们打分。
#
# ⚠️ 全程只碰 val。test 在选完之后单独跑一次，不在这个脚本里。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   MODEL     depth_anything_v2（默认）| pixel_perfect_depth
#   CKPT_ROOT 必填。装着各步 checkpoint 的目录，例如
#             $SCRATCH/runs/da2_thermal/v1   （里面是 step2000/ step4000/ ...）
#   PATTERN   在 CKPT_ROOT 下匹配哪些条目，默认 "step*"
#   ALIGN     打分用的对齐，默认 ssi_disparity（DA2 的原生空间）
#   MANIFEST  val 清单，默认官方 3 序列那份
#   LIMIT     每个 checkpoint 只跑前 N 帧，默认 0 ＝ 全量。冒烟用 40
#   OUT_TAG   输出目录名，默认 select_<MODEL>_<CKPT_ROOT 的末级名>
#
# ── 用法 ────────────────────────────────────────────────────────────
#   冒烟（40 帧 × 10 个点，确认这条链路通）：
#     CKPT_ROOT=$SCRATCH/runs/da2_thermal/v1 LIMIT=40 \
#       sbatch --time=00:40:00 --export=ALL,CKPT_ROOT,LIMIT \
#       ~/Iris/slurm/select_checkpoint.sbatch
#
#   全量：把 LIMIT 去掉即可。
#
# 不 source common.sh 的理由不成立 —— 这里用的是 iris 环境，所以照常 source。

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

: "${CKPT_ROOT:?必须指定 CKPT_ROOT（装着各步 checkpoint 的目录）}"
MODEL="${MODEL:-depth_anything_v2}"
PATTERN="${PATTERN:-step*}"
ALIGN="${ALIGN:-ssi_disparity}"
LIMIT="${LIMIT:-0}"
MANIFEST="${MANIFEST:-$IRIS_MANIFEST_DIR/ms2_val_official3_thermalcap_20260821.jsonl}"
OUT_TAG="${OUT_TAG:-select_${MODEL}_$(basename "$CKPT_ROOT")}"
OUT="$IRIS_RUNS/analysis/$OUT_TAG"

# ⚠️ 与训练时一致，错了不报错只让数变差。
export IRIS_THERMAL_STRETCH="${IRIS_THERMAL_STRETCH:-percentile}"

[[ -d "$CKPT_ROOT" ]] || { echo "!! 找不到 $CKPT_ROOT"; exit 1; }
if [[ ! -f "$MANIFEST" ]]; then
  echo "!! 找不到 val 清单 $MANIFEST"
  ls -1 "$IRIS_MANIFEST_DIR" 2>/dev/null | grep -i val | sed 's/^/     /'
  exit 1
fi

shopt -s nullglob
CKPTS=("$CKPT_ROOT"/$PATTERN)
shopt -u nullglob
[[ "${#CKPTS[@]}" -ge 1 ]] || { echo "!! $CKPT_ROOT 下没有匹配 $PATTERN 的条目"; ls -1 "$CKPT_ROOT"; exit 1; }

mkdir -p "$OUT"
cd "$IRIS_REPO"
git log --oneline -1

echo "=== 配置 ==="
echo "模型      = $MODEL"
echo "checkpoint= ${#CKPTS[@]} 个，来自 $CKPT_ROOT"
echo "对齐      = $ALIGN   (须是该模型的原生空间)"
echo "清单      = $MANIFEST"
echo "帧数上限  = $LIMIT   (0 ＝ 全量)"
echo "thermal   = $IRIS_THERMAL_STRETCH"
echo "输出      = $OUT"
echo "⚠️ 全程只碰 val。test 在选完之后单独跑。"
echo "============"

for ckpt in "${CKPTS[@]}"; do
  name="$(basename "$ckpt")"
  dest="$OUT/$name"
  if [[ -f "$dest/metrics/summary.json" ]]; then
    echo "[skip] $name 已有结果"
    continue
  fi
  echo
  echo "=== $name ==="
  mkdir -p "$dest"

  infer=(--model "$MODEL" --manifest "$MANIFEST" --data-root "$IRIS_MS2_ROOT"
         --out-dir "$dest/raw_predictions" --stretch "$IRIS_THERMAL_STRETCH")
  [[ "$LIMIT" != "0" ]] && infer+=(--limit "$LIMIT")
  case "$MODEL" in
    depth_anything_v2)   infer+=(--weights "$ckpt") ;;
    pixel_perfect_depth) infer+=(--ppd-checkpoint "$ckpt") ;;
  esac
  python tools/run_external_depth_inference.py "${infer[@]}"

  score=(--manifest "$MANIFEST" --data-root "$IRIS_MS2_ROOT"
         --prediction-dir "$dest/raw_predictions" --route iris-lotus
         --align "$ALIGN" --output-dir "$dest")
  [[ "$LIMIT" != "0" ]] && score+=(--limit "$LIMIT")
  python tools/run_official_ms2_evaluation.py "${score[@]}"
done

echo
echo "=== val 曲线 ==="
python tools/summarize_checkpoint_sweep.py "$OUT" --limit "$LIMIT"

echo
echo "[完成] $OUT"
echo "选完之后再单独跑 test —— 别在这个脚本里加。"
