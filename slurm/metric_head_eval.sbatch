#!/bin/bash
#SBATCH --job-name=mheval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 在 test 上给米制头打分。一个作业跑一条臂的三个条件。
#
# 输出两行：align=none（零对齐的真米制，我们自己的数）和 align=median
# （已发表基线用的口径，逐图中位数缩放对齐 test GT）。两者写不同文件，
# 不能合成一张表 —— 它们是两个不同的主张。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   HEAD_DIR  必填。$IRIS_RUNS/metric_head/<臂>，里面要有 head_epoch*.pt
#   REL_STEM  必填。test 相对深度目录的前缀，脚本会拼上 _<条件>；
#             例如 sd2_cap_pct_logtn_s16000_rel
#   EPOCH     用哪一轮的权重，默认 1
#   STRETCH   默认 percentile。⚠️ 须与产生相对深度的那条臂一致
#   CONDS     默认 "day3_common night3 rainy3"
#
# ⚠️ 相对深度必须来自训练这个头时用的同一个 run、同一步、同一个 prompt。
# 换一样，头就是在没见过的分布上被打分，而且不会报错。
#
# ── 用法 ────────────────────────────────────────────────────────────
#   for A in global nocap cap; do HEAD_DIR=$IRIS_RUNS/metric_head/${A}_s42_fix \
#     REL_STEM=sd2_cap_pct_logtn_s16000_rel \
#     sbatch -J mhe_$A --export=ALL,HEAD_DIR,REL_STEM ~/Iris/slurm/metric_head_eval.sbatch; done

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

: "${HEAD_DIR:?必须指定 HEAD_DIR}"
: "${REL_STEM:?必须指定 REL_STEM（test 相对深度目录前缀）}"

EPOCH="${EPOCH:-1}"
STRETCH="${STRETCH:-percentile}"
CONDS="${CONDS:-day3_common night3 rainy3}"
HEAD="$HEAD_DIR/head_epoch${EPOCH}.pt"

[[ -f "$HEAD" ]] || { echo "!! 找不到头权重 $HEAD"; exit 1; }

echo "--- 本次配置 ---"
echo "head      = $HEAD"
echo "相对深度前缀 = $REL_STEM"
echo "thermal   = $STRETCH  (须与产生相对深度的那条臂一致)"
echo "条件       = $CONDS"
echo "----------------"

cd "$IRIS_REPO"

for C in $CONDS; do
  REL="$IRIS_RUNS/eval/${REL_STEM}_${C}/raw_predictions"
  MAN="$IRIS_MANIFEST_DIR/ms2_test_${C}_thermalcap_20260821.jsonl"
  OUT="$HEAD_DIR/eval_${C}"
  if [[ ! -d "$REL" ]]; then
    echo "!! 跳过 $C：找不到 $REL"
    continue
  fi
  echo ""
  echo "=== $C  ($(ls "$REL" | wc -l) 张相对深度) ==="
  python tools/eval_metric_rescale_head.py \
    --head "$HEAD" --manifest "$MAN" --ms2-root "$IRIS_MS2_ROOT" \
    --relative-dir "$REL" --out-dir "$OUT" --stretch "$STRETCH"
done

echo ""
echo "=== 汇总 ==="
python - "$HEAD_DIR" <<'PYEOF'
import json, sys, glob, os
root = sys.argv[1]
print(f"{'条件':<14}{'对齐':<9}{'AbsRel':>10}{'RMSE':>10}{'δ1':>9}")
print("-" * 52)
for path in sorted(glob.glob(os.path.join(root, "eval_*", "metric_head_*.json"))):
    blob = json.load(open(path, encoding="utf-8"))
    cond = os.path.basename(os.path.dirname(path)).replace("eval_", "")
    s = blob["summary"]
    print(f"{cond:<14}{blob['align_mode']:<9}{s['abs_rel']:>10.5f}"
          f"{s['rmse']:>10.4f}{s['a1']:>9.4f}")
PYEOF
