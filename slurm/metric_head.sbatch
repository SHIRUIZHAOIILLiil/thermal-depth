#!/bin/bash
#SBATCH --job-name=mhead
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 训练米制重标定头：从热像和 caption 逐像素恢复 log D = A*y + B。
#
# 三条臂，不是四条。文本到不了两个标量，所以 global-only 那条无论带不带
# caption 都是同一次运行：
#
#   ARM=global   两个标量        —— 基线（度量微调线原本的设计）
#   ARM=nocap    逐像素，空文本  —— 1→2 量的是逐像素值不值
#   ARM=cap      逐像素 + 文本    —— 2→3 量的是文本对尺度恢复有没有额外贡献
#
# ⚠️ 相对深度是带 correct caption 导出的，所以 ARM=nocap 去掉的只是**头自己**
# 的文本分支，底层相对深度里已经吃进去的 caption 还在。这一臂回答的是
# 「文本对尺度恢复有没有额外贡献」，不是「整条管线用不用 caption」。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   ARM       必填：global | nocap | cap
#   REL_DIR   必填。export_relative.sbatch 导出的 raw_predictions 目录
#   MANIFEST  必填。⚠️ 必须与导出时同一份，且是 **train** 划分
#   STRETCH   默认 percentile。⚠️ 须与产生 REL_DIR 的那条臂一致
#   EPOCHS    默认 2
#   SEED      默认 42。复现用：同一 ARM 换 SEED 再跑一次
#   OUT_TAG   输出目录名，默认 ${ARM}_s${SEED}
#
# ── 用法 ────────────────────────────────────────────────────────────
#   三条臂各投一次（互相独立，可并行）：
#     for A in global nocap cap; do ARM=$A REL_DIR=... MANIFEST=... \
#       sbatch -J mh_$A --export=ALL,ARM,REL_DIR,MANIFEST ~/Iris/slurm/metric_head.sbatch; done

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

: "${ARM:?必须指定 ARM（global | nocap | cap）}"
: "${REL_DIR:?必须指定 REL_DIR}"
: "${MANIFEST:?必须指定 MANIFEST}"

STRETCH="${STRETCH:-percentile}"
EPOCHS="${EPOCHS:-2}"
SEED="${SEED:-42}"
OUT_TAG="${OUT_TAG:-${ARM}_s${SEED}}"
OUT="$IRIS_RUNS/metric_head/$OUT_TAG"

case "$ARM" in
  global) EXTRA="--global-only" ;;
  nocap)  EXTRA="--no-captions" ;;
  cap)    EXTRA="" ;;
  *) echo "!! ARM 只能是 global | nocap | cap，收到 '$ARM'"; exit 1 ;;
esac

[[ -d "$REL_DIR" ]]  || { echo "!! 找不到相对深度目录 $REL_DIR"; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "!! 找不到清单 $MANIFEST"; exit 1; }

echo "--- 本次配置 ---"
echo "臂        = $ARM  ($EXTRA)"
echo "相对深度   = $REL_DIR  ($(ls "$REL_DIR" | wc -l) 张)"
echo "manifest  = $(basename "$MANIFEST")"
echo "thermal   = $STRETCH  (须与产生相对深度的那条臂一致)"
echo "epochs    = $EPOCHS   seed = $SEED"
echo "output    = $OUT"
echo "----------------"
echo "门禁：三条臂的 step 0 必须完全一致 —— 逐像素那一路是零初始化的，"
echo "      所以它们从同一个全局仿射出发，任何差距都只能是训练带来的。"

cd "$IRIS_REPO"

python tools/train_metric_rescale_head.py \
  --manifest "$MANIFEST" \
  --ms2-root "$IRIS_MS2_ROOT" \
  --relative-dir "$REL_DIR" \
  --out-dir "$OUT" \
  --stretch "$STRETCH" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  $EXTRA
