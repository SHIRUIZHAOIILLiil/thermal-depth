#!/bin/bash
#SBATCH --job-name=relexp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 把一条臂的相对深度导出成 raw_predictions/<id>.npy，供下游的米制头训练。
#
# 不用 `sbatch --wrap` 跑这件事：common.sh 里有 module load 和 conda activate，
# 而 --wrap 的非交互 shell 里 module 不是已定义的函数，配上它的 set -euo pipefail
# 就是两秒退出、退出码 2。这也是为什么路径要走 env.sh 而不是在提交命令里手拼 ——
# 从登录节点提交时 $IRIS_MANIFEST_DIR 会静默展开成空串。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   RUN       必填。训练输出目录名，如 sd2_cap_pct_logtn
#   STEP      必填。要导出的 checkpoint 步数，如 16000。必须已转换
#             （$IRIS_RUNS/iris_ms2/$RUN/converted/step${STEP}_weights.pt）
#   MANIFEST  必填。要跑哪份清单。⚠️ 训练米制头时给 **train** 划分 ——
#             那个头要学的是从热像和 caption 恢复尺度，拿 test 的预测去学
#             就是拿 test GT 训练
#   PROMPT    empty | correct | permuted，默认 correct。须与该臂训练时一致
#   ALIGN     默认 ssi_log。⚠️ 须与训练的 norm_type 一致：
#             trunc_disparity→ssi_disparity，truncnorm→ssi，log_truncnorm→ssi_log
#   STRETCH   默认 percentile。⚠️ 同样须与训练时一致，错了不报错只让数变差
#   OUT_TAG   输出目录名，默认 ${RUN}_s${STEP}_rel
#   STRIDE    抽帧间隔，默认 1（全量）。冒烟用 1500 → 75,688 帧里抽约 50 张。
#             ⚠️ train_route_suite.py 没有 --limit，只有 --val-stride；
#             「以为有个限量参数」在这个项目里踩过一次了
#
# ── 用法 ────────────────────────────────────────────────────────────
#   冒烟（约 50 帧，先确认导出真的落盘）：
#     RUN=sd2_cap_pct_logtn STEP=16000 STRIDE=1500 \
#       MANIFEST=$IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl \
#       sbatch --time=01:00:00 --export=ALL,RUN,STEP,STRIDE,MANIFEST ~/Iris/slurm/export_relative.sbatch
#
#   全量：STRIDE=1 或不设，其余不变
#
# ⚠️ 全量 75,688 帧约 50 GB（256x640 float32，每张 655 KB）。投之前看配额。

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

: "${RUN:?必须指定 RUN（训练输出目录名）}"
: "${STEP:?必须指定 STEP（checkpoint 步数）}"
: "${MANIFEST:?必须指定 MANIFEST}"

PROMPT="${PROMPT:-correct}"
ALIGN="${ALIGN:-ssi_log}"
STRETCH="${STRETCH:-percentile}"
OUT_TAG="${OUT_TAG:-${RUN}_s${STEP}_rel}"
STRIDE="${STRIDE:-1}"

CKPT="$IRIS_RUNS/iris_ms2/$RUN/converted/step${STEP}_weights.pt"
OUT="$IRIS_RUNS/eval/$OUT_TAG"

[[ -f "$CKPT" ]]     || { echo "!! 找不到转换后的权重 $CKPT"; exit 1; }
[[ -f "$MANIFEST" ]] || { echo "!! 找不到清单 $MANIFEST"; exit 1; }

export IRIS_THERMAL_STRETCH="$STRETCH"

echo "--- 本次配置 ---"
echo "run       = $RUN"
echo "checkpoint= $CKPT"
echo "manifest  = $(basename "$MANIFEST")  ($(wc -l < "$MANIFEST") 帧)"
echo "prompt    = $PROMPT"
echo "对齐       = $ALIGN   (须与训练的 norm_type 一致)"
echo "thermal   = $IRIS_THERMAL_STRETCH  (须与训练时一致)"
echo "抽帧间隔   = $STRIDE  ($([[ "$STRIDE" == "1" ]] && echo 全量 || echo 冒烟))"
echo "output    = $OUT"
echo "----------------"
echo "门禁：日志里必须出现 'saving raw predictions to'，否则这一趟没产出预测，"
echo "      只算了一遍指标。"

cd "$IRIS_REPO"

ARGS=(--route b_thermal_unet --output-dir "$OUT" --ms2-root "$IRIS_MS2_ROOT"
      --train-manifest "$MANIFEST" --val-manifest "$MANIFEST"
      --eval-checkpoint "$CKPT" --val-caption-mode "$PROMPT"
      --align-mode "$ALIGN" --num-inference-steps 1 --condition-latent mode
      --val-stride "$STRIDE" --save-raw-pred --local-files-only)

python tools/train_route_suite.py "${ARGS[@]}"

echo ""
echo "=== 导出了 $(ls "$OUT/raw_predictions" 2>/dev/null | wc -l) 张 ==="
du -sh "$OUT/raw_predictions" 2>/dev/null || true
