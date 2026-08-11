#!/bin/bash
#SBATCH --job-name=irisms2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 「完整伪 GT + Iris 训练方式」线。跑的是 lotus/train_iris_ms2_g.py ——
# Iris 自己那份 trainer 的复制件，只换数据集，超参一个没动。
# 与 route_suite.sbatch 不是一条流水线，别混用。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   START            起点权重，二选一（这条线两个臂唯一的差别）：
#                      sd2  = stabilityai/stable-diffusion-2-base（Iris 自己的起点，
#                             conv_in 做 4→8 扩展，等于在 MS2 上重造一遍 Lotus-G）
#                      ckpt = jingheya/lotus-depth-g-v2-1-disparity（已训好的
#                             Lotus-G，跳过扩展）
#                    默认 sd2。
#   NO_CAPTIONS=1    无文本消融臂（每帧喂空串）。不设＝按 Iris 用真 caption 训练。
#   PSEUDO_DIR       calibrated_pseudo_depth 目录，默认 train_full
#   TRAIN_MANIFEST   默认 clip75_20260728（19,949 帧，≤75 token）
#   MAX_TRAIN_STEPS  默认 20000（Iris 的步数）。SMOKE 时会被覆盖
#   SMOKE=1          20 步就退，只为验开关是否生效
#   RUN_TAG          覆盖输出目录名
#
# ⚠️ lr 3e-05 是 Iris 从 SD2-base 起训的量级。START=ckpt 那一臂沿用它是
#    刻意的（两臂只差起点权重才构成受控对照），但它有把 checkpoint 冲掉的
#    风险 —— 先看 SMOKE 的 SL_A 曲线，再决定要不要投全量。
#
# ── 用法 ────────────────────────────────────────────────────────────
#   冒烟（两臂各一次，各约 10 分钟）：
#     START=sd2  SMOKE=1 sbatch -J sm_sd2  --time=00:40:00 --export=ALL,START,SMOKE ~/Iris/slurm/iris_ms2.sbatch
#     START=ckpt SMOKE=1 sbatch -J sm_ckpt --time=00:40:00 --export=ALL,START,SMOKE ~/Iris/slurm/iris_ms2.sbatch
#
#   全量（两臂并行）：
#     START=sd2  sbatch -J iris_sd2  --export=ALL,START ~/Iris/slurm/iris_ms2.sbatch
#     START=ckpt sbatch -J iris_ckpt --export=ALL,START ~/Iris/slurm/iris_ms2.sbatch
#
# 续跑：同一 RUN_TAG 重投即自动 resume（trainer 的 --resume_from_checkpoint=latest）。

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

START="${START:-sd2}"
case "$START" in
  sd2)  MODEL_NAME="stabilityai/stable-diffusion-2-base" ;;
  ckpt) MODEL_NAME="jingheya/lotus-depth-g-v2-1-disparity" ;;
  *)    echo "!! START 只能是 sd2 或 ckpt，收到 '$START'"; exit 1 ;;
esac

TAG="${RUN_TAG:-iris_ms2_$START}"
[[ "${NO_CAPTIONS:-0}" == "1" && -z "${RUN_TAG:-}" ]] && TAG="${TAG}_nocap"
if [[ "${SMOKE:-0}" == "1" ]]; then
  TAG="smoke_$TAG"
  MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20}"
  CKPT_STEP=20
fi

export MODEL_NAME
export OUTPUT_DIR="$IRIS_RUNS/iris_ms2/$TAG"
export MS2_MANIFEST="${TRAIN_MANIFEST:-$IRIS_MANIFEST_DIR/ms2_train_day2seq_clip75_20260728.jsonl}"
export MS2_ROOT="$IRIS_MS2_ROOT"
export PSEUDO_GT_DIR="${PSEUDO_DIR:-$IRIS_RUNS/pseudo_gt/train_full/calibrated_pseudo_depth}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20000}"
export CKPT_STEP="${CKPT_STEP:-1000}"
export MAIN_PORT="$MAIN_PROCESS_PORT"
export NO_CAPTIONS="${NO_CAPTIONS:-0}"

[[ -f "$MS2_MANIFEST" ]]     || { echo "!! 找不到 manifest: $MS2_MANIFEST"; exit 1; }
[[ -d "$PSEUDO_GT_DIR" ]]    || { echo "!! 找不到伪 GT 目录: $PSEUDO_GT_DIR"; exit 1; }
mkdir -p "$OUTPUT_DIR"

echo "--- 本次配置 ---"
echo "start     = $START ($MODEL_NAME)"
echo "captions  = $([[ "$NO_CAPTIONS" == "1" ]] && echo off || echo on)"
echo "manifest  = $MS2_MANIFEST"
echo "pseudo gt = $PSEUDO_GT_DIR"
echo "steps     = $MAX_TRAIN_STEPS (ckpt every $CKPT_STEP)"
echo "output    = $OUTPUT_DIR"
echo "----------------"
echo "门禁：日志里必须同时出现 SL_A（深度分支）与 SL_R（热像重建分支），"
echo "      以及一行 'target covers ...% of pixels' 且该数应为 100.0%。"

cd "$IRIS_REPO/lotus"
git -C "$IRIS_REPO" log --oneline -1
bash train_scripts/train_iris_ms2_g_depth.sh
