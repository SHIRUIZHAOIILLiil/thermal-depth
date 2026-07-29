#!/bin/bash
#SBATCH --job-name=iris_lotus_d
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --signal=B:USR1@600
#
# Lotus discriminative 全量训练。
# gpu 分区上限 48h；这里报 24h —— backfill 更容易插队，配合 resume 分段跑完。
# 用法：
#   cd $SCRATCH/logs && sbatch $HOME/Iris/slurm/train_lotus_d.sbatch
# 续跑（同一个 RUN_TAG 会自动 --resume_from_checkpoint=latest）：
#   RUN_TAG=b_line sbatch --export=ALL,RUN_TAG $HOME/Iris/slurm/train_lotus_d.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/common.sh"

export RUN_TAG="${RUN_TAG:-lotus_d}"
export OUTPUT_DIR="$IRIS_RUNS/$RUN_TAG"
mkdir -p "$OUTPUT_DIR"

# 覆盖 train_iris_d_depth.sh 里的默认相对路径
export MODEL_NAME="stabilityai/stable-diffusion-2-base"
export TRAIN_DATA_DIR_HYPERSIM="$IRIS_DATA/hypersim"
export TRAIN_DATA_DIR_VKITTI="$IRIS_DATA/vkitti"
export BASE_TEST_DATA_DIR="$IRIS_REPO/datasets/eval/"
export VALIDATION_IMAGES="$IRIS_REPO/datasets/quick_validation/"
export HYPERSIM_TEXT_PATH="$IRIS_REPO/internVL/outputs/lotus/hypersim_depth_descriptions.json"
export VKITTI_TEXT_PATH="$IRIS_REPO/internVL/outputs/lotus/vkitti_depth_descriptions.json"

cd "$IRIS_REPO/lotus"
bash train_scripts/train_iris_d_depth.sh

echo "=== 训练进程退出，状态 $? ==="
ls -la "$OUTPUT_DIR"
