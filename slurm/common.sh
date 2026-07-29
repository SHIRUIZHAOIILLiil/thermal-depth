#!/bin/bash
# Aire 作业的公共环境设置。由各 .sbatch 脚本 source。
# 只放 export 和 module load —— 这些必须留在作业脚本里，不能进 ~/.bashrc（ARC 官方要求）。

set -euo pipefail

module load miniforge/24.7.1 cuda
conda activate iris

# $SCRATCH 由系统提供（/mnt/scratch/<user>）。所有输出、缓存、数据都走这里，
# home 只有 65GB / 150万 inode，写满会拖垮整个登录节点。
: "${SCRATCH:?SCRATCH 未定义 —— 确认已在 Aire 上运行}"

export IRIS_REPO="${IRIS_REPO:-$HOME/Iris}"
export IRIS_DATA="${IRIS_DATA:-$SCRATCH/data}"
export IRIS_RUNS="${IRIS_RUNS:-$SCRATCH/runs}"

# train_route_suite.py 读这两个变量覆盖它的本地硬编码默认值
# （/mnt/e/dataset/ms2 和 /mnt/e/project/thermal-depth/...）。不设时保持本地行为。
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$IRIS_DATA/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$SCRATCH/manifests/sequence_level_internvl3_8b}"

export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH/torch_cache}"

mkdir -p "$IRIS_RUNS" "$HF_HOME" "$TORCH_HOME"

# 起跑前先确认数据和 manifest 真的在，省得排队几小时后才发现路径错
[[ -d "$IRIS_MS2_ROOT/sync_data" ]] || { echo "!! 找不到 $IRIS_MS2_ROOT/sync_data"; exit 1; }
[[ -d "$IRIS_MANIFEST_DIR" ]]       || { echo "!! 找不到 $IRIS_MANIFEST_DIR"; exit 1; }

# 单节点单卡：accelerate 永远只看到一张卡，用 0.yaml
export CUDA=0
# 每张 GPU 分到 8 核（QoS gpulimits: cpu=120 / gpu=15），留 1 核给主进程
export DATALOADER_WORKERS="${DATALOADER_WORKERS:-7}"
# 同节点可能有别人的作业，端口写死会撞
export MAIN_PROCESS_PORT="$((20000 + ${SLURM_JOB_ID:-0} % 20000))"

echo "=== job ${SLURM_JOB_ID:-N/A} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo "repo=$IRIS_REPO"
echo "ms2 =$IRIS_MS2_ROOT"
echo "mfst=$IRIS_MANIFEST_DIR"
echo "runs=$IRIS_RUNS"
