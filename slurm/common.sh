#!/bin/bash
# Aire 作业的公共环境设置。由各 .sbatch 脚本 source。
# 只放 export 和 module load —— 这些必须留在作业脚本里，不能进 ~/.bashrc（ARC 官方要求）。

set -euo pipefail

module load miniforge/24.7.1 cuda
conda activate iris

# $SCRATCH 由系统提供（/mnt/scratch/<user>）。所有输出、缓存、数据都走这里，
# home 只有 65GB / 150万 inode，写满会拖垮整个登录节点。
: "${SCRATCH:?SCRATCH 未定义 —— 确认已在 Aire 上运行}"

# 路径变量集中在 env.sh，登录节点也 source 得了那一份（这里不行：上面的
# set -euo pipefail 和 module load 不该进交互 shell）。两边共用一处定义，
# 就不会再出现「提交时展开成空串、作业里却是对的」这种不一致。
source "${IRIS_REPO:-$HOME/Iris}/slurm/env.sh"

# 其中 IRIS_MS2_ROOT / IRIS_MANIFEST_DIR 是 train_route_suite.py 用来覆盖它那两个
# 本地硬编码默认值的（/mnt/e/dataset/ms2 和 /mnt/e/project/thermal-depth/...）；
# 不设时保持本地行为。

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
# CPU 分区的作业（预取权重、建环境）没有 GPU，set -e 下不能让它中止
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv 2>/dev/null || echo "(本节点无 GPU)"
echo "repo=$IRIS_REPO"
echo "ms2 =$IRIS_MS2_ROOT"
echo "mfst=$IRIS_MANIFEST_DIR"
echo "runs=$IRIS_RUNS"
