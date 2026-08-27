#!/bin/bash
# 在**登录节点**用：  source ~/Iris/slurm/env.sh
#
# common.sh 是给作业用的：它 set -euo pipefail、module load、conda activate，
# 这三样都不该出现在交互 shell 里（-e 会让任何一条失败的命令把你的登录会话踢掉）。
# 但提交命令要用到的路径全都定义在它里面，于是从登录节点提交时，$IRIS_MANIFEST_DIR
# 这类变量会静默展开成空串，路径少掉前缀，作业排到队、拿到卡、两秒后死在前置检查上。
#
# 2026-08-26 一天之内这个根因绊了四次：--ms2-root 传空、torch 缓存写进 $HOME 而作业
# 去 $SCRATCH 找、TRAIN_MANIFEST 少了前缀、以及为了给前一个打补丁又写坏一次。
# 所以把变量单独抽出来，只 export，不带任何副作用。
#
# common.sh 也 source 这个文件，两边因此不可能各说各话。

: "${SCRATCH:?SCRATCH 未定义 —— 确认已在 Aire 上运行}"

export IRIS_REPO="${IRIS_REPO:-$HOME/Iris}"
export IRIS_DATA="${IRIS_DATA:-$SCRATCH/data}"
export IRIS_RUNS="${IRIS_RUNS:-$SCRATCH/runs}"
export IRIS_MS2_ROOT="${IRIS_MS2_ROOT:-$IRIS_DATA/ms2}"
export IRIS_MANIFEST_DIR="${IRIS_MANIFEST_DIR:-$SCRATCH/manifests/sequence_level_internvl3_8b}"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH/torch_cache}"

# 作业的 stdout 是重定向到文件的，所以 Python 默认走块缓冲：logger.info 的输出会
# 攒在缓冲区里，直到进程退出才出现。于是一个正常训练中的作业，从 .out 看起来像是
# 卡在了数据集初始化那一行 —— 2026-08-27 为此查了一轮。
# print(flush=True) 和 tqdm（走 stderr）不受影响，正是这两者能看见而别的看不见的原因。
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# 只在交互式 shell 里打印。作业 source 它的时候这些行只会污染日志。
if [[ $- == *i* ]]; then
  printf 'IRIS env:\n'
  for _v in IRIS_REPO IRIS_DATA IRIS_RUNS IRIS_MS2_ROOT IRIS_MANIFEST_DIR HF_HOME TORCH_HOME; do
    _p="${!_v}"
    if [[ -e "$_p" ]]; then _mark="ok  "; else _mark="MISSING"; fi
    printf '  %-18s %-8s %s\n' "$_v" "$_mark" "$_p"
  done
  unset _v _p _mark
fi
