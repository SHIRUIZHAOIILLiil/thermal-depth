#!/usr/bin/env bash
# 把本地已训好的 checkpoint 和外部权重传到集群。**在 WSL 里运行。**
#
#   bash slurm/transfer_checkpoints.sh check   # 只看本地有没有、多大
#   bash slurm/transfer_checkpoints.sh calib   # 机器差标定用（3.47 GB）
#   bash slurm/transfer_checkpoints.sh stage1  # 任务 1b 阶段二的 --init-from（28 MB）
#   bash slurm/transfer_checkpoints.sh midas   # 任务 4 的 AnyThermal MiDaS 权重
#   bash slurm/transfer_checkpoints.sh all
#
# 这些不走 git —— checkpoint 是产物，.gitignore 已挡掉 *.pt。

set -uo pipefail

REMOTE="${REMOTE:-aire}"
CKPT_DEST="${CKPT_DEST:-/mnt/scratch/sc23sz/checkpoints}"
MODEL_DEST="${MODEL_DEST:-/mnt/scratch/sc23sz/models}"

IRIS_LOCAL="${IRIS_LOCAL:-/mnt/e/project/Iris}"
ANYTHERMAL_LOCAL="${ANYTHERMAL_LOCAL:-/mnt/e/project/AnyThermal}"

CALIB_SRC="$IRIS_LOCAL/outputs/route_suite/b_thermal_unet_20ep/epoch05_weights.pt"
STAGE1_SRC="$IRIS_LOCAL/outputs/route_suite/c1_vae_adapter_20ep/best_weights.pt"
MIDAS_SRC="$ANYTHERMAL_LOCAL/_download/pretrained_checkpoints/depth"

send_file() {
  local src="$1" dstdir="$2"
  if [[ ! -f "$src" ]]; then echo "!! 本地不存在，跳过: $src"; return 1; fi
  local size; size=$(stat -c%s "$src")
  echo ">> 传输 $(basename "$src")  ($(numfmt --to=iec "$size"))"
  ssh "$REMOTE" "mkdir -p '$dstdir'"
  # rsync 单个大文件优于 tar：支持断点续传
  rsync -aP "$src" "$REMOTE:$dstdir/"
}

send_dir() {
  local src="$1" dstdir="$2"
  if [[ ! -d "$src" ]]; then echo "!! 本地不存在，跳过: $src"; return 1; fi
  echo ">> 传输目录 $src -> $dstdir"
  ssh "$REMOTE" "mkdir -p '$dstdir'"
  rsync -aP "$src/" "$REMOTE:$dstdir/"
}

case "${1:-check}" in
  check)
    printf '%-12s %-10s %s\n' 用途 大小 路径
    for pair in "标定:$CALIB_SRC" "阶段一:$STAGE1_SRC"; do
      n="${pair%%:*}"; p="${pair#*:}"
      if [[ -f "$p" ]]; then printf '%-12s %-10s %s\n' "$n" "$(du -h "$p" | cut -f1)" "$p"
      else printf '%-12s %-10s %s\n' "$n" "缺失" "$p"; fi
    done
    if [[ -d "$MIDAS_SRC" ]]; then
      printf '%-12s %-10s %s\n' "MiDaS" "$(du -sh "$MIDAS_SRC" | cut -f1)" "$MIDAS_SRC"
      ls "$MIDAS_SRC"
    else
      printf '%-12s %-10s %s\n' "MiDaS" "缺失" "$MIDAS_SRC"
    fi
    echo; echo "--- 远端已有 ---"
    ssh "$REMOTE" "ls -laR '$CKPT_DEST' '$MODEL_DEST' 2>/dev/null || echo '(尚未创建)'"
    ;;
  calib)  send_file "$CALIB_SRC"  "$CKPT_DEST/calibration" ;;
  stage1) send_file "$STAGE1_SRC" "$CKPT_DEST/c1_stage1" ;;
  midas)  send_dir  "$MIDAS_SRC"  "$MODEL_DEST/anythermal_midas" ;;
  all)
    send_file "$STAGE1_SRC" "$CKPT_DEST/c1_stage1"
    send_file "$CALIB_SRC"  "$CKPT_DEST/calibration"
    send_dir  "$MIDAS_SRC"  "$MODEL_DEST/anythermal_midas"
    ;;
  *) echo "用法: $0 {check|calib|stage1|midas|all}"; exit 1 ;;
esac
