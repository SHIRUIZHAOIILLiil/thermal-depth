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

# NO_MUX=1 绕过 ControlMaster。多路复用省 Duo 认证次数，但 master 一旦变成僵尸
# （socket 还在、通道打不通），ssh 会永远挂着且不报错、也不弹认证。
# 遇到这种情况用 NO_MUX=1，代价是 rash 和 aire 各要认证一次。
if [[ "${NO_MUX:-0}" == "1" ]]; then
  SSH_OPTS="-o ControlMaster=no -o ControlPath=none"
else
  SSH_OPTS=""
fi
SSH="ssh $SSH_OPTS"

IRIS_LOCAL="${IRIS_LOCAL:-/mnt/e/project/Iris}"
ANYTHERMAL_LOCAL="${ANYTHERMAL_LOCAL:-/mnt/e/project/AnyThermal}"

CALIB_SRC="$IRIS_LOCAL/outputs/route_suite/b_thermal_unet_20ep/epoch05_weights.pt"
STAGE1_SRC="$IRIS_LOCAL/outputs/route_suite/c1_vae_adapter_20ep/best_weights.pt"
MIDAS_SRC="$ANYTHERMAL_LOCAL/_download/pretrained_checkpoints/depth"

# 先把连接建起来。不预热的话，认证提示会夹在传输输出里冒出来，
# 看上去像是 rsync 卡死（其实是 ssh 在等 Duo）。
warm_connection() {
  echo "建立 SSH 连接（需要密码 + Duo，之后 8 小时复用）……"
  if ! $SSH -o ConnectTimeout=30 "$REMOTE" 'echo "已连上 $(hostname)"'; then
    echo "!! 连不上 $REMOTE，检查 ~/.ssh/config、VPN 或跳板机"
    exit 1
  fi
}

send_file() {
  local src="$1" dstdir="$2"
  if [[ ! -f "$src" ]]; then echo "!! 本地不存在，跳过: $src"; return 1; fi
  local size; size=$(stat -c%s "$src")
  echo ">> 传输 $(basename "$src")  ($(numfmt --to=iec "$size"))"
  $SSH "$REMOTE" "mkdir -p '$dstdir'"
  # rsync 单个大文件优于 tar：支持断点续传
  rsync -aP -e "$SSH" "$src" "$REMOTE:$dstdir/"
}

send_dir() {
  local src="$1" dstdir="$2"
  if [[ ! -d "$src" ]]; then echo "!! 本地不存在，跳过: $src"; return 1; fi
  echo ">> 传输目录 $src -> $dstdir"
  $SSH "$REMOTE" "mkdir -p '$dstdir'"
  rsync -aP -e "$SSH" "$src/" "$REMOTE:$dstdir/"
}

warm_connection

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
    $SSH "$REMOTE" "ls -laR '$CKPT_DEST' '$MODEL_DEST' 2>/dev/null || echo '(尚未创建)'"
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
