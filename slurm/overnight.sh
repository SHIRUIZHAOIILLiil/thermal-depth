#!/usr/bin/env bash
# 过夜无人值守：传数据 + 在 HPC 上建好 conda 环境。
#
#   cd /mnt/e/project/Iris && bash slurm/overnight.sh
#
# 启动时会要你输一次密码（ControlMaster 之后 8 小时复用，不再打扰）。
# 输完密码看到 "开始无人值守阶段" 就可以去睡了。
#
# 全程写日志到 slurm/overnight_<时间戳>.log，早上直接看结尾的汇总。

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

TS=$(date +%Y%m%d_%H%M)
LOG="slurm/overnight_${TS}.log"
REMOTE="${REMOTE:-aire}"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
declare -a RESULTS=()

stage() {
  local name="$1"; shift
  local t0=$SECONDS
  log "──────── 开始：$name"
  if "$@" >>"$LOG" 2>&1; then
    local d=$((SECONDS - t0))
    log "✅ 完成：$name（用时 $((d/60)) 分 $((d%60)) 秒）"
    RESULTS+=("OK    $name")
  else
    local rc=$? d=$((SECONDS - t0))
    log "❌ 失败：$name（退出码 $rc，用时 $((d/60)) 分）"
    RESULTS+=("FAIL  $name (rc=$rc)")
  fi
}

echo "日志：$LOG"
echo

# ── 阶段 0：交互式建立连接（这一步需要你输密码）────────────────────────
log "建立 SSH 连接，请输入密码……"
if ! ssh -o ConnectTimeout=30 "$REMOTE" 'echo "连接成功：$(hostname)"' | tee -a "$LOG"; then
  log "❌ 连不上 $REMOTE，检查 ~/.ssh/config、VPN 或跳板机。中止。"
  exit 1
fi
log "ControlMaster 已建立，后续 8 小时不再需要密码。"
echo
log "════════ 开始无人值守阶段，可以去睡了 ════════"
echo

# ── 阶段 1：目录 + manifest ──────────────────────────────────────────
stage "创建远端目录" ssh "$REMOTE" \
  'mkdir -p $SCRATCH/{data/ms2,manifests,runs,logs,hf_cache,torch_cache,pip_cache,conda_pkgs} ~/Iris/slurm'
stage "传输 manifest（207 MB）" bash slurm/transfer_ms2.sh manifests

# ── 阶段 2：把建环境需要的文件送过去，先把作业投出去排队 ─────────────
stage "上传 environment.yaml / build_env.sbatch" bash -c \
  'tar -cf - -C slurm environment.yaml build_env.sbatch | ssh '"$REMOTE"' "tar -xf - -C ~/Iris/slurm"'
stage "提交建环境作业" ssh "$REMOTE" \
  'cd $SCRATCH/logs && sbatch ~/Iris/slurm/build_env.sbatch'

# ── 阶段 3：数据（全部 8 个序列，只含用得到的模态）──────────────────
stage "传输数据 8 个序列（56.9 GB / 30.6 万文件）" bash slurm/transfer_ms2.sh data

# ── 阶段 4：核对 ────────────────────────────────────────────────────
stage "逐序列核对文件数" bash slurm/transfer_ms2.sh verify

# ── 汇总 ────────────────────────────────────────────────────────────
{
  echo
  echo "════════════════ 汇总（$(date)）════════════════"
  printf '%s\n' "${RESULTS[@]}"
  echo
  echo "──── scratch 配额 ────"
  ssh "$REMOTE" 'lfs quota -h -u $USER /scratch; echo; echo "--- home ---"; quota -s' 2>&1
  echo
  echo "──── 建环境作业 ────"
  ssh "$REMOTE" 'squeue -u $USER; echo; ls -t $SCRATCH/logs/iris_build_env-*.out 2>/dev/null | head -1 | xargs -r tail -25' 2>&1
  echo "═══════════════════════════════════════════════"
} | tee -a "$LOG"

echo
echo "完整日志：$LOG"
