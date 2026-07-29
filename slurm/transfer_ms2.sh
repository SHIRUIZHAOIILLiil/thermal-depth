#!/usr/bin/env bash
# 把 MS2 数据从本地传到 Aire 的 scratch。**在 WSL 里运行，不是在 HPC 上。**
#
#   bash slurm/transfer_ms2.sh check      # 只看本地/远端现状，不传
#   bash slurm/transfer_ms2.sh manifests  # 传 manifest（207 MB，先跑这个验证链路）
#   bash slurm/transfer_ms2.sh data       # 传全部 8 个序列（56.9 GB）
#   bash slurm/transfer_ms2.sh verify     # 逐序列核对文件数
#
# ── 只传用得到的模态 ───────────────────────────────────────────────
# 扫过全部 27 个 manifest，被引用的路径只有下面 KEEP 里那 5 个。
# nir/、lidar/、gps_imu/、img_right/、depth_multi/、intensity*/ 从未出现，
# 且代码里也没有任何地方读它们。丢掉这些：249.2 GB / 147 万文件 →
# 56.9 GB / 30.6 万文件，省掉 77% 体积和 79% inode。
# 需要完整数据时（右目、NIR、LiDAR）：KEEP_ALL=1 bash slurm/transfer_ms2.sh data
#
# ── 单一根目录 ─────────────────────────────────────────────────────
# 本地 ms2 和 ms2_partial 是下载时分开的，但 manifest 路径全是相对的，
# 且两边序列名不重叠，所以远端合并成一个根，所有 manifest 共用 --ms2-root。
#
# 用 tar 流式传输而非 rsync：几十万个小 PNG，rsync 每文件一次往返，
# 加上跳板机延迟会慢到不可接受。tar 走单条流，快一个数量级。

set -uo pipefail

REMOTE="${REMOTE:-aire}"
DEST="${DEST:-/mnt/scratch/sc23sz/data/ms2}"
MANIFEST_DEST="${MANIFEST_DEST:-/mnt/scratch/sc23sz/manifests}"
CAPTION_SRC="${CAPTION_SRC:-/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b}"
KEEP_ALL="${KEEP_ALL:-0}"

# 序列 -> 本地根目录（这就是"两部分数据"的映射表）
declare -A ROOT=(
  [_2021-08-06-10-59-33]=/mnt/e/dataset/ms2          # caption train / day2seq
  [_2021-08-06-11-23-45]=/mnt/e/dataset/ms2          # caption val / day_inference
  [_2021-08-06-11-37-46]=/mnt/e/dataset/ms2          # caption test / day2seq
  [_2021-08-06-16-19-00]=/mnt/e/dataset/ms2          # test_rainy
  [_2021-08-13-21-36-10]=/mnt/e/dataset/ms2          # ms2_manifest 全局
  [_2021-08-06-16-59-13]=/mnt/e/dataset/ms2_partial  # test rainy
  [_2021-08-13-16-08-46]=/mnt/e/dataset/ms2_partial  # test / rgb_depth clip75
  [_2021-08-13-21-58-13]=/mnt/e/dataset/ms2_partial  # test night
)
SEQS=(_2021-08-06-10-59-33 _2021-08-06-11-23-45 _2021-08-06-11-37-46
      _2021-08-06-16-59-13 _2021-08-13-16-08-46 _2021-08-13-21-58-13
      _2021-08-06-16-19-00 _2021-08-13-21-36-10)

# manifest 里出现过的全部路径模式，SEQ 是占位符
KEEP=(
  "sync_data/SEQ/thr/img_left"        # 热成像输入
  "sync_data/SEQ/rgb/img_left"        # RGB 输入
  "proj_depth/SEQ/thr/depth_filtered" # 热成像深度 GT
  "proj_depth/SEQ/thr/depth"
  "proj_depth/SEQ/rgb/depth_filtered" # RGB 深度（RGB-teacher 线）
)

# 列出某序列本地实际存在的待传路径
paths_for() {
  # 注意：必须分成两条 local。写成 `local seq="$1" root="${ROOT[$seq]}"` 时，
  # bash 在展开 ${ROOT[$seq]} 之前还没给 seq 赋值，set -u 下会直接报 unbound variable。
  local seq="$1"
  local root="${ROOT[$seq]}"
  local k p
  if [[ "$KEEP_ALL" == "1" ]]; then
    for p in "sync_data/$seq" "proj_depth/$seq"; do [[ -d "$root/$p" ]] && echo "$p"; done
    return
  fi
  for k in "${KEEP[@]}"; do p="${k//SEQ/$seq}"; [[ -d "$root/$p" ]] && echo "$p"; done
}

local_count() {
  local seq="$1"
  local root="${ROOT[$seq]}"
  local total=0 p n
  while read -r p; do
    [[ -z "$p" ]] && continue
    n=$(find "$root/$p" -type f 2>/dev/null | wc -l); total=$((total + n))
  done < <(paths_for "$seq")
  echo "$total"
}

remote_count() {
  ssh "$REMOTE" "find '$DEST/sync_data/$1' '$DEST/proj_depth/$1' -type f 2>/dev/null | wc -l" 2>/dev/null || echo 0
}

send_seq() {
  local seq="$1"
  local root="${ROOT[$seq]}"
  local -a paths=(); mapfile -t paths < <(paths_for "$seq")
  if [[ ${#paths[@]} -eq 0 ]]; then echo "!! 本地无数据，跳过 $seq"; return; fi

  local want have
  want=$(local_count "$seq"); have=$(remote_count "$seq")
  if [[ "$have" == "$want" && "$want" != "0" ]]; then
    echo "== $seq 已完整（$have 个文件），跳过"; return
  fi
  [[ "$have" != "0" ]] && echo "== $seq 不完整（$have/$want），重传"

  echo ">> 传输 $seq  （$want 个文件，${#paths[@]} 个子目录）"
  ssh "$REMOTE" "rm -rf '$DEST/sync_data/$seq' '$DEST/proj_depth/$seq'; mkdir -p '$DEST'"
  tar -cf - -C "$root" "${paths[@]}" | ssh "$REMOTE" "tar -xf - -C '$DEST'"
  echo "   完成 $seq  （远端 $(remote_count "$seq") 个文件）"
}

case "${1:-check}" in
  manifests)
    ssh "$REMOTE" "mkdir -p '$DEST' '$MANIFEST_DEST/sequence_level_internvl3_8b'"
    tar -cf - -C /mnt/e/dataset/ms2 $(cd /mnt/e/dataset/ms2 && ls *.jsonl) \
      | ssh "$REMOTE" "tar -xf - -C '$DEST'"
    tar -cf - -C "$CAPTION_SRC" . \
      | ssh "$REMOTE" "tar -xf - -C '$MANIFEST_DEST/sequence_level_internvl3_8b'"
    echo "manifest 传输完成（caption 已内联在 jsonl 里，captioning 项目无需传输）"
    ;;
  data|all|1|2|3)
    [[ "$KEEP_ALL" == "1" ]] && echo "!! KEEP_ALL=1：传输完整模态（249 GB / 147 万文件）"
    for s in "${SEQS[@]}"; do send_seq "$s"; done
    ;;
  check)
    printf '%-24s %-12s %12s %12s\n' 序列 来源 本地文件数 远端文件数
    for s in "${SEQS[@]}"; do
      printf '%-24s %-12s %12s %12s\n' "$s" "$(basename "${ROOT[$s]}")" "$(local_count "$s")" "$(remote_count "$s")"
    done
    ssh "$REMOTE" 'lfs quota -h -u $USER /scratch'
    ;;
  verify)
    fail=0
    for s in "${SEQS[@]}"; do
      l=$(local_count "$s"); r=$(remote_count "$s")
      if [[ "$l" == "$r" && "$l" != "0" ]]; then printf 'OK    %-24s %8s\n' "$s" "$l"
      else printf 'DIFF  %-24s 本地=%s 远端=%s\n' "$s" "$l" "$r"; fail=1; fi
    done
    [[ $fail -eq 0 ]] && echo "全部一致 ✅"
    exit $fail
    ;;
  *) echo "用法: $0 {check|manifests|data|verify}"; exit 1 ;;
esac
