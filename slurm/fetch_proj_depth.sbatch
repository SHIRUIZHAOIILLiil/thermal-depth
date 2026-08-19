#!/bin/bash
#SBATCH --job-name=iris_projdepth
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 直接在集群上取 MS2 的 proj_depth（深度 GT），不经过本地。
#
# 为什么不在本地下再传：本地下完还得把 155 GB 上传到 Aire，而家用宽带的上行
# 比下行还慢几倍 —— 那一段是整条路上最贵的。集群直连省掉它。
# 出站 HTTPS 是通的（prefetch_models.sbatch 拉过 16 GB 的 HF 权重）。
#
# 不申请 GPU：下载和 bunzip2 都用不上，而 GPU 是全集群最稀缺的资源。
#
# ── 配额是硬约束 ────────────────────────────────────────────────────
# 整包 155 GB（15 条），但真正要的只有三个子目录，逻辑体积仅占约 29%。
# 所以**随下随删**：下一条 → 只解这三个 → 立刻删包。
#
# ⚠️ **别用逻辑体积做预算 —— Lustre 按分配块计配额，小文件的实际占用是逻辑
# 体积的数倍。** 实测 _2021-08-13-22-16-02（5,234 帧 × 3 个子目录 = 15,702 个
# 文件）：du 只报 +2 GB，配额却掉了 **8 GB**，平均每个文件吃掉约 533 KB 配额，
# 而逻辑均值只有 66 KB。
#
# 按实测重算：**1.53 GB 配额 / 千帧**。15 条约 12.7 万帧 → **约 194 GB**，
# 而 scratch 余量只有 211 GB。一次下全会把配额榨干，而 iris_ms2 每个 checkpoint
# 的 optimizer.bin 就有 1.7 GB —— 正在跑的训练会因写不出 checkpoint 而失败。
#
# 结论：**不要批量下载，按需一次取一到两条。** 去掉 rgb/depth_filtered
# （INCLUDE_RGB_DEPTH=0）只省约三分之一，省不出量级差别；真正的省法是只取
# 确实要用的序列。
#
# 另注：耗时的大头是 bunzip2 而非下载。实测 6.4 GB 的包：下载 5m17s（20.6 MB/s），
# 解压 11m35s。按包体积线性外推，15 条需约 7.2 小时，超过下面的默认 --time。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   SEQS   必填。空格分隔的序列名，如 "_2021-08-13-22-16-02 _2021-08-13-22-36-41"
#   INCLUDE_RGB_DEPTH  默认 1（与 transfer_ms2.sh 一致，RGB-teacher 线要）。
#                      设 0 只取热像那两个，体积减半。
#   MIN_FREE_GB        低于此剩余空间就停手，默认 40
#
# ── 用法 ────────────────────────────────────────────────────────────
#   SEQS="_2021-08-13-22-16-02 _2021-08-13-22-36-41" \
#     sbatch --export=ALL,SEQS $HOME/Iris/slurm/fetch_proj_depth.sbatch
#
# 中断后重投即可：已解压好的序列会跳过，下到一半的包用 curl -C - 续传。

set -euo pipefail

: "${SCRATCH:?SCRATCH 未定义 —— 确认已在 Aire 上运行}"
: "${SEQS:?必须指定 SEQS（空格分隔的序列名）}"
INCLUDE_RGB_DEPTH="${INCLUDE_RGB_DEPTH:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-40}"

DEST="${DEST:-$SCRATCH/data/ms2/proj_depth}"
TMP="${TMP:-$SCRATCH/tmp/projdepth}"
mkdir -p "$DEST" "$TMP"

# Dropbox 直链，抄自数据集自带的 download_MS2_proj_depth_sequences.sh
declare -A URL=(
  [_2021-08-06-10-59-33]=a2nprnbsbra1xfu [_2021-08-06-11-23-45]=8l0ncccgewjvwhw
  [_2021-08-06-11-37-46]=clikuwhzc5jx5ii [_2021-08-06-16-19-00]=fsnp9qxce6bo1iy
  [_2021-08-06-16-45-28]=dpnnxhiit4gwjup [_2021-08-06-16-59-13]=0saug4cg7l6rp3g
  [_2021-08-06-17-21-04]=39j5fjpwqkky6yo [_2021-08-06-17-44-55]=yxzn75lr6eudoqh
  [_2021-08-13-15-46-56]=z29o18vvm2oa8gd [_2021-08-13-16-08-46]=5h2t3acg40iszel
  [_2021-08-13-16-14-48]=31jz6atqp34owh5 [_2021-08-13-16-31-10]=ltdbjpy7d5yan3n
  [_2021-08-13-16-50-57]=p3puadu9033wtg7 [_2021-08-13-17-06-04]=86e8bajcpd1twky
  [_2021-08-13-21-18-04]=64yripmlcmpu0uq [_2021-08-13-21-36-10]=zd9hui1qt74vyoe
  [_2021-08-13-21-58-13]=ttpki1aoasjq0y9 [_2021-08-13-22-03-03]=bmyo97xb08xm7pi
  [_2021-08-13-22-16-02]=j7r70325a3ypbu8 [_2021-08-13-22-36-41]=atbvl0riin33xky
)

PATTERNS=( "*/thr/depth_filtered/*" "*/thr/depth/*" )
[[ "$INCLUDE_RGB_DEPTH" == "1" ]] && PATTERNS+=( "*/rgb/depth_filtered/*" )

# ⚠️ 不能用 df：Lustre 上它报的是整个文件系统的可用量（2.8 PB），不是你的配额。
# 用它做守卫等于没有守卫 —— 永远不会触发，而 scratch 一旦被填满，正在跑的训练
# 作业写 checkpoint 也会失败，那才是真正的损失。per-user 配额只能问 lfs quota
# （prefetch_models.sbatch 里也是这么查的）。
# 输出列：Filesystem kbytes quota limit grace files ...  超软配额时数字后会带 '*'。
free_gb() {
  lfs quota -u "$USER" /scratch 2>/dev/null | awk '
    $1 ~ /scratch/ {
      used = $2; limit = $4
      gsub(/[^0-9.]/, "", used); gsub(/[^0-9.]/, "", limit)
      if (limit + 0 > 0) { printf "%d", (limit - used) / 1024 / 1024; found = 1 }
    }
    END { if (!found) print -1 }'
}

echo "=== 目标 $DEST"
echo "=== 取这些子目录: ${PATTERNS[*]}"
echo "=== 起始剩余 $(free_gb) GB"

for seq in $SEQS; do
  echo
  echo "───── $seq ─────"

  if [[ -z "${URL[$seq]:-}" ]]; then
    echo "!! 未知序列名 $seq —— 跳过（拼写？）"; continue
  fi
  if [[ -d "$DEST/$seq/thr/depth_filtered" ]]; then
    n=$(find "$DEST/$seq/thr/depth_filtered" -maxdepth 1 -type f | wc -l)
    echo "已存在，$n 个文件 —— 跳过"; continue
  fi

  avail=$(free_gb)
  if (( avail < 0 )); then
    echo "!! 读不到 lfs quota —— 拒绝在配额未知的情况下下载几十 GB"
    echo "   手动确认后用 MIN_FREE_GB=0 覆盖，或修 free_gb()"
    exit 1
  fi
  if (( avail < MIN_FREE_GB )); then
    echo "!! 配额剩余 ${avail}GB 低于 MIN_FREE_GB=${MIN_FREE_GB}，停手"
    echo "   填满 scratch 会连带让正在跑的训练作业写不出 checkpoint。"
    exit 1
  fi

  tarball="$TMP/$seq.tar.bz2"
  echo "[下载] $(date +%H:%M:%S)  剩余 ${avail}GB"
  # -C - 续传，-L 跟随重定向，--retry 应对偶发断流
  curl -fL -C - --retry 5 --retry-delay 10 \
       -o "$tarball" \
       "https://dl.dropboxusercontent.com/s/${URL[$seq]}/${seq}.tar.bz2"

  echo "[解压] $(date +%H:%M:%S)  包大小 $(du -h "$tarball" | cut -f1)"
  # 只解需要的三个子目录。--wildcards 必须在模式之前。
  tar -xjf "$tarball" -C "$DEST" --wildcards "${PATTERNS[@]}"

  # 删包之前先确认真的解出来了 —— 删掉一个没解成功的包就得重下十几 GB
  if [[ ! -d "$DEST/$seq/thr/depth_filtered" ]]; then
    echo "!! $seq 解压后找不到 thr/depth_filtered，保留压缩包供排查"; exit 1
  fi
  n_thr=$(find "$DEST/$seq/thr/depth_filtered" -maxdepth 1 -type f | wc -l)
  echo "[完成] thr/depth_filtered = $n_thr 个文件"

  rm -f "$tarball"
  echo "[清理] 已删压缩包，剩余 $(free_gb) GB"
done

echo
echo "=== 全部结束 $(date) ==="
echo "=== 最终剩余 $(free_gb) GB ==="
du -sh "$DEST" 2>/dev/null || true
