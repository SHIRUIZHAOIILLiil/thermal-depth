#!/bin/bash
#SBATCH --job-name=migrate
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# 把这个账号的产物推到另一个账号的 scratch。同一个文件系统，不走网络。
#
# 顺序是按「丢了要花多少 GPU 时间重来 ÷ 大小」排的，所以作业中途被砍也不会
# 先丢最贵的那几样：
#
#   1. manifests        几百 MB —— 76,547 帧的 VLM caption 推理都在里面
#   2. metric_adapt     几 KB   —— 冻结常数，主表的尺度靠它
#   3. baseline_bench   小      —— 四个基线的复现值
#   4. converted/*.pt   ~23 GB  —— 进论文的那几个模型，每个背后 20 小时训练
#   5. pseudo_gt        ~47 GB  —— 76,547 个 .npy（2026-09-07 实测，不是表头
#                                  曾经写的 13 GB）
#   6. data/ms2         ~57 GB  —— 数据集本身，文件数最多、最慢
#   7. eval             ~1 GB   —— 约 29 GPU 小时的评估结果
#                                  ⚠️ 只有 SKIP_EVAL=0 时才走，见下
#
# rsync 可断点续传：作业被墙钟砍了原样重投即可接着跑。
#
# ── 环境变量 ────────────────────────────────────────────────────────
#   DEST       必填。对方 scratch 里的目标目录，对方需先
#              `setfacl -m u:<你>:rwx` 开到那一层
#   SKIP_EVAL  1（默认）跳过 eval/。评估作业还在往里写的时候必须跳过，
#              否则会复制到写了一半的结果。全部评估完成后再用 SKIP_EVAL=0
#              单独跑一次，rsync 只补新增的。
#   ARCHIVE    1 = 把文件数多的目录打成单个 tar 写到目标，而不是逐文件复制。
#              目标账号的 inode 配额可能远小于空间配额（实测对方剩 50,641 个
#              inode，而这一趟要搬 117 万个文件），这时逐文件复制根本装不下。
#              打包后 data/ms2 的 810,497 个文件只占 1 个 inode，字节数不变。
#              ⚠️ 归档手段，不是「搬过去就能接着跑」：要用得先解开，而解开
#              需要把 inode 还回来。真正的解法是提配额，这是等配额期间的做法。
#   FULL       1 = 连 runs/iris_ms2 下全部训练中间 checkpoint 一起搬（约 1.7 TB）。
#              默认 0：只搬第 4 段里被评估过的 converted/*.pt。中间点的用处是
#              「将来想换选点而不重训」，所以它排在最后一段。
#   DRY        1 = 只列要传什么、不动手
#
# ── 用法 ────────────────────────────────────────────────────────────
#   DEST=/mnt/scratch/dwdh0363/from_sc23sz DRY=1 \
#     sbatch --export=ALL,DEST,DRY ~/Iris/slurm/migrate_to.sbatch
#
#   DEST=/mnt/scratch/dwdh0363/from_sc23sz \
#     sbatch --export=ALL,DEST ~/Iris/slurm/migrate_to.sbatch

source "${IRIS_REPO:-$HOME/Iris}/slurm/env.sh"

: "${DEST:?必须指定 DEST（对方 scratch 里的目标目录）}"
SKIP_EVAL="${SKIP_EVAL:-1}"
FLAGS="-a --info=progress2 --no-inc-recursive"
# --itemize-changes 会为每个文件打一行，data/ms2 有几十万个：空跑的目的是
# 看总量，不是看清单。--stats 给的正是每段的文件数与字节数。
[[ "${DRY:-0}" == "1" ]] && FLAGS="$FLAGS --dry-run --stats"

# 写得进去吗。这一步失败的话后面每一条都会失败，早知道比晚知道好。
probe="$DEST/.write_probe_$$"
if ! touch "$probe" 2>/dev/null; then
  echo "!! 写不进 $DEST —— 让对方跑："
  echo "   setfacl -m u:$USER:rwx \$(dirname $DEST) $DEST"
  exit 1
fi
rm -f "$probe"
echo "=== 目标可写：$DEST ==="
df -h "$DEST" 2>/dev/null | tail -1

copy () {  # 标签 源 目标子目录
  local label="$1" src="$2" sub="$3"
  if [[ ! -e "$src" ]]; then
    echo "-- 跳过 $label：找不到 $src"
    return 0
  fi
  echo
  echo "################ $label"
  mkdir -p "$DEST/$sub"
  rsync $FLAGS "$src" "$DEST/$sub/" || echo "!! $label 未完成（重投可接着跑）"
}

copy "1/7 manifests（caption 全在这里）" "$IRIS_MANIFEST_DIR/" "manifests"
copy "2/7 metric_adapt（冻结常数）"       "$IRIS_RUNS/metric_adapt/" "runs/metric_adapt"
copy "3/7 baseline_bench（基线复现值）"   "$IRIS_RUNS/baseline_bench/" "runs/baseline_bench"

echo
echo "################ 4/7 converted 权重（进论文的那几个）"
for run in ma_r3ow_cap ma_r3ow_nocap r3ow_cap r3ow_nocap \
           iris_ms2_full8_thermalcap iris_ms2_full8_nocap \
           lotusd_full8_thermalcap lotusd_full8_nocap; do
  src="$IRIS_RUNS/iris_ms2/$run/converted"
  [[ -d "$src" ]] || { echo "-- 跳过 $run：没有 converted/"; continue; }
  mkdir -p "$DEST/runs/iris_ms2/$run"
  rsync $FLAGS "$src/" "$DEST/runs/iris_ms2/$run/converted/" \
    || echo "!! $run 未完成"
done

archive () {  # 标签 源目录 目标tar名 文件数说明
  local label="$1" src="$2" name="$3" n="$4"
  if [[ ! -d "$src" ]]; then echo "-- 跳过 $label：找不到 $src"; return 0; fi
  echo
  echo "################ $label"
  if [[ -s "$DEST/$name" ]]; then
    echo "[skip] $DEST/$name 已存在（$(du -h "$DEST/$name" 2>/dev/null | cut -f1)）"
    return 0
  fi
  echo "打包 $n 个文件 -> $DEST/$name（占 1 个 inode）"
  if [[ "${DRY:-0}" == "1" ]]; then echo "[dry] 不实际打包"; return 0; fi
  # 直接写到目标，源端不需要额外空间。⚠️ tar 没有断点续传，被砍就得重来，
  # 所以打包的两段排在小东西后面。
  if tar -cf "$DEST/$name" -C "$src" . ; then
    ls -lh "$DEST/$name"
  else
    echo "!! $label 打包失败，删掉半成品"
    rm -f "$DEST/$name"
  fi
}

if [[ "${ARCHIVE:-0}" == "1" ]]; then
  archive "5/8 pseudo_gt（打包）" "$IRIS_RUNS/pseudo_gt/official_train" \
          "pseudo_gt_official_train.tar" "76,551"
  archive "6/8 data/ms2（打包）" "$IRIS_MS2_ROOT" "data_ms2.tar" "810,497"
else
  copy "5/8 pseudo_gt（76,547 个 npy，约 47 GB）" \
       "$IRIS_RUNS/pseudo_gt/official_train/" "runs/pseudo_gt/official_train"
  copy "6/8 data/ms2（文件数最多，最慢）" "$IRIS_MS2_ROOT/" "data/ms2"
fi

if [[ "$SKIP_EVAL" == "1" ]]; then
  echo
  echo "-- 7/8 eval：按 SKIP_EVAL=1 跳过。评估作业跑完后用 SKIP_EVAL=0 单独再跑一次。"
else
  copy "7/8 eval（约 29 GPU 小时的结果）" "$IRIS_RUNS/eval/" "runs/eval"
fi

if [[ "${FULL:-0}" == "1" ]]; then
  # 训练中间点。约 1.7 TB，而复现论文里的数一个都用不上 —— 上面第 4 段的
  # converted/*.pt 才是被评估过的那些。它们的用处是「将来想换一个选点而
  # 不重训」，那是真实的用处，只是排在最后：作业被砍时先丢这一段，损失
  # 最小。
  echo
  if [[ "${ARCHIVE:-0}" == "1" ]]; then
    archive "8/8 iris_ms2 全部 checkpoint（打包，约 1.7 TB）" \
            "$IRIS_RUNS/iris_ms2" "iris_ms2_all.tar" "约 27 万"
  else
    echo "################ 8/8 iris_ms2 全部 checkpoint（约 1.7 TB，最后走）"
    rsync $FLAGS "$IRIS_RUNS/iris_ms2/" "$DEST/runs/iris_ms2/" \
      || echo "!! 未完成（重投可接着跑）"
  fi
else
  echo
  echo "-- 8/8 iris_ms2 中间 checkpoint：按 FULL=0 跳过。要全搬用 FULL=1。"
fi

echo
echo "################ 完成 ################"
du -sh "$DEST" 2>/dev/null
