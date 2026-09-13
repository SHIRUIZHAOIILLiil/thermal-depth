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
#   4. converted/*.pt   —— converted/ 非空的臂。⚠️ 09-10 清理之后，进论文的那
#                          几条臂在这一段里已经是空的（权重只剩 checkpoint-*，
#                          见第 8 段），这段现在主要是别处没有副本的那些步数
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

# 写得进去 ≠ 对方读得到。2026-09-13 实测：from_sc23sz 的 default ACL 里只有
# user:: 和一个第三方，没有属主本人的命名条目。于是这里新建的子树属主是**我**、
# 组权限 ---，对方作为上级目录的属主照样进不去；而 setfacl 只有文件属主能跑,
# 等 117 万个文件搬完再补就是递归一百多万次。
# 整件事的目的是让对方能用，所以这一条不通过就不开搬。
owner=$(stat -c %U "$DEST" 2>/dev/null)
# 拉的人就是目标属主本人时这一整条检查是空的：新建的东西本来就归他，
# 而 getfacl 会把他记成 user::（属主位），不是 user:<名字>: 的命名条目。
# 从他账号拉回去正是推荐的做法（文件从一开始就归会留下来的那个账号），
# 所以这里放行，不然这道闸只会挡住最该走的那条路。
if [[ "$USER" == "$owner" ]]; then
  echo "=== 目标属主本人在拉，新文件直接归 $owner，ACL 免检 ==="
else
aclprobe="$DEST/.aclprobe_$$"
if mkdir -p "$aclprobe" 2>/dev/null; then
  if getfacl -p "$aclprobe" 2>/dev/null | grep -q "^user:$owner:"; then
    echo "=== 对方（$owner）读得到新建内容：是 ==="
  else
    echo "!! 新建的目录里没有 $owner 的 ACL 条目 —— 搬过去他读不到。"
    echo "!! 让他在自己账号上跑一次（只需一次，往后新建的自动继承）："
    echo "     setfacl -m d:u:$owner:rwx -m u:$owner:rwx $DEST"
    echo "!! 确认不需要这一步就用 SKIP_ACL_CHECK=1 再投。"
    rmdir "$aclprobe"
    [[ "${SKIP_ACL_CHECK:-0}" == "1" ]] || exit 1
  fi
  rmdir "$aclprobe" 2>/dev/null
fi
fi
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

# 目标要的是 $SCRATCH 的镜像：那边已有 manifests/sequence_level_internvl3_8b/，
# 而且所有 sbatch 都按 $IRIS_MANIFEST_DIR 这个相对布局找清单。铺到 manifests/
# 根下会在上一层多出一份重复的，并且路径对不上。
copy "1/8 manifests（caption 全在这里）" "$IRIS_MANIFEST_DIR/" "manifests/sequence_level_internvl3_8b"
copy "2/8 metric_adapt（冻结常数）"       "$IRIS_RUNS/metric_adapt/" "runs/metric_adapt"
copy "3/8 baseline_bench（基线复现值）"   "$IRIS_RUNS/baseline_bench/" "runs/baseline_bench"

echo
echo "################ 4/8 converted 权重"
# 2026-09-13 改：这里原先写死八条「进论文的臂」。09-10 清过一轮 step*_weights.pt
# （规则是「有对应 checkpoint-<step> 才删」）之后，那八条的 converted/ 全空了 ——
# 它们的权重只剩 checkpoint-<step>/，属于第 8 段。同一次清理反而让另一批成了
# 唯一副本：iris_ms2_ckpt(7)、iris_ms2_ckpt_nocap(5)、iris_ms2_thermalcap(5)、
# r2/r3/r4 里超出 checkpoint 数的那几步。
# 写死名单在这种局面下两头落空：该搬的一个都搬不到，唯一副本一个都没覆盖。
for run in $(ls -1 "$IRIS_RUNS/iris_ms2" 2>/dev/null); do
  src="$IRIS_RUNS/iris_ms2/$run/converted"
  [[ -d "$src" ]] || continue
  [[ -n "$(ls -A "$src" 2>/dev/null)" ]] || { echo "-- 跳过 $run：converted/ 空（权重在 checkpoint-*，第 8 段）"; continue; }
  mkdir -p "$DEST/runs/iris_ms2/$run"
  rsync $FLAGS "$src/" "$DEST/runs/iris_ms2/$run/converted/" \
    || echo "!! $run 未完成"
done

# tar 的结尾是两个 512 字节全零块。被墙钟砍在半路的 tar 没有这个尾巴 —— 但它
# 非空，所以「已存在就跳过」会把半成品当成完成品放过去，而这个脚本存在的全部
# 意义就是「被砍了原样重投接着跑」。O(1)：只读最后 1024 字节。
tar_complete () {
  local f="$1"
  [[ -s "$f" ]] || return 1
  [[ $(tail -c 1024 "$f" 2>/dev/null | tr -d '\000' | wc -c) -eq 0 ]]
}

archive () {  # 标签 源目录 目标tar名 文件数说明
  local label="$1" src="$2" name="$3" n="$4"
  local final="$DEST/$name" part="$DEST/$name.part"
  if [[ ! -d "$src" ]]; then echo "-- 跳过 $label：找不到 $src"; return 0; fi
  echo
  echo "################ $label"
  if tar_complete "$final"; then
    echo "[skip] $final 已完整（$(du -h "$final" 2>/dev/null | cut -f1)）"
    return 0
  fi
  if [[ -e "$final" ]]; then
    # 09-07 之前的版本直接写最终名，所以旧的半成品长着完成品的名字。
    echo "!! $final 存在但结尾不完整（被砍在半路），删掉重打"
    rm -f "$final"
  fi
  [[ -e "$part" ]] && { echo "-- 上一趟的 $part.part 作废，重来"; rm -f "$part"; }
  echo "打包 $n 个文件 -> $final（占 1 个 inode）"
  if [[ "${DRY:-0}" == "1" ]]; then echo "[dry] 不实际打包"; return 0; fi
  # 先写 .part 再改名：作业在打包途中被砍，留下的是 .part，下一趟一眼看得出
  # 它不是成品。改名是同一文件系统内的元数据操作，不搬字节。
  # ⚠️ tar 仍然没有断点续传 —— 被砍就是从零重来，所以打包的段排在小东西后面。
  if tar -cf "$part" -C "$src" . && tar_complete "$part"; then
    mv -f "$part" "$final"
    ls -lh "$final"
  else
    echo "!! $label 打包失败，删掉半成品"
    rm -f "$part"
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
    # 一条臂一个 tar，不是整个目录一个。tar 不能续传：作为单个 1.7 TB 的文件，
    # 作业被砍在八成处留下的是不可用的半成品，而且要从零重来。按臂拆开之后，
    # 同样被砍时每条跑完的臂都是完整的，重投时上面那个「已存在就跳过」的检查
    # 会从断掉的那条接着走。
    echo
    echo "################ 8/8 iris_ms2 全部 checkpoint（按臂打包，约 1.7 TB）"
    for run in $(ls -1 "$IRIS_RUNS/iris_ms2" 2>/dev/null); do
      [[ -d "$IRIS_RUNS/iris_ms2/$run" ]] || continue
      archive "  $run" "$IRIS_RUNS/iris_ms2/$run" "iris_ms2__$run.tar" "整条臂"
    done
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
