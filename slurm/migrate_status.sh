#!/bin/bash
# 在**登录节点**跑：  DEST=/mnt/scratch/dwdh0363/from_sc23sz bash ~/Iris/slurm/migrate_status.sh
#
# 回答一个问题：迁移到哪儿了，下一步该投什么。
#
# ⚠️ 不用 du 走 data/ms2 那种几十万文件的目录 —— 登录节点上会挂住好几分钟
# （HANDOFF_20260823 §里记过）。tar 是单个文件，ls 就够；rsync 段只看目录在不在
# 和顶层条目数，要精确比对交给 rsync 自己（它本来就会跳过一致的）。

: "${DEST:?必须指定 DEST，例如 DEST=/mnt/scratch/dwdh0363/from_sc23sz}"
source "${IRIS_REPO:-$HOME/Iris}/slurm/env.sh"

# tar 的结尾是两个 512 字节全零块。被墙钟砍在半路的 tar 没有这个尾巴，但它
# 非空，所以 migrate_to.sbatch 里「已存在就跳过」会把它当成完成品放过去。
# 这个检查是 O(1) 的：只读最后 1024 字节。
tar_ok () {
  local f="$1"
  [[ -s "$f" ]] || return 1
  [[ $(tail -c 1024 "$f" 2>/dev/null | tr -d '\000' | wc -c) -eq 0 ]]
}

row () { printf '  %-38s %s\n' "$1" "$2"; }

echo "=== 目标 $DEST ==="
[[ -d "$DEST" ]] || { echo "!! 目标目录不存在或看不见（对方跑过 setfacl 了吗）"; exit 1; }
touch "$DEST/.probe_$$" 2>/dev/null && { echo "可写：是"; rm -f "$DEST/.probe_$$"; } || echo "可写：⛔ 否"
echo

echo "=== 逐文件段（rsync，可续传）==="
for spec in "1 manifests:manifests" \
            "2 metric_adapt:runs/metric_adapt" \
            "3 baseline_bench:runs/baseline_bench" \
            "7 eval:runs/eval"; do
  label="${spec%%:*}"; sub="${spec##*:}"
  if [[ -d "$DEST/$sub" ]]; then
    row "$label" "在（顶层 $(ls -1 "$DEST/$sub" 2>/dev/null | wc -l) 项，$(du -sh "$DEST/$sub" 2>/dev/null | cut -f1)）"
  else
    row "$label" "—— 没有"
  fi
done

echo
echo "=== 4 converted 权重（进论文的那几条臂）==="
for run in ma_r3ow_cap ma_r3ow_nocap r3ow_cap r3ow_nocap \
           iris_ms2_full8_thermalcap iris_ms2_full8_nocap \
           lotusd_full8_thermalcap lotusd_full8_nocap; do
  s="$IRIS_RUNS/iris_ms2/$run/converted"; d="$DEST/runs/iris_ms2/$run/converted"
  ns=$([[ -d "$s" ]] && ls -1 "$s" | wc -l || echo 0)
  nd=$([[ -d "$d" ]] && ls -1 "$d" | wc -l || echo 0)
  if   [[ "$ns" == 0 ]];        then row "$run" "源上没有 converted/"
  elif [[ "$nd" == "$ns" ]];    then row "$run" "✅ $nd/$ns"
  else                               row "$run" "⚠️ $nd/$ns（未完）"
  fi
done

echo
echo "=== 打包段（tar，⚠️ 不能续传）==="
shopt -s nullglob
tars=("$DEST"/*.tar)
if [[ ${#tars[@]} -eq 0 ]]; then
  echo "  （目标下没有 .tar）"
else
  for f in "${tars[@]}"; do
    if tar_ok "$f"; then row "$(basename "$f")" "✅ 完整 $(du -h "$f" | cut -f1)"
    else                 row "$(basename "$f")" "⛔ 截断/半成品 $(du -h "$f" | cut -f1) —— 删掉重打"
    fi
  done
fi
parts=("$DEST"/*.tar.part)
for f in "${parts[@]}"; do row "$(basename "$f")" "⏳ 上一趟没打完（重投会自动重来）"; done

echo
echo "=== 还没搬的臂（FULL=1 才走的第 8 段）==="
missing=0
for run in $(ls -1 "$IRIS_RUNS/iris_ms2" 2>/dev/null); do
  [[ -d "$IRIS_RUNS/iris_ms2/$run" ]] || continue
  t="$DEST/iris_ms2__$run.tar"
  tar_ok "$t" && continue
  [[ -d "$DEST/runs/iris_ms2/$run/checkpoint-"* ]] 2>/dev/null && continue
  echo "  - $run"; missing=$((missing+1))
done
echo "  共 $missing 条未搬"

echo
echo "=== 配额 ==="
echo "--- 源 sc23sz ---"; lfs quota -h -u "$USER" /scratch 2>/dev/null | sed -n '2,4p'
echo "--- 目标（看得到就打，看不到要对方自己跑 lfs quota -h -u dwdh0363 /scratch）---"
lfs quota -h -u "$(stat -c %U "$DEST" 2>/dev/null)" /scratch 2>/dev/null | sed -n '2,4p' || echo "  读不到"
