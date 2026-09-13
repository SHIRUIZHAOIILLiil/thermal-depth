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

# 「读不到」和「是空的」在 ls 里长得一模一样，只要 stderr 被吞掉。2026-09-13
# 就是这么把一个已经搬进去几百 GB 的目标读成了零：from_sc23sz 的 default ACL 里
# 没有 sc23sz，凡是在对方账号下建的子目录，属主是他、组 ---，我们连 ls 都不行。
# 所以每一处都先问「我读得进去吗」，读不进去就照实说，别报成没有。
peek () {  # 目录 -> 一句话状态
  local d="$1"
  [[ -e "$d" ]] || { echo "—— 没有"; return; }
  if ! ls -1 "$d" >/dev/null 2>&1; then
    echo "⛔ 在，但我读不到（属主 $(stat -c %U "$d" 2>/dev/null)）—— 要他 setfacl -R -m u:$USER:rwX"
    return
  fi
  echo "在（顶层 $(ls -1 "$d" | wc -l) 项，$(du -sh "$d" 2>/dev/null | cut -f1)）"
}

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
  row "$label" "$(peek "$DEST/$sub")"
done

echo
echo "=== 4 converted 权重（进论文的那几条臂）==="
# ⚠️ converted/ 空 ≠ 这条臂没东西。2026-09-10 清过一轮 step*_weights.pt，
# 规则是「有对应 checkpoint-<step> 才删」——所以权重可能只以 checkpoint-<step>/
# 的形式存在，而那属于第 8 段（FULL=1 才走）。这两种情况必须分开报，
# 不然会以为模型已经没了。
for run in ma_r3ow_cap ma_r3ow_nocap r3ow_cap r3ow_nocap \
           iris_ms2_full8_thermalcap iris_ms2_full8_nocap \
           lotusd_full8_thermalcap lotusd_full8_nocap; do
  s="$IRIS_RUNS/iris_ms2/$run/converted"; d="$DEST/runs/iris_ms2/$run/converted"
  ns=0; [[ -d "$s" ]] && ns=$(ls -1 "$s" 2>/dev/null | wc -l)
  nd=0; [[ -d "$d" ]] && nd=$(ls -1 "$d" 2>/dev/null | wc -l)
  if [[ -d "$d" ]] && ! ls -1 "$d" >/dev/null 2>&1; then row "$run" "⛔ 目标侧读不到（属主 $(stat -c %U "$d"))"; continue; fi
  nc=$(ls -d "$IRIS_RUNS/iris_ms2/$run/checkpoint-"* 2>/dev/null | wc -l)
  if   [[ ! -d "$IRIS_RUNS/iris_ms2/$run" ]]; then row "$run" "源上没有这条臂"
  elif [[ "$ns" == 0 && "$nc" != 0 ]]; then row "$run" "⚠️ converted/ 已清空，只剩 $nc 个 checkpoint-* → 要 FULL=1 才搬得走"
  elif [[ "$ns" == 0 ]];     then row "$run" "⛔ converted/ 空且无 checkpoint-*"
  elif [[ "$nd" == "$ns" ]]; then row "$run" "✅ $nd/$ns"
  else                            row "$run" "⚠️ $nd/$ns（未完）"
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
  # 非打包模式下这条臂可能是逐文件搬过去的
  compgen -G "$DEST/runs/iris_ms2/$run/checkpoint-*" >/dev/null 2>&1 && continue
  echo "  - $run"; missing=$((missing+1))
done
echo "  共 $missing 条未搬"

echo
echo "=== 配额 ==="
echo "--- 源 sc23sz ---"; lfs quota -h -u "$USER" /scratch 2>/dev/null | sed -n '2,4p'
echo "--- 目标（看得到就打，看不到要对方自己跑 lfs quota -h -u dwdh0363 /scratch）---"
lfs quota -h -u "$(stat -c %U "$DEST" 2>/dev/null)" /scratch 2>/dev/null | sed -n '2,4p' || echo "  读不到"
