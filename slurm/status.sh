#!/usr/bin/env bash
# 集群自查工具。**在 Aire 上运行。**
#
#   bash ~/Iris/slurm/status.sh            # 总览：队列 + 近期作业 + 训练进度 + 配额
#   bash ~/Iris/slurm/status.sh log 12345  # 跟某个作业的日志（Ctrl+C 退出）
#   bash ~/Iris/slurm/status.sh err 12345  # 看某个作业的报错
#   bash ~/Iris/slurm/status.sh why 12345  # 排队中的作业为什么还没跑 + 预计启动时间
#   bash ~/Iris/slurm/status.sh results    # 所有评估数字
#   bash ~/Iris/slurm/status.sh regions    # 所有分层报告
#   bash ~/Iris/slurm/status.sh runs       # 各训练目录的进度与体积
#   bash ~/Iris/slurm/status.sh wait       # 阻塞到全部作业结束，然后打印汇总
#
# 原生命令速查（不用这个脚本时）：
#   squeue -u $USER                     我的队列。ST: PD=排队 R=运行 CG=收尾
#   squeue -u $USER --start             排队中作业的预计启动时间
#   scancel 12345                       取消
#   sacct -j 12345 --format=JobID,State,Elapsed,ExitCode,MaxRSS
#   scontrol show job 12345             作业的全部细节
#   tail -f $SCRATCH/logs/<名字>-<id>.out
#
# 作业结束的判断：squeue 里没有了 = 结束。成功要看 sacct 的
# State=COMPLETED 且 ExitCode=0:0。TIMEOUT 表示撞了墙钟，重投即续跑。

set -uo pipefail
: "${SCRATCH:?这个脚本要在 Aire 上运行}"
L="$SCRATCH/logs"

case "${1:-overview}" in

  log)  tail -f "$(ls -t "$L"/*-"${2:?给个 job id}".out | head -1)" ;;
  err)  f=$(ls -t "$L"/*-"${2:?给个 job id}".err | head -1); echo "--- $f"; tail -40 "$f" ;;

  why)
    id="${2:?给个 job id}"
    squeue -j "$id" -o "%.12i %.10T %.20R %.20S" 2>/dev/null
    echo "（R 列是等待原因，S 列是预计启动时间；Resources=等卡，Priority=排在别人后面）"
    ;;

  results)
    echo "=== 评估结果（test/val AbsRel）==="
    for f in "$SCRATCH"/runs/eval/*/eval_*.json; do
      [ -f "$f" ] || continue
      python3 - "$f" <<'PY'
import json, sys, os
d = json.load(open(sys.argv[1]))
name = os.path.basename(os.path.dirname(sys.argv[1]))
print("  %-22s abs_rel %.4f  rmse %.3f  a1 %.4f  (%s 帧, %s)" % (
    name, d["abs_rel"], d["rmse"], d["a1"], d["val_samples"],
    os.path.basename(str(d.get("val_manifest", "?")))[:38]))
PY
    done
    ;;

  regions)
    for d in "$SCRATCH"/runs/regions/*/region_report.json; do
      [ -f "$d" ] || continue
      echo "=== $(basename "$(dirname "$d")")"
      python3 - "$d" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))["strata"]
rows = ("all", "depth/far >30m", "row/top", "structure/boundary", "structure/interior")
for r in rows:
    if r in s:
        m = s[r]["means"]
        print("  %-20s" % r + "  ".join("%s=%.4f" % (k, v) for k, v in m.items()))
b, i = s.get("structure/boundary"), s.get("structure/interior")
if b and i:
    print("  %-20s" % "锐利度 (b/i)" + "  ".join(
        "%s=%.2f" % (k, b["means"][k] / i["means"][k]) for k in b["means"]))
PY
    done
    ;;

  runs)
    echo "=== 训练目录 ==="
    for d in "$SCRATCH"/runs/route_suite/*/; do
      [ -d "$d" ] || continue
      n=$(basename "$d")
      last=$(tail -1 "$d/epoch_metrics.jsonl" 2>/dev/null)
      ep=$(echo "$last" | grep -o '"epoch": *[0-9]*' | grep -o '[0-9]*$')
      ar=$(echo "$last" | grep -o '"val_abs_rel": *[0-9.]*' | grep -o '[0-9.]*$')
      printf '  %-28s %6s  epoch=%-4s val_abs_rel=%s\n' "$n" "$(du -sh "$d" 2>/dev/null | cut -f1)" "${ep:-?}" "${ar:-?}"
    done
    ;;

  wait)
    echo "等待全部作业结束……（Ctrl+C 可随时退出，不影响作业）"
    while squeue -h -u "$USER" | grep -q .; do sleep 120; done
    echo "=== 全部结束 ==="
    sacct -S "$(date -d '2 days ago' +%F)" -X --format=JobID%12,JobName%18,State%14,Elapsed,ExitCode
    ;;

  overview|*)
    echo "=== 队列 ==="
    squeue -u "$USER" -o "%.10i %.16j %.2t %.11M %.11L %.18R"
    echo "  (t: PD=排队 R=运行 CG=收尾 | M=已运行 L=剩余时限 | R=节点或等待原因)"
    echo
    echo "=== 近两天作业 ==="
    sacct -S "$(date -d '2 days ago' +%F)" -X --format=JobID%12,JobName%18,State%14,Elapsed,ExitCode
    echo
    echo "=== 运行中作业的最新进度 ==="
    for id in $(squeue -h -u "$USER" -t R -o "%i"); do
      f=$(ls -t "$L"/*-"$id".out 2>/dev/null | head -1)
      [ -n "$f" ] && { echo "--- $(basename "$f")"; tail -2 "$f"; }
    done
    echo
    echo "=== 配额 ==="
    lfs quota -h -u "$USER" /scratch | tail -2
    ;;
esac
