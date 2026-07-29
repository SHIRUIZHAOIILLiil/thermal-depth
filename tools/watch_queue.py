#!/usr/bin/env python3
"""One-screen status for a running experiment queue.

``tools/watch_training.py`` follows a single run directory, but a queue moves
between arms and between phases (train -> inference -> rescore), so a fixed
directory stops being the interesting one after a few minutes. This finds the
currently-active work by modification time instead.

Shows: the queue's current stage line, the live training step + losses + ETA
(total taken from the run's own frozen_config.json, not guessed), the live
inference npy count, and which cells have already been scored.

Usage (WSL, single line):
    watch -n 30 python3 tools/watch_queue.py

Or point it at a different queue:
    python3 tools/watch_queue.py --queue-log outputs/lotus_line_v2/<name>.log
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue-log", type=Path,
                   default=Path("outputs/lotus_line_v2/rgbdt500_six_route_completion.log"))
    p.add_argument("--runs-root", type=Path, default=Path("outputs/lotus_line_v2"))
    p.add_argument("--official-root", type=Path, default=Path("outputs/ms2_official"))
    p.add_argument("--pattern", default="rgbdt500")
    p.add_argument("--stale-seconds", type=float, default=600,
                   help="Treat a directory older than this as not the active one.")
    return p.parse_args()


def human(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def newest(paths, stale: float):
    live = [(p, p.stat().st_mtime) for p in paths if p.exists()]
    if not live:
        return None, 0.0
    path, mtime = max(live, key=lambda x: x[1])
    age = time.time() - mtime
    return (path, age) if age <= stale else (None, age)


def show_training(runs_root: Path, stale: float) -> bool:
    logs = list(runs_root.glob("*/training_metrics.jsonl"))
    log, age = newest(logs, stale)
    if log is None:
        return False
    last = ""
    with log.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = line
    if not last:
        return False
    record = json.loads(last)
    run = log.parent
    total = None
    config = run / "frozen_config.json"
    if config.is_file():
        try:
            total = json.loads(config.read_text(encoding="utf-8")).get("optimizer_updates")
        except (ValueError, OSError):
            total = None
    step = record.get("step", 0)
    elapsed = record.get("elapsed_seconds", 0.0)
    print(f"\n[训练中] {run.name}   (最后更新 {human(age)} 前)")
    if total:
        rate = elapsed / step if step else 0.0
        eta = rate * (total - step)
        bar = int(30 * step / total)
        print(f"  进度 {step}/{total} ({step / total:6.1%})  [{'#' * bar}{'.' * (30 - bar)}]"
              f"  已用 {human(elapsed)}  预计剩余 {human(eta)}")
    else:
        print(f"  step {step}  已用 {human(elapsed)}")
    keys = [k for k in ("gt_ssi_l1", "dense_teacher_l1", "condition_total",
                        "response_total", "total") if k in record]
    print("  " + "   ".join(f"{k}={record[k]:.4f}" for k in keys))
    for grad in ("unet_grad_norm", "adapter_grad_norm"):
        if record.get(grad) is not None:
            print(f"  {grad}={record[grad]:.3f}", end="")
    print()
    return True


def show_inference(runs_root: Path, pattern: str, stale: float) -> bool:
    dirs = [d for d in runs_root.glob(f"*{pattern}*/raw_predictions") if d.is_dir()]
    directory, age = newest(dirs, stale)
    if directory is None:
        return False
    count = sum(1 for _ in directory.glob("*.npy"))
    print(f"\n[推理中] {directory.parent.name}   (最后更新 {human(age)} 前)")
    print(f"  已产出 {count} 个 npy")
    return True


def main() -> int:
    args = parse_args()
    print("=" * 72)
    print(f"队列监控  {time.strftime('%H:%M:%S')}")
    print("=" * 72)

    if args.queue_log.is_file():
        stages = [l.rstrip() for l in args.queue_log.open(encoding="utf-8") if l.startswith("===")]
        age = time.time() - args.queue_log.stat().st_mtime
        print(f"\n[队列] {args.queue_log.name}   (最后写入 {human(age)} 前)")
        for line in stages[-3:]:
            print(f"  {line}")
        if not stages:
            print("  (还没有阶段标记)")
    else:
        print(f"\n[队列] 日志尚未创建: {args.queue_log}")

    busy = show_training(args.runs_root, args.stale_seconds)
    busy = show_inference(args.runs_root, args.pattern, args.stale_seconds) or busy
    if not busy:
        print("\n[空闲] 没有近期活动的训练或推理目录 —— 队列可能已结束或已中断")

    scored = sorted(p.parent.parent.name
                    for p in args.official_root.glob(f"*{args.pattern}*/metrics/summary.json"))
    print(f"\n[已完成官方复算] {len(scored)} 格")
    for name in scored[-6:]:
        try:
            stats = json.loads(
                (args.official_root / name / "metrics" / "summary.json").read_text()
            )["image_wise"]["statistics"]
            print(f"  {name:52} AbsRel={stats['abs_rel']['mean']:.4f} d1={stats['a1']['mean']:.4f}")
        except (OSError, ValueError, KeyError):
            print(f"  {name:52} (读取失败)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
