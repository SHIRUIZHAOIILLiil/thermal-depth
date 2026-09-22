#!/usr/bin/env python3
"""把 select_checkpoint 扫出来的一堆 summary.json 汇成一条 val 曲线。

单独成脚本而不是内联在 sbatch 里，因为扫描本身很贵而汇总不要钱：分数已经在盘上，
重新排一次队只为了再打印一遍表格是荒唐的。这个在登录节点上跑。

    python tools/summarize_checkpoint_sweep.py $IRIS_RUNS/analysis/select_... [--limit 40]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

METRICS = ("abs_rel", "rmse", "a1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--limit", default="0",
                        help="扫描时每个点用了多少帧；非 0 会在表下标明这是冒烟口径。")
    args = parser.parse_args()

    rows = []
    for d in sorted(args.sweep_dir.iterdir()):
        path = d / "metrics" / "summary.json"
        if not path.is_file():
            continue
        # statistics[name] 是 {"mean","std","median","count"}，不是标量。
        # 评估器自己打印的也是 mean，所以这里跟着取 mean。
        stats = json.loads(path.read_text(encoding="utf-8"))["image_wise"]["statistics"]
        step = int(m.group()) if (m := re.search(r"\d+", d.name)) else 0
        rows.append((step, d.name, *(stats[k]["mean"] for k in METRICS)))
    if not rows:
        raise SystemExit(f"{args.sweep_dir} 下没有 metrics/summary.json")
    rows.sort()

    print(f"\n{'checkpoint':>14s}{'AbsRel':>10s}{'RMSE (m)':>12s}{'δ1':>10s}")
    best = min(rows, key=lambda r: r[2])
    for _, name, abs_rel, rmse, delta1 in rows:
        mark = "  <- val 最优" if name == best[1] else ""
        print(f"{name:>14s}{abs_rel:>10.5f}{rmse:>12.3f}{delta1:>10.5f}{mark}")

    print(f"\n选点：{best[1]}   AbsRel {best[2]:.5f}")
    # 「后面的点更差」对纯噪声也成立，所以先问这条曲线分不分得开。相邻两点的
    # 典型跳动就是这个采样量下的噪声尺度；最优与次优的差比它还小，说明名次是
    # 抖出来的，报一个选点只是在给噪声起名字。
    if len(rows) == 1:
        return

    jumps = sorted(abs(rows[k][2] - rows[k - 1][2]) for k in range(1, len(rows)))
    noise = jumps[len(jumps) // 2] if jumps else 0.0
    ordered = sorted(rows, key=lambda r: r[2])
    margin = (ordered[1][2] - ordered[0][2]) if len(ordered) > 1 else float("inf")
    print(f"相邻点跳动（中位）{noise:.5f}　最优与次优之差 {margin:.5f}")
    if margin < noise:
        print("⛔ 差距小于噪声尺度 —— 这条曲线分不出名次，不能据此选点。")
        print("   下一步取决于这个跳动是哪来的：")
        print("     · 采样不足 → 加帧数（噪声约按 1/√N 降）")
        print("     · checkpoint 自己在抖 → 加帧数没用，需要一条处理并列的规则，")
        print("       而且规则要在跑 test 之前定下来")
        print("   分辨方法：换一个帧数重跑，看跳动有没有按 1/√N 降下去")
        return
    # 最优落在最后一个点，说明 val 可能还在下降 —— 那个「最优」也许只是训练被截断
    # 的地方，和一条真正翻了头的曲线是两回事，报的时候不能混为一谈。
    if best[1] == rows[-1][1]:
        print("⚠️ 最优落在最后一个 checkpoint —— val 可能还在下降，预算没花完，"
              "这个「最优」也许只是训练停在了那里。")
    else:
        later = sum(1 for r in rows if r[0] > best[0])
        print(f"val 在此之后翻头（其后 {later} 个点都更差），不是被截断的。")
    if str(args.limit) != "0":
        print(f"\n⛔ 冒烟口径（每点只用了 {args.limit} 帧），这条曲线不能用来定选点。")


if __name__ == "__main__":
    main()
