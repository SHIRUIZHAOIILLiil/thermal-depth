"""Paired per-sample analysis of the RGBDT500 caption 2x2 (dense-GT venue).

Mirrors tools/analyze_caption_ablation_20cell.py so the effect sizes here are
directly comparable with the MS2 numbers: same metric, same pairing, same
bootstrap. The research question is whether a DENSE GT (RGBDT500 ~78% valid)
changes the caption verdict that sparse MS2 GT (~29%) produced.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ms2_eval.aggregate import paired_comparison  # noqa: E402

OFFICIAL = ROOT / "outputs" / "ms2_official"
METRICS = ("abs_rel", "a1")
LOWER_IS_BETTER = {"abs_rel"}

# NOTE: the `_mind01` runs use --min-depth 0.1. The first pass used the MS2
# default of 1e-3, which on this consumer depth sensor admits millimetre-level
# noise pixels; |err|/gt then explodes and inflated every cell to ~0.82.
CELLS_BY_LINE = {
    "d": {  # thermal -> frozen VAE -> trained U-Net
        "emptytrain_empty": "rgbdt500_emptytrain_empty_mind01",
        "emptytrain_correct": "rgbdt500_emptytrain_correct_mind01",
        "capttrain_empty": "rgbdt500_capttrain_empty_mind01",
        "capttrain_correct": "rgbdt500_capttrain_correct_mind01",
    },
    "f": {  # AnyThermal features -> trained Adapter -> frozen U-Net
        "emptytrain_empty": "rgbdt500_fline_emptytrain_empty_mind01",
        "emptytrain_correct": "rgbdt500_fline_emptytrain_correct_mind01",
        "capttrain_empty": "rgbdt500_fline_capttrain_empty_mind01",
        "capttrain_correct": "rgbdt500_fline_capttrain_correct_mind01",
    },
}
CELLS = CELLS_BY_LINE["d"]  # overridden by --line

# (label, arm A, arm B) -- positive improvement always means A beats B
COMPARISONS = (
    ("推理注入价值 (empty训): correct vs empty", "emptytrain_correct", "emptytrain_empty"),
    ("推理注入价值 (caption训): correct vs empty", "capttrain_correct", "capttrain_empty"),
    ("caption训练的依赖代价: capttrain vs emptytrain @empty推理", "capttrain_empty", "emptytrain_empty"),
    ("caption训练的净收益: capttrain vs emptytrain @correct推理", "capttrain_correct", "emptytrain_correct"),
    ("训练+使用总效果: capttrain_correct vs emptytrain_empty", "capttrain_correct", "emptytrain_empty"),
)


def load(name: str):
    path = OFFICIAL / CELLS[name] / "metrics" / "per_image.csv"
    if not path.is_file():
        raise SystemExit(f"缺少评估结果: {path}\n先跑 tools/run_rgbdt500_caption_eval_queue.sh")
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({"sample_id": row["sample_id"],
                         "abs_rel": float(row["abs_rel"]), "a1": float(row["a1"])})
    return rows


def main() -> int:
    global CELLS
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line", default="d", choices=sorted(CELLS_BY_LINE),
                        help="d = thermal->VAE->trained U-Net; f = AnyThermal->trained Adapter->frozen U-Net")
    args = parser.parse_args()
    CELLS = CELLS_BY_LINE[args.line]
    print(f"路线: {args.line} 线\n")

    cells = {name: load(name) for name in CELLS}
    print("=== RGBDT500 各格均值 (官方协议, 稠密 GT) ===")
    for name, rows in cells.items():
        ar = sum(r["abs_rel"] for r in rows) / len(rows)
        d1 = sum(r["a1"] for r in rows) / len(rows)
        print(f"  {name:22s} n={len(rows):5d}  abs_rel={ar:.4f}  δ1={d1:.4f}")

    print("\n=== 逐样本配对 (* = bootstrap 95% CI 排除零) ===")
    report = {}
    for label, a, b in COMPARISONS:
        result = paired_comparison(cells[a], cells[b], label_a=a, label_b=b,
                                   metrics=list(METRICS), lower_is_better=LOWER_IS_BETTER)
        report[label] = result
        print(f"\n{label}")
        for metric, stats in result["metrics"].items():
            sig = "*" if stats["ci_low"] > 0 or stats["ci_high"] < 0 else " "
            print(f"  {metric:8s} {stats['mean']:+.5f}{sig} "
                  f"[{stats['ci_low']:+.5f},{stats['ci_high']:+.5f}]  胜率 {stats['win_rate']*100:.1f}%")

    out = ROOT / "outputs" / "lotus_line_v2" / f"rgbdt500_caption_paired_{args.line}line.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n保存: {out}")
    print("\nMS2 对照 (稀疏 GT ~29%): RGB线注入 -0.0041* | thermal线 ~0 n.s. | f线 +0.0013*")
    print("f线 caption训练依赖代价 -0.0041* , 净收益 -0.0001 n.s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
