"""Does the caption's benefit come only from what thermal can actually see?

Our captions are written from the RGB frame but fed to a model that sees only
thermal. Roughly 81% of them name colours ("green recycling bins", "blue
newspaper vending machines") which have no thermal correlate at all, while
other content -- spatial layering, and warm objects like people and vehicles --
is in principle groundable in a thermal image.

Hypothesis: the measured caption benefit concentrates in captions whose content
is groundable, so colour-heavy captions should help LESS on thermal.

Built-in falsification control: on the RGB route colour IS groundable, so the
same lexical score must NOT predict the caption effect there. If colour density
predicts the effect equally on both routes, the score is just tracking some
confound (caption length, scene complexity) and the reading is void.

Per-image effect = abs_rel(empty) - abs_rel(caption); positive = caption helped.
CPU only, reads existing official per_image.csv outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


COLOUR = ("red", "green", "blue", "yellow", "white", "black", "orange", "purple",
          "brown", "gray", "grey", "pink", "beige", "tan", "golden", "silver")
# Objects that are warm (or reliably cooler than surroundings) and therefore
# carry a real thermal signature the model can attach the word to.
WARM = ("person", "people", "man", "woman", "men", "women", "child", "pedestrian",
        "car", "vehicle", "bus", "truck", "bicycle", "motorcycle", "dog", "animal")
SPATIAL = ("foreground", "midground", "background", "near", "far", "behind",
           "front", "left", "right", "distant", "beyond", "layered")

BASE = Path("outputs/ms2_official")
PAIRS = {
    "c 线 冻结热像 (零训练)": ("rgbdt500_thermal_frozen_empty_mind01",
                              "rgbdt500_thermal_frozen_capt_mind01"),
    "a 线 冻结RGB (对照)": ("rgbdt500_rgb_direct_mind01",
                            "rgbdt500_rgb_direct_capt_mind01"),
    "d 线 端到端": ("rgbdt500_emptytrain_empty_mind01",
                    "rgbdt500_capttrain_correct_mind01"),
    "RGB-teacher 线 端到端": ("rgbdt500_rgbteacher_l1_mind01",
                              "rgbdt500_eval_rgbteacher_l1_capttrain_correct_mind01"),
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("E:/dataset/RGBDT500/clean_test/"
                                     "rgbdt500_test_manifest_iris_prose.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/lotus_line_v2/caption_groundability.json"))
    return parser.parse_args()


def load_abs_rel(name: str) -> dict[str, float]:
    path = BASE / name / "metrics" / "per_image.csv"
    out = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[row.get("sample_id") or row.get("id")] = float(row["abs_rel"])
    return out


def features(caption: str) -> dict[str, float]:
    words = re.findall(r"[a-z]+", caption.lower())
    total = max(len(words), 1)
    counts = {
        "colour": sum(words.count(w) for w in COLOUR),
        "warm": sum(words.count(w) for w in WARM),
        "spatial": sum(words.count(w) for w in SPATIAL),
    }
    return {
        "words": float(total),
        "colour": float(counts["colour"]),
        "warm": float(counts["warm"]),
        "colour_density": counts["colour"] / total,
        "warm_density": counts["warm"] / total,
        "spatial_density": counts["spatial"] / total,
    }


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Rank correlation plus a normal-approximation two-sided p (n is ~2800)."""
    def rank(v):
        order = v.argsort()
        ranks = np.empty(len(v), float)
        ranks[order] = np.arange(len(v), dtype=float)
        return ranks
    rx, ry = rank(x), rank(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x)
    if n < 10 or abs(rho) >= 1:
        return rho, float("nan")
    t = rho * np.sqrt((n - 2) / max(1 - rho * rho, 1e-12))
    from math import erfc, sqrt
    return rho, float(erfc(abs(t) / sqrt(2)))


def terciles(score: np.ndarray, effect: np.ndarray) -> list[dict]:
    edges = np.quantile(score, [1 / 3, 2 / 3])
    bins = [effect[score <= edges[0]],
            effect[(score > edges[0]) & (score <= edges[1])],
            effect[score > edges[1]]]
    out = []
    for label, values in zip(("低", "中", "高"), bins):
        if len(values) == 0:
            continue
        se = values.std(ddof=1) / np.sqrt(len(values))
        out.append({"bin": label, "n": int(len(values)), "mean_effect": float(values.mean()),
                    "ci_lo": float(values.mean() - 1.96 * se),
                    "ci_hi": float(values.mean() + 1.96 * se)})
    return out


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8") if line.strip()]
    feats = {r["id"]: features(r.get("caption", "")) for r in rows}

    report = {}
    for label, (empty_dir, capt_dir) in PAIRS.items():
        if not (BASE / empty_dir / "metrics" / "per_image.csv").is_file() or \
           not (BASE / capt_dir / "metrics" / "per_image.csv").is_file():
            print(f"skip {label}: missing per_image.csv")
            continue
        empty, capt = load_abs_rel(empty_dir), load_abs_rel(capt_dir)
        ids = sorted(set(empty) & set(capt) & set(feats))
        effect = np.array([empty[i] - capt[i] for i in ids])   # >0 = caption helped
        entry = {"n": len(ids), "mean_effect": float(effect.mean()), "correlations": {}, "terciles": {}}
        print(f"\n=== {label}   n={len(ids)}   平均效应 {effect.mean():+.4f} "
              f"({'caption 有益' if effect.mean() > 0 else 'caption 有害'})")
        for key in ("colour_density", "warm_density", "spatial_density", "words"):
            score = np.array([feats[i][key] for i in ids])
            rho, p = spearman(score, effect)
            entry["correlations"][key] = {"spearman_rho": rho, "p": p}
            entry["terciles"][key] = terciles(score, effect)
            flag = "显著" if p < 0.05 else "n.s."
            print(f"  {key:16} rho={rho:+.4f}  p={p:.3g}  {flag}")
            cells = "  ".join(f"{b['bin']}:{b['mean_effect']:+.4f}" for b in entry["terciles"][key])
            print(f"      三分位平均效应   {cells}")
        report[label] = entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.output}")
    print("\n读法: 假说成立需要 colour_density 在热像线上显著为负, 且在 a 线(RGB)上不显著。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
