"""Decompose P3 spatial-caption damage into object / depth-band / lateral-tag parts.

Joins per_sample_metrics.csv of four eval runs on the same checkpoint & seeds
(empty / objects-only / no-positions / correct) and reports, per transform,
the paired AbsRel damage relative to the empty baseline plus the marginal
contribution of each spatial-scaffold component:

    objects            = damage(objects-only)
    depth bands (A 载体) = damage(no-positions) - damage(objects-only)
    lateral tags (B 载体) = damage(correct)      - damage(no-positions)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = {
    "empty": "outputs/lotus_line_v2/p3_official_val_full_empty",
    "objects-only": "outputs/lotus_line_v2/p3_official_val_full_objectsonly",
    "no-positions": "outputs/lotus_line_v2/p3_official_val_full_nopositions",
    "correct": "outputs/lotus_line_v2/p3_official_val_full_correct",
}


def read_absrel(run_dir: Path) -> dict[str, float]:
    path = run_dir / "per_sample_metrics.csv"
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows[row["filename"]] = float(row["abs_relative_difference"])
    if not rows:
        raise SystemExit(f"No per-sample rows in {path}")
    return rows


def sign_test_z(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 0.0
    return (wins - n / 2) / math.sqrt(n / 4)


def paired_stats(mode: dict[str, float], base: dict[str, float]):
    keys = sorted(set(mode) & set(base))
    if len(keys) != len(mode) or len(keys) != len(base):
        raise SystemExit(
            f"Sample sets differ: {len(mode)} vs {len(base)} rows, {len(keys)} shared"
        )
    diffs = [mode[k] - base[k] for k in keys]
    diffs_sorted = sorted(diffs)
    n = len(diffs)
    median = (
        diffs_sorted[n // 2]
        if n % 2
        else 0.5 * (diffs_sorted[n // 2 - 1] + diffs_sorted[n // 2])
    )
    wins = sum(1 for d in diffs if d < 0)  # mode better (lower AbsRel)
    losses = sum(1 for d in diffs if d > 0)
    return {
        "n": n,
        "mean_damage": sum(diffs) / n,
        "median_damage": median,
        "win_rate_vs_base": wins / n,
        "sign_test_z": sign_test_z(wins, losses),
        "mean_absrel": sum(mode[k] for k in keys) / n,
    }


def main():
    parser = argparse.ArgumentParser()
    for name, default in DEFAULT_RUNS.items():
        flag = "--" + name.replace(" ", "-") + "-dir"
        parser.add_argument(flag, type=Path, default=ROOT / default)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    dirs = {
        "empty": args.empty_dir,
        "objects-only": args.objects_only_dir,
        "no-positions": args.no_positions_dir,
        "correct": args.correct_dir,
    }
    absrel = {name: read_absrel(path) for name, path in dirs.items()}
    base = absrel["empty"]

    report = {"baseline_mean_absrel": sum(base.values()) / len(base), "modes": {}}
    for name in ("objects-only", "no-positions", "correct"):
        report["modes"][name] = paired_stats(absrel[name], base)

    damage = {name: report["modes"][name]["mean_damage"] for name in report["modes"]}
    report["decomposition"] = {
        "objects": damage["objects-only"],
        "depth_bands_marginal": damage["no-positions"] - damage["objects-only"],
        "lateral_tags_marginal": damage["correct"] - damage["no-positions"],
        "total_damage_correct_vs_empty": damage["correct"],
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output_json:
        args.output_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
