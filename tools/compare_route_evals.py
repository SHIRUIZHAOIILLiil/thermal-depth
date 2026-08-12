"""Paired comparison between per-sample evaluation dumps.

`train_route_suite.py --eval-checkpoint` writes `eval_<tag>_per_sample.csv`, one
row per validation frame.  Two such files scored on the same manifest are paired
by frame id, so the right test is a paired one: differences are computed
frame-by-frame and the confidence interval comes from bootstrapping those
differences, not from the two marginal means.

This matters here.  The project has been misled three times by unpaired mean
directions (the far-field false positive, the val32 false positive, the region
stratification sign flip).  A 0.003 gap between epoch 1 and epoch 5 is exactly
the size where paired and unpaired disagree.

    python tools/compare_route_evals.py outputs/route_suite/b_thermal_unet_20ep/eval_full_ep01_per_sample.csv outputs/route_suite/b_thermal_unet_20ep/eval_full_ep05_per_sample.csv outputs/route_suite/b_thermal_unet_20ep/eval_full_ep20_per_sample.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

# metric -> lower is better
METRICS = {
    "abs_rel": True,
    "rmse": True,
    "sq_rel": True,
    "log10": True,
    "a1": False,
    "a2": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csvs", nargs="+", type=Path, help="Two or more eval_*_per_sample.csv files.")
    parser.add_argument("--metrics", nargs="*", default=["abs_rel", "rmse", "a1"])
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def load(path: Path) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values = {}
            for key, value in row.items():
                if key in ("id", "sequence"):
                    continue
                try:
                    values[key] = float(value)
                except (TypeError, ValueError):
                    continue
            table[row["id"]] = values
    if not table:
        raise SystemExit(f"{path} has no rows")
    return table


def label_for(path: Path) -> str:
    name = path.stem
    for prefix, suffix in (("eval_", ""), ("", "_per_sample")):
        if prefix and name.startswith(prefix):
            name = name[len(prefix):]
        if suffix and name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def paired_stats(left: np.ndarray, right: np.ndarray, lower_is_better: bool, bootstrap: int, rng) -> dict:
    """right - left, plus a bootstrap CI over the per-frame differences."""
    difference = right - left
    mean = float(difference.mean())
    indices = rng.integers(0, len(difference), size=(bootstrap, len(difference)))
    samples = difference[indices].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    improved = difference < 0 if lower_is_better else difference > 0
    return {
        "n": int(len(difference)),
        "left_mean": float(left.mean()),
        "right_mean": float(right.mean()),
        "mean_difference": mean,
        "ci95": [float(low), float(high)],
        "significant": bool(low > 0 or high < 0),
        "right_win_rate": float(improved.mean()),
    }


def main() -> None:
    args = parse_args()
    if len(args.csvs) < 2:
        raise SystemExit("Need at least two per-sample CSVs to compare.")
    rng = np.random.default_rng(args.seed)

    # eval.sbatch leaves --eval-tag at its default, so every run it produces writes
    # `eval_eval_per_sample.csv`: the file name identifies nothing and two arms of
    # one experiment differ only by directory. Keyed on the stem alone they collapse
    # into a single entry and the tool prints one run and no comparison at all --
    # which reads like "no difference" rather than like a mistake.
    names = [label_for(path) for path in args.csvs]
    if len(set(names)) != len(names):
        names = [path.parent.name for path in args.csvs]
        if len(set(names)) != len(names):
            raise SystemExit(
                "Two CSVs share both a file name and a parent directory name; pass "
                f"distinct files. Got: {[str(p) for p in args.csvs]}"
            )
        print(f"[label] file names collide, using directory names instead: {names}\n")
    tables = {name: load(path) for name, path in zip(names, args.csvs)}
    labels = list(tables)
    shared = set.intersection(*(set(table) for table in tables.values()))
    if not shared:
        raise SystemExit("The CSVs share no frame ids -- were they scored on the same manifest?")
    ordered_ids = sorted(shared)
    for label, table in tables.items():
        if len(table) != len(shared):
            print(f"[warn] {label}: {len(table)} rows, {len(shared)} shared with the others")
    print(f"{len(labels)} runs, {len(ordered_ids)} paired frames\n")

    report: dict = {"frames": len(ordered_ids), "runs": labels, "comparisons": []}

    for metric in args.metrics:
        if metric not in METRICS:
            print(f"[warn] unknown metric {metric}, skipping")
            continue
        lower_is_better = METRICS[metric]
        print(f"=== {metric} ({'lower' if lower_is_better else 'higher'} is better)")
        for label in labels:
            column = np.array([tables[label][i][metric] for i in ordered_ids])
            print(f"    {label:16s} mean {column.mean():.4f}")
        for left_label, right_label in itertools.combinations(labels, 2):
            left = np.array([tables[left_label][i][metric] for i in ordered_ids])
            right = np.array([tables[right_label][i][metric] for i in ordered_ids])
            stats = paired_stats(left, right, lower_is_better, args.bootstrap, rng)
            marker = "*" if stats["significant"] else " "
            print(
                f"    {right_label} - {left_label}: {stats['mean_difference']:+.5f}{marker} "
                f"CI[{stats['ci95'][0]:+.5f},{stats['ci95'][1]:+.5f}] "
                f"win {stats['right_win_rate']:.1%}"
            )
            report["comparisons"].append({"metric": metric, "left": left_label, "right": right_label, **stats})
        print()

    print("* = 95% bootstrap CI excludes zero")
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWritten to {args.report}")


if __name__ == "__main__":
    main()
