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
    parser.add_argument(
        "--block-lengths", nargs="*", type=int, default=[1, 50, 200, 500],
        help=(
            "Moving-block bootstrap block lengths, in frames. 1 is the IID "
            "resample, kept only for comparison: MS2 runs at 10 Hz, so "
            "neighbouring frames are near-duplicates and an IID interval comes "
            "out too narrow. Blocks never span two drives. Read the "
            "autocorrelation printed beside each result and take a block long "
            "enough to contain it."
        ),
    )
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def load(path: Path) -> tuple[dict[str, dict[str, float]], dict[str, str]]:
    table: dict[str, dict[str, float]] = {}
    sequences: dict[str, str] = {}
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
            # A block of frames may not span two drives. Where the column is
            # absent, fall back to the id prefix -- ids are <sequence>_<frame>.
            sequences[row["id"]] = row.get("sequence") or row["id"].rsplit("_", 1)[0]
    if not table:
        raise SystemExit(f"{path} has no rows")
    return table, sequences


def label_for(path: Path) -> str:
    name = path.stem
    for prefix, suffix in (("eval_", ""), ("", "_per_sample")):
        if prefix and name.startswith(prefix):
            name = name[len(prefix):]
        if suffix and name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def sequence_runs(sequence_of: list[str]) -> list[tuple[int, int]]:
    """Half-open [start, stop) index ranges, one per contiguous run of one drive."""
    runs, start = [], 0
    for i in range(1, len(sequence_of) + 1):
        if i == len(sequence_of) or sequence_of[i] != sequence_of[start]:
            runs.append((start, i))
            start = i
    return runs


def block_starts(runs: list[tuple[int, int]], length: int) -> np.ndarray:
    """Every index at which a block of `length` frames fits inside one drive."""
    starts = [np.arange(a, b - length + 1) for a, b in runs if b - a >= length]
    if not starts:
        raise SystemExit(f"no drive is {length} frames long; pick a shorter block")
    return np.concatenate(starts)


def autocorrelation(difference: np.ndarray, runs: list[tuple[int, int]], lags: list[int]) -> dict[int, float]:
    """Autocorrelation of the per-frame difference, computed within drives only.

    This is the quantity that decides the block length. Neighbouring frames of a
    10 Hz drive are near-duplicates, so the difference series carries correlation
    that IID resampling silently assumes away.
    """
    centred = difference - difference.mean()
    out = {}
    for lag in lags:
        num = den = 0.0
        for a, b in runs:
            seg = centred[a:b]
            if len(seg) <= lag:
                continue
            num += float(np.dot(seg[:-lag], seg[lag:]))
            den += float(np.dot(seg, seg))
        out[lag] = num / den if den else float("nan")
    return out


def paired_stats(left: np.ndarray, right: np.ndarray, lower_is_better: bool, bootstrap: int, rng,
                 runs: list[tuple[int, int]] | None = None, block: int = 1) -> dict:
    """right - left, plus a bootstrap CI over the per-frame differences.

    `block=1` is the ordinary IID resample. Any larger value resamples contiguous
    blocks instead, never spanning two drives, so the correlation between
    neighbouring frames survives the resampling instead of being averaged away.
    """
    difference = right - left
    mean = float(difference.mean())
    n = len(difference)
    if block <= 1 or runs is None:
        indices = rng.integers(0, n, size=(bootstrap, n))
        samples = difference[indices].mean(axis=1)
    else:
        starts = block_starts(runs, block)
        count = int(np.ceil(n / block))
        offsets = np.arange(block)
        samples = np.empty(bootstrap)
        for b in range(bootstrap):
            picked = starts[rng.integers(0, len(starts), size=count)]
            idx = (picked[:, None] + offsets[None, :]).ravel()[:n]
            samples[b] = difference[idx].mean()
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
    loaded = {name: load(path) for name, path in zip(names, args.csvs)}
    tables = {name: pair[0] for name, pair in loaded.items()}
    sequence_of_id = next(iter(loaded.values()))[1]
    labels = list(tables)
    shared = set.intersection(*(set(table) for table in tables.values()))
    if not shared:
        raise SystemExit("The CSVs share no frame ids -- were they scored on the same manifest?")
    ordered_ids = sorted(shared)
    for label, table in tables.items():
        if len(table) != len(shared):
            print(f"[warn] {label}: {len(table)} rows, {len(shared)} shared with the others")
    runs = sequence_runs([sequence_of_id.get(i, "?") for i in ordered_ids])
    print(f"{len(labels)} runs, {len(ordered_ids)} paired frames, "
          f"{len(runs)} drive(s)")
    print()

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
            acf = autocorrelation(right - left, runs, [1, 10, 50, 200])
            per_block = {}
            for block in args.block_lengths:
                per_block[block] = paired_stats(
                    left, right, lower_is_better, args.bootstrap, rng,
                    runs=runs, block=block,
                )
            head = per_block[args.block_lengths[0]]
            acf_text = " ".join(f"lag{k}={v:+.2f}" for k, v in acf.items())
            print(
                f"    {right_label} - {left_label}: "
                f"{head['mean_difference']:+.5f}  win {head['right_win_rate']:.1%}   "
                f"acf {acf_text}"
            )
            for block, stats in per_block.items():
                marker = "*" if stats["significant"] else " "
                tag = "IID " if block == 1 else f"blk{block:<4d}"
                print(f"        {tag} CI[{stats['ci95'][0]:+.5f},"
                      f"{stats['ci95'][1]:+.5f}]{marker}")
            report["comparisons"].append({
                "metric": metric, "left": left_label, "right": right_label,
                "autocorrelation": acf,
                "by_block": {str(b): s for b, s in per_block.items()},
                **head,
            })
        print()

    print("* = 95% bootstrap CI excludes zero.  IID is kept for comparison only;",
          "on 10 Hz video the honest interval is the one from a block long enough",
          "to contain the autocorrelation printed above.")
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWritten to {args.report}")


if __name__ == "__main__":
    main()
