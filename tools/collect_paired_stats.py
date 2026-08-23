#!/usr/bin/env python3
"""Run the paired block bootstrap across the whole prompt grid and write one CSV.

`compare_route_evals.py` already computes the right statistic and can already
persist it with `--report`.  What has been missing is the step after: every
"separable from zero" in the paper came from reading a terminal, and a terminal
is not a record.  This drives that tool over each (arm, condition) triple and
flattens the reports into a single table the paper can be checked against.

Two details it exists to get right:

  block order  `compare_route_evals.py` fills its top-level `mean_difference`,
               `ci95` and `significant` from **the first** `--block-lengths`
               value.  Passing `1 200` therefore heads every row with the IID
               interval -- the one this project already established is too
               narrow on 10 Hz video.  We pass `200 1`, so the headline is the
               block interval and the IID one survives beside it for the
               comparison the supplementary makes.
  condition    Manifest fingerprints in the directory names identify a frame
               set but say nothing about which condition it is, and the mapping
               lives only in someone's memory.  We read the sequences actually
               present in the per-sample CSV and name the condition from those,
               so a renamed or re-fingerprinted exam still lands in the right
               row.

    python tools/collect_paired_stats.py --eval-root $SCRATCH/runs/eval \
        --output docs/data/paired_stats_20260823.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

# MS2's own condition grouping for the nine official test sequences.
CONDITION_OF_SEQUENCE = {
    "2021-08-06-11-23-45": "day",   "2021-08-13-15-46-56": "day",   "2021-08-13-16-31-10": "day",
    "2021-08-06-16-19-00": "rain",  "2021-08-06-16-45-28": "rain",  "2021-08-06-16-59-13": "rain",
    "2021-08-13-21-18-04": "night", "2021-08-13-22-03-03": "night", "2021-08-13-22-16-02": "night",
}

# Passed to compare_route_evals.py in this order, which fixes the direction of
# every pair it forms: itertools.combinations preserves input order and the tool
# reports `right - left`.
PROMPT_ORDER = ["empty", "correct", "permuted"]

# `right - left` -> what the paper calls it.  A caption arm's own caption against
# another frame's is the content question; a real caption against no caption at
# all is the presence question; the two together are what a naive
# matched-versus-empty comparison would conflate.
COMPARISON_NAME = {
    ("empty", "correct"): "total",
    ("empty", "permuted"): "presence",
    ("correct", "permuted"): "content",
}

DIR_PATTERN = re.compile(r"^(?P<run>.+?)_test_(?P<fingerprint>[0-9a-f]+)_s(?P<step>\d+)_(?P<prompt>\w+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-root", type=Path, required=True,
                        help="Directory holding the <run>_test_<fp>_s<step>_<prompt> eval dirs.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", nargs="*", default=None,
                        help="Restrict to these run names. Default: every run found.")
    parser.add_argument("--metrics", nargs="*", default=["abs_rel", "rmse", "a1"])
    parser.add_argument("--block-lengths", nargs="*", type=int, default=[200, 1],
                        help="First value heads each row; see the module docstring.")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare-tool", type=Path,
                        default=Path(__file__).with_name("compare_route_evals.py"))
    return parser.parse_args()


def condition_of(per_sample: Path) -> str:
    """Name the condition from the drives present, not from the fingerprint."""
    seen = set()
    with per_sample.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sequence = row.get("sequence") or row.get("id", "").rsplit("_", 1)[0]
            seen.add(CONDITION_OF_SEQUENCE.get(sequence.lstrip("_")))
    seen.discard(None)
    if len(seen) == 1:
        return seen.pop()
    # A mixed or unrecognised set is not something to guess at: it means the exam
    # is not one of the three benchmark groups, and silently filing it under one
    # of them would put a wrong row in the paper's evidence table.
    return "mixed" if seen else "unknown"


def discover(eval_root: Path, runs: list[str] | None) -> dict[tuple[str, str, str], dict[str, Path]]:
    """(run, fingerprint, step) -> {prompt: per_sample.csv}"""
    groups: dict[tuple[str, str, str], dict[str, Path]] = defaultdict(dict)
    for directory in sorted(eval_root.iterdir()):
        if not directory.is_dir():
            continue
        match = DIR_PATTERN.match(directory.name)
        if not match:
            continue
        if runs and match["run"] not in runs:
            continue
        per_sample = directory / "eval_eval_per_sample.csv"
        if per_sample.is_file():
            groups[(match["run"], match["fingerprint"], match["step"])][match["prompt"]] = per_sample
    return groups


def main() -> int:
    args = parse_args()
    groups = discover(args.eval_root, args.runs)
    if not groups:
        raise SystemExit(f"no eval directories matched under {args.eval_root}")

    rows: list[dict] = []
    skipped: list[str] = []
    for (run, fingerprint, step), by_prompt in sorted(groups.items()):
        present = [p for p in PROMPT_ORDER if p in by_prompt]
        if len(present) < 2:
            skipped.append(f"{run} {fingerprint} s{step}: only {present}")
            continue
        condition = condition_of(by_prompt[present[0]])
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.json"
            command = [
                sys.executable, str(args.compare_tool),
                *[str(by_prompt[p]) for p in present],
                "--metrics", *args.metrics,
                "--block-lengths", *[str(b) for b in args.block_lengths],
                "--bootstrap", str(args.bootstrap),
                "--seed", str(args.seed),
                "--report", str(report_path),
            ]
            print(f"[run] {run} s{step} {condition} ({fingerprint}): {'/'.join(present)}", flush=True)
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0 or not report_path.is_file():
                sys.stderr.write(result.stdout + result.stderr)
                raise SystemExit(f"compare_route_evals.py failed on {run} {fingerprint} s{step}")
            report = json.loads(report_path.read_text(encoding="utf-8"))

        for comparison in report["comparisons"]:
            left = comparison["left"].rsplit("_", 1)[-1]
            right = comparison["right"].rsplit("_", 1)[-1]
            head_block = args.block_lengths[0]
            head = comparison["by_block"][str(head_block)]
            iid = comparison["by_block"].get("1")
            rows.append({
                "arm": run,
                "condition": condition,
                "step": step,
                "manifest_fingerprint": fingerprint,
                "comparison": COMPARISON_NAME.get((left, right), f"{right}-{left}"),
                "direction": f"{right} - {left}",
                "metric": comparison["metric"],
                "frames": comparison["n"],
                "left_mean": f"{comparison['left_mean']:.6f}",
                "right_mean": f"{comparison['right_mean']:.6f}",
                "mean_difference": f"{head['mean_difference']:.6f}",
                "block": head_block,
                "ci95_low": f"{head['ci95'][0]:.6f}",
                "ci95_high": f"{head['ci95'][1]:.6f}",
                "separable_from_zero": head["significant"],
                "iid_ci95_low": f"{iid['ci95'][0]:.6f}" if iid else "",
                "iid_ci95_high": f"{iid['ci95'][1]:.6f}" if iid else "",
                "iid_separable": iid["significant"] if iid else "",
                "acf_lag1": f"{comparison['autocorrelation'].get('1', float('nan')):.3f}",
                "acf_lag200": f"{comparison['autocorrelation'].get('200', float('nan')):.3f}",
                # Kept because it is what the tool computed, not because it belongs
                # in a table anyone reads: win rate does not go into deliverables.
                "right_win_rate": f"{comparison['right_win_rate']:.4f}",
            })

    if not rows:
        raise SystemExit("every group had fewer than two prompt conditions; nothing to compare")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[done] {len(rows)} rows -> {args.output}")
    separable = sum(1 for r in rows if r["separable_from_zero"])
    flipped = sum(1 for r in rows if r["separable_from_zero"] != r["iid_separable"])
    print(f"       {separable}/{len(rows)} separable from zero at block {args.block_lengths[0]}")
    print(f"       {flipped} rows where the block verdict differs from the IID one")
    for line in skipped:
        print(f"       [skipped] {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
