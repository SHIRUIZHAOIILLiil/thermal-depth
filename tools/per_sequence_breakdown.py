"""Break a pooled evaluation back apart, one row per test sequence.

The condition-level numbers pool sequences that are not equally hard. The daytime
set alone mixes a morning drive with two afternoon ones, and MS2's own sequence
table gives each drive a thermal contrast that spans a factor of three across the
test split. A pooled AbsRel cannot say whether a condition is hard or whether one
drive inside it is.

Nothing is re-run: `train_route_suite.py --eval-checkpoint` already writes
`eval_eval_per_sample.csv`, one row per frame, carrying the `sequence` column. This
regroups those rows.

Aggregation matches the protocol: the official evaluator macro-averages per image,
unweighted by pixel count, so the per-sequence figure is the plain mean over that
sequence's frames and the pooled figure is the frame-count-weighted mean of the
per-sequence ones. That identity is checked rather than assumed -- pass
`--verify-against` a pooled CSV and any drift over 1e-9 is reported.

    python tools/per_sequence_breakdown.py --eval-root $IRIS_RUNS/eval \\
        --output per_sequence_20260822.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

TAG_TEST = re.compile(r"^(?P<arm>.*)_test_(?P<fp>[0-9a-f]{8})_s(?P<step>\d+)_(?P<prompt>[a-z]+)$")
METRICS = ("abs_rel", "rmse", "a1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-against", type=Path, default=None,
                        help="Pooled grid CSV; per-sequence means are recombined and compared.")
    return parser.parse_args()


def load_per_sample(path: Path) -> dict[str, list[dict[str, float]]]:
    by_sequence: dict[str, list[dict[str, float]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sequence = row.get("sequence") or ""
            if not sequence:
                # Fall back to the id prefix; ids look like <sequence>_<frame>.
                sequence = (row.get("id") or "").rsplit("_", 1)[0]
            values = {}
            for metric in METRICS:
                try:
                    values[metric] = float(row[metric])
                except (KeyError, TypeError, ValueError):
                    continue
            if len(values) == len(METRICS):
                by_sequence.setdefault(sequence, []).append(values)
    return by_sequence


def main() -> int:
    args = parse_args()
    rows = []
    for path in sorted(args.eval_root.glob("*/eval_eval_per_sample.csv")):
        match = TAG_TEST.match(path.parent.name)
        if not match:
            continue
        for sequence, frames in sorted(load_per_sample(path).items()):
            row = {
                "arm": match["arm"], "step": match["step"], "prompt": match["prompt"],
                "manifest_fp": match["fp"], "sequence": sequence, "frames": len(frames),
            }
            for metric in METRICS:
                row[metric] = sum(f[metric] for f in frames) / len(frames)
            row["source"] = str(path)
            rows.append(row)

    if not rows:
        raise SystemExit(f"no per-sample CSVs under {args.eval_root}")

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] {len(rows)} rows -> {args.output}")

    if args.verify_against:
        # Recombining the per-sequence means must reproduce the pooled number. If it
        # does not, the grouping is wrong and every per-sequence figure is suspect.
        #
        # The key must include the manifest fingerprint. Without it, one arm's day,
        # rain and night sequences collapse into a single group and get compared
        # against whichever pooled row happened to be read last -- which is how this
        # check first reported a drift of 0.26 while the grouping was in fact correct.
        pooled = {}
        with args.verify_against.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if "manifest_fp" not in (reader.fieldnames or []):
                raise SystemExit(
                    f"{args.verify_against} has no manifest_fp column; regenerate it "
                    f"with tools/collect_eval_grid.py"
                )
            for row in reader:
                if row.get("split") != "test":
                    continue
                pooled[(row["arm"], row["step"], row["prompt"], row["manifest_fp"])] = row
        worst = 0.0
        checked = 0
        groups: dict[tuple[str, str, str, str], list[dict]] = {}
        for row in rows:
            groups.setdefault(
                (row["arm"], row["step"], row["prompt"], row["manifest_fp"]), []
            ).append(row)
        for key, group in groups.items():
            if key not in pooled:
                continue
            total = sum(g["frames"] for g in group)
            for metric in METRICS:
                recombined = sum(g[metric] * g["frames"] for g in group) / total
                drift = abs(recombined - float(pooled[key][metric]))
                worst = max(worst, drift)
                checked += 1
        print(f"[verify] {checked} comparisons, worst drift {worst:.3e}")
        if worst > 1e-9:
            raise SystemExit("per-sequence means do not recombine to the pooled value")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
