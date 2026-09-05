"""Re-read one of our per-frame results on the official MS2 evaluation subset.

    python tools/subset_to_official.py \
        --per-sample $IRIS_RUNS/eval/<dir>/eval_eval_per_sample.csv \
        --ms2-root $IRIS_MS2_ROOT --test-env test_day

Our evaluations score every frame of the test sequences; the published benchmark
scores every tenth (`tools/run_ms2_supdepth_baselines.official_frames`).  The two
measurements differ by under 0.0003 AbsRel -- measured, not assumed -- so this
exists to put both sides of a comparison table on one frame set rather than to
correct anything.

No inference and no GPU: the per-frame metrics are already on disk, and a macro
average over a subset is an average over a subset.

Frames our manifests dropped (six day frames whose caption generation looped)
cannot be scored and are reported rather than quietly skipped: a table footnote
saying "2331 of the official 2332" is honest, a silent 2331 is not.
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_ms2_supdepth_baselines import official_frames  # noqa: E402

METRICS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "a1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--per-sample", type=Path, required=True, nargs="+",
                        help="One or more eval_*_per_sample.csv files.")
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--test-env", choices=("test_day", "test_night", "test_rain"), required=True)
    parser.add_argument("--sample-step", type=int, default=10)
    parser.add_argument("--prefix", default="",
                        help="Column-name prefix, for the baseline CSVs whose columns "
                             "are '<align>_abs_rel'. Empty for ours.")
    parser.add_argument(
        "--write-subset",
        type=Path,
        default=None,
        help=(
            "Also write each result restricted to the official frames, as "
            "<dir>/<eval dir name>_official_<env>_per_sample.csv. Downstream paired "
            "statistics should run on these rather than on the full-frame files: the "
            "table reports the official subset, and a moving-block bootstrap over ten "
            "times the frames is ten times the work for a number nobody prints."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wanted = {f["id"] for f in official_frames(args.ms2_root, args.test_env, args.sample_step)}
    print(f"[official] {len(wanted)} frames for {args.test_env}\n")

    for path in args.per_sample:
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        if not rows:
            print(f"{path}: empty"); continue
        # Our route evaluator names the column "id"; the official one writes
        # "sample_id" for the same thing. Detecting it beats a flag nobody
        # remembers to pass, and a wrong flag would silently score no frames.
        id_column = "id" if "id" in rows[0] else "sample_id"
        if id_column not in rows[0]:
            print(f"{path}: no id column (has {sorted(rows[0])[:6]})"); continue
        have = {r[id_column] for r in rows}
        subset = [r for r in rows if r[id_column] in wanted]
        missing = wanted - have
        column = {m: f"{args.prefix}{m}" if args.prefix else m for m in METRICS}
        available = [m for m in METRICS if column[m] in rows[0]]

        def mean(sample, metric):
            return st.mean(float(r[column[metric]]) for r in sample)

        print(f"=== {path.parent.name}/{path.name}")
        print(f"    all frames      {len(rows):>6}: " + "  ".join(
            f"{m} {mean(rows, m):.5f}" for m in available))
        print(f"    official subset {len(subset):>6}: " + "  ".join(
            f"{m} {mean(subset, m):.5f}" for m in available))
        if missing:
            print(f"    ⚠️ {len(missing)} official frames are absent from this result "
                  f"(e.g. {sorted(missing)[:3]}); the subset row is over "
                  f"{len(subset)} of {len(wanted)}")
        extra = len(rows) - len(subset)
        print(f"    ({extra} frames outside the official subset were dropped)\n")

        if args.write_subset is not None:
            args.write_subset.mkdir(parents=True, exist_ok=True)
            out = args.write_subset / f"{path.parent.name}_official_{args.test_env}_per_sample.csv"
            with out.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(subset)
            print(f"    -> {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
