"""Estimate the two global inverse-depth constants on the TRAINING split.

    python tools/fit_metric_norm.py \
        --manifest $IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl \
        --ms2-root $IRIS_MS2_ROOT \
        --split-list $IRIS_MS2_ROOT/train_list.txt \
        --output docs/data/metric_norm_train_full8.json

What it computes
----------------
Over every valid lidar pixel of every training frame -- valid meaning the
official `min_depth < D < max_depth` -- it takes `q = 1/D` and reports

    q_lo = Q_0.02(q)      q_hi = Q_0.98(q)

These two numbers are then frozen and reused for every train, validation and
test frame by `tools/metric_depth_norm.MetricNorm`.

Why a histogram rather than a sample
------------------------------------
The eight official training sequences hold about 76k frames with roughly 40k
valid lidar pixels each: three billion values, which neither fits in memory nor
sorts in reasonable time.  Subsampling would work but makes the constants
depend on a seed, and these two numbers end up baked into the model's units --
they should be reproducible to the digit.

So: a fixed-bin histogram over `log10(q)`, every valid pixel counted exactly
once, quantiles read off the cumulative counts with linear interpolation inside
the containing bin.  Log spacing because `q` spans 1/80 to 1/1e-3, five orders
of magnitude, and the interesting end is the small one.  At the default 200,000
bins over [-3, 3] each bin is 3e-5 wide in log10, i.e. 7e-5 in relative terms --
four digits finer than anything downstream reads.

The split guard
---------------
`--split-list` is not optional by accident.  This project has twice shipped a
number computed on the wrong split, and constants estimated on val or test would
leak the reported split into the model's units where no later check could see
it.  Every manifest row's sequence must appear in the list, or this exits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.metric_depth_norm import MetricNorm  # noqa: E402


LOG_MIN, LOG_MAX = -3.0, 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, required=True, help="TRAIN split manifest (jsonl).")
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument(
        "--split-list",
        type=Path,
        required=True,
        help="train_list.txt. Every manifest sequence must appear here; see module docstring.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON.")
    parser.add_argument("--quantile-lo", type=float, default=0.02)
    parser.add_argument("--quantile-hi", type=float, default=0.98)
    parser.add_argument("--min-depth", type=float, default=1e-3, help="Official MS2 lower bound.")
    parser.add_argument("--max-depth", type=float, default=80.0, help="Official MS2 upper bound.")
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help=(
            "Take every Nth frame. 1 reads all of them and is what the shipped "
            "constants must be fitted with; larger values are for a quick look only "
            "and are recorded in the artifact."
        ),
    )
    parser.add_argument("--bins", type=int, default=200_000)
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--allow-partial-split",
        action="store_true",
        help=(
            "Permit a manifest covering only some of the sequences in --split-list. "
            "Off by default: a manifest short of the full training split gives "
            "constants that are not the ones the paper should quote."
        ),
    )
    return parser.parse_args()


def read_rows(manifest: Path, stride: int) -> list[dict]:
    rows = []
    with manifest.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            if index % stride:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{manifest} produced no rows at stride {stride}")
    return rows


def check_split(rows: list[dict], split_list: Path, allow_partial: bool) -> list[str]:
    allowed = {
        line.strip().lstrip("_")
        for line in split_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if not allowed:
        raise SystemExit(f"{split_list} is empty")
    seen = sorted({str(row.get("sequence", "")).lstrip("_") for row in rows})
    stray = [s for s in seen if s not in allowed]
    if stray:
        raise SystemExit(
            f"These manifest sequences are not in {split_list.name}: {stray}\n"
            "The normalisation constants may only be estimated on the training split."
        )
    missing = sorted(allowed - set(seen))
    if missing and not allow_partial:
        raise SystemExit(
            f"{split_list.name} lists {len(allowed)} sequences, the manifest covers "
            f"{len(seen)}. Missing: {missing}\n"
            "Pass --allow-partial-split only for an exploratory fit; the constants "
            "the paper quotes must come from the whole training split."
        )
    if missing:
        print(f"[warn] partial split: missing {missing}", flush=True)
    return seen


def quantile_from_histogram(counts: np.ndarray, edges: np.ndarray, q: float) -> float:
    """Linear interpolation inside the bin that contains the qth quantile."""
    total = counts.sum()
    if total <= 0:
        raise SystemExit("No valid pixels: nothing to take a quantile of")
    target = q * total
    cumulative = np.cumsum(counts)
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(index, counts.size - 1)
    below = cumulative[index - 1] if index else 0.0
    within = counts[index]
    frac = 0.0 if within <= 0 else (target - below) / within
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(edges[index] + frac * (edges[index + 1] - edges[index]))


def main() -> int:
    args = parse_args()
    if not 0.0 < args.quantile_lo < args.quantile_hi < 1.0:
        raise SystemExit("Need 0 < --quantile-lo < --quantile-hi < 1")

    rows = read_rows(args.manifest, args.frame_stride)
    sequences = check_split(rows, args.split_list, args.allow_partial_split)
    print(
        f"[fit] {len(rows):,} frames (stride {args.frame_stride}) over "
        f"{len(sequences)} sequences from {args.manifest.name}",
        flush=True,
    )

    edges = np.linspace(LOG_MIN, LOG_MAX, args.bins + 1, dtype=np.float64)
    counts = np.zeros(args.bins, dtype=np.int64)
    total_valid = 0
    q_min_observed, q_max_observed = np.inf, -np.inf
    out_of_histogram = 0
    frames_without_lidar = 0

    for position, row in enumerate(rows):
        field = row.get("thermal_depth_path") or row.get("depth_path")
        if not field:
            raise SystemExit(f"Row {row.get('id')} carries no thermal-view GT path")
        if "/thr/" not in str(field).replace("\\", "/"):
            # Conclusion 15 of the frozen document: thermal-view and RGB-view GT
            # are different exams, and mixing them here would put RGB-view
            # geometry into the thermal model's units.
            raise SystemExit(f"Row {row.get('id')}: GT {field} is not the thermal view")
        depth = np.asarray(Image.open(args.ms2_root / field), dtype=np.float32) / args.depth_scale
        valid = np.isfinite(depth) & (depth > args.min_depth) & (depth < args.max_depth)
        n_valid = int(np.count_nonzero(valid))
        if not n_valid:
            frames_without_lidar += 1
            continue
        q = 1.0 / depth[valid].astype(np.float64)
        q_min_observed = min(q_min_observed, float(q.min()))
        q_max_observed = max(q_max_observed, float(q.max()))
        log_q = np.log10(q)
        inside = (log_q >= LOG_MIN) & (log_q < LOG_MAX)
        out_of_histogram += int(np.count_nonzero(~inside))
        counts += np.bincount(
            np.clip(((log_q[inside] - LOG_MIN) / (LOG_MAX - LOG_MIN) * args.bins).astype(np.int64),
                    0, args.bins - 1),
            minlength=args.bins,
        )
        total_valid += n_valid
        if (position + 1) % 2000 == 0:
            print(f"[fit]   {position + 1:,}/{len(rows):,} frames, {total_valid:,} pixels", flush=True)

    if out_of_histogram:
        # Cannot happen for D in (1e-3, 80): q is then in (0.0125, 1000), inside
        # 10^[-3, 3]. Reported rather than assumed away.
        print(f"[warn] {out_of_histogram} pixels fell outside the histogram range", flush=True)
    if frames_without_lidar:
        print(f"[warn] {frames_without_lidar} frames had no valid lidar pixel", flush=True)

    q_lo = 10.0 ** quantile_from_histogram(counts, edges, args.quantile_lo)
    q_hi = 10.0 ** quantile_from_histogram(counts, edges, args.quantile_hi)

    norm = MetricNorm(
        q_lo=q_lo,
        q_hi=q_hi,
        quantile_lo=args.quantile_lo,
        quantile_hi=args.quantile_hi,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        source_split="train",
        source_manifest=str(args.manifest),
        valid_pixels=total_valid,
        frames=len(rows),
        frame_stride=args.frame_stride,
        depth_scale=args.depth_scale,
        sequences=sequences,
        q_min_observed=float(q_min_observed),
        q_max_observed=float(q_max_observed),
        histogram_bins=args.bins,
        notes=args.notes,
    )
    norm.save(args.output)
    print(f"[done] {norm.summary()}")
    print(f"[done] -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
