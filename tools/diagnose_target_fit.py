"""What the model fits, as opposed to how accurate it is.

Two questions get confused because both produce an AbsRel:

  * *Is the depth right?* Scored against the LiDAR returns. This is the number
    the benchmark reports and the only one comparable with published models,
    because it is the only reference that is a measurement.
  * *Did the model learn what we asked it to learn?* Scored against the
    completed pseudo-depth, which is the training target. A model that copied
    the teacher perfectly would score perfectly here and be exactly as wrong as
    the teacher, so this is a fit diagnostic and never a result.

The second question is the one that explains mechanism. The metre-space loss
term supervises only where a return exists -- twenty-six percent of pixels --
and at those pixels the target was overwritten with the measurement itself. So
the term should pull the model off the teacher's opinion precisely where the two
disagree, which predicts a worse fit to the pseudo-depth alongside a better
score against LiDAR. That is checkable, and this is what checks it.

A third column falls out for free. Restricting the pseudo-depth reference to the
pixels that carry no return scores the seventy-four percent the benchmark can
never see. It is still a fit measure -- there is no ground truth there, only the
teacher -- but it is the only view we have of the region where the metre-space
term does not reach.

No network runs here. It reads raw predictions that
``train_route_suite.py --save-raw-pred`` already wrote, so it needs neither a
GPU nor torch, and it calls the same ``evaluate_sample`` the pipeline calls
rather than reimplementing an alignment that would then drift.

    python tools/diagnose_target_fit.py \
        --raw-dir $IRIS_RUNS/eval/<run>/raw_predictions \
        --manifest $IRIS_MANIFEST_DIR/ms2_train_official8_thermalcap_...jsonl \
        --pseudo-dir $IRIS_RUNS/pseudo_gt/official_train/calibrated_pseudo_depth \
        --ms2-root $IRIS_MS2_ROOT --limit 500
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

from ms2_eval.official_protocol import evaluate_sample  # noqa: E402

METRICS = ("abs_rel", "rmse", "a1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="Directory of <id>.npy written by --save-raw-pred.")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="The manifest those predictions were made on.")
    parser.add_argument("--pseudo-dir", type=Path, required=True,
                        help="calibrated_pseudo_depth, i.e. the training target.")
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=500,
                        help="Frames to score, sampled uniformly. 0 uses all of them.")
    parser.add_argument("--align", default="ssi_disparity",
                        choices=("ssi_disparity", "ssi", "median", "none"))
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--label", default="", help="Printed with the table.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON.")
    return parser.parse_args()


def rows_from(manifest: Path, limit: int) -> list[dict]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if limit and limit < len(rows):
        step = len(rows) / limit
        rows = [rows[int(i * step)] for i in range(limit)]
    return rows


def depth_path(root: Path, row: dict) -> Path:
    for key in ("thermal_depth_path", "depth_path", "gt_path"):
        if key in row and row[key]:
            path = Path(row[key])
            return path if path.is_absolute() else root / path
    raise KeyError(f"no thermal depth path in {sorted(row)[:8]}")


def main() -> int:
    args = parse_args()
    rows = rows_from(args.manifest, args.limit)

    # Three references, three accumulators. The names say what each one answers.
    totals: dict[str, dict[str, float]] = {k: {} for k in
                                           ("lidar", "pseudo_all", "pseudo_unmeasured")}
    counts = {k: 0 for k in totals}
    skipped = {"no_prediction": 0, "no_pseudo": 0, "shape": 0, "empty_mask": 0}
    measured_fraction: list[float] = []

    for row in rows:
        pred_path = args.raw_dir / f"{row['id']}.npy"
        pseudo_path = args.pseudo_dir / f"{row['id']}.npy"
        if not pred_path.is_file():
            skipped["no_prediction"] += 1
            continue
        if not pseudo_path.is_file():
            skipped["no_pseudo"] += 1
            continue

        pred = np.load(pred_path, allow_pickle=False).astype(np.float32)
        lidar = np.asarray(Image.open(depth_path(args.ms2_root, row)),
                           dtype=np.float32) / args.depth_scale
        pseudo = np.load(pseudo_path, allow_pickle=False).astype(np.float32)

        # Refuse rather than resize. The pipeline interpolates with torch and a
        # particular corner convention; reproducing it approximately here would
        # make the LiDAR column disagree with the pipeline's own number for a
        # reason nobody would find.
        if pred.shape != lidar.shape or pseudo.shape != lidar.shape:
            skipped["shape"] += 1
            continue

        measured = (np.isfinite(lidar) & (lidar > args.min_depth)
                    & (lidar < args.max_depth))
        measured_fraction.append(float(measured.mean()))

        # evaluate_sample derives its valid mask from the reference it is given,
        # so zeroing a pixel in the reference is how a pixel is excluded. That
        # is the whole trick behind the third column.
        pseudo_unmeasured = np.where(measured, 0.0, pseudo).astype(np.float32)

        for name, reference in (("lidar", lidar),
                                ("pseudo_all", pseudo),
                                ("pseudo_unmeasured", pseudo_unmeasured)):
            try:
                scored = evaluate_sample(
                    pred, reference, align=args.align,
                    min_depth=args.min_depth, max_depth=args.max_depth,
                )
            except Exception:
                skipped["empty_mask"] += 1
                continue
            for metric in METRICS:
                totals[name][metric] = totals[name].get(metric, 0.0) + float(scored[metric])
            counts[name] += 1

    if not counts["lidar"]:
        raise SystemExit("No frame produced a measurement -- check --raw-dir and --manifest.")

    header = f"{'reference':<20}{'frames':>8}{'AbsRel':>10}{'RMSE':>10}{'delta1':>10}   what it answers"
    if args.label:
        print(f"=== {args.label} ===")
    print(header)
    print("-" * len(header))
    ANSWERS = {
        "lidar": "is the depth right (26% of pixels, the benchmark)",
        "pseudo_all": "did it learn the target (fit, not accuracy)",
        "pseudo_unmeasured": "fit where no return exists (74%, fit only)",
    }
    for name in ("lidar", "pseudo_all", "pseudo_unmeasured"):
        n = counts[name]
        if not n:
            print(f"{name:<20}{0:>8}{'  --  ':>10}{'  --  ':>10}{'  --  ':>10}   {ANSWERS[name]}")
            continue
        m = {k: totals[name][k] / n for k in METRICS}
        print(f"{name:<20}{n:>8}{m['abs_rel']:>10.4f}{m['rmse']:>10.4f}"
              f"{m['a1']:>10.4f}   {ANSWERS[name]}")

    print(f"\nmeasured pixels per frame: {100 * float(np.mean(measured_fraction)):.1f}%")
    if any(skipped.values()):
        print("skipped:", {k: v for k, v in skipped.items() if v})
    print("\nThe pseudo rows are fit, not accuracy: the reference is a model's output,")
    print("and it is the same output the model was trained on. Only the LiDAR row is")
    print("comparable with a published number.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "label": args.label,
            "align": args.align,
            "frames": counts,
            "skipped": skipped,
            "measured_fraction": float(np.mean(measured_fraction)),
            "metrics": {name: {k: totals[name][k] / counts[name] for k in METRICS}
                        for name in totals if counts[name]},
        }, indent=2), encoding="utf-8")
        print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
