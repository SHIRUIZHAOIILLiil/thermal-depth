"""Stratified error for models we can only see as saved predictions.

`analyze_route_regions.py` answers "where does the error live" for our own
route-suite checkpoints by running their forward path.  Task 4 asks the same
question about a model we did not train -- the rebuilt AnyThermal MiDaS teacher
(AbsRel 0.0929 on our val) -- to find out whether the sky band that our thermal
routes get wrong is wrong for the fully supervised baseline too.  That model is
not a RouteModel, so it enters through the same contract the official evaluator
uses: a manifest plus a directory of raw `*.npy` predictions.

The strata are imported from `analyze_route_regions`, not re-implemented, so the
depth bands, the row thirds and the boundary test are identical and the two
tools' tables can be read side by side.

Alignment mirrors `ms2_eval.official_protocol.evaluate_sample` branch for branch
(the protocol module is frozen and does not hand back the aligned map).  To make
sure the mirror never drifts, every frame is also scored by the real
`evaluate_sample` and the two AbsRel values are compared; a disagreement above
--tolerance aborts the run.

    python tools/analyze_prediction_regions.py \
        --manifest outputs/lotus_line_v2/anythermal_midas_val_full/selected_manifest.jsonl \
        --data-root /mnt/e/dataset/ms2 \
        --predictions outputs/lotus_line_v2/anythermal_midas_val_full/raw_predictions:anythermal \
        --route anythermal-midas --output-dir outputs/route_suite/anythermal_regions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "lotus", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ms2_eval.official_protocol import (  # noqa: E402
    ALIGN_MODES,
    collapse_channels,
    evaluate_sample,
    fit_scale_shift,
    median_scale_ratio,
    official_valid_mask,
)
from ms2_eval.resize import resize_dense_prediction  # noqa: E402
from analyze_route_regions import paired_difference, strata_for  # noqa: E402
from run_official_ms2_evaluation import ROUTE_DEFAULT_ALIGN, prediction_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        metavar="DIR[:LABEL[:ALIGN]]",
        help="Raw *.npy prediction directory, optionally labelled. Repeat to compare models. "
        "A third field overrides --align for that directory alone, which is how a "
        "disparity-space route and a depth-space one enter the same paired table.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--route",
        default=None,
        choices=sorted(ROUTE_DEFAULT_ALIGN),
        help="Picks the official alignment for these predictions (same table as the evaluator).",
    )
    parser.add_argument(
        "--align",
        default="auto",
        choices=("auto", *ALIGN_MODES),
        help="auto = look --route up in the official table. The default for every "
        "prediction dir that does not name its own.",
    )
    parser.add_argument(
        "--gt-view",
        default="thermal",
        choices=("thermal", "rgb"),
        help="Conclusion 15: score a route against the GT of the view it sees.",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="0 = all frames.")
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Max allowed AbsRel gap between this tool's aligned map and evaluate_sample.",
    )
    return parser.parse_args()


def parse_prediction_sources(values: list[str], default_align: str) -> list[tuple[str, Path, str]]:
    sources = []
    for value in values:
        rest, align = value, default_align
        # DIR:LABEL:ALIGN -- the alignment names are a closed set, so a trailing
        # field that is one of them can only have been meant as the alignment
        head, _, tail = value.rpartition(":")
        if tail in ALIGN_MODES and ":" in head[2:]:
            rest, align = head, tail
        # a Windows drive letter is not a label separator
        if ":" in rest[2:]:
            head, _, label = rest.rpartition(":")
            sources.append((label, Path(head), align))
        else:
            sources.append((Path(rest).name, Path(rest), align))
    labels = [label for label, _, _ in sources]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"Prediction labels must be unique, got {labels}")
    for label, directory, _ in sources:
        if not directory.is_dir():
            raise SystemExit(f"{label}: not a directory: {directory}")
    return sources


def read_rows(manifest: Path, data_root: Path, gt_view: str, stride: int, limit: int) -> list[dict]:
    depth_field = "rgb_depth_path" if gt_view == "rgb" else "thermal_depth_path"
    rows = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            relative = row.get(depth_field) or row.get("depth_path")
            if not relative:
                raise SystemExit(f"Row {row.get('id')} has no {gt_view}-view GT path")
            view = "rgb" if gt_view == "rgb" else "thr"
            if row.get("rgb_depth_path") and f"/{view}/" not in str(relative).replace("\\", "/"):
                raise SystemExit(f"Row {row.get('id')}: GT {relative} is not the {view} view")
            rows.append({"id": str(row["id"]), "depth_path": data_root / relative})
    rows = rows[:: max(1, stride)]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(f"{manifest} produced no rows")
    return rows


def load_gt(path: Path, depth_scale: float) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path), dtype=np.float32) / depth_scale


def align_prediction(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, align: str) -> np.ndarray:
    """Mirror of evaluate_sample's alignment branches, returning the aligned map."""
    if align == "ssi":
        scale, shift = fit_scale_shift(pred, gt, valid)
        return pred.astype(np.float64) * scale + shift
    if align == "ssi_disparity":
        gt_disparity = np.zeros_like(gt, np.float64)
        gt_disparity[valid] = 1.0 / gt[valid].astype(np.float64)
        scale, shift = fit_scale_shift(pred, gt_disparity.astype(np.float32), valid)
        return 1.0 / np.clip(pred.astype(np.float64) * scale + shift, 1e-3, None)
    if align == "median":
        return pred.astype(np.float64) * median_scale_ratio(pred, gt, valid)
    return pred.astype(np.float64)


def score_predictions(label: str, directory: Path, rows: list[dict], align: str, args) -> dict:
    per_frame: dict[str, list[float]] = {}
    pixel_totals: dict[str, list[float]] = {}
    official_abs_rel: list[float] = []
    worst_gap = 0.0
    for index, row in enumerate(rows):
        raw = collapse_channels(np.load(prediction_path(directory, row["id"]), allow_pickle=False))
        gt = load_gt(row["depth_path"], args.depth_scale)
        pred = resize_dense_prediction(raw, tuple(gt.shape))

        valid = official_valid_mask(gt, args.min_depth, args.max_depth)
        if not valid.any():
            continue
        aligned = np.clip(
            align_prediction(pred, gt, valid, align), args.min_depth, args.max_depth
        )
        error = np.abs(aligned - gt) / np.maximum(gt, 1e-6)

        official = evaluate_sample(
            pred, gt, align=align, min_depth=args.min_depth, max_depth=args.max_depth
        )
        official_abs_rel.append(float(official["abs_rel"]))
        gap = abs(float(error[valid].mean()) - float(official["abs_rel"]))
        worst_gap = max(worst_gap, gap)
        if gap > args.tolerance:
            raise SystemExit(
                f"{label}/{row['id']}: stratified AbsRel {error[valid].mean():.8f} disagrees with "
                f"the official evaluator {official['abs_rel']:.8f} (gap {gap:.2e}); the alignment "
                "mirror has drifted from ms2_eval.official_protocol."
            )

        for name, mask in strata_for(gt, valid).items():
            if not mask.any():
                continue
            per_frame.setdefault(name, []).append(float(error[mask].mean()))
            pixel_totals.setdefault(name, []).append(float(mask.sum()))
        if (index + 1) % 500 == 0:
            print(f"    {index + 1}/{len(rows)}", flush=True)

    return {
        "per_frame": per_frame,
        "pixel_counts": {name: float(np.sum(values)) for name, values in pixel_totals.items()},
        "official_abs_rel": float(np.mean(official_abs_rel)),
        "worst_alignment_gap": worst_gap,
        "frames": len(official_abs_rel),
    }


def main() -> None:
    args = parse_args()
    if args.align == "auto":
        if args.route is None:
            raise SystemExit("--align auto needs --route to look the alignment up.")
        default_align = ROUTE_DEFAULT_ALIGN[args.route]
    else:
        default_align = args.align
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources = parse_prediction_sources(args.predictions, default_align)
    rows = read_rows(args.manifest, args.data_root, args.gt_view, args.stride, args.limit)
    print(f"[data] {len(rows)} frames, GT view {args.gt_view}", flush=True)
    if len({align for _, _, align in sources}) > 1:
        # Everything downstream of alignment is shared, so the table stays readable
        # only if the one step that is not shared is stated where the numbers are.
        print("[align] mixed -- each model removes its own scale/shift ambiguity:", flush=True)
        for label, _, align in sources:
            print(f"        {label}: {align}", flush=True)

    results = {}
    for label, directory, align in sources:
        print(f"[scan] {label} <- {directory} (align {align})", flush=True)
        results[label] = score_predictions(label, directory, rows, align, args)
        print(
            f"       official AbsRel {results[label]['official_abs_rel']:.4f} "
            f"({results[label]['frames']} frames, alignment gap <= "
            f"{results[label]['worst_alignment_gap']:.1e})",
            flush=True,
        )

    names = list(results)
    strata = sorted(results[names[0]]["per_frame"])
    rng = np.random.default_rng(args.seed)

    print(f"\n{'stratum':24s} " + "".join(f"{name[:14]:>16s}" for name in names))
    print("-" * (24 + 16 * len(names)))
    for stratum in strata:
        cells = "".join(f"{np.mean(results[name]['per_frame'][stratum]):>16.4f}" for name in names)
        print(f"{stratum:24s} {cells}")

    report: dict = {
        "strata": {},
        "predictions": {label: str(directory) for label, directory, _ in sources},
        "manifest": str(args.manifest),
        "frames": len(rows),
        "gt_view": args.gt_view,
        "align": {label: align for label, _, align in sources},
        "official_abs_rel": {name: results[name]["official_abs_rel"] for name in names},
    }
    for stratum in strata:
        entry = {
            "means": {name: float(np.mean(results[name]["per_frame"][stratum])) for name in names},
            "pixels": results[names[0]]["pixel_counts"].get(stratum),
            "paired": {},
        }
        for i in range(len(names) - 1):
            for j in range(i + 1, len(names)):
                entry["paired"][f"{names[j]} - {names[i]}"] = paired_difference(
                    results[names[i]]["per_frame"][stratum],
                    results[names[j]]["per_frame"][stratum],
                    args.bootstrap,
                    rng,
                )
        report["strata"][stratum] = entry

    if len(names) >= 2:
        print("\npaired differences (AbsRel, negative = the right-hand model is better)")
        for stratum in strata:
            for key, stats in report["strata"][stratum]["paired"].items():
                marker = "*" if stats["significant"] else " "
                print(
                    f"  {stratum:24s} {key:34s} {stats['mean_difference']:+.5f}{marker} "
                    f"CI[{stats['ci95'][0]:+.5f},{stats['ci95'][1]:+.5f}] win {stats['right_win_rate']:.1%}"
                )

    path = args.output_dir / "region_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
