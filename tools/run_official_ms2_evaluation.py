#!/usr/bin/env python3
"""Recompute exported MS2 raw predictions under the official protocol.

Reads the same manifest + raw ``predictions/*.npy`` contract as the unified
v1 evaluator, but scores them with the official BridgeMultiSpectralDepth /
AnyThermal protocol (``ms2_eval.official_protocol``). Never modifies v1
outputs; writes an independent output tree so both report tables can sit
side by side.

Smoke first, then full (WSL single-line examples):

  python tools/run_official_ms2_evaluation.py --manifest data/ms2/manifest.jsonl --data-root /mnt/e/datasets/MS2 --prediction-dir outputs/lotus_line_v2/<run>/raw_predictions --route iris-lotus --output-dir outputs/ms2_official/<run>_smoke --limit 20

  python tools/run_official_ms2_evaluation.py --manifest data/ms2/manifest.jsonl --data-root /mnt/e/datasets/MS2 --prediction-dir outputs/lotus_line_v2/<run>/raw_predictions --route iris-lotus --output-dir outputs/ms2_official/<run> --compare-per-image outputs/ms2_unified/<run>/metrics/per_image.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ms2_eval.aggregate import summarize_by_condition, summarize_rows  # noqa: E402
from ms2_eval.io import load_manifest, load_ms2_gt, sha256_file  # noqa: E402
from ms2_eval.official_protocol import (  # noqa: E402
    ALIGN_MODES,
    DEFAULT_MAX_DEPTH_M,
    DEFAULT_MIN_DEPTH_M,
    PROTOCOL_REFERENCE,
    OfficialProtocolError,
    collapse_channels,
    evaluate_sample,
    official_valid_mask,
)
from ms2_eval.resize import resize_dense_prediction  # noqa: E402


ROUTE_DEFAULT_ALIGN = {
    "iris-lotus": "ssi",
    "adapter-only": "ssi",
    "adapter+u-net": "ssi",
    "adapter-unet": "ssi",
    "rgb-frozen": "ssi",
    "rgb-unet": "ssi",
    "thermal-frozen": "ssi",
    "thermal-unet": "ssi",
    "vae-adapter": "ssi",
    "anythermal-midas": "ssi",
    "sp-dit": "median",
    "spdit": "median",
}

# official metric -> unified-v1 per_image.csv column carrying the comparable number
V1_METRIC_PAIRS = (
    ("abs_rel", "aligned_abs_rel"),
    ("rmse", "aligned_rmse_m"),
    ("rmse_log", "aligned_rmse_log"),
    ("a1", "aligned_delta1"),
)

# official metric -> lotus evaluator per_sample_metrics.csv column
LOTUS_METRIC_PAIRS = (
    ("abs_rel", "abs_relative_difference"),
    ("sq_rel", "squared_relative_difference"),
    ("rmse", "rmse_linear"),
    ("rmse_log", "rmse_log"),
    ("log10", "log10"),
    ("a1", "delta1_acc"),
    ("a2", "delta2_acc"),
    ("a3", "delta3_acc"),
)

SEQUENCE_ID_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_(.+)$")
SEQUENCE_DIR_RE = re.compile(r"^_?(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})$")


def sequence_from_sample_id(sample_id: str) -> str | None:
    match = SEQUENCE_ID_RE.match(sample_id)
    return match.group(1) if match else None


def lotus_filename_to_sample_id(filename: str) -> str | None:
    """Map lotus per-sample 'filename' (…/_<seq>/thr/img_left/pred_000000) to '<seq>_000000'."""
    parts = filename.replace("\\", "/").split("/")
    sequence = next((m.group(1) for p in parts if (m := SEQUENCE_DIR_RE.match(p))), None)
    frame = re.search(r"(\d+)$", parts[-1])
    if sequence is None or frame is None:
        return None
    return f"{sequence}_{frame.group(1)}"


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def prediction_path(directory: Path, sample_id: str) -> Path:
    direct, safe = directory / f"{sample_id}.npy", directory / f"{safe_id(sample_id)}.npy"
    if direct.is_file():
        return direct
    if safe.is_file():
        return safe
    raise FileNotFoundError(f"No raw prediction for sample {sample_id!r}; tried {direct} and {safe}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


@dataclass(frozen=True)
class RGBManifestSample:
    """RGB-route sample: input and GT are both in the RGB camera view."""
    sample_id: str
    thermal_gt_path: str  # attribute name kept for loop compatibility; holds the RGB-view GT
    sequence: str
    condition: str
    split: str


def load_rgb_manifest(path: Path, data_root: Path):
    """RGB-view loader (six-route lines a/b); ms2_eval.io stays thermal-only."""
    manifest, root = Path(path).resolve(), Path(data_root).resolve()
    samples = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, text in enumerate(handle, 1):
            if not text.strip():
                continue
            row = json.loads(text)
            depth = row.get("rgb_depth_path")
            if not depth or "/rgb/" not in str(depth).replace("\\", "/"):
                raise ValueError(f"Manifest line {line_number} lacks RGB-view GT (rgb_depth_path)")
            depth_path = Path(depth)
            samples.append(RGBManifestSample(
                sample_id=str(row.get("id") or row.get("sample_id")),
                thermal_gt_path=str(depth_path if depth_path.is_absolute() else root / depth_path),
                sequence=str(row.get("sequence") or "unknown"),
                condition=str(row.get("condition") or "unknown").lower(),
                split=str(row.get("split") or "unknown"),
            ))
    if not samples:
        raise ValueError(f"Manifest contains no samples: {manifest}")
    return samples, {"path": str(manifest), "sha256": sha256_file(manifest), "sample_count": len(samples)}


def summarize_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return {name: summarize_rows(items) for name, items in sorted(grouped.items())}


def compare_with_csv(rows: list[dict[str, Any]], other_csv: Path, *, label: str,
                     metric_pairs, id_column: str | None = "sample_id",
                     id_from_row=None) -> dict[str, Any]:
    """Pair official per-image rows with another evaluator's per-image CSV."""
    other_rows: dict[str, dict[str, str]] = {}
    with other_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row.get(id_column) if id_from_row is None else id_from_row(row)
            if key:
                other_rows[key] = row
    official_rows = {row["sample_id"]: row for row in rows}
    shared = sorted(set(other_rows) & set(official_rows))
    if not shared:
        raise ValueError(f"No shared sample IDs between official run and {other_csv}")
    comparison: dict[str, Any] = {
        "compared_against": label,
        "per_image_csv": str(other_csv),
        "paired_sample_count": len(shared),
        "official_only_count": len(official_rows) - len(shared),
        "other_only_count": len(other_rows) - len(shared),
        "metrics": {},
    }
    for official_name, other_name in metric_pairs:
        pairs = []
        for sample_id in shared:
            official_value = official_rows[sample_id].get(official_name)
            try:
                other_value = float(other_rows[sample_id].get(other_name, ""))
            except (TypeError, ValueError):
                continue
            if official_value is None or not np.isfinite(official_value) or not np.isfinite(other_value):
                continue
            pairs.append((float(official_value), other_value))
        if not pairs:
            continue
        official_values = np.asarray([p[0] for p in pairs], np.float64)
        other_values = np.asarray([p[1] for p in pairs], np.float64)
        comparison["metrics"][official_name] = {
            "other_column": other_name,
            "official_mean": float(official_values.mean()),
            "other_mean": float(other_values.mean()),
            "mean_difference_official_minus_other": float((official_values - other_values).mean()),
            "paired_count": int(official_values.size),
        }
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path, help="JSONL manifest (same file the v1 evaluator used)")
    parser.add_argument("--data-root", required=True, type=Path, help="Dataset root that manifest paths resolve against")
    parser.add_argument("--prediction-dir", required=True, type=Path, help="Directory of RAW route predictions (*.npy by sample id)")
    parser.add_argument("--route", required=True, choices=sorted(ROUTE_DEFAULT_ALIGN), help="Route name; picks the official alignment path")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--align", default="auto", choices=("auto", *ALIGN_MODES),
                        help="auto: ssi for relative routes (official MiDaS/DPT path), median for sp-dit (official metric path)")
    parser.add_argument("--pred-transform", default="raw", choices=("raw", "inverse"),
                        help="raw: feed network output as-is (official-faithful). inverse: 1/max(x,eps) before alignment (diagnostic only)")
    parser.add_argument("--min-depth", type=float, default=DEFAULT_MIN_DEPTH_M)
    parser.add_argument("--max-depth", type=float, default=DEFAULT_MAX_DEPTH_M)
    parser.add_argument("--depth-scale", type=float, default=256.0, help="uint16 GT units per metre")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="Floor used only by --pred-transform inverse")
    parser.add_argument("--limit", type=int, default=0,
                        help="SMOKE ONLY: evaluate the first N manifest samples; summary is marked and must not be reported")
    parser.add_argument("--strict-manifest", action="store_true",
                        help="Refuse legacy 'depth_path' manifests (default allows them, matching build_ms2_manifest.py output)")
    parser.add_argument("--gt-shift-x", type=int, default=0,
                        help="DIAGNOSTIC ONLY (default 0 = untouched official behaviour). "
                             "Translate the GT map horizontally by this many pixels before "
                             "scoring; vacated columns become invalid. Used to test whether a "
                             "dataset's depth map is registered to the camera the prediction "
                             "lives in -- a metric minimum away from 0 means it is not. "
                             "Never use a non-zero value for a reportable number.")
    parser.add_argument("--gt-shift-mode", choices=("shift", "mask"), default="shift",
                        help="Companion to --gt-shift-x. 'mask' is the control arm: it "
                             "drops the same edge columns without translating, isolating "
                             "the alignment effect from the edge-column effect.")
    parser.add_argument("--min-gt-valid-fraction", type=float, default=0.0,
                        help=(
                            "Skip frames whose valid-GT pixels fall below this fraction of the "
                            "frame. RGBDT500 has a handful of near-empty depth maps whose "
                            "per-image metrics are pure noise; MS2 is unaffected at 0.0."
                        ))
    parser.add_argument("--gt-view", default="thermal", choices=("thermal", "rgb"),
                        help="GT camera view: thermal (default) or rgb for six-route lines a/b. "
                             "Never mix views in one table without a labelled GT-view column.")
    parser.add_argument("--compare-per-image", type=Path, default=None,
                        help="Optional unified-v1 metrics/per_image.csv to join for a side-by-side comparison table")
    parser.add_argument("--compare-lotus-per-sample", type=Path, default=None,
                        help="Optional lotus-evaluator per_sample_metrics.csv (e.g. from run_ms2_lotus_trained_official.py) to join")
    return parser.parse_args()


def shift_gt_horizontally(gt: np.ndarray, dx: int, mode: str = "shift") -> np.ndarray:
    """Translate GT by dx columns, filling vacated ones with 0 (= invalid).

    np.roll would wrap the far edge back in and quietly score garbage against
    real predictions, so the shift is done with an explicit zero fill.

    ``mode="mask"`` is the control arm: it invalidates exactly the columns the
    shift would vacate but leaves the content in place. A shift only proves
    misregistration if it beats this control -- otherwise the apparent gain is
    just the metric losing the (harder) edge columns.
    """
    out = np.zeros_like(gt)
    if dx > 0:
        if mode == "shift":
            out[:, dx:] = gt[:, :-dx]
        else:
            out[:, dx:] = gt[:, dx:]
    elif dx < 0:
        if mode == "shift":
            out[:, :dx] = gt[:, -dx:]
        else:
            out[:, :dx] = gt[:, :dx]
    else:
        out = gt
    return out


def main() -> int:
    args = parse_args()
    align = ROUTE_DEFAULT_ALIGN[args.route] if args.align == "auto" else args.align
    if args.pred_transform == "inverse" and align != "ssi":
        raise SystemExit("--pred-transform inverse is only meaningful with the plain ssi path")

    if args.gt_view == "rgb":
        samples, manifest_info = load_rgb_manifest(args.manifest, args.data_root)
    else:
        samples, manifest_info = load_manifest(args.manifest, args.data_root,
                                               allow_legacy_depth_path=not args.strict_manifest)
    smoke = args.limit > 0 and args.limit < len(samples)
    if smoke:
        print(f"[SMOKE] Evaluating only the first {args.limit} of {len(samples)} samples; "
              "these numbers are for pipeline validation, never for reporting.")
        samples = samples[: args.limit]

    pred_dir = args.prediction_dir.resolve()
    output = args.output_dir.resolve()
    (output / "metrics").mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    resolution_log: list[dict[str, Any]] = []
    skipped_sparse: list[dict[str, Any]] = []
    for sample in samples:
        raw = np.load(prediction_path(pred_dir, sample.sample_id), allow_pickle=False)
        pred = collapse_channels(raw)
        if args.pred_transform == "inverse":
            pred = (1.0 / np.maximum(pred, args.epsilon)).astype(np.float32)
        _, gt = load_ms2_gt(sample.thermal_gt_path, args.depth_scale)
        if args.gt_shift_x:
            gt = shift_gt_horizontally(gt, args.gt_shift_x, args.gt_shift_mode)
        if args.min_gt_valid_fraction > 0:
            fraction = float(official_valid_mask(gt, args.min_depth, args.max_depth).mean())
            if fraction < args.min_gt_valid_fraction:
                skipped_sparse.append({"sample_id": sample.sample_id, "gt_valid_fraction": round(fraction, 4)})
                continue
        native = resize_dense_prediction(pred, tuple(gt.shape))  # official: bilinear to GT resolution
        try:
            metrics = evaluate_sample(native, gt, align=align,
                                      min_depth=args.min_depth, max_depth=args.max_depth)
        except OfficialProtocolError as error:
            raise SystemExit(f"Sample {sample.sample_id!r} failed the official protocol: {error}") from error
        sequence = sample.sequence
        if sequence in ("unknown", "", None):
            sequence = sequence_from_sample_id(sample.sample_id) or "unknown"
        rows.append({"sample_id": sample.sample_id, "sequence": sequence,
                     "condition": sample.condition, "split": sample.split,
                     "route": args.route, "pred_transform": args.pred_transform, **metrics})
        resolution_log.append({"sample_id": sample.sample_id, "raw_prediction_hw": list(pred.shape),
                               "gt_hw": list(gt.shape)})

    fields = list(rows[0].keys())
    with (output / "metrics" / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "protocol": PROTOCOL_REFERENCE,
        "gt_view": args.gt_view,
        "gt_shift_x": args.gt_shift_x,
        "gt_shift_mode": args.gt_shift_mode,
        "align_mode": align,
        "pred_transform": args.pred_transform,
        "min_depth_m": args.min_depth,
        "max_depth_m": args.max_depth,
        "depth_scale_uint16_per_metre": args.depth_scale,
        "route": args.route,
        "smoke_limit": args.limit if smoke else None,
        "reportable": (not smoke) and args.gt_shift_x == 0,
        "sample_count": len(rows),
        "min_gt_valid_fraction": args.min_gt_valid_fraction,
        "skipped_sparse_gt": {"count": len(skipped_sparse), "samples": skipped_sparse[:20]},
        "manifest": manifest_info,
        "image_wise": summarize_rows(rows),
    }
    write_json(output / "metrics" / "summary.json", summary)
    write_json(output / "metrics" / "summary_by_condition.json", summarize_by_condition(rows))
    write_json(output / "metrics" / "summary_by_sequence.json", summarize_by_key(rows, "sequence"))
    write_json(output / "logs" / "resolutions.json", resolution_log)
    write_json(output / "run_metadata.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(),
        "argv": sys.argv, "prediction_dir": str(pred_dir), "output_dir": str(output),
    })

    if args.compare_per_image is not None:
        comparison = compare_with_csv(rows, args.compare_per_image.resolve(),
                                      label="ms2_eval-unified-v1", metric_pairs=V1_METRIC_PAIRS)
        write_json(output / "metrics" / "comparison_v1_vs_official.json", comparison)

    if args.compare_lotus_per_sample is not None:
        comparison = compare_with_csv(
            rows, args.compare_lotus_per_sample.resolve(),
            label="lotus-least_square_disparity", metric_pairs=LOTUS_METRIC_PAIRS,
            id_column=None, id_from_row=lambda row: lotus_filename_to_sample_id(row.get("filename", "")))
        write_json(output / "metrics" / "comparison_lotus_vs_official.json", comparison)

    stats = summary["image_wise"]["statistics"]
    names = [n for n in ("abs_diff", "abs_rel", "sq_rel", "log10", "rmse", "rmse_log", "a1", "a2", "a3") if n in stats]
    print(f"\nOfficial protocol ({align}, {args.pred_transform}) over {len(rows)} images"
          + (f"  [SMOKE limit={args.limit}]" if smoke else ""))
    print("  " + ("{:>9} | " * len(names)).format(*names))
    print("  " + ("{:9.4f} | " * len(names)).format(*(stats[n]["mean"] for n in names)))
    print(f"\nOutputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
