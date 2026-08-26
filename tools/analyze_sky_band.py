"""Sky-band readout over a whole split, from saved raw predictions.

`export_qualitative.py` reports `top_third_pred_median_m` while it draws
figures, which capped every sky claim we have made at five frames. This tool
does the same measurement with no GPU and no model: it reads the `.npy`
predictions an evaluation already wrote, aligns each one the way the official
protocol does, and reports the band statistic over every frame in the manifest.

The band statistic deliberately carries **no valid mask**. Sky has no lidar
return, so every masked metric is blind there; the median of the aligned
prediction over the top rows is the only number that sees it at all. It cannot
separate "sky judged near" from "near object judged correctly" -- a branch
overhanging the road belongs in that band -- so read it as a same-frame
comparison between models, never as an error.

Predictions are paired per frame, so the difference between two models comes
with a bootstrap interval and a win rate rather than two bare averages.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ms2_eval.official_protocol import fit_scale_shift, official_valid_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--preds",
        nargs="+",
        required=True,
        metavar="DIR[:LABEL]",
        help="Directories of <id>.npy raw predictions. The first one is the reference "
             "every other is paired against.",
    )
    parser.add_argument(
        "--sky-mask-dir",
        type=Path,
        default=None,
        help=(
            "Directory of <id>.png sky masks from tools/build_sky_masks.py. Given one, "
            "the region measured is the mask instead of the top rows -- which is the "
            "only way the number means 'sky'. On MS2 daytime the top 32 rows are not "
            "mostly sky: where lidar returns in that band at all, its median is 12.5 m, "
            "i.e. buildings and overhanging branches. A band reading cannot separate "
            "'sky judged near' from 'near object judged correctly'; a mask reading can."
        ),
    )
    parser.add_argument("--band", type=int, default=32,
                        help="Rows counted from the top. 32 is ~17-21 degrees above the "
                             "optical axis on MS2 thermal, i.e. sky unless something tall "
                             "is in frame; 85 is the top third used by earlier reports.")
    parser.add_argument("--gt-view", default="thermal", choices=("thermal", "rgb"))
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--near-threshold", type=float, default=10.0,
                        help="Predictions below this count as 'called near'.")
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def region_label(args) -> str:
    """What was actually measured. A header saying "top 32 rows" while a mask is
    in use is the worst kind of wrong label: the numbers under it are right, so
    nothing looks broken, and the line is what ends up copied into a caption."""
    if args.sky_mask_dir is not None:
        return f"sky mask from {args.sky_mask_dir.name}"
    return f"top {args.band} rows"


def read_manifest(path: Path, gt_view: str) -> list[dict]:
    key = "rgb_depth_path" if gt_view == "rgb" else "thermal_depth_path"
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            depth = row.get(key) or row.get("depth_path")
            if not depth:
                raise ValueError(f"Row {row.get('id')} has no {key}")
            marker = "/rgb/" if gt_view == "rgb" else "/thr/"
            if marker not in str(depth).replace("\\", "/"):
                raise ValueError(
                    f"Row {row.get('id')}: {depth} is not a {gt_view}-view map. "
                    "Mixing GT views is a protocol violation."
                )
            rows.append({"id": row["id"], "depth_path": depth})
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


def align_to_metric_depth(prediction, gt, min_depth, max_depth):
    """The official ssi_disparity path: fit in disparity space, invert, clamp.

    Kept byte-identical to `export_qualitative.py::align_to_metric_depth` so the
    numbers here and the numbers on the figures are the same quantity.
    """
    valid = official_valid_mask(gt, min_depth, max_depth)
    gt_disparity = np.zeros_like(gt, np.float64)
    gt_disparity[valid] = 1.0 / gt[valid].astype(np.float64)
    scale, shift = fit_scale_shift(prediction, gt_disparity.astype(np.float32), valid)
    aligned = 1.0 / np.clip(prediction.astype(np.float64) * scale + shift, 1e-3, None)
    return np.clip(aligned, min_depth, max_depth), valid


def resize_to(prediction: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear, matching what the evaluator does before it scores."""
    if prediction.shape == shape:
        return prediction
    source = Image.fromarray(prediction.astype(np.float32), mode="F")
    return np.asarray(source.resize((shape[1], shape[0]), Image.BILINEAR), dtype=np.float32)


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(samples)
    for i in range(samples):
        means[i] = values[rng.integers(0, n, n)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest, args.gt_view)

    specs = []
    for entry in args.preds:
        directory, _, label = entry.partition(":")
        path = Path(directory)
        if not path.is_dir():
            raise SystemExit(f"Not a directory: {path}")
        specs.append((path, label or path.parent.name))

    per_frame: dict[str, dict[str, float]] = {label: {} for _, label in specs}
    gt_band: dict[str, float] = {}
    gt_band_coverage: dict[str, float] = {}
    missing: list[str] = []

    for index, row in enumerate(rows):
        gt = np.asarray(Image.open(args.data_root / row["depth_path"]), dtype=np.float32)
        gt = gt / args.depth_scale
        if args.sky_mask_dir is not None:
            mask_path = args.sky_mask_dir / f"{row['id']}.png"
            if not mask_path.exists():
                missing.append(f"skymask:{row['id']}")
                continue
            region = np.asarray(Image.open(mask_path)) > 127
            if region.shape != gt.shape:
                raise SystemExit(
                    f"{row['id']}: sky mask {region.shape} vs GT {gt.shape}"
                )
            if not region.any():                    # a frame with no sky is normal
                continue
        else:
            region = np.zeros(gt.shape, dtype=bool)
            region[: args.band] = True
        band = region
        for path, label in specs:
            npy = path / f"{row['id']}.npy"
            if not npy.exists():
                missing.append(f"{label}:{row['id']}")
                continue
            prediction = np.load(npy, allow_pickle=False).astype(np.float32)
            if prediction.ndim == 3:
                prediction = prediction.mean(axis=0 if prediction.shape[0] <= 4 else -1)
            prediction = resize_to(prediction, gt.shape)
            aligned, valid = align_to_metric_depth(
                prediction, gt, args.min_depth, args.max_depth
            )
            per_frame[label][row["id"]] = {
                "band_median_m": float(np.median(aligned[band])),
                "band_p90_m": float(np.percentile(aligned[band], 90)),
                "band_frac_near": float((aligned[band] < args.near_threshold).mean()),
            }
            if row["id"] not in gt_band:
                gt_valid = gt[band][valid[band]]
                gt_band[row["id"]] = float(np.median(gt_valid)) if gt_valid.size else float("nan")
                gt_band_coverage[row["id"]] = float(valid[band].mean())
        if (index + 1) % 250 == 0:
            print(f"[{index + 1}/{len(rows)}]", flush=True)

    if missing:
        print(f"!! {len(missing)} predictions missing, e.g. {missing[:3]}", flush=True)

    shared = sorted(set.intersection(*(set(per_frame[label]) for _, label in specs)))
    if not shared:
        raise SystemExit("No frame is present in every prediction directory.")
    print(f"\n{len(shared)} frames scored in all {len(specs)} directories "
          f"(region = {region_label(args)}, no valid mask)\n")

    header = f"{'model':>28} {'band median':>12} {'band p90':>10} {'frac < ' + str(args.near_threshold) + 'm':>12}"
    print(header)
    print("-" * len(header))
    summary = {}
    for _, label in specs:
        med = np.array([per_frame[label][i]["band_median_m"] for i in shared])
        p90 = np.array([per_frame[label][i]["band_p90_m"] for i in shared])
        near = np.array([per_frame[label][i]["band_frac_near"] for i in shared])
        summary[label] = {"band_median_m": float(np.median(med)),
                          "band_median_mean": float(med.mean()),
                          "band_p90_m": float(np.median(p90)),
                          "frac_near": float(near.mean())}
        print(f"{label:>28} {np.median(med):11.2f}m {np.median(p90):9.2f}m {near.mean()*100:11.1f}%")
    gt_values = np.array([gt_band[i] for i in shared])
    finite = gt_values[np.isfinite(gt_values)]
    print(f"{'GT (returns only)':>28} {np.median(finite):11.2f}m {'':>10} "
          f"   coverage {np.mean([gt_band_coverage[i] for i in shared])*100:.2f}%")

    reference = specs[0][1]
    print(f"\npaired against {reference} (positive = judged further away)\n")
    print(f"{'model':>28} {'delta':>10} {'95% CI':>24} {'win rate':>9}")
    comparisons = {}
    for _, label in specs[1:]:
        diff = np.array([per_frame[label][i]["band_median_m"]
                         - per_frame[reference][i]["band_median_m"] for i in shared])
        lo, hi = bootstrap_ci(diff, args.bootstrap, args.seed)
        win = float((diff > 0).mean())
        comparisons[label] = {"delta_m": float(diff.mean()), "ci95": [lo, hi],
                              "further_rate": win, "n": len(shared)}
        star = "*" if lo * hi > 0 else " "
        print(f"{label:>28} {diff.mean():+9.2f}m{star} [{lo:+8.2f}, {hi:+8.2f}] {win*100:8.1f}%")

    (args.output_dir / "sky_band_summary.json").write_text(
        json.dumps({"band_rows": args.band, "frames": len(shared),
                    "gt_view": args.gt_view, "manifest": str(args.manifest),
                    "region": region_label(args),
                    "align": "official ssi_disparity, no valid mask on the region",
                    "per_model": summary,
                    "gt_band_median_m": float(np.median(finite)),
                    "paired_vs": reference, "comparisons": comparisons},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")
    with (args.output_dir / "sky_band_per_frame.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "gt_band_median_m", "gt_band_coverage"]
                        + [f"{label}_{field}" for _, label in specs
                           for field in ("band_median_m", "band_p90_m", "band_frac_near")])
        for frame in shared:
            writer.writerow([frame, gt_band[frame], gt_band_coverage[frame]]
                            + [per_frame[label][frame][field] for _, label in specs
                               for field in ("band_median_m", "band_p90_m", "band_frac_near")])
    print(f"\n-> {args.output_dir}/sky_band_summary.json")


if __name__ == "__main__":
    main()
