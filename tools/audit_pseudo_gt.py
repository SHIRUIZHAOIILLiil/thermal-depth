"""Measure calibrated AnyThermal pseudo depth before anything is trained on it.

Two questions decide whether dense completion can do what it is being built for,
and both are answerable from arithmetic alone.

Where does the sky land? The pseudo depth above the horizon is the answer to
whether this supervision can reach the region the lidar never describes, and
what range handling may do to it. Cutting everything past the evaluation ceiling
would delete exactly the pixels the experiment exists to supervise if that is
where the sky sits; pinning them to the ceiling would smuggle in a distance
prior and answer the sky question before the experiment asks it. So the bands
are reported untouched -- non-positive, under the ceiling, ceiling to 150 m,
beyond -- and the policy is chosen afterwards.

Does the fit carry? Lidar returns in the top rows cover 2.37% of them and sit at
8-11 m, so pseudo depth in the sky is an affine anchored on near ground and
extrapolated far past it. Held-out pixels measure the fit where lidar exists;
fitting on near returns and scoring far ones measures the extrapolation itself.
A fit that cannot reach 30 m from 15 m will not reach the sky either, and would
inject error precisely where the sky is supposed to be repaired.

Train and val only. The test split stays untouched until the final evaluation.

    python tools/audit_pseudo_gt.py \
        --manifest <train.jsonl> --ms2-root <root> \
        --raw-pred-dir <runs>/anythermal_train_subset/raw_predictions \
        --output-dir <runs>/pseudo_gt/audit_train
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_anythermal_pseudo_gt import (  # noqa: E402
    calibrate_pseudo_depth,
    fit_affine_depth_space,
    load_gt_depth,
    read_rows,
    resize_to,
)

PERCENTILES = (1, 10, 50, 90, 99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="Train or val. Never test.")
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--raw-pred-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--gt-min-depth", type=float, default=1e-3)
    parser.add_argument("--gt-max-depth", type=float, default=80.0,
                        help="Evaluation ceiling. Reported as a band edge, never applied here.")
    parser.add_argument("--top-rows", type=int, default=32,
                        help="A band of rows, not a segmentation: canopy, gantries and "
                             "buildings live here too, so its numbers are a mixture.")
    parser.add_argument("--sky-mask-dir", type=Path, default=None,
                        help="build_sky_masks.py output. Only these pixels answer the sky "
                             "question; the row band cannot, being a mixture.")
    parser.add_argument("--near-max-depth", type=float, default=15.0, help="Extrapolation gate: fit below this.")
    parser.add_argument("--far-min-depth", type=float, default=30.0, help="Extrapolation gate: score above this.")
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--min-fit-pixels", type=int, default=100)
    parser.add_argument("--min-gate-pixels", type=int, default=200,
                        help="Frames with fewer near or far returns cannot run the gate.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260808)
    return parser.parse_args()


def band_counts(values: np.ndarray, ceiling: float) -> dict[str, int]:
    """Split by distance, with the evaluation ceiling as one of the edges.

    Whether pseudo depth in the sky reads as near structure or as distance is the
    whole question, so the bands are fine enough to tell those apart rather than
    collapsing everything admissible into one bucket.
    """
    finite = np.isfinite(values)
    ok = finite & (values > 0)
    return {
        "pixels": int(values.size),
        "nonfinite": int((~finite).sum()),
        "nonpositive": int((finite & (values <= 0)).sum()),
        "under_20": int((ok & (values < 20.0)).sum()),
        "20_to_40": int((ok & (values >= 20.0) & (values < 40.0)).sum()),
        f"40_to_{ceiling:g}": int((ok & (values >= 40.0) & (values < ceiling)).sum()),
        f"{ceiling:g}_to_150": int((ok & (values >= ceiling) & (values < 150.0)).sum()),
        "beyond_150": int((ok & (values >= 150.0)).sum()),
    }


def add_counts(total: dict[str, int], part: dict[str, int]) -> None:
    for key, value in part.items():
        total[key] = total.get(key, 0) + value


def shares(counts: dict[str, int]) -> dict[str, float]:
    pixels = max(1, counts.get("pixels", 0))
    return {k: v / pixels for k, v in counts.items() if k != "pixels"}


def abs_rel(prediction: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - gt) / np.maximum(gt, 1e-6)))


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if "test" in args.manifest.name.lower():
        raise SystemExit(f"Refusing a test manifest: {args.manifest.name}. Train or val only.")

    rows = read_rows(args.manifest.resolve(), args.ms2_root.resolve(), args.stride, args.limit)
    rng = np.random.default_rng(args.seed)
    print(f"[data] {len(rows)} frames from {args.manifest.name}, ceiling {args.gt_max_depth} m", flush=True)

    region_names = ["all", "pseudo", "top_rows", "top_rows_pseudo"]
    if args.sky_mask_dir is not None:
        region_names += ["sky", "sky_pseudo"]
    regions: dict[str, dict] = {name: {} for name in region_names}
    percentiles: dict[str, list] = {name: [] for name in region_names}
    holdout, gate_extrapolated, gate_in_sample, in_sample = [], [], [], []
    real_gt_share, top_row_real_share, sky_share = [], [], []
    gate_skipped = 0
    frames_without_sky = 0
    used = 0

    for row in rows:
        raw_path = args.raw_pred_dir / f"{row['id']}.npy"
        if not raw_path.is_file():
            continue
        raw = np.load(raw_path, allow_pickle=False)
        gt_depth = load_gt_depth(row["gt_path"], args.depth_scale)
        real_gt_mask = (
            np.isfinite(gt_depth) & (gt_depth > args.gt_min_depth) & (gt_depth < args.gt_max_depth)
        )
        if int(real_gt_mask.sum()) < args.min_fit_pixels:
            continue
        try:
            calibrated, diagnostics = calibrate_pseudo_depth(
                raw, gt_depth, real_gt_mask, min_fit_pixels=args.min_fit_pixels)
        except RuntimeError:
            continue
        used += 1
        in_sample.append(diagnostics["fit_abs_rel"])
        real_gt_share.append(float(real_gt_mask.mean()))

        top = slice(0, args.top_rows)
        views = {
            "all": calibrated,
            "pseudo": calibrated[~real_gt_mask],
            "top_rows": calibrated[top],
            "top_rows_pseudo": calibrated[top][~real_gt_mask[top]],
        }
        top_row_real_share.append(float(real_gt_mask[top].mean()))
        if args.sky_mask_dir is not None:
            mask_path = args.sky_mask_dir / f"{row['id']}.png"
            if mask_path.is_file():
                sky = np.asarray(Image.open(mask_path), dtype=np.uint8) > 127
                if sky.shape != calibrated.shape:
                    raise SystemExit(f"{row['id']}: sky mask {sky.shape} vs GT {calibrated.shape}")
                sky_share.append(float(sky.mean()))
                views["sky"] = calibrated[sky]
                # Where lidar spoke, lidar wins: only this part becomes supervision.
                views["sky_pseudo"] = calibrated[sky & ~real_gt_mask]
            else:
                # Tunnels, canopy and tall buildings leave frames with no sky at
                # all -- 23.5% of the corpus -- so a missing mask is not an error.
                frames_without_sky += 1
        for name, values in views.items():
            add_counts(regions[name], band_counts(values, args.gt_max_depth))
            finite = values[np.isfinite(values)]
            if finite.size:
                percentiles[name].append(np.percentile(finite, PERCENTILES))

        # The fit is in-sample by construction, so hold pixels out of it.
        prediction = resize_to(raw, gt_depth.shape)
        indices = np.flatnonzero(real_gt_mask.ravel())
        rng.shuffle(indices)
        cut = int(len(indices) * (1.0 - args.holdout_fraction))
        train_idx, test_idx = indices[:cut], indices[cut:]
        if len(train_idx) >= args.min_fit_pixels and len(test_idx) >= args.min_fit_pixels:
            flat_pred, flat_gt = prediction.ravel(), gt_depth.ravel()
            try:
                scale, shift = fit_affine_depth_space(flat_pred[train_idx], flat_gt[train_idx])
                holdout.append(abs_rel(flat_pred[test_idx] * scale + shift, flat_gt[test_idx]))
            except RuntimeError:
                pass

        # The sky is reached by extrapolation, not interpolation: near returns
        # are almost all this fit has to stand on.
        near = real_gt_mask & (gt_depth < args.near_max_depth)
        far = real_gt_mask & (gt_depth > args.far_min_depth)
        if int(near.sum()) >= args.min_gate_pixels and int(far.sum()) >= args.min_gate_pixels:
            try:
                scale, shift = fit_affine_depth_space(prediction[near], gt_depth[near])
                gate_extrapolated.append(abs_rel(prediction[far] * scale + shift, gt_depth[far]))
                gate_in_sample.append(abs_rel(calibrated[far], gt_depth[far]))
            except RuntimeError:
                gate_skipped += 1
        else:
            gate_skipped += 1

    if not used:
        raise SystemExit("No frame produced a calibration; check --raw-pred-dir")

    def summarize(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        array = np.asarray(values, np.float64)
        return {"frames": int(array.size), "p50": float(np.median(array)),
                "p90": float(np.percentile(array, 90)), "mean": float(array.mean()),
                "max": float(array.max())}

    report = {
        "manifest": str(args.manifest), "frames_used": used, "gate_skipped": gate_skipped,
        "ceiling_m": args.gt_max_depth, "top_rows": args.top_rows,
        "sky_mask_dir": str(args.sky_mask_dir) if args.sky_mask_dir else None,
        "frames_without_sky_mask": frames_without_sky,
        "real_gt_share": summarize(real_gt_share),
        "real_gt_share_top_rows": summarize(top_row_real_share),
        "sky_share_of_frame": summarize(sky_share),
        "bands": {name: {"counts": counts, "shares": shares(counts)} for name, counts in regions.items()},
        "percentiles_median_over_frames": {
            name: dict(zip((f"p{p}" for p in PERCENTILES),
                           np.median(np.stack(values), axis=0).tolist()))
            for name, values in percentiles.items() if values
        },
        "error": {
            "in_sample_abs_rel": summarize(in_sample),
            "holdout_abs_rel": summarize(holdout),
            "gate_fit_near_score_far_abs_rel": summarize(gate_extrapolated),
            "gate_fit_all_score_far_abs_rel": summarize(gate_in_sample),
        },
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[frames] {used} calibrated, {gate_skipped} without enough near/far returns for the gate")
    print(f"[lidar]  real GT share: p50={report['real_gt_share']['p50']:.3f}  "
          f"top {args.top_rows} rows: p50={report['real_gt_share_top_rows']['p50']:.4f}")
    if sky_share:
        print(f"[sky]    mask covers p50={report['sky_share_of_frame']['p50']:.3%} of a frame; "
              f"{frames_without_sky} frames have no mask")

    ceiling = f"{args.gt_max_depth:g}"
    columns = ["nonfinite", "nonpositive", "under_20", "20_to_40",
               f"40_to_{ceiling}", f"{ceiling}_to_150", "beyond_150"]
    headers = ["nonfinite", "<=0", "<20m", "20-40m", f"40-{ceiling}m", f"{ceiling}-150m", ">150m"]
    print(f"\n{'region':18s}" + "".join(f"{h:>11s}" for h in headers))
    for name in region_names:
        s = shares(regions[name])
        print(f"{name:18s}" + "".join(f"{s.get(c, 0.0):>11.2%}" for c in columns))

    print(f"\n{'region':18s}" + "".join(f"{f'p{p}':>12s}" for p in PERCENTILES))
    for name, values in report["percentiles_median_over_frames"].items():
        print(f"{name:18s}" + "".join(f"{values[f'p{p}']:>12.2f}" for p in PERCENTILES))

    print(f"\n{'fit':34s}{'frames':>8s}{'p50':>10s}{'p90':>10s}{'max':>10s}")
    for label, key in (("in-sample (fitted pixels)", "in_sample_abs_rel"),
                       ("held-out lidar pixels", "holdout_abs_rel"),
                       ("EXTRAPOLATION: near-fit -> far", "gate_fit_near_score_far_abs_rel"),
                       ("reference: all-fit -> far", "gate_fit_all_score_far_abs_rel")):
        entry = report["error"][key]
        if entry:
            print(f"{label:34s}{entry['frames']:>8d}{entry['p50']:>10.4f}"
                  f"{entry['p90']:>10.4f}{entry['max']:>10.4f}")

    print("\nReading it: sky_pseudo is the row that answers the sky question -- top_rows is a"
          "\nmixture of canopy, gantries and buildings and cannot. Its bands set range handling:"
          f"\na sky beyond {args.gt_max_depth:g} m cannot be cut without deleting the supervision this"
          "\nexperiment is for, nor pinned there without asserting the distance prior the"
          "\nexperiment is supposed to test for. And the extrapolation line says whether any of"
          "\nit means anything: the sky is the same reach past the lidar, measured where lidar"
          "\ncan still check the answer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
