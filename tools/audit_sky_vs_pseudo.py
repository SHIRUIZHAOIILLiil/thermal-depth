"""What the completed pseudo-GT target says inside the sky mask.

    python tools/audit_sky_vs_pseudo.py \
        --manifest $IRIS_MANIFEST_DIR/ms2_train_day2seq_clip75_20260728.jsonl \
        --ms2-root $IRIS_MS2_ROOT \
        --pseudo-dir $IRIS_RUNS/pseudo_gt/official_train/calibrated_pseudo_depth \
        --sky-mask-dir $IRIS_RUNS/sky_masks/skymask_train_full/masks \
        --frames 400

Why this has to be measured before a sky loss is added to this line
-------------------------------------------------------------------
On the sparse-GT line a sky term filled a vacuum: lidar returns nothing from
sky, so those pixels had no target at all and the network drifted them near.

This line's target is the *completed* map -- lidar where it spoke, calibrated
AnyThermal pseudo depth everywhere else -- so sky already has a target. Adding
a sky term therefore does not fill a hole; it **contradicts an existing
supervision signal** wherever the pseudo map reads nearer than the cap. That
conflict has no rule yet, and inventing a weight to average two disagreeing
targets is worse than deciding which one is right.

So: read what the pseudo map actually says up there. If it already sits at the
cap, the sky term has nothing to add to this line and the question is closed.
If it reads near, that is both the cause of the model's own sky reading and the
thing to fix -- and the fix is a precedence rule on the target, not a weight.

Everything here is a property of the *target*, not of any model. No GPU, no
checkpoint.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--pseudo-dir", type=Path, required=True)
    parser.add_argument("--sky-mask-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=400,
                        help="Uniformly spaced across the manifest. 0 = every frame.")
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--d-min", type=float, default=1e-3)
    parser.add_argument("--d-max", type=float, default=80.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8") if line.strip()]
    if 0 < args.frames < len(rows):
        rows = [rows[int(i)] for i in np.linspace(0, len(rows) - 1, args.frames, dtype=int)]

    sky_share, sky_pseudo_median, sky_at_cap, sky_below_20 = [], [], [], []
    sky_with_lidar, lidar_share, frames_without_sky, missing = [], [], 0, 0
    pooled_pseudo: list[np.ndarray] = []

    for row in rows:
        rid = str(row["id"])
        mask_path = args.sky_mask_dir / f"{rid}.png"
        pseudo_path = args.pseudo_dir / f"{rid}.npy"
        if not mask_path.is_file() or not pseudo_path.is_file():
            missing += 1
            continue
        depth_field = row.get("thermal_depth_path") or row.get("depth_path")
        gt = np.asarray(Image.open(args.ms2_root / depth_field), dtype=np.float32) / args.depth_scale
        lidar = np.isfinite(gt) & (gt > args.d_min) & (gt < args.d_max)
        sky = np.asarray(Image.open(mask_path)) > 127
        pseudo = np.load(pseudo_path, allow_pickle=False).astype(np.float32)
        if sky.shape != pseudo.shape or sky.shape != gt.shape:
            raise SystemExit(f"{rid}: shapes disagree {sky.shape} {pseudo.shape} {gt.shape}")

        lidar_share.append(float(lidar.mean()))
        sky_share.append(float(sky.mean()))
        sky_with_lidar.append(float((sky & lidar).mean() / max(sky.mean(), 1e-9)))

        # The region the sky term would actually own: predicted sky where lidar
        # is silent. Where lidar spoke, lidar wins and there is nothing to argue.
        region = sky & ~lidar
        if not region.any():
            frames_without_sky += 1
            continue
        values = np.clip(pseudo[region], args.d_min, args.d_max)
        sky_pseudo_median.append(float(np.median(values)))
        sky_at_cap.append(float((values >= args.d_max - 0.5).mean()))
        sky_below_20.append(float((values < 20.0).mean()))
        if len(pooled_pseudo) < 200:                 # bounded memory, plenty for a quantile
            pooled_pseudo.append(values[:: max(1, values.size // 2000)])

    if not sky_pseudo_median:
        raise SystemExit("No frame had sky outside the lidar mask; nothing to report.")

    pooled = np.concatenate(pooled_pseudo)
    report = {
        "frames_read": len(sky_pseudo_median),
        "frames_missing_inputs": missing,
        "frames_without_sky_outside_lidar": frames_without_sky,
        "lidar_share_of_frame_mean": float(np.mean(lidar_share)),
        "sky_share_of_frame_mean": float(np.mean(sky_share)),
        "sky_pixels_carrying_lidar_mean": float(np.mean(sky_with_lidar)),
        "pseudo_in_sky_median_m": float(np.median(sky_pseudo_median)),
        "pseudo_in_sky_p10_m": float(np.percentile(pooled, 10)),
        "pseudo_in_sky_p90_m": float(np.percentile(pooled, 90)),
        "pseudo_in_sky_frac_at_cap": float(np.mean(sky_at_cap)),
        "pseudo_in_sky_frac_below_20m": float(np.mean(sky_below_20)),
        "d_max": args.d_max,
    }

    print(f"frames read {report['frames_read']}  "
          f"(missing inputs {missing}, no sky outside lidar {frames_without_sky})")
    print(f"lidar covers {report['lidar_share_of_frame_mean'] * 100:.1f}% of the frame; "
          f"predicted sky {report['sky_share_of_frame_mean'] * 100:.2f}%; "
          f"{report['sky_pixels_carrying_lidar_mean'] * 100:.3f}% of sky carries a return")
    print()
    print("--- what the completed target says where the sky term would own the pixel ---")
    print(f"  median          {report['pseudo_in_sky_median_m']:.2f} m   (cap is {args.d_max:.0f} m)")
    print(f"  p10 / p90       {report['pseudo_in_sky_p10_m']:.2f} / {report['pseudo_in_sky_p90_m']:.2f} m")
    print(f"  already at cap  {report['pseudo_in_sky_frac_at_cap'] * 100:.1f}% of sky pixels")
    print(f"  below 20 m      {report['pseudo_in_sky_frac_below_20m'] * 100:.1f}% of sky pixels")
    print()
    if report["pseudo_in_sky_frac_at_cap"] > 0.9:
        print("=> the target already puts sky at the cap; a sky term adds nothing here.")
    elif report["pseudo_in_sky_median_m"] < 40:
        print("=> the target itself calls sky near. Fix the target by precedence "
              "(lidar > sky mask > pseudo), not by weighting two disagreeing losses.")
    else:
        print("=> partial: the target is far but not at the cap. Report both numbers.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
