"""Turn raw Lotus-RGB disparity predictions into calibrated dense teacher labels.

The experiment-5 recipe needs a *dense* disparity map that already lives in the
GT's value range: its bare-L1 term is what pins the student's output scale
(the "range anchor"), and that only works if the teacher is anchored first.

This is the disparity-space sibling of ``calibrated_teacher_disparity`` in
``tools/run_ms2_anythermal_midas.py``.  That one fits a MiDaS-style prediction
against GT *depth* and inverts; Lotus already emits disparity, so the fit runs
directly in disparity space -- the same 2-parameter budget, and the same space
the official ``ssi_disparity`` evaluator uses, so a correctly calibrated teacher
reproduces that evaluator's AbsRel (printed below as a self-check).

Sparse GT pixels drive the fit; the fitted affine map is then applied to *every*
pixel, which is what makes the label dense.

Written for RGBDT500 (depth_scale 1000, 20 m cap) but the units are all flags.
CPU only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True,
                        help="Directory the manifest's relative depth paths resolve against.")
    parser.add_argument("--prediction-dir", type=Path, required=True,
                        help="raw_predictions/ from run_ms2_lotus_rgb_official.py --save-raw-pred")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="teacher_disparity/ is created inside this directory.")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--clamp-min-depth", type=float, default=1.0,
                        help="Near clamp on the calibrated teacher, mirroring the "
                             "AnyThermal teacher's --teacher-clamp-min. Bounds the "
                             "disparity the bare L1 can be asked to match.")
    parser.add_argument("--min-fit-pixels", type=int, default=100)
    return parser.parse_args()


def fit_affine(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Closed-form least squares target ~ scale * prediction + shift.

    Same normal-equation form as the AnyThermal teacher path, kept identical so
    the two teacher families stay comparable.
    """
    a_00 = float(np.sum(prediction * prediction))
    a_01 = float(np.sum(prediction))
    a_11 = float(prediction.size)
    b_0 = float(np.sum(prediction * target))
    b_1 = float(np.sum(target))
    determinant = a_00 * a_11 - a_01 * a_01
    if determinant <= 0:
        raise ValueError("Degenerate calibration fit")
    scale = (a_11 * b_0 - a_01 * b_1) / determinant
    shift = (-a_01 * b_0 + a_00 * b_1) / determinant
    return scale, shift


def main() -> int:
    args = parse_args()
    from PIL import Image

    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8") if line.strip()]
    teacher_dir = args.output_dir / "teacher_disparity"
    teacher_dir.mkdir(parents=True, exist_ok=True)

    disparity_floor = 1.0 / args.max_depth
    disparity_ceiling = 1.0 / args.clamp_min_depth

    written, skipped = 0, []
    abs_rels, clamp_fractions, scales, shifts = [], [], [], []
    for index, row in enumerate(rows):
        prediction_path = args.prediction_dir / f"{row['id']}.npy"
        if not prediction_path.is_file():
            skipped.append({"id": row["id"], "reason": "missing prediction"})
            continue
        prediction = np.load(prediction_path).astype(np.float64)
        gt_depth = np.asarray(Image.open(args.root / row["depth_path"])).astype(np.float64)
        gt_depth = gt_depth / args.depth_scale
        if gt_depth.shape != prediction.shape:
            skipped.append({"id": row["id"], "reason":
                            f"shape {gt_depth.shape} vs pred {prediction.shape}"})
            continue

        valid = np.isfinite(gt_depth) & (gt_depth > args.min_depth) & (gt_depth < args.max_depth)
        if int(valid.sum()) < args.min_fit_pixels:
            skipped.append({"id": row["id"], "reason": f"only {int(valid.sum())} valid GT px"})
            continue

        try:
            scale, shift = fit_affine(prediction[valid], 1.0 / gt_depth[valid])
        except ValueError as error:
            skipped.append({"id": row["id"], "reason": str(error)})
            continue

        calibrated = prediction * scale + shift
        clamp_fractions.append(float(np.mean(
            (calibrated < disparity_floor) | (calibrated > disparity_ceiling)
        )))
        calibrated = np.clip(calibrated, disparity_floor, disparity_ceiling)
        np.save(teacher_dir / f"{row['id']}.npy", calibrated.astype(np.float32))
        written += 1
        scales.append(scale)
        shifts.append(shift)

        # Self-check: on GT pixels this is exactly what ssi_disparity scores, so
        # the median here should land on the official AbsRel for this run.
        teacher_depth = 1.0 / np.clip(calibrated[valid], 1e-6, None)
        abs_rels.append(float(np.mean(np.abs(teacher_depth - gt_depth[valid]) / gt_depth[valid])))

        if (index + 1) % 500 == 0:
            print(f"  {index + 1}/{len(rows)}")

    summary = {
        "manifest": str(args.manifest),
        "prediction_dir": str(args.prediction_dir),
        "teacher_dir": str(teacher_dir),
        "written": written,
        "skipped": skipped,
        "clamp": {"disparity_floor": disparity_floor, "disparity_ceiling": disparity_ceiling,
                  "mean_clamped_pixel_fraction": float(np.mean(clamp_fractions)) if clamp_fractions else None},
        "fit": {"median_scale": float(np.median(scales)) if scales else None,
                "median_shift": float(np.median(shifts)) if shifts else None},
        "self_check_abs_rel": {"median": float(np.median(abs_rels)) if abs_rels else None,
                               "mean": float(np.mean(abs_rels)) if abs_rels else None},
    }
    (args.output_dir / "teacher_calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\nwrote {written} teacher labels to {teacher_dir}")
    if skipped:
        print(f"skipped {len(skipped)}: {skipped[:5]}")
    print(f"fit: median scale {summary['fit']['median_scale']:.4f}, "
          f"shift {summary['fit']['median_shift']:.4f}")
    print(f"clamped pixels: {summary['clamp']['mean_clamped_pixel_fraction']:.3%}")
    print(f"self-check AbsRel: mean {summary['self_check_abs_rel']['mean']:.4f} "
          f"(median {summary['self_check_abs_rel']['median']:.4f})  "
          f"<- should match the official ssi_disparity score for this run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
