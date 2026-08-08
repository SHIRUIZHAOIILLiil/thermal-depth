"""Complete MS2's sparse LiDAR GT with calibrated AnyThermal pseudo depth.

MS2's lidar leaves roughly three quarters of every frame without a number, and
`masked_ssi_l1` gives those pixels exactly zero gradient. This builds a dense
pseudo depth map to stand in where the lidar said nothing, so the loss can reach
regions -- the sky above all -- that are currently unconstrained.

AnyThermal is used strictly as an offline pseudo-depth generator. It never
enters a Lotus forward pass, nothing is matched between the two models, and no
loss compares their outputs; its prediction is calibrated against this frame's
own lidar and then read as if it were another GT source.

The calibration follows the representation AnyThermal actually has: a relative
depth-like map, larger-is-farther, whose sign carries no meaning until the fit.
So the affine is solved in DEPTH space against real lidar depth -- the same fit
BridgeMultiSpectralDepth scores this family under -- and nothing is clamped
beforehand, because clamping unfitted output would discard the very range the
two parameters exist to place. Inverting first and fitting in disparity space
would misspecify the family outright: it scores 0.2724 where the depth-space fit
scores 0.0821.

What this writes is calibrated depth in metres, not the disparity the loss
consumes. Range handling -- what to do with pseudo depth beyond the evaluation
ceiling, which is most of the sky -- decides whether the supervision carries a
distance prior, and that belongs to a training flag that can be moved in a
minute, not baked into a corpus that costs hours to rebuild.

    python tools/build_anythermal_pseudo_gt.py \
        --manifest <train.jsonl> --ms2-root /mnt/scratch/<user>/data/ms2 \
        --raw-pred-dir <runs>/anythermal_train/raw_predictions \
        --output-dir <runs>/pseudo_gt/train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="Training split only.")
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--raw-pred-dir", type=Path, required=True,
                        help="AnyThermal raw *.npy, from run_ms2_anythermal_midas.py.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--gt-min-depth", type=float, default=1e-3,
                        help="Validity rule for the lidar the fit is anchored on.")
    parser.add_argument("--gt-max-depth", type=float, default=80.0)
    parser.add_argument("--min-fit-pixels", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0, help="0 = every row.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32",
                        help="float16 halves the corpus; its 0.06 m step at 80 m is far "
                             "below the disagreement this supervision carries anyway.")
    return parser.parse_args()


def read_rows(manifest: Path, ms2_root: Path, stride: int, limit: int) -> list[dict]:
    rows = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            relative = row.get("thermal_depth_path") or row.get("depth_path")
            if not relative:
                raise SystemExit(f"Manifest row {row.get('id')} lacks GT depth")
            # Conclusion 15: the thermal-view and RGB-view GT are different exams.
            if row.get("rgb_depth_path") and "/thr/" not in str(relative).replace("\\", "/"):
                raise SystemExit(f"Row {row.get('id')}: GT {relative} is not the thermal view")
            rows.append({"id": str(row["id"]), "gt_path": ms2_root / relative})
    rows = rows[:: max(1, stride)]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise SystemExit(f"{manifest} produced no rows")
    return rows


def load_gt_depth(path: Path, depth_scale: float) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.float64) / float(depth_scale)


def resize_to(prediction: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear resize without pulling torch in for one interpolation."""
    if prediction.shape == shape:
        return prediction.astype(np.float64)
    source = Image.fromarray(prediction.astype(np.float32), mode="F")
    resized = source.resize((shape[1], shape[0]), resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.float64)


def fit_affine_depth_space(prediction_pixels: np.ndarray, gt_depth_pixels: np.ndarray) -> tuple[float, float]:
    """Solve depth ~= scale * raw + shift in closed form. Shared with the audit,
    which fits on pixel subsets the calibration never sees."""
    p, g = np.asarray(prediction_pixels, np.float64), np.asarray(gt_depth_pixels, np.float64)
    a_00, a_01, a_11 = float(np.sum(p * p)), float(np.sum(p)), float(p.size)
    b_0, b_1 = float(np.sum(p * g)), float(np.sum(g))
    det = a_00 * a_11 - a_01 * a_01
    if det <= 0:
        raise RuntimeError("Degenerate affine fit: the raw map is constant on these pixels")
    return (a_11 * b_0 - a_01 * b_1) / det, (-a_01 * b_0 + a_00 * b_1) / det


def calibrate_pseudo_depth(
    anythermal_pseudo_depth: np.ndarray,
    gt_depth: np.ndarray,
    real_gt_mask: np.ndarray,
    *,
    min_fit_pixels: int,
) -> tuple[np.ndarray, dict]:
    """Fit s, t in depth space on this frame's real lidar and apply them.

    No clamping happens here. The raw map is relative and may be negative; the
    two parameters are what place it on a metric scale, so anything done to its
    range beforehand corrupts the fit rather than protecting it. The result is
    returned exactly as the affine produced it -- negatives, poles and all --
    because deciding what counts as an admissible depth is a separate question
    that the audit has to see the untouched distribution to answer.
    """
    prediction = resize_to(anythermal_pseudo_depth, gt_depth.shape)
    fit_pixels = int(real_gt_mask.sum())
    if fit_pixels < min_fit_pixels:
        raise RuntimeError(f"Only {fit_pixels} lidar pixels to fit on; need {min_fit_pixels}")
    p = prediction[real_gt_mask]
    g = gt_depth[real_gt_mask]
    scale, shift = fit_affine_depth_space(p, g)
    calibrated_pseudo_depth = prediction * scale + shift
    residual = calibrated_pseudo_depth[real_gt_mask] - g
    diagnostics = {
        "scale": scale,
        "shift": shift,
        "fit_pixels": fit_pixels,
        "fit_abs_rel": float(np.mean(np.abs(residual) / np.maximum(g, 1e-6))),
        "raw_nonpositive_fraction": float(np.mean(prediction <= 0)),
        "calibrated_nonpositive_fraction": float(np.mean(calibrated_pseudo_depth <= 0)),
        "calibrated_p50": float(np.median(calibrated_pseudo_depth)),
        "calibrated_max": float(np.max(calibrated_pseudo_depth)),
    }
    return calibrated_pseudo_depth, diagnostics


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    depth_dir = output / "calibrated_pseudo_depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.manifest.resolve(), args.ms2_root.resolve(), args.stride, args.limit)
    print(f"[data] {len(rows)} frames from {args.manifest.name}", flush=True)

    records = []
    failures = []
    started = time.time()
    for index, row in enumerate(rows):
        raw_path = args.raw_pred_dir / f"{row['id']}.npy"
        if not raw_path.is_file():
            failures.append({"id": row["id"], "error": f"no raw prediction at {raw_path}"})
            continue
        gt_depth = load_gt_depth(row["gt_path"], args.depth_scale)
        real_gt_mask = (
            np.isfinite(gt_depth) & (gt_depth > args.gt_min_depth) & (gt_depth < args.gt_max_depth)
        )
        try:
            calibrated_pseudo_depth, diagnostics = calibrate_pseudo_depth(
                np.load(raw_path, allow_pickle=False), gt_depth, real_gt_mask,
                min_fit_pixels=args.min_fit_pixels,
            )
        except RuntimeError as error:
            failures.append({"id": row["id"], "error": str(error)})
            continue
        np.save(depth_dir / f"{row['id']}.npy", calibrated_pseudo_depth.astype(args.dtype))
        diagnostics.update({"id": row["id"], "real_gt_fraction": float(real_gt_mask.mean())})
        records.append(diagnostics)
        if index % 200 == 0 or index == len(rows) - 1:
            print(f"[{index + 1}/{len(rows)}] {row['id']} "
                  f"s={diagnostics['scale']:.4f} t={diagnostics['shift']:.3f} "
                  f"fit_abs_rel={diagnostics['fit_abs_rel']:.4f} "
                  f"{(time.time() - started) / (index + 1):.3f}s/frame", flush=True)

    (output / "calibration.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    (output / "build_config.json").write_text(json.dumps({
        "manifest": str(args.manifest), "manifest_sha256": file_sha256(args.manifest),
        "raw_pred_dir": str(args.raw_pred_dir), "ms2_root": str(args.ms2_root),
        "depth_scale": args.depth_scale, "gt_min_depth": args.gt_min_depth,
        "gt_max_depth": args.gt_max_depth, "min_fit_pixels": args.min_fit_pixels,
        "stride": args.stride, "limit": args.limit, "dtype": args.dtype,
        "written": len(records), "failed": len(failures),
        "stored": "calibrated depth in metres, no range handling applied",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    if failures:
        (output / "failures.jsonl").write_text(
            "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in failures), encoding="utf-8")

    print(f"\n[done] wrote {len(records)}, failed {len(failures)} -> {depth_dir}")
    if records:
        fit = np.asarray([r["fit_abs_rel"] for r in records])
        print(f"[fit] AbsRel on the lidar pixels the affine was fitted to: "
              f"p50={np.median(fit):.4f} p90={np.percentile(fit, 90):.4f} max={fit.max():.4f}")
        print("      (in-sample by construction -- audit_pseudo_gt.py holds pixels out)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
