"""Calibration experiment A: run AnyThermal's released MS2 depth model on our data.

Loads the reconstructed Midas/DPT-DINOv2 net (``tools/build_anythermal_midas.py``,
strict-verified against ``Midas_anythermal`` et al.) and runs it over a manifest,
exporting raw predictions for tools/run_official_ms2_evaluation.py.

Preprocessing mirrors their eval chain (hparams + BMSD custom_transforms):
    16-bit thermal -> bilinear resize 256x640 -> 252x630 (patch-14 multiple;
    the private fork must resize too, exact policy unknown -- recorded in
    metadata) -> /2^14 -> per-image 1%/99% clip + min-max (TensorIWMM)
    -> Normalize(0.45, 0.225) -> replicate to 3 channels.

Their MiDaS path is evaluated with a per-image scale+shift fit against GT
depth, i.e. our official CLI's ``--align ssi`` (route ``anythermal-midas``).

Example (smoke then full):
  python tools/run_ms2_anythermal_midas.py --manifest <val.jsonl> --ms2-root /mnt/e/dataset/ms2 --output-dir outputs/lotus_line_v2/anythermal_midas_val_smoke8 --max-samples 8
  python tools/run_ms2_anythermal_midas.py --manifest <val.jsonl> --ms2-root /mnt/e/dataset/ms2 --output-dir outputs/lotus_line_v2/anythermal_midas_val_full --max-samples 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = ROOT / "tools"
for path in (ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_anythermal_midas import build_anythermal_midas  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Defaults to Midas_anythermal; point at Midas_dinov2/Midas_small for the other Table IV rows")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--norm-mean", type=float, default=0.45)
    parser.add_argument("--norm-std", type=float, default=0.225)
    parser.add_argument("--raw-divisor", type=float, default=float(2**14))
    parser.add_argument(
        "--calibrate-to-gt",
        action="store_true",
        help=(
            "Task-4 teacher mode: per-sample fit scale+shift of the prediction "
            "against the sparse GT depth (their MiDaS eval path), clamp, invert "
            "to disparity, and save dense teacher_disparity/<id>.npy at GT "
            "resolution for train-time distillation."
        ),
    )
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--calib-min-depth", type=float, default=1e-3,
                        help="Valid-GT mask lower bound for the calibration fit")
    parser.add_argument("--calib-max-depth", type=float, default=80.0)
    parser.add_argument("--teacher-clamp-min", type=float, default=1.0,
                        help="Clamp calibrated depth below this (metres) so 1/depth teacher targets have no near-field spikes")
    return parser.parse_args()


def read_manifest(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            depth = row.get("thermal_depth_path") or row.get("depth_path")
            if not depth:
                raise ValueError(f"Manifest row {row.get('id')} lacks GT depth")
            # MS2 ships thermal-view and RGB-view GT and mixing them is a protocol
            # violation (frozen conclusion 15). Datasets with a single registered
            # depth (RGBDT500) declare themselves and are exempt.
            if str(row.get("dataset", "MS2")).upper() == "MS2" and "/thr/" not in str(depth).replace("\\", "/"):
                raise ValueError(f"Manifest row {row.get('id')} lacks thermal-view GT")
            rows.append({"id": str(row["id"]), "thermal_path": row["thermal_path"],
                         "thermal_depth_path": depth,
                         "sequence": row.get("sequence"), "condition": row.get("condition"),
                         "split": row.get("split"), "caption": row.get("caption")})
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


def choose_uniform(rows, count: int):
    if count <= 0 or count >= len(rows):
        return rows
    return [rows[int(i)] for i in np.linspace(0, len(rows) - 1, count, dtype=int)]


def tensor_iwmm(image: torch.Tensor) -> torch.Tensor:
    """Port of BMSD TensorIWMM: per-image 1%/99% clip then min-max to [0,1]."""
    flat, _ = torch.sort(image.reshape(-1))
    count = flat.numel()
    tmax = flat[round(count * 0.99) - 1]
    tmin = flat[round(count * 0.01)]
    if not torch.isfinite(tmax) or not torch.isfinite(tmin) or float(tmax) <= float(tmin):
        raise ValueError("Degenerate thermal percentiles; refusing constant image")
    clipped = image.clamp(min=float(tmin), max=float(tmax))
    return (clipped - tmin) / (tmax - tmin)


def preprocess_thermal(path: Path, *, raw_divisor: float, mean: float, std: float,
                       patch: int = 14) -> tuple[torch.Tensor, tuple[int, int]]:
    import cv2

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(path)
    raw = raw.astype(np.float32)
    if raw.ndim == 3:
        raw = raw[..., 0]
    height, width = raw.shape
    tensor = torch.from_numpy(raw)[None, None]  # 1,1,H,W
    target = (height // patch * patch, width // patch * patch)  # 256x640 -> 252x630
    if target != (height, width):
        tensor = F.interpolate(tensor, target, mode="bilinear", align_corners=False)
    tensor = tensor / raw_divisor
    tensor = tensor_iwmm(tensor)
    tensor = (tensor - mean) / std
    tensor = tensor.repeat(1, 3, 1, 1)  # replicate after 1-channel normalize, like their Midas path
    return tensor, (height, width)


def calibrated_teacher_disparity(prediction: np.ndarray, gt_path: Path, *,
                                 depth_scale: float, min_depth: float, max_depth: float,
                                 clamp_min: float) -> np.ndarray:
    """Fit pred to sparse GT depth (2-param, their MiDaS path), clamp, invert."""
    from PIL import Image

    gt_raw = np.asarray(Image.open(gt_path))
    gt = gt_raw.astype(np.float64) / depth_scale
    pred = torch.from_numpy(prediction)[None, None]
    pred = F.interpolate(pred, gt.shape, mode="bilinear", align_corners=False)[0, 0].numpy().astype(np.float64)
    valid = np.isfinite(gt) & (gt > min_depth) & (gt < max_depth)
    if valid.sum() < 100:
        raise RuntimeError(f"Too few valid GT pixels for calibration: {gt_path}")
    p, g = pred[valid], gt[valid]
    a_00, a_01, a_11 = float(np.sum(p * p)), float(np.sum(p)), float(p.size)
    b_0, b_1 = float(np.sum(p * g)), float(np.sum(g))
    det = a_00 * a_11 - a_01 * a_01
    if det <= 0:
        raise RuntimeError(f"Degenerate calibration fit: {gt_path}")
    scale = (a_11 * b_0 - a_01 * b_1) / det
    shift = (-a_01 * b_0 + a_00 * b_1) / det
    depth = np.clip(pred * scale + shift, clamp_min, max_depth)
    return (1.0 / depth).astype(np.float32)


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output}")
    (output / "raw_predictions").mkdir(parents=True, exist_ok=True)
    if args.calibrate_to_gt:
        (output / "teacher_disparity").mkdir(parents=True, exist_ok=True)

    manifest = args.manifest.resolve()
    ms2_root = args.ms2_root.resolve()
    selected = choose_uniform(read_manifest(manifest), args.max_samples)
    (output / "selected_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")

    model, load_info = build_anythermal_midas(args.checkpoint, device=args.device)
    device = torch.device(args.device)

    start = time.time()
    for index, row in enumerate(selected):
        tensor, native_hw = preprocess_thermal(
            ms2_root / row["thermal_path"], raw_divisor=args.raw_divisor,
            mean=args.norm_mean, std=args.norm_std)
        with torch.no_grad():
            prediction = model(tensor.to(device)).float().cpu().numpy()[0]
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"Non-finite prediction for {row['id']}")
        np.save(output / "raw_predictions" / f"{row['id']}.npy", prediction.astype(np.float32))
        if args.calibrate_to_gt:
            teacher = calibrated_teacher_disparity(
                prediction, ms2_root / row["thermal_depth_path"],
                depth_scale=args.depth_scale, min_depth=args.calib_min_depth,
                max_depth=args.calib_max_depth, clamp_min=args.teacher_clamp_min)
            np.save(output / "teacher_disparity" / f"{row['id']}.npy", teacher)
        if index % 200 == 0 or index == len(selected) - 1:
            print(f"[{index + 1}/{len(selected)}] {row['id']}  out={prediction.shape}  "
                  f"{(time.time() - start) / (index + 1):.2f}s/img", flush=True)

    metadata = {
        "experiment": "calibration-A: AnyThermal released depth model on our manifest",
        "model": load_info,
        "model_reconstruction": "tools/build_anythermal_midas.py (strict 231/231 load)",
        "preprocessing": {
            "resize": "bilinear to floor-multiple-of-14 (256x640 -> 252x630); exact fork policy unknown",
            "raw_divisor": args.raw_divisor,
            "per_image_clip": "1%/99% percentile + min-max (TensorIWMM port)",
            "normalize": {"mean": args.norm_mean, "std": args.norm_std},
            "channels": "1-channel normalize, then replicate to 3 (their Midas path order)",
        },
        "output_space": "MiDaS-style affine depth; evaluate with run_official_ms2_evaluation.py --route anythermal-midas (ssi)",
        "calibrate_to_gt": bool(args.calibrate_to_gt),
        "teacher_disparity": (
            {
                "definition": "1 / clip(scale*pred+shift, clamp_min, max_depth); per-sample 2-param fit on sparse GT",
                "clamp_min_m": args.teacher_clamp_min,
                "max_depth_m": args.calib_max_depth,
            }
            if args.calibrate_to_gt else None
        ),
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "sample_count": len(selected),
        "elapsed_seconds": time.time() - start,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in ("experiment", "sample_count", "elapsed_seconds")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
