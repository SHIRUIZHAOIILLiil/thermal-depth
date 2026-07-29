"""Region-level caption effect analysis on the P1 Arm 1 checkpoint.

For every Val sample this tool runs the trained pipeline twice — correct
caption vs empty caption — with identical noise, seed, and condition, then
compares depth error inside mode-independent region masks:

  global        all valid GT pixels
  thermal_edge  valid pixels near strong thermal intensity edges
                (dilated Sobel; fine structures such as poles/pedestrians/cars)
  depth_near    valid pixels with GT depth in [min, 10) m
  depth_mid     valid pixels with GT depth in [10, 30) m
  depth_far     valid pixels with GT depth in [30, max] m

Masks derive only from the thermal input and GT, never from predictions or
captions, so the correct/empty comparison inside each region is fair.
Alignment mirrors the official evaluator (per-sample least squares scale/shift
in disparity space over all valid pixels), applied independently per mode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = ROOT / "lotus"
TOOLS_ROOT = ROOT / "tools"
for path in (ROOT, LOTUS_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.anythermal_encoder import AnyThermalEncoder  # noqa: E402
from models.anythermal_lotus_adapter_v2_3 import AnyThermalLotusAdapterV23  # noqa: E402
from models.anythermal_lotus_model import extract_anythermal_feature_pyramid  # noqa: E402
from models.anythermal_lotus_v2 import thermal_to_lotus_input  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--edge-percentile", type=float, default=90.0)
    parser.add_argument("--edge-dilation", type=int, default=4)
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            caption = str(row.get("caption", "")).strip()
            if not caption:
                raise ValueError(f"Empty caption in manifest row {row.get('id')}")
            rows.append(
                {
                    "id": str(row["id"]),
                    "thermal_path": str(row["thermal_path"]),
                    "depth_path": str(row["thermal_depth_path"]),
                    "caption": caption,
                }
            )
    if not rows:
        raise ValueError("Manifest is empty.")
    return rows


def choose_uniform(rows, count: int):
    if count <= 0 or count >= len(rows):
        return rows
    return [rows[int(i)] for i in np.linspace(0, len(rows) - 1, count, dtype=int)]


def task_embedding(device, dtype):
    task = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
    return torch.cat([torch.sin(task), torch.cos(task)], dim=-1)


def sobel_edge_mask(gray: np.ndarray, percentile: float, dilation: int) -> np.ndarray:
    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    magnitude = np.abs(gx) + np.abs(gy)
    threshold = np.percentile(magnitude, percentile)
    edge = magnitude >= max(threshold, 1e-6)
    if dilation > 0:
        pad = dilation
        padded = np.pad(edge, pad, mode="constant")
        stacked = np.zeros_like(edge, dtype=bool)
        for dy in range(-pad, pad + 1):
            for dx in range(-pad, pad + 1):
                stacked |= padded[
                    pad + dy : pad + dy + edge.shape[0],
                    pad + dx : pad + dx + edge.shape[1],
                ]
        edge = stacked
    return edge


def predict_disparity(lotus, adapter, features, thermal_tensor, prompt, noise, device, dtype):
    target = tuple(noise.shape[-2:])
    condition = adapter(
        [feature.to(device=device, dtype=torch.float32) for feature in features],
        thermal_tensor.to(device=device, dtype=torch.float32),
        target_size=target,
    ).to(dtype=dtype)
    timestep = torch.tensor(999, device=device, dtype=torch.long)
    latent_input = lotus.scheduler.scale_model_input(noise, timestep)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
    ):
        x0 = lotus.unet(
            torch.cat([condition, latent_input], dim=1),
            timestep,
            encoder_hidden_states=prompt.to(dtype=dtype),
            class_labels=task_embedding(device, dtype),
            return_dict=False,
        )[0]
        decoded = lotus.vae.decode(x0 / lotus.vae.config.scaling_factor, return_dict=False)[0]
        image = lotus.image_processor.postprocess(decoded, output_type="np", do_denormalize=[True])[0]
    disparity = np.asarray(image, np.float32).mean(axis=-1)
    if not np.isfinite(disparity).all():
        raise RuntimeError("Prediction contains NaN/Inf")
    return disparity


def aligned_depth(pred_disparity: np.ndarray, gt_depth: np.ndarray, valid: np.ndarray, min_depth: float, max_depth: float):
    gt_disp = np.zeros_like(gt_depth)
    gt_disp[valid] = 1.0 / gt_depth[valid]
    pred = pred_disparity[valid].astype(np.float64)
    design = np.stack([pred, np.ones_like(pred)], axis=1)
    (scale, shift), *_ = np.linalg.lstsq(design, gt_disp[valid].astype(np.float64), rcond=None)
    aligned_disp = scale * pred_disparity.astype(np.float64) + shift
    aligned_disp = np.clip(aligned_disp, 1.0 / max_depth, 1.0 / min_depth)
    return 1.0 / aligned_disp


def region_abs_rel(pred_depth: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray):
    count = int(mask.sum())
    if count == 0:
        return None, 0
    error = np.abs(pred_depth[mask] - gt_depth[mask]) / gt_depth[mask]
    return float(error.mean()), count


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows = choose_uniform(read_manifest(args.manifest.resolve()), args.max_samples)
    ms2_root = args.ms2_root.resolve()
    device = torch.device("cuda")
    dtype = torch.float16

    anythermal = AnyThermalEncoder(
        model_path=args.anythermal_model_path, device="cuda", local_files_only=args.local_files_only
    )
    from pipeline import LotusGPipeline

    lotus = LotusGPipeline.from_pretrained(
        args.lotus_model_path, torch_dtype=dtype, local_files_only=args.local_files_only
    ).to(device)
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    if checkpoint.get("adapter_architecture") != "v2_3_thermal_detail_skip":
        raise RuntimeError("This analysis expects an Adapter V2.3 checkpoint.")
    adapter = AnyThermalLotusAdapterV23().to(device=device, dtype=torch.float32)
    adapter.load_state_dict(checkpoint["adapter"], strict=True)
    if "lotus_unet_state_dict" in checkpoint:
        lotus.unet.load_state_dict(checkpoint["lotus_unet_state_dict"], strict=True)
    for module in (adapter, lotus.vae, lotus.text_encoder, lotus.unet):
        module.requires_grad_(False).eval()

    empty_prompt, _ = lotus.encode_prompt(
        prompt="", device=device, num_images_per_prompt=1, do_classifier_free_guidance=None
    )
    empty_prompt = empty_prompt.detach()

    regions = ("global", "thermal_edge", "depth_near", "depth_mid", "depth_far")
    per_sample = []
    for index, row in enumerate(rows):
        thermal_path = ms2_root / row["thermal_path"]
        features, info, _ = extract_anythermal_feature_pyramid(anythermal, thermal_path)
        thermal = thermal_to_lotus_input(thermal_path, processing_res=0)
        height, width = map(int, info.original_shape[:2])
        target = (height // int(lotus.vae_scale_factor), width // int(lotus.vae_scale_factor))
        generator = torch.Generator(device=device).manual_seed(args.seed + index)
        noise = torch.randn(
            (1, int(lotus.unet.config.in_channels) // 2, *target),
            generator=generator,
            device=device,
            dtype=dtype,
        ) * lotus.scheduler.init_noise_sigma

        correct_prompt, _ = lotus.encode_prompt(
            prompt=row["caption"], device=device, num_images_per_prompt=1,
            do_classifier_free_guidance=None,
        )
        disp_correct = predict_disparity(
            lotus, adapter, features, thermal.tensor, correct_prompt.detach(), noise, device, dtype
        )
        disp_empty = predict_disparity(
            lotus, adapter, features, thermal.tensor, empty_prompt, noise, device, dtype
        )

        gt_depth = np.asarray(Image.open(ms2_root / row["depth_path"]), dtype=np.float32)
        gt_depth = gt_depth / args.depth_scale
        valid = (gt_depth > args.min_depth) & (gt_depth < args.max_depth) & np.isfinite(gt_depth)
        if int(valid.sum()) < 100:
            continue

        gray = np.asarray(
            Image.open(thermal_path).convert("I;16"), dtype=np.float32
        )
        gray = (gray - gray.min()) / max(float(gray.max() - gray.min()), 1e-6)
        edge = sobel_edge_mask(gray, args.edge_percentile, args.edge_dilation)

        masks = {
            "global": valid,
            "thermal_edge": valid & edge,
            "depth_near": valid & (gt_depth < 10.0),
            "depth_mid": valid & (gt_depth >= 10.0) & (gt_depth < 30.0),
            "depth_far": valid & (gt_depth >= 30.0),
        }
        depth_correct = aligned_depth(disp_correct, gt_depth, valid, args.min_depth, args.max_depth)
        depth_empty = aligned_depth(disp_empty, gt_depth, valid, args.min_depth, args.max_depth)

        record = {"id": row["id"], "thermal_path": row["thermal_path"]}
        for name in regions:
            correct_value, count = region_abs_rel(depth_correct, gt_depth, masks[name])
            empty_value, _ = region_abs_rel(depth_empty, gt_depth, masks[name])
            record[f"{name}_correct"] = correct_value
            record[f"{name}_empty"] = empty_value
            record[f"{name}_pixels"] = count
        per_sample.append(record)
        if (index + 1) % 50 == 0 or index + 1 == len(rows):
            print(f"processed {index + 1}/{len(rows)}", flush=True)

    csv_path = output / "region_per_sample.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_sample[0].keys()))
        writer.writeheader()
        writer.writerows(per_sample)

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.resolve().read_bytes()).hexdigest(),
        "manifest": str(args.manifest),
        "samples": len(per_sample),
        "seed": args.seed,
        "edge_percentile": args.edge_percentile,
        "edge_dilation": args.edge_dilation,
        "alignment": "per-mode least_square_disparity on all valid pixels",
        "regions": {},
    }
    for name in regions:
        pairs = [
            (r[f"{name}_correct"], r[f"{name}_empty"])
            for r in per_sample
            if r[f"{name}_correct"] is not None and r[f"{name}_empty"] is not None
        ]
        if not pairs:
            continue
        correct_mean = float(np.mean([p[0] for p in pairs]))
        empty_mean = float(np.mean([p[1] for p in pairs]))
        wins = sum(1 for c, e in pairs if c < e)
        diffs = sorted(e - c for c, e in pairs)
        summary["regions"][name] = {
            "samples": len(pairs),
            "abs_rel_correct": correct_mean,
            "abs_rel_empty": empty_mean,
            "correct_win_rate": wins / len(pairs),
            "median_improvement": float(diffs[len(diffs) // 2]),
            "mean_pixels": float(np.mean([r[f"{name}_pixels"] for r in per_sample])),
        }
    (output / "region_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary["regions"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
