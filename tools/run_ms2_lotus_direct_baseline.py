"""Zero-parameter AnyThermal -> Lotus-G Direct baseline on MS2 thermal view."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LOTUS = ROOT / "lotus"
for path in (ROOT, LOTUS):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

from models.anythermal_encoder import AnyThermalEncoder
from models.anythermal_lotus_bridge import AnyThermalLotusBridge
from pipeline import LotusGPipeline
from ms2_eval.io import load_manifest, load_ms2_gt, sha256_file
from ms2_eval.lotus_official import OFFICIAL_METRICS, aggregate_imagewise, align_lotus_disparity_to_ms2_depth, lotus_official_metrics
from ms2_eval.visualize import save_shared_visualization


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def choose_uniform(samples, count: int):
    if count <= 0 or count >= len(samples): return samples
    return [samples[int(index)] for index in np.linspace(0, len(samples) - 1, count, dtype=int)]


def task_embedding(device, dtype):
    task = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
    return torch.cat([torch.sin(task), torch.cos(task)], dim=-1)


def infer_direct(anythermal, bridge, lotus, thermal_path: str, *, seed: int, device: torch.device):
    encoded = anythermal.encode(Path(thermal_path))
    spatial = encoded["spatial_features"]
    height, width = map(int, encoded["original_shape"][:2])
    target = (height // int(lotus.vae_scale_factor), width // int(lotus.vae_scale_factor))
    dtype = lotus.unet.dtype
    condition = bridge(spatial.to(device=device, dtype=dtype), target_size=target,
                       output_channels=int(lotus.unet.config.in_channels) // 2)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn((1, int(lotus.unet.config.in_channels) // 2, *target), generator=generator,
                        device=device, dtype=dtype) * lotus.scheduler.init_noise_sigma
    timestep = torch.tensor(999, device=device, dtype=torch.long)
    prompt, _ = lotus.encode_prompt(prompt="", device=device, num_images_per_prompt=1,
                                    do_classifier_free_guidance=None)
    latent_input = lotus.scheduler.scale_model_input(noise, timestep)
    unet_input = torch.cat([condition, latent_input], dim=1)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        x0 = lotus.unet(unet_input, timestep, encoder_hidden_states=prompt.to(dtype=dtype),
                        class_labels=task_embedding(device, dtype), return_dict=False)[0]
        decoded = lotus.vae.decode(x0 / lotus.vae.config.scaling_factor, return_dict=False)[0]
        image = lotus.image_processor.postprocess(decoded, output_type="np", do_denormalize=[True])[0]
    disparity = np.asarray(image, np.float32).mean(axis=-1)
    if disparity.shape != (height, width):
        raise RuntimeError(f"Decoded shape {disparity.shape} != thermal shape {(height, width)}")
    if not np.isfinite(disparity).all(): raise RuntimeError("Direct prediction contains NaN/Inf")
    diagnostics = {"condition_shape": list(condition.shape), "noise_shape": list(noise.shape),
                   "unet_input_shape": list(unet_input.shape), "x0_shape": list(x0.shape),
                   "bridge_parameters": sum(p.numel() for p in bridge.parameters()), "seed": seed}
    return disparity, diagnostics


def main():
    args = parse_args(); output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()): raise SystemExit(f"Refusing non-empty output directory: {output}")
    for name in ("predictions", "metrics", "visualizations", "logs"): (output / name).mkdir(parents=True, exist_ok=True)
    all_samples, manifest_info = load_manifest(args.manifest, args.ms2_root)
    samples = choose_uniform(all_samples, args.max_samples)
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    anythermal = AnyThermalEncoder(model_path=args.anythermal_model_path, device=args.device,
                                   local_files_only=args.local_files_only)
    lotus = LotusGPipeline.from_pretrained(args.lotus_model_path, torch_dtype=dtype,
                                           local_files_only=args.local_files_only).to(device)
    for module in (lotus.vae, lotus.text_encoder, lotus.unet): module.requires_grad_(False).eval()
    bridge = AnyThermalLotusBridge().to(device).eval()
    if sum(p.numel() for p in bridge.parameters()) != 0: raise RuntimeError("Direct bridge is not zero-parameter")

    rows, diagnostics = [], []
    for index, sample in enumerate(samples):
        sample_seed = args.seed + index
        disparity, diag = infer_direct(anythermal, bridge, lotus, sample.thermal_path,
                                        seed=sample_seed, device=device)
        raw, gt = load_ms2_gt(sample.thermal_gt_path, args.depth_scale)
        valid = np.isfinite(gt) & (gt > args.min_depth) & (gt < args.max_depth)
        aligned, scale, shift = align_lotus_disparity_to_ms2_depth(
            disparity, gt, valid, min_depth_m=args.min_depth, max_depth_m=args.max_depth)
        metrics = lotus_official_metrics(aligned, gt, valid)
        row = {"sample_id": sample.sample_id, "sequence": sample.sequence, "condition": sample.condition,
               "seed": sample_seed, "alignment_scale": scale, "alignment_shift": shift, **metrics}
        rows.append(row); diagnostics.append({"sample_id": sample.sample_id, **diag,
            "thermal_hw": list(gt.shape), "gt_raw_dtype": str(raw.dtype), "valid_pixels": int(valid.sum())})
        safe = sample.sample_id.replace("/", "_").replace("\\", "_")
        np.save(output / "predictions" / f"{safe}__raw_disparity.npy", disparity.astype(np.float32))
        np.save(output / "predictions" / f"{safe}__aligned_depth_m.npy", aligned.astype(np.float32))
        native_relative_depth = 1.0 / np.clip(disparity, 1e-6, None)
        save_shared_visualization(output / "visualizations" / safe, sample_id=sample.sample_id,
            thermal_path=sample.thermal_path, gt_m=gt, valid=valid,
            native_depth=native_relative_depth, aligned_depth_m=aligned,
            raw_is_metric=False, depth_range_m=(args.min_depth, args.max_depth), colormap="magma_r")

    with (output / "metrics" / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = aggregate_imagewise(rows)
    metadata = {"protocol": "Lotus-official-MS2-internal-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "route": "direct-zero-parameter-AnyThermal-to-Lotus-G", "caption": "empty", "manifest": manifest_info,
        "selected_sample_ids": [s.sample_id for s in samples], "sample_count": len(samples),
        "models": {"anythermal": args.anythermal_model_path, "lotus": args.lotus_model_path},
        "representation": "Lotus-G disparity", "alignment": "least_square_disparity",
        "metrics": list(OFFICIAL_METRICS), "depth_scale": args.depth_scale,
        "valid_rule": f"finite(gt) & gt > {args.min_depth} & gt < {args.max_depth}",
        "noise_policy": "seed + uniform-selection index; future routes must reuse exact per-sample seeds",
        "all_models_frozen": True, "bridge_parameters": 0}
    write_json(output / "run_metadata.json", metadata); write_json(output / "metrics" / "summary.json", summary)
    write_json(output / "logs" / "inference_diagnostics.json", diagnostics)
    print(json.dumps({"status": "complete", "output": str(output), "samples": len(samples),
                      "absrel": summary["metrics"]["abs_relative_difference"]["mean"],
                      "rmse": summary["metrics"]["rmse_linear"]["mean"],
                      "delta1": summary["metrics"]["delta1_acc"]["mean"]}, indent=2))


if __name__ == "__main__": main()
