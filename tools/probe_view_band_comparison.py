"""Band-wise fair re-examination of RGB vs thermal zero-shot depth quality.

For the same N scenes, runs three zero-shot routes through pretrained Lotus:
  rgb_native   RGB at native resolution (cropped to /8)
  rgb_768      RGB resized so the long side is ~768 (Lotus's comfort zone)
  thermal_vae  thermal image through the VAE condition (640x256 native)

Each route is scored against its own view's LiDAR GT, but broken down by GT
depth band (near <10 m / mid 10-30 m / far >=30 m) in addition to the overall
mean, plus each view's valid-pixel composition per band.  This separates
"composition effect" (different exam papers) from true quality differences.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = ROOT / "lotus"
for path in (ROOT, LOTUS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.anythermal_lotus_v2 import thermal_to_lotus_input  # noqa: E402

BANDS = (("near", 0.1, 10.0), ("mid", 10.0, 30.0), ("far", 30.0, 80.0))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def task_embedding(device, dtype):
    task = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
    return torch.cat([torch.sin(task), torch.cos(task)], dim=-1)


@torch.no_grad()
def predict_disparity(lotus, tensor, seed, device, dtype):
    with torch.autocast(device_type="cuda", dtype=dtype):
        latent = lotus.vae.encode(tensor.to(device=device, dtype=dtype)).latent_dist.mode()
        latent = latent * lotus.vae.config.scaling_factor
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(latent.shape, generator=generator, device=device, dtype=dtype)
        noise = noise * lotus.scheduler.init_noise_sigma
        timestep = torch.tensor(999, device=device, dtype=torch.long)
        latent_input = lotus.scheduler.scale_model_input(noise, timestep)
        x0 = lotus.unet(
            torch.cat([latent, latent_input], dim=1),
            timestep,
            encoder_hidden_states=predict_disparity.prompt,
            class_labels=task_embedding(device, dtype),
            return_dict=False,
        )[0]
        decoded = lotus.vae.decode(x0 / lotus.vae.config.scaling_factor, return_dict=False)[0]
        image = lotus.image_processor.postprocess(decoded, output_type="np", do_denormalize=[True])[0]
    return np.asarray(image, np.float64).mean(axis=-1)


def banded_abs_rel(pred_disparity, gt_depth, args):
    """Align once on all valid pixels, then report AbsRel per depth band."""
    valid = (gt_depth > args.min_depth) & (gt_depth < args.max_depth) & np.isfinite(gt_depth)
    if valid.sum() < 100:
        return None
    if pred_disparity.shape != gt_depth.shape:
        pred_disparity = np.asarray(
            Image.fromarray(pred_disparity.astype(np.float32)).resize(
                (gt_depth.shape[1], gt_depth.shape[0]), Image.BILINEAR
            ),
            np.float64,
        )
    gt_disp = np.zeros_like(gt_depth)
    gt_disp[valid] = 1.0 / gt_depth[valid]
    pred = pred_disparity[valid]
    design = np.stack([pred, np.ones_like(pred)], axis=1)
    (scale, shift), *_ = np.linalg.lstsq(design, gt_disp[valid], rcond=None)
    aligned = np.clip(scale * pred_disparity + shift, 1.0 / args.max_depth, 1.0 / args.min_depth)
    pred_depth = 1.0 / aligned
    rel = np.abs(pred_depth - gt_depth) / np.maximum(gt_depth, 1e-6)
    out = {"overall": float(rel[valid].mean()), "valid_px": int(valid.sum())}
    for name, lo, hi in BANDS:
        band = valid & (gt_depth >= lo) & (gt_depth < hi)
        out[name] = float(rel[band].mean()) if band.sum() > 50 else None
        out[f"{name}_px"] = int(band.sum())
    return out


def load_rgb(path, long_side=None):
    image = Image.open(path).convert("RGB")
    if long_side is not None:
        width, height = image.size
        ratio = long_side / max(width, height)
        image = image.resize(
            (int(width * ratio) // 8 * 8, int(height * ratio) // 8 * 8), Image.BILINEAR
        )
    else:
        width, height = image.size
        image = image.crop((0, 0, width - width % 8, height - height % 8))
    array = np.asarray(image, np.float32) / 255.0
    return torch.from_numpy(array * 2 - 1).permute(2, 0, 1)[None]


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in args.manifest.open(encoding="utf-8") if l.strip()]
    picks = [rows[int(i)] for i in np.linspace(0, len(rows) - 1, args.samples, dtype=int)]

    device = torch.device("cuda")
    dtype = torch.float16
    from pipeline import LotusGPipeline

    lotus = LotusGPipeline.from_pretrained(
        args.lotus_model_path, torch_dtype=dtype, local_files_only=args.local_files_only
    ).to(device)
    for module in (lotus.vae, lotus.text_encoder, lotus.unet):
        module.requires_grad_(False).eval()
    prompt, _ = lotus.encode_prompt(
        prompt="", device=device, num_images_per_prompt=1, do_classifier_free_guidance=None
    )
    predict_disparity.prompt = prompt.detach().to(dtype=dtype)

    results = {"rgb_native": [], "rgb_768": [], "thermal_vae": []}
    for index, row in enumerate(picks):
        gt_rgb = np.asarray(Image.open(args.ms2_root / row["rgb_depth_path"]), np.float64) / args.depth_scale
        gt_thr = np.asarray(Image.open(args.ms2_root / row["thermal_depth_path"]), np.float64) / args.depth_scale
        seed = args.seed + index

        disp = predict_disparity(lotus, load_rgb(args.ms2_root / row["rgb_path"]), seed, device, dtype)
        record = banded_abs_rel(disp, gt_rgb, args)
        if record: results["rgb_native"].append(record)

        disp = predict_disparity(
            lotus, load_rgb(args.ms2_root / row["rgb_path"], long_side=768), seed, device, dtype
        )
        record = banded_abs_rel(disp, gt_rgb, args)
        if record: results["rgb_768"].append(record)

        thermal = thermal_to_lotus_input(args.ms2_root / row["thermal_path"], processing_res=0)
        disp = predict_disparity(lotus, thermal.tensor, seed, device, dtype)
        record = banded_abs_rel(disp, gt_thr, args)
        if record: results["thermal_vae"].append(record)
        if (index + 1) % 8 == 0:
            print(f"processed {index + 1}/{len(picks)}", flush=True)

    summary = {"samples": args.samples, "routes": {}}
    for route, records in results.items():
        entry = {"n": len(records)}
        for key in ("overall", "near", "mid", "far"):
            values = [r[key] for r in records if r.get(key) is not None]
            entry[key] = float(np.mean(values)) if values else None
        total_px = sum(r["valid_px"] for r in records)
        for name, _, _ in BANDS:
            entry[f"{name}_px_share"] = float(sum(r[f"{name}_px"] for r in records) / max(total_px, 1))
        summary["routes"][route] = entry
    (output / "band_comparison.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary["routes"], indent=2))


if __name__ == "__main__":
    main()
