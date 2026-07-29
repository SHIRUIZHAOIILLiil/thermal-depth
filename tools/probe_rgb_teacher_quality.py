"""Probe 2: how good is pretrained Lotus as an RGB depth teacher on MS2?

Runs the pretrained Lotus-G pipeline on paired RGB frames at native
resolution (1224x384), single-step x0 with empty prompt, aligns predictions
to the RGB-view LiDAR GT (least squares in disparity space), and reports
depth metrics in the RGB view.  No training, no thermal involvement.

Judgment: the teacher must be clearly stronger in its own view than our
thermal champion (AbsRel 0.1226) for dense cross-view distillation to have
profit margin.
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
    parser.add_argument(
        "--unet-checkpoint",
        type=Path,
        default=None,
        help="Optional trained U-Net checkpoint (loads lotus_unet_state_dict).",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def task_embedding(device, dtype):
    task = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
    return torch.cat([torch.sin(task), torch.cos(task)], dim=-1)


def metrics_from_depth(pred_depth, gt_depth, valid):
    gt = gt_depth[valid]
    pred = pred_depth[valid]
    abs_rel = float(np.mean(np.abs(pred - gt) / gt))
    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    ratio = np.maximum(pred / gt, gt / pred)
    d1 = float((ratio < 1.25).mean())
    log_diff = np.log(pred) - np.log(gt)
    silog = float(np.sqrt(np.mean(log_diff**2) - np.mean(log_diff) ** 2) * 100)
    return {"abs_rel": abs_rel, "rmse": rmse, "delta1": d1, "silog": silog, "valid_px": int(valid.sum())}


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    picks = [rows[int(i)] for i in np.linspace(0, len(rows) - 1, args.samples, dtype=int)]

    device = torch.device("cuda")
    dtype = torch.float16
    from pipeline import LotusGPipeline

    lotus = LotusGPipeline.from_pretrained(
        args.lotus_model_path, torch_dtype=dtype, local_files_only=args.local_files_only
    ).to(device)
    unet_checkpoint_info = {}
    if args.unet_checkpoint is not None:
        checkpoint = torch.load(
            args.unet_checkpoint.resolve(), map_location="cpu", weights_only=False
        )
        lotus.unet.load_state_dict(checkpoint["lotus_unet_state_dict"], strict=True)
        unet_checkpoint_info = {
            "unet_checkpoint": str(args.unet_checkpoint.resolve()),
            "unet_checkpoint_format": str(checkpoint.get("format", "?")),
            "unet_checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        }
    for module in (lotus.vae, lotus.text_encoder, lotus.unet):
        module.requires_grad_(False).eval()
    prompt, _ = lotus.encode_prompt(
        prompt="", device=device, num_images_per_prompt=1, do_classifier_free_guidance=None
    )
    prompt = prompt.detach().to(dtype=dtype)

    records = []
    for index, row in enumerate(picks):
        rgb = Image.open(args.ms2_root / row["rgb_path"]).convert("RGB")
        width, height = rgb.size
        width -= width % 8
        height -= height % 8
        rgb = rgb.crop((0, 0, width, height))
        array = np.asarray(rgb, np.float32) / 255.0
        tensor = torch.from_numpy(array * 2 - 1).permute(2, 0, 1)[None].to(device=device, dtype=dtype)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
            latent = lotus.vae.encode(tensor).latent_dist.mode() * lotus.vae.config.scaling_factor
            generator = torch.Generator(device=device).manual_seed(args.seed + index)
            noise = torch.randn(latent.shape, generator=generator, device=device, dtype=dtype)
            noise = noise * lotus.scheduler.init_noise_sigma
            timestep = torch.tensor(999, device=device, dtype=torch.long)
            latent_input = lotus.scheduler.scale_model_input(noise, timestep)
            x0 = lotus.unet(
                torch.cat([latent, latent_input], dim=1),
                timestep,
                encoder_hidden_states=prompt,
                class_labels=task_embedding(device, dtype),
                return_dict=False,
            )[0]
            decoded = lotus.vae.decode(x0 / lotus.vae.config.scaling_factor, return_dict=False)[0]
            image = lotus.image_processor.postprocess(decoded, output_type="np", do_denormalize=[True])[0]
        pred_disparity = np.asarray(image, np.float64).mean(axis=-1)

        gt_depth = np.asarray(
            Image.open(args.ms2_root / row["rgb_depth_path"]), np.float64
        )[:height, :width] / args.depth_scale
        valid = (gt_depth > args.min_depth) & (gt_depth < args.max_depth) & np.isfinite(gt_depth)
        if valid.sum() < 100:
            continue
        gt_disp = np.zeros_like(gt_depth)
        gt_disp[valid] = 1.0 / gt_depth[valid]
        pred = pred_disparity[valid]
        design = np.stack([pred, np.ones_like(pred)], axis=1)
        (scale, shift), *_ = np.linalg.lstsq(design, gt_disp[valid], rcond=None)
        aligned = np.clip(
            scale * pred_disparity + shift, 1.0 / args.max_depth, 1.0 / args.min_depth
        )
        pred_depth = 1.0 / aligned
        record = {"id": row["id"], **metrics_from_depth(pred_depth, gt_depth, valid)}
        records.append(record)
        if index < 4:
            vis = ((pred_disparity - pred_disparity.min()) /
                   max(pred_disparity.max() - pred_disparity.min(), 1e-6) * 255).astype(np.uint8)
            Image.fromarray(vis).save(output / f"rgb_teacher_disp_{index}.png")
            rgb.save(output / f"rgb_input_{index}.png")

    summary = {
        "samples": len(records),
        **unet_checkpoint_info,
        "mean": {
            key: float(np.mean([record[key] for record in records]))
            for key in ("abs_rel", "rmse", "delta1", "silog")
        },
        "median_abs_rel": float(np.median([record["abs_rel"] for record in records])),
        "reference": {
            "thermal_champion_val_full": 0.1226,
            "thermal_vae_direct_val_full": 0.1291,
            "note": "different view/GT; magnitude comparison only",
        },
        "records": records,
    }
    (output / "rgb_teacher_quality.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in ("samples", "mean", "median_abs_rel")}, indent=2))


if __name__ == "__main__":
    main()
