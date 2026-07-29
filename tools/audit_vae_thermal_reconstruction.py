"""Probe: how much does the RGB-pretrained Lotus VAE lose on thermal images?

Encodes N thermal images (pipeline preprocessing) and their paired RGB images
(resized to the same budget) through the frozen VAE, decodes them back, and
reports reconstruction L1 / PSNR plus a high-frequency retention ratio
(Sobel energy of reconstruction vs original).  Saves side-by-side panels
(original | reconstruction | 5x residual) for eyeballing.

If thermal reconstruction is on par with RGB, the "VAE drops thermal detail"
hypothesis is weak and encoder fine-tuning should not be funded.
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def sobel_energy(gray: np.ndarray) -> float:
    gx = np.zeros_like(gray); gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return float((np.abs(gx) + np.abs(gy)).mean())


def metrics(original: np.ndarray, reconstruction: np.ndarray):
    """original/reconstruction: float arrays in [0,1], shape [H,W,3]."""
    l1 = float(np.abs(original - reconstruction).mean())
    mse = float(((original - reconstruction) ** 2).mean())
    psnr = float(10 * np.log10(1.0 / max(mse, 1e-12)))
    gray_o = original.mean(axis=-1)
    gray_r = reconstruction.mean(axis=-1)
    hf_o = sobel_energy(gray_o)
    hf_r = sobel_energy(gray_r)
    return {
        "l1": l1,
        "psnr": psnr,
        "hf_energy_original": hf_o,
        "hf_energy_reconstruction": hf_r,
        "hf_retention_ratio": hf_r / max(hf_o, 1e-9),
    }


def panel(original: np.ndarray, reconstruction: np.ndarray, out_path: Path):
    residual = np.clip(np.abs(original - reconstruction) * 5.0, 0, 1)
    strip = np.concatenate([original, reconstruction, residual], axis=1)
    Image.fromarray((strip * 255).astype(np.uint8)).save(out_path)


@torch.no_grad()
def reconstruct(vae, image_tensor, device):
    """image_tensor: [1,3,H,W] in [-1,1]. Returns [H,W,3] float in [0,1]."""
    latent = vae.encode(image_tensor.to(device=device, dtype=vae.dtype)).latent_dist.mode()
    decoded = vae.decode(latent, return_dict=False)[0]
    out = (decoded.float().clamp(-1, 1) / 2 + 0.5)[0].permute(1, 2, 0).cpu().numpy()
    return out


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
    from pipeline import LotusGPipeline

    lotus = LotusGPipeline.from_pretrained(
        args.lotus_model_path,
        torch_dtype=torch.float16,
        local_files_only=args.local_files_only,
    ).to(device)
    vae = lotus.vae.requires_grad_(False).eval()

    records = []
    for index, row in enumerate(picks):
        thermal = thermal_to_lotus_input(args.ms2_root / row["thermal_path"], processing_res=0)
        thermal_original = (thermal.tensor.float()[0].permute(1, 2, 0).numpy() / 2 + 0.5).clip(0, 1)
        thermal_recon = reconstruct(vae, thermal.tensor, device)
        thermal_metrics = metrics(thermal_original, thermal_recon)

        rgb = Image.open(args.ms2_root / row["rgb_path"]).convert("RGB").resize(
            (thermal_original.shape[1], thermal_original.shape[0]), Image.BILINEAR
        )
        rgb_original = np.asarray(rgb, np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb_original * 2 - 1).permute(2, 0, 1)[None]
        rgb_recon = reconstruct(vae, rgb_tensor, device)
        rgb_metrics = metrics(rgb_original, rgb_recon)

        records.append({"id": row["id"], "thermal": thermal_metrics, "rgb": rgb_metrics})
        if index < 4:
            panel(thermal_original, thermal_recon, output / f"panel_thermal_{index}.png")
            panel(rgb_original, rgb_recon, output / f"panel_rgb_{index}.png")

    def aggregate(key):
        return {
            metric: float(np.mean([record[key][metric] for record in records]))
            for metric in records[0][key]
        }

    summary = {
        "samples": len(records),
        "thermal_mean": aggregate("thermal"),
        "rgb_mean": aggregate("rgb"),
        "records": records,
    }
    (output / "vae_reconstruction_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in ("samples", "thermal_mean", "rgb_mean")}, indent=2))


if __name__ == "__main__":
    main()
