"""Count inference-time parameters for the four MS2 baselines and for our stack.

Neither side needs trained weights: `build_network` takes `pre_trained=False,
ckpt_path=None`, and our three modules can be instantiated from their configs.
So this runs on CPU in seconds and can be re-run anywhere the code is checked out.

The number that belongs in the paper is what runs at inference. For us that is
U-Net + VAE decoder + CLIP text tower; the VAE encoder never sees a test image
(we encode the thermal frame, so it does), and the text tower runs once per caption.
Both columns are reported so the choice is visible rather than assumed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def baselines(bmsd_root: Path) -> dict[str, int]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_ms2_supdepth_baselines import build_network

    out = {}
    for name in ("dorn", "bts", "adabins", "newcrf"):
        try:
            out[name] = count(build_network(name, bmsd_root))
        except Exception as exc:                      # one missing dep must not hide the rest
            out[name] = f"FAILED: {type(exc).__name__}: {exc}"
    return out


def ours(model_name: str) -> dict[str, int]:
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel

    parts = {}
    for key, cls, sub in (
        ("unet", UNet2DConditionModel, "unet"),
        ("vae", AutoencoderKL, "vae"),
        ("text_encoder", CLIPTextModel, "text_encoder"),
    ):
        try:
            net = cls.from_pretrained(model_name, subfolder=sub, torch_dtype=torch.float32)
            parts[key] = count(net)
            if key == "vae":
                parts["vae_encoder"] = count(net.encoder) + count(net.quant_conv)
                parts["vae_decoder"] = count(net.decoder) + count(net.post_quant_conv)
            del net
        except Exception as exc:
            parts[key] = f"FAILED: {type(exc).__name__}: {exc}"
    return parts


def fmt(n) -> str:
    return f"{n/1e6:8.1f}M" if isinstance(n, int) else str(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bmsd-root", type=Path,
                    default=Path("/mnt/e/project/2026-6-6-depth-estimate/SupDepth4Thermal"))
    ap.add_argument("--model-name", default="jingheya/lotus-depth-g-v2-1-disparity")
    ap.add_argument("--skip-ours", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    if not args.skip_baselines:
        print("=== MS2 监督基线（SupDepth4Thermal 发布配置，无需权重）===")
        if not args.bmsd_root.is_dir():
            print(f"!! 找不到 {args.bmsd_root}")
        else:
            for name, n in baselines(args.bmsd_root).items():
                print(f"  {name:10} {fmt(n)}")

    if not args.skip_ours:
        print(f"\n=== 我们（{args.model_name}）===")
        parts = ours(args.model_name)
        for key in ("unet", "vae", "vae_encoder", "vae_decoder", "text_encoder"):
            if key in parts:
                print(f"  {key:14} {fmt(parts[key])}")
        nums = [parts.get(k) for k in ("unet", "vae", "text_encoder")]
        if all(isinstance(v, int) for v in nums):
            print(f"  {'合计':14} {fmt(sum(nums))}")


if __name__ == "__main__":
    main()
