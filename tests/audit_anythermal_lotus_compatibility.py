"""Compare bridged AnyThermal features with the Lotus-D VAE latent interface."""

from __future__ import annotations

import argparse
import gc
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = REPO_ROOT / "lotus"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.anythermal_encoder import AnyThermalEncoder  # noqa: E402
from models.anythermal_lotus_bridge import AnyThermalLotusBridge  # noqa: E402

if str(LOTUS_ROOT) not in sys.path:
    sys.path.insert(0, str(LOTUS_ROOT))

from pipeline import LotusDPipeline  # noqa: E402
from utils.image_utils import get_tv_resample_method, resize_max_res  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit AnyThermal bridge and Lotus-D latent statistics."
    )
    parser.add_argument("--thermal-image", required=True)
    parser.add_argument("--rgb-image", required=True)
    parser.add_argument("--anythermal-model-path", required=True)
    parser.add_argument("--lotus-model-path", required=True)
    parser.add_argument("--anythermal-revision", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--processing-res", type=int, default=768)
    parser.add_argument("--half-precision", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> None:
    for name in ("thermal_image", "rgb_image"):
        path = Path(getattr(args, name))
        if not path.is_file():
            raise SystemExit(f"{name.replace('_', ' ')} does not exist: {path}")
    if args.processing_res < 0:
        raise SystemExit("--processing-res must be greater than or equal to zero.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"CUDA is unavailable; cannot use --device {args.device}.")


def encode_lotus_rgb(
    pipeline: LotusDPipeline,
    image_path: Path,
    processing_res: int,
    device: torch.device,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image, dtype=np.float32)
    rgb_in = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    rgb_in = rgb_in / 127.5 - 1.0
    if processing_res > 0:
        rgb_in = resize_max_res(
            rgb_in,
            max_edge_resolution=processing_res,
            resample_method=get_tv_resample_method("bilinear"),
        )

    autocast_context = torch.autocast(device.type) if device.type == "cuda" else nullcontext()
    with torch.inference_mode(), autocast_context:
        rgb_latents = pipeline.vae.encode(rgb_in.to(device)).latent_dist.sample()
        return rgb_latents * pipeline.vae.config.scaling_factor


def tensor_statistics(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().float()
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "absolute_mean": float(values.abs().mean()),
        "l2_norm": float(torch.linalg.vector_norm(values)),
    }


def print_statistics(name: str, statistics: dict[str, float]) -> None:
    print(f"{name}:")
    for key in ("mean", "std", "min", "max", "absolute_mean", "l2_norm"):
        print(f"- {key}: {statistics[key]}")


def main() -> int:
    args = parse_args()
    validate_inputs(args)
    device = torch.device(args.device)

    anythermal = AnyThermalEncoder(
        model_path=args.anythermal_model_path,
        device=args.device,
        revision=args.anythermal_revision,
        local_files_only=args.local_files_only,
    )
    anythermal_output = anythermal.encode(args.thermal_image)
    patch_token_shape = tuple(anythermal_output["patch_tokens"].shape)
    spatial_feature_shape = tuple(anythermal_output["spatial_features"].shape)
    anythermal_resolution = tuple(anythermal_output["grid_size"])
    anythermal_spatial_features = anythermal_output["spatial_features"].detach().cpu()

    del anythermal_output, anythermal
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    lotus_dtype = torch.float16 if args.half_precision else torch.float32
    lotus = LotusDPipeline.from_pretrained(
        args.lotus_model_path,
        torch_dtype=lotus_dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    lotus_rgb_latent = encode_lotus_rgb(
        lotus,
        Path(args.rgb_image),
        args.processing_res,
        device,
    )
    lotus_latent_shape = tuple(lotus_rgb_latent.shape)
    lotus_expected_channels = int(lotus.unet.config.in_channels)
    lotus_spatial_resolution = lotus_latent_shape[-2:]
    lotus_conv_in_shape = tuple(lotus.unet.conv_in.weight.shape)
    lotus_vae_scaling_factor = float(lotus.vae.config.scaling_factor)

    bridge = AnyThermalLotusBridge().to(device)
    bridged_anythermal_latent = bridge(
        anythermal_spatial_features.to(device),
        target_size=lotus_spatial_resolution,
        output_channels=lotus_expected_channels,
    )
    bridged_latent_shape = tuple(bridged_anythermal_latent.shape)
    channels_per_group = spatial_feature_shape[1] // lotus_expected_channels
    bridge_parameters = sum(parameter.numel() for parameter in bridge.parameters())
    bridged_statistics = tensor_statistics(bridged_anythermal_latent)
    lotus_statistics = tensor_statistics(lotus_rgb_latent)

    channel_shape_compatible = bridged_latent_shape[1] == lotus_expected_channels
    spatial_shape_compatible = bridged_latent_shape[-2:] == lotus_spatial_resolution
    representation_compatible = False
    direct_insertion_possible = False

    print(f"AnyThermal patch token shape: {patch_token_shape}")
    print(f"AnyThermal spatial feature shape: {spatial_feature_shape}")
    print(f"AnyThermal spatial resolution: {anythermal_resolution}")
    print("AnyThermal representation: DINOv2 patch feature")
    print(f"Bridged AnyThermal latent shape: {bridged_latent_shape}")
    print(f"Channels per group: {channels_per_group}")
    print(f"Bridge parameters: {bridge_parameters}")
    print(f"Lotus RGB latent shape: {lotus_latent_shape}")
    print(f"Lotus U-Net expected input channels: {lotus_expected_channels}")
    print(f"Lotus U-Net spatial resolution: {lotus_spatial_resolution}")
    print(f"Lotus U-Net conv_in weight shape: {lotus_conv_in_shape}")
    print(f"Lotus VAE scaling factor: {lotus_vae_scaling_factor}")
    print("Lotus representation: Stable-Diffusion VAE latent")
    print_statistics("Bridged AnyThermal latent", bridged_statistics)
    print_statistics("Lotus RGB VAE latent", lotus_statistics)
    print(f"Bridged channel shape compatible: {channel_shape_compatible}")
    print(f"Bridged spatial shape compatible: {spatial_shape_compatible}")
    print(f"Representation compatible: {representation_compatible}")
    print(f"Direct insertion possible: {direct_insertion_possible}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())