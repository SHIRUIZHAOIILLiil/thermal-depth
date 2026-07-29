"""Smoke test for the zero-parameter AnyThermal-to-Lotus bridge."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.anythermal_encoder import AnyThermalEncoder  # noqa: E402
from models.anythermal_lotus_bridge import AnyThermalLotusBridge  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test AnyThermal bridge for the Lotus-D latent shape."
    )
    parser.add_argument("--thermal-image", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-height", type=int, default=30)
    parser.add_argument("--target-width", type=int, default=96)
    parser.add_argument("--output-channels", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-fallback-preprocess", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.thermal_image)
    if not image_path.is_file():
        print(f"[path error] Thermal image does not exist: {image_path}", file=sys.stderr)
        return 2

    try:
        encoder = AnyThermalEncoder(
            model_path=args.model_path,
            device=args.device,
            revision=args.revision,
            local_files_only=args.local_files_only,
            fallback_preprocess=not args.no_fallback_preprocess,
        )
        bridge = AnyThermalLotusBridge().to(encoder.device)
        anythermal_output = encoder.encode(image_path)
        spatial_features = anythermal_output["spatial_features"]
        target_size = (args.target_height, args.target_width)
        resized_features = bridge.resize_spatial_features(
            spatial_features,
            target_size,
        )
        projected_features = bridge.project_channels(
            resized_features,
            output_channels=args.output_channels,
        )
        forward_features = bridge(
            spatial_features,
            target_size=target_size,
            output_channels=args.output_channels,
        )
    except Exception as exc:
        print(f"[bridge error] {exc}", file=sys.stderr)
        return 3

    bridge_parameters = sum(parameter.numel() for parameter in bridge.parameters())
    output_is_finite = bool(torch.isfinite(projected_features).all())
    channels_per_group = resized_features.shape[1] // args.output_channels
    expected_shape = (
        spatial_features.shape[0],
        args.output_channels,
        args.target_height,
        args.target_width,
    )

    print(f"原始 AnyThermal spatial shape: {tuple(spatial_features.shape)}")
    print(f"resize 后 shape: {tuple(resized_features.shape)}")
    print(f"channel projection 后 shape: {tuple(projected_features.shape)}")
    print(f"每组通道数: {channels_per_group}")
    print(f"bridge 参数量: {bridge_parameters}")
    print(f"输出 dtype: {projected_features.dtype}")
    print(f"输出 device: {projected_features.device}")
    print(f"输出 requires_grad: {projected_features.requires_grad}")
    print(f"finite 检查: {output_is_finite}")
    print(f"输出 mean: {float(projected_features.mean())}")
    print(f"输出 std: {float(projected_features.std())}")
    print(f"输出 min: {float(projected_features.min())}")
    print(f"输出 max: {float(projected_features.max())}")

    checks_passed = (
        tuple(projected_features.shape) == expected_shape
        and torch.equal(projected_features, forward_features)
        and bridge_parameters == 0
        and projected_features.dtype == spatial_features.dtype
        and projected_features.device == spatial_features.device
        and projected_features.requires_grad == spatial_features.requires_grad
        and not projected_features.requires_grad
        and output_is_finite
    )
    if not checks_passed:
        print("[bridge error] Bridge checks failed.", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())