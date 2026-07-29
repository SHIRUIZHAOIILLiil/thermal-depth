"""Smoke test for the standalone AnyThermal encoder wrapper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.anythermal_encoder import AnyThermalEncoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AnyThermal encoder smoke test.")
    parser.add_argument("--image", required=True, help="Path to a thermal image.")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local HF model directory/snapshot or Hugging Face repo id, e.g. theairlabcmu/AnyThermal.",
    )
    parser.add_argument(
        "--processor-path",
        default=None,
        help="Optional processor path. Defaults to --model-path.",
    )
    parser.add_argument("--device", default="cuda", help="Device, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face model revision.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download files from Hugging Face.",
    )
    parser.add_argument(
        "--no-fallback-preprocess",
        action="store_true",
        help="Fail if AutoImageProcessor cannot be loaded.",
    )
    return parser.parse_args()


def shape(value: Any) -> tuple:
    return tuple(value.shape)


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)

    if not image_path.exists():
        print(f"[path error] Thermal image does not exist: {image_path}", file=sys.stderr)
        return 2

    try:
        image = Image.open(image_path)
        original_shape = (image.height, image.width, len(image.getbands()))
        print(f"输入 thermal image 原始 shape: {original_shape}")
    except Exception as exc:
        print(f"[image error] Failed to open image '{image_path}': {exc}", file=sys.stderr)
        return 2

    try:
        encoder = AnyThermalEncoder(
            model_path=args.model_path,
            processor_path=args.processor_path,
            device=args.device,
            revision=args.revision,
            local_files_only=args.local_files_only,
            fallback_preprocess=not args.no_fallback_preprocess,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "CUDA" in message or "cuda" in message:
            print(f"[cuda error] {message}", file=sys.stderr)
        else:
            print(f"[model loading error] {message}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"[model loading error] {exc}", file=sys.stderr)
        return 3

    try:
        features = encoder.encode(image)
    except RuntimeError as exc:
        message = str(exc)
        if "CUDA" in message or "cuda" in message:
            print(f"[cuda error] Forward pass failed: {message}", file=sys.stderr)
        else:
            print(f"[forward error] {message}", file=sys.stderr)
        return 4
    except Exception as exc:
        print(f"[forward error] {exc}", file=sys.stderr)
        return 4

    last_hidden_state = features["last_hidden_state"]
    cls_token = features["cls_token"]
    register_tokens = features["register_tokens"]
    patch_tokens = features["patch_tokens"]
    spatial_features = features["spatial_features"]
    parameter_summary = encoder.parameter_summary()

    print(f"预处理后的 tensor shape: {features['preprocessed_shape']}")
    print(f"模型所在 device: {features['model_device']}")
    print(f"register token 数量: {features['num_register_tokens']}")
    print(f"last_hidden_state shape: {shape(last_hidden_state)}")
    print(f"CLS token shape: {shape(cls_token)}")
    print(f"register token shape: {shape(register_tokens)}")
    print(f"patch token shape: {shape(patch_tokens)}")
    print(f"patch size: {encoder.patch_size}")
    print(f"patch grid size: {features['grid_size']}")
    print(f"spatial feature shape: {shape(spatial_features)}")
    print(f"spatial feature mean: {float(spatial_features.mean())}")
    print(f"spatial feature std: {float(spatial_features.std())}")
    print(f"spatial feature finite: {bool(torch.isfinite(spatial_features).all())}")
    print(f"特征 mean: {float(last_hidden_state.mean())}")
    print(f"特征 std: {float(last_hidden_state.std())}")
    print(f"所有特征是否为 finite: {bool(torch.isfinite(last_hidden_state).all())}")
    print(f"模型总参数量: {parameter_summary['total_parameters']}")
    print(f"可训练参数量: {parameter_summary['trainable_parameters']}")
    print(f"冻结参数量: {parameter_summary['frozen_parameters']}")
    print(f"模型是否完全冻结: {parameter_summary['is_fully_frozen']}")
    print(f"模型 training 状态: {encoder.model.training}")
    print(f"last_hidden_state.requires_grad: {last_hidden_state.requires_grad}")
    print(f"patch_tokens.requires_grad: {patch_tokens.requires_grad}")
    print(f"spatial_features.requires_grad: {spatial_features.requires_grad}")

    freeze_checks_passed = (
        parameter_summary["is_fully_frozen"]
        and not encoder.model.training
        and not last_hidden_state.requires_grad
        and not patch_tokens.requires_grad
        and not spatial_features.requires_grad
        and bool(torch.isfinite(spatial_features).all())
    )
    if not freeze_checks_passed:
        print("[freeze error] AnyThermal encoder freeze checks failed.", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

