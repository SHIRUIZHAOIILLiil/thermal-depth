"""One-batch forward/backward smoke test for AnyThermal -> Adapter -> Lotus-G."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = REPO_ROOT / "lotus"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LOTUS_ROOT) not in sys.path:
    sys.path.insert(0, str(LOTUS_ROOT))

from diffusers import DDPMScheduler  # noqa: E402

from models.anythermal_encoder import AnyThermalEncoder  # noqa: E402
from models.anythermal_lotus_adapter import AnyThermalLotusAdapter  # noqa: E402
from models.anythermal_lotus_model import AnyThermalLotusModel  # noqa: E402
from pipeline import LotusGPipeline  # noqa: E402
from utils.image_utils import get_tv_resample_method, resize_max_res  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thermal-image", required=True)
    parser.add_argument("--depth-image", required=True)
    parser.add_argument("--rgb-reference-image", default=None)
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument(
        "--lotus-model-path",
        default="jingheya/lotus-depth-g-v2-1-disparity",
    )
    parser.add_argument("--anythermal-revision", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--processing-res", type=int, default=768)
    parser.add_argument("--prediction-type", default="sample")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--half-precision", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", default="outputs/adapter_v0")
    return parser.parse_args()


def tensor_statistics(tensor: torch.Tensor) -> Dict[str, Any]:
    values = tensor.detach().float().cpu()
    return {
        "shape": list(values.shape),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "l2_norm": float(torch.linalg.vector_norm(values)),
        "finite": bool(torch.isfinite(values).all()),
    }


def module_grad_status(module: torch.nn.Module) -> Dict[str, Any]:
    gradients = [p.grad.detach() for p in module.parameters() if p.grad is not None]
    trainable_parameters = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if not gradients:
        return {
            "trainable_parameters": trainable_parameters,
            "has_grad": False,
            "finite": True,
            "max_abs_grad": 0.0,
            "l2_norm": 0.0,
        }
    max_abs = max(float(g.float().abs().max()) for g in gradients)
    l2_norm = float(
        torch.sqrt(sum((g.float() ** 2).sum() for g in gradients))
    )
    return {
        "trainable_parameters": trainable_parameters,
        "has_grad": any(bool((g != 0).any()) for g in gradients),
        "finite": all(bool(torch.isfinite(g).all()) for g in gradients),
        "max_abs_grad": max_abs,
        "l2_norm": l2_norm,
    }


def any_parameter_has_grad(module: torch.nn.Module) -> bool:
    return any(p.grad is not None for p in module.parameters())


def load_depth_target(path: Path) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        depth = np.asarray(image).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth image, got {depth.shape}.")
    valid_mask = np.isfinite(depth) & (depth > 0)
    if valid_mask.any():
        valid_values = depth[valid_mask]
        near = float(valid_values.min())
        far = float(valid_values.max())
        if far > near:
            normalized = (depth - near) / (far - near)
        else:
            normalized = np.zeros_like(depth, dtype=np.float32)
    else:
        normalized = np.zeros_like(depth, dtype=np.float32)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~valid_mask] = 0.0
    target = torch.from_numpy(normalized).view(1, 1, *normalized.shape)
    target = target.repeat(1, 3, 1, 1) * 2.0 - 1.0
    return target.float(), depth, valid_mask


def save_minmax_image(array: np.ndarray, path: Path) -> None:
    values = np.asarray(array, dtype=np.float32)
    if values.ndim == 3:
        values = values[..., 0]
    finite = np.isfinite(values)
    if finite.any():
        lo = float(values[finite].min())
        hi = float(values[finite].max())
        if hi > lo:
            values = (values - lo) / (hi - lo)
        else:
            values = np.zeros_like(values)
    else:
        values = np.zeros_like(values)
    values = np.clip(values, 0.0, 1.0)
    Image.fromarray((values * 255.0).round().astype(np.uint8), mode="L").save(path)


def image_to_lotus_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor / 127.5 - 1.0


def encode_reference_latent(
    lotus: LotusGPipeline,
    image: Image.Image,
    *,
    processing_res: int,
    device: torch.device,
) -> torch.Tensor:
    tensor = image_to_lotus_tensor(image)
    if processing_res > 0:
        tensor = resize_max_res(
            tensor,
            max_edge_resolution=processing_res,
            resample_method=get_tv_resample_method("bilinear"),
        )
    dtype = next(lotus.vae.parameters()).dtype
    autocast_context = torch.autocast(device.type) if device.type == "cuda" else nullcontext()
    with torch.no_grad(), autocast_context:
        latents = lotus.vae.encode(tensor.to(device=device, dtype=dtype)).latent_dist.sample()
        return latents * lotus.vae.config.scaling_factor


def main() -> int:
    args = parse_args()
    thermal_path = Path(args.thermal_image)
    depth_path = Path(args.depth_image)
    output_dir = Path(args.output_dir)
    if not thermal_path.is_file():
        raise SystemExit(f"Thermal image not found: {thermal_path}")
    if not depth_path.is_file():
        raise SystemExit(f"Depth image not found: {depth_path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"CUDA is not available for --device {args.device}")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)

    anythermal = AnyThermalEncoder(
        model_path=args.anythermal_model_path,
        device=args.device,
        revision=args.anythermal_revision,
        local_files_only=args.local_files_only,
    )
    lotus_dtype = torch.float16 if args.half_precision else torch.float32
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.lotus_model_path,
        subfolder="scheduler",
        local_files_only=args.local_files_only,
    )
    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
    lotus = LotusGPipeline.from_pretrained(
        args.lotus_model_path,
        scheduler=noise_scheduler,
        torch_dtype=lotus_dtype,
        safety_checker=None,
        local_files_only=args.local_files_only,
    ).to(device)
    if int(lotus.unet.config.in_channels) != 8:
        raise SystemExit(
            "This smoke test expects Lotus-G with 8 U-Net input channels; "
            f"got {lotus.unet.config.in_channels}."
        )

    adapter = AnyThermalLotusAdapter().to(device)
    model = AnyThermalLotusModel(
        anythermal_encoder=anythermal,
        lotus_pipeline=lotus,
        adapter=adapter,
        noise_scheduler=noise_scheduler,
    ).to(device)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.learning_rate)

    depth_values, gt_depth, valid_mask = load_depth_target(depth_path)
    depth_values = depth_values.to(device)
    timesteps = torch.tensor([args.timestep], device=device, dtype=torch.long)

    outputs = model(
        thermal_image=thermal_path,
        depth_values=depth_values,
        timesteps=timesteps,
        return_decoded=True,
    )
    loss = outputs["loss"]
    checks_before_backward = {
        "adapter_output_shape_correct": list(outputs["condition_latent"].shape)
        == list(outputs["target_latents"].shape),
        "adapter_output_finite": bool(torch.isfinite(outputs["condition_latent"]).all()),
        "loss_finite": bool(torch.isfinite(loss.detach())),
        "unet_input_channels": int(outputs["unet_input"].shape[1]),
    }
    if not all(
        [
            checks_before_backward["adapter_output_shape_correct"],
            checks_before_backward["adapter_output_finite"],
            checks_before_backward["loss_finite"],
            checks_before_backward["unet_input_channels"] == 8,
        ]
    ):
        raise SystemExit(f"Pre-backward checks failed: {checks_before_backward}")

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    adapter_grad = module_grad_status(adapter)
    anythermal_has_grad = any_parameter_has_grad(anythermal.model)
    lotus_has_grad = any_parameter_has_grad(lotus.unet) or any_parameter_has_grad(lotus.vae)
    lotus_has_grad = lotus_has_grad or any_parameter_has_grad(lotus.text_encoder)
    optimizer.step()

    with torch.no_grad():
        decoded = outputs["decoded"].detach().float().cpu()
        predicted_depth = decoded.mean(dim=1)[0].numpy()
        adapter_output = outputs["condition_latent"].detach().float().cpu()

    raw_thermal, processor_rgb = anythermal.get_last_thermal_artifacts()
    reference_path = Path(args.rgb_reference_image) if args.rgb_reference_image else None
    if reference_path is not None:
        reference_image = Image.open(reference_path).convert("RGB")
        reference_source = str(reference_path)
    else:
        reference_image = processor_rgb
        reference_source = "AnyThermal processor RGB thermal image"
    rgb_reference_latent = encode_reference_latent(
        lotus,
        reference_image,
        processing_res=args.processing_res,
        device=device,
    )

    tensor_stats = {
        "rgb_vae_latent": tensor_statistics(rgb_reference_latent),
        "adapter_output": tensor_statistics(outputs["condition_latent"]),
        "target_depth_latent": tensor_statistics(outputs["target_latents"]),
        "noisy_depth_latent": tensor_statistics(outputs["noisy_depth_latents"]),
        "unet_input": tensor_statistics(outputs["unet_input"]),
        "lotus_prediction_latent": tensor_statistics(outputs["model_pred"]),
        "decoded_prediction": tensor_statistics(outputs["decoded"]),
        "anythermal_features": [
            tensor_statistics(feature) for feature in model.extract_features(thermal_path)[0]
        ],
        "adapter_output_per_channel": [
            tensor_statistics(adapter_output[:, channel : channel + 1])
            for channel in range(adapter_output.shape[1])
        ],
    }

    np.save(output_dir / "gt_depth_raw.npy", gt_depth)
    np.save(output_dir / "valid_mask.npy", valid_mask.astype(np.bool_))
    np.save(output_dir / "adapter_output.npy", adapter_output.numpy())
    np.save(output_dir / "lotus_predicted_depth_raw.npy", predicted_depth)
    with Image.open(thermal_path) as thermal_image:
        thermal_image.save(output_dir / "thermal_input.png")
    save_minmax_image(raw_thermal, output_dir / "thermal_raw_visualization.png")
    processor_rgb.save(output_dir / "thermal_processor_rgb.png")
    save_minmax_image(gt_depth, output_dir / "gt_depth_visualization.png")
    Image.fromarray((valid_mask.astype(np.uint8) * 255), mode="L").save(
        output_dir / "valid_mask.png"
    )
    save_minmax_image(predicted_depth, output_dir / "lotus_predicted_depth_visualization.png")

    feature_info = outputs["feature_info"]
    run_info = {
        "status": "passed"
        if (
            adapter_grad["has_grad"]
            and adapter_grad["finite"]
            and not anythermal_has_grad
            and not lotus_has_grad
        )
        else "failed",
        "models": {
            "anythermal": args.anythermal_model_path,
            "anythermal_revision": args.anythermal_revision,
            "lotus": args.lotus_model_path,
        },
        "inputs": {
            "thermal": str(thermal_path),
            "depth": str(depth_path),
            "rgb_reference": reference_source,
        },
        "settings": {
            "seed": args.seed,
            "timestep": args.timestep,
            "processing_res": args.processing_res,
            "prediction_type": args.prediction_type,
            "learning_rate": args.learning_rate,
            "caption": "",
        },
        "anythermal": {
            "feature_transformer_block_indices": list(
                feature_info.transformer_block_indices
            ),
            "hidden_state_indices": list(feature_info.hidden_state_indices),
            "input_thermal_original_shape": list(feature_info.original_shape),
            "preprocessed_shape": list(feature_info.preprocessed_shape),
            "token_grid_size": list(feature_info.grid_size),
            "has_cls_token": feature_info.has_cls_token,
            "num_register_tokens": feature_info.num_register_tokens,
            "diagnostics": outputs["thermal_diagnostics"],
        },
        "checks": {
            **checks_before_backward,
            "adapter_has_grad": adapter_grad["has_grad"],
            "adapter_grad_finite": adapter_grad["finite"],
            "anythermal_has_grad": anythermal_has_grad,
            "lotus_frozen_has_grad": lotus_has_grad,
            "optimizer_step_executed": True,
        },
        "gradient": {
            "adapter": adapter_grad,
        },
        "saved_files": sorted(path.name for path in output_dir.iterdir()),
    }
    with (output_dir / "tensor_stats.json").open("w", encoding="utf-8") as file:
        json.dump(tensor_stats, file, indent=2)
    with (output_dir / "run_info.json").open("w", encoding="utf-8") as file:
        json.dump(run_info, file, indent=2)

    print(json.dumps(run_info["checks"], indent=2))
    print(f"loss: {float(loss.detach().cpu())}")
    print(f"output_dir: {output_dir.resolve()}")
    return 0 if run_info["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
