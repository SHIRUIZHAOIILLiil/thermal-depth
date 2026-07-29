"""Direct frozen AnyThermal-to-Lotus-D U-Net/VAE smoke test."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

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


PROMPT = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a direct frozen AnyThermal bridge through Lotus-D U-Net and VAE."
    )
    parser.add_argument("--thermal-image", required=True)
    parser.add_argument("--anythermal-model-path", required=True)
    parser.add_argument(
        "--lotus-model-path",
        default="jingheya/lotus-depth-d-v2-0-disparity",
    )
    parser.add_argument("--anythermal-revision", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--half-precision", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="outputs/anythermal_lotus_direct_smoke",
    )
    return parser.parse_args()


def tensor_statistics(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().float()
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "finite": bool(torch.isfinite(values).all()),
    }


def module_summary(module: torch.nn.Module) -> dict[str, Any]:
    total_parameters = sum(parameter.numel() for parameter in module.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "training": module.training,
        "is_fully_frozen": trainable_parameters == 0 and not module.training,
    }


def min_max_uint8(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 3 and values.shape[-1] in (1, 3):
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D thermal array, got {tuple(values.shape)}.")
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Thermal visualization input contains NaN or Inf.")
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value > min_value:
        values = (values - min_value) / (max_value - min_value)
    else:
        values = np.zeros_like(values)
    return (values * 255.0).round().astype(np.uint8)


def save_depth_visualization(depth_like: torch.Tensor, path: Path) -> None:
    image = min_max_uint8(depth_like.detach().float().cpu().numpy())
    Image.fromarray(image, mode="L").save(path)


def main() -> int:
    args = parse_args()
    thermal_image_path = Path(args.thermal_image)
    output_dir = Path(args.output_dir)
    if not thermal_image_path.is_file():
        print(f"[path error] Thermal image does not exist: {thermal_image_path}", file=sys.stderr)
        return 2
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[cuda error] CUDA is unavailable for device {args.device}.", file=sys.stderr)
        return 2

    device = torch.device(args.device)
    try:
        anythermal = AnyThermalEncoder(
            model_path=args.anythermal_model_path,
            device=args.device,
            revision=args.anythermal_revision,
            local_files_only=args.local_files_only,
        )
        anythermal_summary = anythermal.parameter_summary()
        anythermal_training = anythermal.model.training
        anythermal_frozen = (
            anythermal_summary["is_fully_frozen"] and not anythermal_training
        )
        anythermal_output = anythermal.encode(thermal_image_path)
        diagnostics = anythermal_output["thermal_diagnostics"]
        raw_thermal_array, processor_rgb_image = anythermal.get_last_thermal_artifacts()
        spatial_features = anythermal_output["spatial_features"]
        spatial_features_cpu = spatial_features.detach().cpu()
        spatial_feature_shape = tuple(spatial_features.shape)
        spatial_statistics = tensor_statistics(spatial_features)
        original_shape = tuple(anythermal_output["original_shape"])
    except Exception as exc:
        print(f"[AnyThermal error] {exc}", file=sys.stderr)
        return 3

    del anythermal_output, spatial_features, anythermal
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    try:
        lotus_dtype = torch.float16 if args.half_precision else torch.float32
        lotus = LotusDPipeline.from_pretrained(
            args.lotus_model_path,
            torch_dtype=lotus_dtype,
            local_files_only=args.local_files_only,
        ).to(device)

        lotus.vae.requires_grad_(False)
        lotus.text_encoder.requires_grad_(False)
        lotus.unet.requires_grad_(False)
        lotus.vae.eval()
        lotus.text_encoder.eval()
        lotus.unet.eval()

        vae_summary = module_summary(lotus.vae)
        text_encoder_summary = module_summary(lotus.text_encoder)
        unet_summary = module_summary(lotus.unet)
        vae_scale_factor = int(lotus.vae_scale_factor)
        thermal_height, thermal_width = original_shape[:2]
        if thermal_height % vae_scale_factor or thermal_width % vae_scale_factor:
            raise ValueError(
                "Thermal dimensions must be divisible by the Lotus VAE scale factor: "
                f"size={(thermal_height, thermal_width)}, "
                f"vae_scale_factor={vae_scale_factor}."
            )
        target_size = (
            thermal_height // vae_scale_factor,
            thermal_width // vae_scale_factor,
        )

        bridge = AnyThermalLotusBridge().to(device).eval()
        bridged_latent = bridge(
            spatial_features_cpu.to(device),
            target_size=target_size,
            output_channels=int(lotus.unet.config.in_channels),
        )
        bridge_parameters = sum(parameter.numel() for parameter in bridge.parameters())
        bridge_frozen = bridge_parameters == 0 and not bridge.training

        all_models_frozen = (
            anythermal_frozen
            and bridge_frozen
            and vae_summary["is_fully_frozen"]
            and text_encoder_summary["is_fully_frozen"]
            and unet_summary["is_fully_frozen"]
        )
        if not all_models_frozen:
            raise RuntimeError("One or more models are not fully frozen in eval mode.")

        task_emb = torch.tensor([[1.0, 0.0]], device=device)
        task_emb = torch.cat(
            [torch.sin(task_emb), torch.cos(task_emb)],
            dim=-1,
        )
        timestep = torch.tensor([args.timestep], device=device, dtype=torch.long)
        unet_input = bridged_latent.to(device=device, dtype=lotus.unet.dtype)

        autocast_context = (
            torch.autocast(device.type) if device.type == "cuda" else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            prompt_embeds, _ = lotus.encode_prompt(
                prompt=PROMPT,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=None,
            )
            prediction = lotus.unet(
                unet_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                class_labels=task_emb,
                return_dict=False,
            )[0]
            decoded = lotus.vae.decode(
                prediction / lotus.vae.config.scaling_factor,
                return_dict=False,
            )[0]
            postprocessed = lotus.image_processor.postprocess(
                decoded,
                output_type="np",
                do_denormalize=[True] * decoded.shape[0],
            )
    except Exception as exc:
        print(f"[Lotus forward error] {exc}", file=sys.stderr)
        return 4

    if tuple(decoded.shape[-2:]) != (thermal_height, thermal_width):
        print(
            "[shape error] Decoded output does not match thermal resolution: "
            f"decoded={tuple(decoded.shape[-2:])}, "
            f"thermal={(thermal_height, thermal_width)}.",
            file=sys.stderr,
        )
        return 5

    bridged_statistics = tensor_statistics(unet_input)
    prediction_statistics = tensor_statistics(prediction)
    decoded_statistics = tensor_statistics(decoded)
    depth_like = decoded.detach().float().mean(dim=1)
    depth_statistics = tensor_statistics(depth_like)
    finite_checks = {
        "anythermal_spatial_features": spatial_statistics["finite"],
        "bridged_latent": bridged_statistics["finite"],
        "unet_prediction": prediction_statistics["finite"],
        "decoded": decoded_statistics["finite"],
        "depth_like": depth_statistics["finite"],
        "postprocessed": bool(np.isfinite(postprocessed).all()),
    }
    if not all(finite_checks.values()):
        print(f"[finite error] Non-finite output detected: {finite_checks}", file=sys.stderr)
        return 6

    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(thermal_image_path) as thermal_image:
        thermal_image.save(output_dir / "thermal_input.png")
    Image.fromarray(
        min_max_uint8(raw_thermal_array),
        mode="L",
    ).save(output_dir / "thermal_raw_visualization.png")
    processor_rgb_image.save(output_dir / "thermal_rgb_input.png")
    np.save(
        output_dir / "predicted_depth_raw.npy",
        depth_like[0].cpu().numpy(),
    )
    save_depth_visualization(
        depth_like[0],
        output_dir / "predicted_depth_visualization.png",
    )

    run_info = {
        "models": {
            "anythermal": args.anythermal_model_path,
            "anythermal_revision": args.anythermal_revision,
            "lotus": args.lotus_model_path,
        },
        "timestep": args.timestep,
        "prompt": PROMPT,
        "thermal_loading_mode": {
            "source": diagnostics["loading_mode"],
            "pil_mode": diagnostics["pil_mode"],
            "numpy_dtype": diagnostics["original_numpy_dtype"],
        },
        "raw_thermal_statistics": {
            "min": diagnostics["raw_min"],
            "max": diagnostics["raw_max"],
            "mean": diagnostics["raw_mean"],
            "std": diagnostics["raw_std"],
        },
        "converted_thermal_statistics": {
            "min": diagnostics["converted_uint8_min"],
            "max": diagnostics["converted_uint8_max"],
            "mean": diagnostics["converted_uint8_mean"],
            "std": diagnostics["converted_uint8_std"],
            "rgb_channels_equal": diagnostics["rgb_channels_equal"],
        },
        "processor_pixel_values": {
            "shape": list(diagnostics["processor_pixel_values_shape"]),
            "mean": diagnostics["processor_pixel_values_mean"],
            "std": diagnostics["processor_pixel_values_std"],
        },
        "legacy_path_conversion": {
            "min": diagnostics["legacy_rgb_min"],
            "max": diagnostics["legacy_rgb_max"],
            "mean": diagnostics["legacy_rgb_mean"],
            "std": diagnostics["legacy_rgb_std"],
            "saturated_fraction": diagnostics["legacy_rgb_saturated_fraction"],
        },
        "vae_scale_factor": vae_scale_factor,
        "thermal_original_size": [thermal_height, thermal_width],
        "dynamic_bridge_target_size": list(target_size),
        "decoded_output_size": list(decoded.shape[-2:]),
        "shapes": {
            "anythermal_spatial_features": list(spatial_feature_shape),
            "bridged_latent": list(unet_input.shape),
            "prompt_embedding": list(prompt_embeds.shape),
            "task_embedding": list(task_emb.shape),
            "unet_output": list(prediction.shape),
            "decoded_output": list(decoded.shape),
            "postprocessed_output": list(postprocessed.shape),
            "depth_like": list(depth_like.shape),
        },
        "statistics": {
            "anythermal_spatial_features": spatial_statistics,
            "bridged_latent": bridged_statistics,
            "unet_output": prediction_statistics,
            "decoded_output": decoded_statistics,
            "depth_like": depth_statistics,
        },
        "finite_checks": finite_checks,
        "frozen": {
            "all_models_frozen": all_models_frozen,
            "anythermal": {
                **anythermal_summary,
                "training": anythermal_training,
            },
            "bridge": {
                "parameters": bridge_parameters,
                "training": bridge.training,
            },
            "lotus_vae": vae_summary,
            "lotus_text_encoder": text_encoder_summary,
            "lotus_unet": unet_summary,
        },
        "visualization_notes": {
            "thermal_raw_visualization.png": (
                "Per-image min-max visualization; not a calibrated temperature map."
            ),
            "thermal_rgb_input.png": (
                "Actual uint8 three-channel image passed to the AnyThermal processor; "
                "all channels are identical."
            ),
            "predicted_depth_visualization.png": (
                "Min-max visualization of decoded channel mean; not metric-aligned "
                "physical depth."
            ),
        },
    }
    with (output_dir / "run_info.json").open("w", encoding="utf-8") as file:
        json.dump(run_info, file, indent=2, ensure_ascii=False)

    print(
        "raw thermal dtype/mode: "
        f"{diagnostics['original_numpy_dtype']} / {diagnostics['pil_mode']}"
    )
    print(
        "raw thermal min/max/mean/std: "
        f"{diagnostics['raw_min']} / {diagnostics['raw_max']} / "
        f"{diagnostics['raw_mean']} / {diagnostics['raw_std']}"
    )
    print(
        "converted thermal min/max/mean/std: "
        f"{diagnostics['converted_uint8_min']} / "
        f"{diagnostics['converted_uint8_max']} / "
        f"{diagnostics['converted_uint8_mean']} / "
        f"{diagnostics['converted_uint8_std']}"
    )
    print(f"final RGB channel equality: {diagnostics['rgb_channels_equal']}")
    print(
        "processor pixel_values shape/mean/std: "
        f"{diagnostics['processor_pixel_values_shape']} / "
        f"{diagnostics['processor_pixel_values_mean']} / "
        f"{diagnostics['processor_pixel_values_std']}"
    )
    print(f"AnyThermal spatial shape: {spatial_feature_shape}")
    print(f"VAE scale factor: {vae_scale_factor}")
    print(f"dynamic Lotus target latent size: {target_size}")
    print(f"bridge output shape: {tuple(unet_input.shape)}")
    print(f"U-Net output shape: {tuple(prediction.shape)}")
    print(f"decoded output shape: {tuple(decoded.shape)}")
    print(f"decoded output finite: {decoded_statistics['finite']}")
    print(f"all models frozen: {all_models_frozen}")
    print(
        "legacy RGB saturated fraction: "
        f"{diagnostics['legacy_rgb_saturated_fraction']}"
    )
    print(f"output directory: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())