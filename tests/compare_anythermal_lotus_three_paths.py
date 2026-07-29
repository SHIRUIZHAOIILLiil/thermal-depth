"""Compare three frozen Lotus-D visual-input paths on one paired sample."""

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
import torch.nn.functional as F
from PIL import Image, ImageDraw


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


PATH_SPECS = {
    "A": ("A_rgb_lotus", "RGB VAE latent"),
    "B": ("B_thermal_vae_lotus", "Thermal VAE latent"),
    "C": ("C_anythermal_lotus", "AnyThermal bridge latent"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-image", required=True)
    parser.add_argument("--thermal-image", required=True)
    parser.add_argument("--depth-image", required=True)
    parser.add_argument("--anythermal-model-path", required=True)
    parser.add_argument("--lotus-model-path", required=True)
    parser.add_argument("--anythermal-revision", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--half-precision", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", default="outputs/compare_three_paths")
    parser.add_argument(
        "--prompt",
        default="",
        help="Caption shared by the RGB, thermal-VAE, and AnyThermal paths.",
    )
    parser.add_argument(
        "--caption-source",
        default=None,
        help="Optional caption source path recorded in summary.json.",
    )
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--rgb-processing-res", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--native-output-frames",
        action="store_true",
        help="Keep RGB output in the RGB frame; thermal paths remain in the thermal frame.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("rgb_image", "thermal_image", "depth_image"):
        path = Path(getattr(args, name))
        if not path.is_file():
            raise SystemExit(f"{name.replace('_', ' ')} does not exist: {path}")
    if args.rgb_processing_res <= 0:
        raise SystemExit("--rgb-processing-res must be greater than zero.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"CUDA is unavailable; cannot use --device {args.device}.")


def tensor_statistics(tensor: torch.Tensor) -> dict[str, Any]:
    values = tensor.detach().float()
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "finite": bool(torch.isfinite(values).all()),
    }


def numpy_statistics(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array, dtype=np.float32)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "finite": bool(np.isfinite(values).all()),
    }


def module_summary(module: torch.nn.Module) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "training": module.training,
        "is_fully_frozen": trainable == 0 and not module.training,
    }


def image_to_lotus_tensor(image: Image.Image) -> torch.Tensor:
    array = np.array(image.convert("RGB"), dtype=np.float32, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor / 127.5 - 1.0


def min_max_uint8(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 3 and values.shape[-1] in (1, 3):
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D array for visualization, got {values.shape}.")
    values = values.astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Visualization input contains NaN or infinite values.")
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum > minimum:
        values = (values - minimum) / (maximum - minimum)
    else:
        values = np.zeros_like(values)
    return (values * 255.0).round().astype(np.uint8)


def load_depth_array(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(path) as image:
        mode = image.mode
        array = np.asarray(image).copy()
    if array.ndim == 3:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected 2D GT depth, got {array.shape}.")
    return array, {
        "mode": mode,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "statistics": numpy_statistics(array),
    }


def make_autocast(device: torch.device):
    return torch.autocast(device.type) if device.type == "cuda" else nullcontext()


def encode_vae_latent(
    lotus: LotusDPipeline,
    image_tensor: torch.Tensor,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    with torch.inference_mode(), make_autocast(device):
        distribution = lotus.vae.encode(image_tensor.to(device)).latent_dist
        latent = distribution.sample(generator=generator)
        return latent * lotus.vae.config.scaling_factor


def resize_depth_array(
    depth: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    if depth.shape == target_size:
        return depth.astype(np.float32, copy=False)
    tensor = torch.from_numpy(depth.astype(np.float32)).view(1, 1, *depth.shape)
    resized = F.interpolate(
        tensor,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    return resized[0, 0].numpy()


def run_lotus_path(
    lotus: LotusDPipeline,
    visual_latent: torch.Tensor,
    prompt_embeds: torch.Tensor,
    task_embedding: torch.Tensor,
    timestep: torch.Tensor,
    output_size: tuple[int, int],
    device: torch.device,
    frozen_state: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    unet_input = visual_latent.to(device=device, dtype=lotus.unet.dtype)
    with torch.inference_mode(), make_autocast(device):
        prediction = lotus.unet(
            unet_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            class_labels=task_embedding,
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

    depth_like = postprocessed[0].mean(axis=-1).astype(np.float32)
    final_depth = resize_depth_array(depth_like, output_size)
    finite_checks = {
        "visual_input_latent": bool(torch.isfinite(unet_input).all()),
        "unet_output": bool(torch.isfinite(prediction).all()),
        "decoded_output": bool(torch.isfinite(decoded).all()),
        "postprocessed_output": bool(np.isfinite(postprocessed).all()),
        "final_depth_like": bool(np.isfinite(final_depth).all()),
    }
    if not all(finite_checks.values()):
        raise RuntimeError(f"Non-finite values in Lotus path: {finite_checks}")

    run_info = {
        "visual_input_latent": {
            "shape": list(unet_input.shape),
            "statistics": tensor_statistics(unet_input),
        },
        "prompt_embedding_shape": list(prompt_embeds.shape),
        "task_embedding_shape": list(task_embedding.shape),
        "unet_input_shape": list(unet_input.shape),
        "unet_output": {
            "shape": list(prediction.shape),
            "statistics": tensor_statistics(prediction),
        },
        "decoded_output_shape": list(decoded.shape),
        "postprocessed_output_shape": list(postprocessed.shape),
        "final_depth_like": {
            "shape": list(final_depth.shape),
            "statistics": numpy_statistics(final_depth),
        },
        "finite_checks": finite_checks,
        "models_frozen": frozen_state,
        "visualization_note": (
            "prediction_visualization.png is independently min-max visualized and "
            "is not metric-aligned physical depth."
        ),
    }
    return final_depth, run_info


def save_path_result(
    root: Path,
    directory_name: str,
    prediction: np.ndarray,
    run_info: dict[str, Any],
) -> None:
    path = root / directory_name
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "prediction.npy", prediction.astype(np.float32))
    Image.fromarray(min_max_uint8(prediction), mode="L").save(
        path / "prediction_visualization.png"
    )
    with (path / "run_info.json").open("w", encoding="utf-8") as file:
        json.dump(run_info, file, indent=2, ensure_ascii=False)


def create_comparison(
    output_path: Path,
    images: list[tuple[str, Image.Image]],
    size: tuple[int, int],
) -> None:
    label_height = 28
    canvas = Image.new("RGB", (size[1] * len(images), size[0] + label_height), "black")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(images):
        panel = image.convert("RGB").resize((size[1], size[0]), Image.Resampling.BILINEAR)
        x = index * size[1]
        canvas.paste(panel, (x, label_height))
        draw.text((x + 8, 8), label, fill="white")
    canvas.save(output_path)


def main() -> int:
    args = parse_args()
    validate_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    anythermal = AnyThermalEncoder(
        model_path=args.anythermal_model_path,
        device=args.device,
        revision=args.anythermal_revision,
        local_files_only=args.local_files_only,
    )
    anythermal_summary = anythermal.parameter_summary()
    anythermal_output = anythermal.encode(args.thermal_image)
    thermal_raw, thermal_rgb = anythermal.get_last_thermal_artifacts()
    thermal_diagnostics = anythermal_output["thermal_diagnostics"]
    thermal_size = tuple(anythermal_output["original_shape"][:2])
    anythermal_spatial = anythermal_output["spatial_features"].detach().cpu()
    anythermal_spatial_shape = tuple(anythermal_spatial.shape)
    anythermal_frozen = (
        anythermal_summary["is_fully_frozen"] and not anythermal.model.training
    )

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
    lotus.vae.requires_grad_(False).eval()
    lotus.text_encoder.requires_grad_(False).eval()
    lotus.unet.requires_grad_(False).eval()

    frozen_state = {
        "anythermal": anythermal_frozen,
        "lotus_vae": module_summary(lotus.vae),
        "lotus_text_encoder": module_summary(lotus.text_encoder),
        "lotus_unet": module_summary(lotus.unet),
    }
    all_models_frozen = anythermal_frozen and all(
        value["is_fully_frozen"]
        for key, value in frozen_state.items()
        if key != "anythermal"
    )
    if not all_models_frozen:
        raise RuntimeError(f"Models are not fully frozen: {frozen_state}")

    vae_scale_factor = int(lotus.vae_scale_factor)
    if thermal_size[0] % vae_scale_factor or thermal_size[1] % vae_scale_factor:
        raise ValueError(
            f"Thermal size {thermal_size} is not divisible by VAE scale factor "
            f"{vae_scale_factor}."
        )
    thermal_latent_size = (
        thermal_size[0] // vae_scale_factor,
        thermal_size[1] // vae_scale_factor,
    )

    with Image.open(args.rgb_image) as image:
        rgb_image = image.convert("RGB").copy()
    rgb_size = (rgb_image.height, rgb_image.width)
    rgb_tensor = image_to_lotus_tensor(rgb_image)
    rgb_tensor = resize_max_res(
        rgb_tensor,
        max_edge_resolution=args.rgb_processing_res,
        resample_method=get_tv_resample_method("bilinear"),
    )
    thermal_tensor = image_to_lotus_tensor(thermal_rgb)

    generator_a = torch.Generator(device=device).manual_seed(args.seed)
    generator_b = torch.Generator(device=device).manual_seed(args.seed)
    rgb_vae_latent = encode_vae_latent(lotus, rgb_tensor, device, generator_a)
    thermal_vae_latent = encode_vae_latent(lotus, thermal_tensor, device, generator_b)

    bridge = AnyThermalLotusBridge().to(device).eval()
    bridged_latent = bridge(
        anythermal_spatial.to(device),
        target_size=thermal_latent_size,
        output_channels=int(lotus.unet.config.in_channels),
    )
    bridge_parameters = sum(parameter.numel() for parameter in bridge.parameters())
    if bridge_parameters != 0:
        raise RuntimeError(f"Bridge unexpectedly has {bridge_parameters} parameters.")

    task_embedding = torch.tensor([[1.0, 0.0]], device=device)
    task_embedding = torch.cat(
        [torch.sin(task_embedding), torch.cos(task_embedding)],
        dim=-1,
    )
    timestep = torch.tensor([args.timestep], device=device, dtype=torch.long)
    with torch.inference_mode(), make_autocast(device):
        prompt_embeds, _ = lotus.encode_prompt(
            prompt=args.prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=None,
        )

    predictions: dict[str, np.ndarray] = {}
    path_infos: dict[str, dict[str, Any]] = {}
    path_latents = {
        "A": rgb_vae_latent,
        "B": thermal_vae_latent,
        "C": bridged_latent,
    }
    for key, latent in path_latents.items():
        output_size = (
            rgb_size
            if args.native_output_frames and key == "A"
            else thermal_size
        )
        prediction, info = run_lotus_path(
            lotus=lotus,
            visual_latent=latent,
            prompt_embeds=prompt_embeds,
            task_embedding=task_embedding,
            timestep=timestep,
            output_size=output_size,
            device=device,
            frozen_state={
                **frozen_state,
                "all_models_frozen": all_models_frozen,
            },
        )
        directory_name, visual_input_name = PATH_SPECS[key]
        info["path"] = key
        info["visual_input"] = visual_input_name
        if key == "A":
            info["rgb_processing_resolution"] = args.rgb_processing_res
            info["rgb_processed_tensor_shape"] = list(rgb_tensor.shape)
            info["final_output_size"] = list(output_size)
            info["output_frame"] = "rgb" if args.native_output_frames else "thermal"
        if key == "C":
            info["anythermal_spatial_feature_shape"] = list(anythermal_spatial_shape)
            info["bridge_output_shape"] = list(bridged_latent.shape)
            info["bridge_parameters"] = bridge_parameters
        save_path_result(output_dir, directory_name, prediction, info)
        predictions[key] = prediction
        path_infos[key] = info

    rgb_image.save(output_dir / "rgb_input.png")
    thermal_rgb.save(output_dir / "thermal_input.png")
    depth_array, depth_info = load_depth_array(Path(args.depth_image))
    gt_visualization = Image.fromarray(min_max_uint8(depth_array), mode="L")
    gt_visualization.save(output_dir / "gt_depth_visualization.png")

    comparison_images = [
        ("RGB", rgb_image),
        ("Thermal", thermal_rgb),
        ("GT", gt_visualization),
        (
            "A: RGB Lotus",
            Image.fromarray(min_max_uint8(predictions["A"]), mode="L"),
        ),
        (
            "B: Thermal VAE",
            Image.fromarray(min_max_uint8(predictions["B"]), mode="L"),
        ),
        (
            "C: AnyThermal",
            Image.fromarray(min_max_uint8(predictions["C"]), mode="L"),
        ),
    ]
    create_comparison(output_dir / "comparison.png", comparison_images, thermal_size)

    summary = {
        "models": {
            "anythermal": args.anythermal_model_path,
            "anythermal_revision": args.anythermal_revision,
            "lotus": args.lotus_model_path,
        },
        "inputs": {
            "rgb": str(Path(args.rgb_image)),
            "thermal": str(Path(args.thermal_image)),
            "depth": str(Path(args.depth_image)),
        },
        "settings": {
            "prompt": args.prompt,
            "prompt_shared_across_paths": ["A", "B", "C"],
            "caption_source": args.caption_source,
            "timestep": args.timestep,
            "seed": args.seed,
            "thermal_size": list(thermal_size),
            "thermal_latent_size": list(thermal_latent_size),
            "rgb_processing_resolution": args.rgb_processing_res,
            "native_output_frames": args.native_output_frames,
            "rgb_size": list(rgb_size),
            "vae_scale_factor": vae_scale_factor,
            "all_models_frozen": all_models_frozen,
            "bridge_parameters": bridge_parameters,
        },
        "thermal_diagnostics": thermal_diagnostics,
        "gt_depth": depth_info,
        "paths": path_infos,
        "geometry_assessment": {
            "A": "pending manual visual inspection",
            "B": "pending manual visual inspection",
            "C": "pending manual visual inspection",
        },
        "visualization_note": (
            "Every prediction panel is independently min-max visualized. These are "
            "not metric-aligned physical depth maps."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(f"Thermal size: {thermal_size}")
    print(f"Thermal latent size: {thermal_latent_size}")
    print(f"Shared prompt: {args.prompt}")
    for key in ("A", "B", "C"):
        info = path_infos[key]
        print(
            f"Path {key}: latent={tuple(info['visual_input_latent']['shape'])}, "
            f"unet_output={tuple(info['unet_output']['shape'])}, "
            f"decoded={tuple(info['decoded_output_shape'])}, "
            f"finite={all(info['finite_checks'].values())}"
        )
    print(f"All models frozen: {all_models_frozen}")
    print(f"Output directory: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
