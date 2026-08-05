"""Run the RGB->Lotus route (six-route line a/b) through the upstream Lotus evaluator.

Line a: MS2 left RGB -> frozen Lotus VAE condition -> frozen pretrained
Lotus-G U-Net -> disparity (zero training, Lotus as designed).
Line b: same, with --unet-checkpoint loading a U-Net fine-tuned on RGB.

GT VIEW WARNING (frozen conclusion 15): the MS2 RGB and thermal cameras do
NOT share a viewpoint, so this route is evaluated against the RGB-view
filtered LiDAR GT (``proj_depth/<seq>/rgb/depth_filtered``).  Its numbers may
share a table with thermal-view routes only if the GT-view column is labelled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = ROOT / "lotus"
for path in (ROOT, LOTUS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluation.evaluation import evaluation_depth  # noqa: E402
from models.anythermal_lotus_v2 import encode_condition_latent  # noqa: E402
from pipeline import LotusGPipeline  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--caption-mode", choices=("empty", "correct"), default="empty")
    parser.add_argument(
        "--unet-checkpoint",
        type=Path,
        default=None,
        help="Optional trained U-Net checkpoint (loads lotus_unet_state_dict) -- line b.",
    )
    parser.add_argument(
        "--condition-posterior",
        choices=("sample", "mode", "mean"),
        default="mode",
        help="mode matches the zero-training thermal baseline (line c) policy.",
    )
    parser.add_argument(
        "--dataset",
        choices=("ms2", "rgbdt500"),
        default="ms2",
        help=(
            "ms2 (default) keeps the two-GT-view guards of conclusion 15. "
            "rgbdt500 reads the single registered depth_path instead."
        ),
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=999,
        help=(
            "Timestep for the single-step path, matching upstream lotus/infer.py "
            "(--timestep 999, num_inference_steps=1). Ignored when "
            "--num-inference-steps > 1, where the scheduler picks the schedule."
        ),
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=1,
        help=(
            "Denoising steps. 1 (default) keeps the historical single forward at "
            "--timestep, so every published number is reproduced bit for bit. "
            ">1 runs LotusGPipeline's real DDIM loop (prediction_type='sample', "
            "so the U-Net's x0 goes straight into scheduler.step). "
            "NOTE the scheduler's own 1-step schedule would be t=1, not 999 -- "
            "the two single-step conventions are not the same experiment, which "
            "is why 1 keeps the fixed-timestep path instead of set_timesteps(1)."
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--save-raw-pred",
        action="store_true",
        help=(
            "Also dump each raw decoded disparity as raw_predictions/<id>.npy "
            "(float32, pre-alignment), for protocol recomputation such as "
            "tools/run_official_ms2_evaluation.py."
        ),
    )
    return parser.parse_args()


def read_manifest(path: Path, dataset: str = "ms2"):
    """Rows must carry rgb_path and the RGB-view GT.

    On MS2 the ``/rgb/`` guards are load-bearing: that set ships two GT views and
    silently falling back to the thermal-view map would violate conclusion 15.
    RGBDT500 ships a single registered depth map (measured 2026-07-21: thermal
    sits at (0,0) against RGB), so there is no second view to confuse and the GT
    key is the plain ``depth_path``.
    """
    rows = []
    gt_key = "rgb_depth_path" if dataset == "ms2" else "depth_path"
    require_rgb_dir = dataset == "ms2"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rgb = row.get("rgb_path")
            depth = row.get(gt_key)
            if not rgb or (require_rgb_dir and "/rgb/" not in str(rgb).replace("\\", "/")):
                raise ValueError(f"Manifest row {row.get('id')} lacks an RGB input path")
            if not depth or (require_rgb_dir and "/rgb/" not in str(depth).replace("\\", "/")):
                raise ValueError(
                    f"Manifest row {row.get('id')} lacks RGB-view GT ({gt_key}); "
                    "thermal-view GT is deliberately not a fallback for the RGB route"
                )
            rows.append(
                {
                    "id": row["id"],
                    "rgb_path": rgb,
                    "rgb_depth_path": depth,
                    "caption": str(row.get("caption", "")),
                }
            )
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


def choose_uniform(rows, count: int):
    if count <= 0 or count >= len(rows):
        return rows
    return [rows[int(i)] for i in np.linspace(0, len(rows) - 1, count, dtype=int)]


def task_embedding(device, dtype):
    task = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
    return torch.cat([torch.sin(task), torch.cos(task)], dim=-1)


def rgb_to_lotus_input(path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load MS2 left RGB at native resolution as a ``[-1,1]`` [1,3,H,W] tensor."""
    with Image.open(path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    height, width = rgb.shape[:2]
    if height % 8 or width % 8:
        raise ValueError(
            f"RGB resolution {height}x{width} is not divisible by the VAE factor 8: {path}"
        )
    if int(rgb.min()) == int(rgb.max()):
        raise ValueError(f"RGB image is constant: {path}")
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.astype(np.float32)))
    tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    return tensor, (height, width)


@dataclass
class RGBPipelineBundle:
    lotus: LotusGPipeline
    ms2_root: Path
    seeds: dict[str, int]
    prompts: dict[str, str]
    condition_posterior: str
    timestep: int = 999
    num_inference_steps: int = 1


def generate_rgb_prediction(input_rgb, bundle, image_path=None, dataset_name=None):
    del input_rgb, dataset_name
    if image_path is None:
        raise ValueError("Lotus evaluator did not provide image_path")
    rgb_tensor, (height, width) = rgb_to_lotus_input(bundle.ms2_root / image_path)
    lotus = bundle.lotus
    device = lotus._execution_device
    dtype = lotus.unet.dtype
    target = (height // int(lotus.vae_scale_factor), width // int(lotus.vae_scale_factor))

    sample_seed = bundle.seeds[image_path]
    condition_seed = sample_seed + 1_000_000
    condition = encode_condition_latent(
        lotus.vae,
        rgb_tensor,
        posterior=bundle.condition_posterior,
        seed=condition_seed if bundle.condition_posterior == "sample" else None,
    ).to(device=device, dtype=dtype)
    expected_shape = (1, int(lotus.unet.config.in_channels) // 2, *target)
    if tuple(condition.shape) != expected_shape:
        raise RuntimeError(f"RGB condition shape {tuple(condition.shape)} != {expected_shape}")

    generator = torch.Generator(device=device).manual_seed(sample_seed)
    noise = torch.randn(
        expected_shape,
        generator=generator,
        device=device,
        dtype=dtype,
    ) * lotus.scheduler.init_noise_sigma
    prompt, _ = lotus.encode_prompt(
        prompt=bundle.prompts.get(image_path, ""),
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=None,
    )
    steps = max(1, int(bundle.num_inference_steps))
    if steps == 1:
        # The historical path, and upstream's own default: one forward at a fixed
        # timestep, x0 taken as the answer. Kept byte-identical so the frozen
        # baselines do not move when this file grows a scheduler.
        timesteps = [torch.tensor(bundle.timestep, device=device, dtype=torch.long)]
    else:
        lotus.scheduler.set_timesteps(steps, device=device)
        timesteps = list(lotus.scheduler.timesteps)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda",
    ):
        latents = noise
        for timestep in timesteps:
            latent_input = lotus.scheduler.scale_model_input(latents, timestep)
            x0 = lotus.unet(
                torch.cat([condition, latent_input], dim=1),
                timestep,
                encoder_hidden_states=prompt.to(dtype=dtype),
                class_labels=task_embedding(device, dtype),
                return_dict=False,
            )[0]
            # Mirrors LotusGPipeline.__call__: with more than one timestep the
            # scheduler walks x_t -> x_t-1 (the checkpoint's DDIM config declares
            # prediction_type='sample', so x0 is the right thing to hand it);
            # with one, x0 *is* the latent.
            latents = (
                lotus.scheduler.step(x0, timestep, latents, return_dict=False)[0]
                if len(timesteps) > 1
                else x0
            )
        decoded = lotus.vae.decode(
            latents / lotus.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        image = lotus.image_processor.postprocess(
            decoded,
            output_type="np",
            do_denormalize=[True],
        )[0]
    disparity = np.asarray(image, np.float32).mean(axis=-1)
    if disparity.shape != (height, width):
        raise RuntimeError(f"Decoded shape {disparity.shape} != RGB shape {(height, width)}")
    if not np.isfinite(disparity).all():
        raise RuntimeError("Prediction contains NaN/Inf")
    return disparity


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest = args.manifest.resolve()
    ms2_root = args.ms2_root.resolve()
    selected = choose_uniform(read_manifest(manifest, args.dataset), args.max_samples)
    filename_list = output / "official_ms2_filename_list.txt"
    filename_list.write_text(
        "".join(f"{row['rgb_path']} {row['rgb_depth_path']}\n" for row in selected),
        encoding="utf-8",
    )
    config = output / "official_ms2_dataset.yaml"
    config.write_text(
        "\n".join(
            [
                # NOTE: `name` is a dispatch key into
                # lotus/evaluation/dataset_depth/__init__.py::dataset_name_class_dict,
                # NOT a label -- it must stay `ms2_rgb`, whose MS2RGBDataset is a
                # generic uint16-depth RGB loader that RGBDT500 also rides on.
                # Only `disp_name` below is free text.
                "name: ms2_rgb",
                f"disp_name: {'MS2 RGB-view filtered LiDAR' if args.dataset == 'ms2' else 'RGBDT500 registered sensor depth'}",
                "dir: .",
                f"filenames: {filename_list.as_posix()}",
                f"depth_scale: {args.depth_scale}",
                f"min_depth: {args.min_depth}",
                f"max_depth: {args.max_depth}",
                "processing_res: null",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output / "selected_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )

    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    lotus = LotusGPipeline.from_pretrained(
        args.lotus_model_path,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    unet_checkpoint_info = {}
    if args.unet_checkpoint is not None:
        unet_checkpoint_path = args.unet_checkpoint.resolve()
        unet_checkpoint = torch.load(unet_checkpoint_path, map_location="cpu", weights_only=False)
        lotus.unet.load_state_dict(unet_checkpoint["lotus_unet_state_dict"], strict=True)
        unet_checkpoint_info = {
            "unet_checkpoint": str(unet_checkpoint_path),
            "unet_checkpoint_sha256": hashlib.sha256(unet_checkpoint_path.read_bytes()).hexdigest(),
            "unet_checkpoint_global_step": int(unet_checkpoint.get("global_step", -1)),
            "unet_checkpoint_format": str(unet_checkpoint.get("format", "?")),
        }
        del unet_checkpoint
    for module in (lotus.vae, lotus.text_encoder, lotus.unet):
        module.requires_grad_(False).eval()
    seeds = {row["rgb_path"]: args.seed + index for index, row in enumerate(selected)}
    if args.caption_mode == "correct":
        empty_ids = [row["id"] for row in selected if not row["caption"].strip()]
        if empty_ids:
            raise RuntimeError(f"Correct-caption mode found empty captions: {empty_ids[:5]}")
        prompts = {row["rgb_path"]: row["caption"] for row in selected}
    else:
        prompts = {row["rgb_path"]: "" for row in selected}
    bundle = RGBPipelineBundle(
        lotus,
        ms2_root,
        seeds,
        prompts,
        args.condition_posterior,
        timestep=args.timestep,
        num_inference_steps=args.num_inference_steps,
    )

    gen_prediction = generate_rgb_prediction
    if args.save_raw_pred:
        raw_dir = output / "raw_predictions"
        raw_dir.mkdir(exist_ok=True)
        id_by_rgb_path = {row["rgb_path"]: row["id"] for row in selected}

        def gen_prediction(input_rgb, pipeline, image_path=None, dataset_name=None):
            disparity = generate_rgb_prediction(input_rgb, pipeline, image_path, dataset_name)
            np.save(raw_dir / f"{id_by_rgb_path[image_path]}.npy", disparity.astype(np.float32))
            return disparity

    tracker = evaluation_depth(
        str(output),
        str(config),
        str(ms2_root),
        eval_mode="generate_prediction",
        gen_prediction=gen_prediction,
        pipeline=bundle,
        alignment="least_square_disparity",
        save_pred_vis=True,
        processing_res=None,
    )
    metadata = {
        "evaluator": "lotus/evaluation/evaluation.py::evaluation_depth",
        "visualizer": "lotus/utils/image_utils.py::colorize_depth_map",
        "route": (
            "rgb-vae-condition-plus-trained-UNet"
            if args.unet_checkpoint is not None
            else "rgb-vae-condition-plus-pretrained-Lotus-G-UNet"
        ),
        "six_route_line": "b" if args.unet_checkpoint is not None else "a",
        **unet_checkpoint_info,
        "dataset": args.dataset,
        "gt_view": (
            "rgb-view-filtered-lidar" if args.dataset == "ms2"
            else "rgbdt500-registered-sensor-depth"
        ),
        "gt_view_warning": (
            "RGB and thermal cameras do not share a viewpoint (frozen conclusion 15); "
            "numbers may join thermal-view tables only with a labelled GT-view column"
            if args.dataset == "ms2" else
            "RGBDT500 ships one registered depth map and RGB sits at (0,0) against "
            "thermal (measured 2026-07-21), so this number shares a table with the "
            "thermal routes directly. Caveat: that depth is itself offset ~+7 px "
            "horizontally from both cameras, a handicap borne equally by every route."
        ),
        "metric_scope": "official upstream Lotus depth-quality metrics",
        "caption_mode": args.caption_mode,
        "num_inference_steps": args.num_inference_steps,
        "timestep": args.timestep if args.num_inference_steps == 1 else None,
        "denoising": (
            f"single forward at t={args.timestep} (upstream lotus/infer.py default)"
            if args.num_inference_steps == 1
            else f"DDIM loop, {args.num_inference_steps} steps, scheduler schedule"
        ),
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "sample_count": len(selected),
        "alignment": "least_square_disparity",
        "raw_predictions_saved": bool(args.save_raw_pred),
        "dtype": args.dtype,
        "condition_posterior": args.condition_posterior,
        "seed_policy": "same diffusion noise seeds as Direct/Thermal-VAE; condition seed = sample seed + 1000000",
        "metrics": tracker.result(),
    }
    (output / "official_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
