"""Run the Thermal-VAE sanity route through the upstream Lotus evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = ROOT / "lotus"
for path in (ROOT, LOTUS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluation.evaluation import evaluation_depth  # noqa: E402
from models.anythermal_lotus_v2 import (  # noqa: E402
    encode_condition_latent,
    thermal_to_lotus_input,
)
from models.thermal_vae_latent_adapter import ThermalVAELatentAdapter  # noqa: E402
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
    parser.add_argument(
        "--caption-mode", choices=("empty", "correct", "hard-wrong"), default="empty"
    )
    parser.add_argument(
        "--unet-checkpoint",
        type=Path,
        default=None,
        help="Optional trained U-Net checkpoint (loads lotus_unet_state_dict).",
    )
    parser.add_argument(
        "--latent-adapter-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional trained VAE-latent adapter checkpoint (six-route line e; "
            "loads adapter_state_dict from train_ms2_vae_latent_adapter_gt.py)."
        ),
    )
    parser.add_argument(
        "--condition-posterior",
        choices=("sample", "mode", "mean"),
        default="sample",
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


def read_manifest(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            depth = row.get("thermal_depth_path") or row.get("depth_path")
            if not depth:
                raise ValueError(f"Manifest row {row.get('id')} lacks GT depth")
            # MS2 ships both a thermal-view and an RGB-view GT and mixing them is a
            # protocol violation (frozen conclusion 15), hence the hard `/thr/`
            # check. Datasets that publish a single registered depth (RGBDT500)
            # declare themselves and are exempt.
            if str(row.get("dataset", "MS2")).upper() == "MS2" and "/thr/" not in depth.replace("\\", "/"):
                raise ValueError(f"Manifest row {row.get('id')} lacks thermal-view GT")
            rows.append(
                {
                    "id": row["id"],
                    "thermal_path": row["thermal_path"],
                    "thermal_depth_path": depth,
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


def tensor_stats(tensor: torch.Tensor):
    values = tensor.detach().float().cpu()
    return {
        "shape": list(values.shape),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "finite": bool(torch.isfinite(values).all()),
        "channel_mean": values.mean((0, 2, 3)).tolist(),
        "channel_std": values.std((0, 2, 3), unbiased=False).tolist(),
    }


@dataclass
class ThermalVAEPipelineBundle:
    lotus: LotusGPipeline
    ms2_root: Path
    seeds: dict[str, int]
    prompts: dict[str, str]
    condition_posterior: str
    latent_adapter: ThermalVAELatentAdapter | None = None
    condition_diagnostics: list[dict] = field(default_factory=list)


def generate_thermal_vae_prediction(input_thermal, bundle, image_path=None, dataset_name=None):
    del input_thermal, dataset_name
    if image_path is None:
        raise ValueError("Lotus evaluator did not provide image_path")
    thermal_path = bundle.ms2_root / image_path
    thermal = thermal_to_lotus_input(thermal_path, processing_res=0)
    height, width = map(int, thermal.diagnostics["raw_shape"])
    lotus = bundle.lotus
    device = lotus._execution_device
    dtype = lotus.unet.dtype
    target = (height // int(lotus.vae_scale_factor), width // int(lotus.vae_scale_factor))

    sample_seed = bundle.seeds[image_path]
    condition_seed = sample_seed + 1_000_000
    condition = encode_condition_latent(
        lotus.vae,
        thermal.tensor,
        posterior=bundle.condition_posterior,
        seed=condition_seed if bundle.condition_posterior == "sample" else None,
    ).to(device=device, dtype=dtype)
    if bundle.latent_adapter is not None:
        with torch.inference_mode():
            condition = bundle.latent_adapter(condition.float()).to(dtype=dtype)
    expected_shape = (1, int(lotus.unet.config.in_channels) // 2, *target)
    if tuple(condition.shape) != expected_shape:
        raise RuntimeError(
            f"Thermal-VAE condition shape {tuple(condition.shape)} != {expected_shape}"
        )

    # This is an interface sanity value, not a depth-quality loss.  The frozen
    # Thermal-VAE condition is the teacher itself, so its reference loss is zero.
    self_mse = float(torch.mean((condition.float() - condition.detach().float()) ** 2).cpu())
    bundle.condition_diagnostics.append(
        {
            "image_path": image_path,
            "condition_seed": condition_seed,
            "condition_posterior": bundle.condition_posterior,
            "thermal": thermal.diagnostics,
            "condition_latent": tensor_stats(condition),
            "condition_reference_self_mse": self_mse,
            "depth_target_loss_available": False,
            "depth_target_loss_reason": (
                "Sparse MS2 GT is not zero-filled and VAE-encoded in Adapter V2."
            ),
        }
    )

    generator = torch.Generator(device=device).manual_seed(sample_seed)
    noise = torch.randn(
        expected_shape,
        generator=generator,
        device=device,
        dtype=dtype,
    ) * lotus.scheduler.init_noise_sigma
    timestep = torch.tensor(999, device=device, dtype=torch.long)
    prompt, _ = lotus.encode_prompt(
        prompt=bundle.prompts.get(image_path, ""),
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=None,
    )
    latent_input = lotus.scheduler.scale_model_input(noise, timestep)
    unet_input = torch.cat([condition, latent_input], dim=1)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda",
    ):
        x0 = lotus.unet(
            unet_input,
            timestep,
            encoder_hidden_states=prompt.to(dtype=dtype),
            class_labels=task_embedding(device, dtype),
            return_dict=False,
        )[0]
        decoded = lotus.vae.decode(
            x0 / lotus.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        image = lotus.image_processor.postprocess(
            decoded,
            output_type="np",
            do_denormalize=[True],
        )[0]
    disparity = np.asarray(image, np.float32).mean(axis=-1)
    if disparity.shape != (height, width):
        raise RuntimeError(f"Decoded shape {disparity.shape} != thermal shape {(height, width)}")
    if not np.isfinite(disparity).all():
        raise RuntimeError("Prediction contains NaN/Inf")
    return disparity


def main():
    args = parse_args()
    if args.latent_adapter_checkpoint is not None and args.unet_checkpoint is not None:
        raise SystemExit(
            "Line e keeps the U-Net frozen: --latent-adapter-checkpoint and "
            "--unet-checkpoint are mutually exclusive."
        )
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest = args.manifest.resolve()
    ms2_root = args.ms2_root.resolve()
    selected = choose_uniform(read_manifest(manifest), args.max_samples)
    filename_list = output / "official_ms2_filename_list.txt"
    filename_list.write_text(
        "".join(f"{row['thermal_path']} {row['thermal_depth_path']}\n" for row in selected),
        encoding="utf-8",
    )
    config = output / "official_ms2_dataset.yaml"
    config.write_text(
        "\n".join(
            [
                "name: ms2_thermal",
                "disp_name: MS2 thermal-view filtered LiDAR",
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
        unet_checkpoint = torch.load(
            unet_checkpoint_path, map_location="cpu", weights_only=False
        )
        lotus.unet.load_state_dict(
            unet_checkpoint["lotus_unet_state_dict"], strict=True
        )
        loaded_vae_encoder = False
        if "vae_encoder_state_dict" in unet_checkpoint:
            lotus.vae.encoder.load_state_dict(
                unet_checkpoint["vae_encoder_state_dict"], strict=True
            )
            lotus.vae.quant_conv.load_state_dict(
                unet_checkpoint["vae_quant_conv_state_dict"], strict=True
            )
            loaded_vae_encoder = True
        unet_checkpoint_info = {
            "unet_checkpoint": str(unet_checkpoint_path),
            "unet_checkpoint_sha256": hashlib.sha256(
                unet_checkpoint_path.read_bytes()
            ).hexdigest(),
            "unet_checkpoint_global_step": int(unet_checkpoint.get("global_step", -1)),
            "unet_checkpoint_format": str(unet_checkpoint.get("format", "?")),
            "loaded_trained_vae_encoder": loaded_vae_encoder,
        }
        del unet_checkpoint
    for module in (lotus.vae, lotus.text_encoder, lotus.unet):
        module.requires_grad_(False).eval()
    latent_adapter = None
    latent_adapter_info = {}
    if args.latent_adapter_checkpoint is not None:
        adapter_path = args.latent_adapter_checkpoint.resolve()
        adapter_checkpoint = torch.load(adapter_path, map_location="cpu", weights_only=False)
        if "adapter_state_dict" not in adapter_checkpoint:
            raise RuntimeError("Latent-adapter checkpoint lacks adapter_state_dict")
        latent_adapter = ThermalVAELatentAdapter(
            hidden_channels=int(adapter_checkpoint.get("adapter_hidden_channels", 256)),
            blocks=int(adapter_checkpoint.get("adapter_blocks", 6)),
        ).to(device=device, dtype=torch.float32)
        latent_adapter.load_state_dict(adapter_checkpoint["adapter_state_dict"], strict=True)
        latent_adapter.requires_grad_(False).eval()
        latent_adapter_info = {
            "latent_adapter_checkpoint": str(adapter_path),
            "latent_adapter_checkpoint_sha256": hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
            "latent_adapter_global_step": int(adapter_checkpoint.get("global_step", -1)),
            "latent_adapter_format": str(adapter_checkpoint.get("format", "?")),
        }
        del adapter_checkpoint
    seeds = {row["thermal_path"]: args.seed + index for index, row in enumerate(selected)}
    if args.caption_mode in ("correct", "hard-wrong"):
        empty_ids = [row["id"] for row in selected if not row["caption"].strip()]
        if empty_ids:
            raise RuntimeError(
                f"{args.caption_mode}-caption mode found empty captions: {empty_ids[:5]}"
            )
    if args.caption_mode == "correct":
        prompts = {row["thermal_path"]: row["caption"] for row in selected}
    elif args.caption_mode == "hard-wrong":
        # Ported verbatim from run_ms2_lotus_trained_official.py so the two routes
        # produce the same derangement for the same seed and sample set.
        # Deterministic derangement: cyclic shift along a seeded shuffle, so no
        # sample keeps its own caption and the assignment is reproducible.
        if len(selected) < 2:
            raise RuntimeError("hard-wrong caption mode needs at least 2 samples.")
        order = list(range(len(selected)))
        random.Random(args.seed + 777).shuffle(order)
        prompts = {}
        for position, sample_index in enumerate(order):
            donor_index = order[(position + 1) % len(order)]
            prompts[selected[sample_index]["thermal_path"]] = selected[donor_index]["caption"]
    else:
        prompts = {row["thermal_path"]: "" for row in selected}
    bundle = ThermalVAEPipelineBundle(
        lotus,
        ms2_root,
        seeds,
        prompts,
        args.condition_posterior,
        latent_adapter=latent_adapter,
    )

    gen_prediction = generate_thermal_vae_prediction
    if args.save_raw_pred:
        raw_dir = output / "raw_predictions"
        raw_dir.mkdir(exist_ok=True)
        id_by_thermal_path = {row["thermal_path"]: row["id"] for row in selected}

        def gen_prediction(input_thermal, pipeline, image_path=None, dataset_name=None):
            disparity = generate_thermal_vae_prediction(input_thermal, pipeline, image_path, dataset_name)
            np.save(raw_dir / f"{id_by_thermal_path[image_path]}.npy", disparity.astype(np.float32))
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
    diagnostics_path = output / "condition_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(
            {
                "metric_scope": "condition-latent interface diagnostics only",
                "not_depth_quality_metrics": True,
                "condition_reference_self_mse_mean": float(
                    np.mean(
                        [item["condition_reference_self_mse"] for item in bundle.condition_diagnostics]
                    )
                ),
                "records": bundle.condition_diagnostics,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    metadata = {
        "evaluator": "lotus/evaluation/evaluation.py::evaluation_depth",
        "visualizer": "lotus/utils/image_utils.py::colorize_depth_map",
        "route": (
            "thermal-vae-latent-adapter-plus-pretrained-Lotus-G-UNet"
            if latent_adapter is not None
            else "thermal-vae-condition-plus-trained-UNet"
            if args.unet_checkpoint is not None
            else "thermal-vae-condition-plus-pretrained-Lotus-G-UNet"
        ),
        "six_route_line": (
            "e" if latent_adapter is not None
            else "d" if args.unet_checkpoint is not None
            else "c"
        ),
        **unet_checkpoint_info,
        **latent_adapter_info,
        "metric_scope": "official upstream Lotus depth-quality metrics",
        "distillation_diagnostics_included": False,
        "condition_diagnostics": str(diagnostics_path),
        "condition_reference_self_mse": 0.0,
        "condition_reference_self_mse_scope": "interface sanity only; not depth quality",
        "depth_target_loss_available": False,
        "caption_mode": args.caption_mode,
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "sample_count": len(selected),
        "alignment": "least_square_disparity",
        "raw_predictions_saved": bool(args.save_raw_pred),
        "dtype": args.dtype,
        "condition_posterior": args.condition_posterior,
        "seed_policy": "same diffusion noise seeds as Direct/V2; condition seed = sample seed + 1000000",
        "metrics": tracker.result(),
    }
    (output / "official_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
