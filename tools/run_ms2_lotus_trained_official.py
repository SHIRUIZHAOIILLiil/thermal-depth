"""Evaluate a trained AnyThermal adapter route with the upstream Lotus evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = ROOT / "lotus"
for path in (ROOT, LOTUS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluation.evaluation import evaluation_depth
from models.anythermal_encoder import AnyThermalEncoder
from models.anythermal_lotus_adapter import AnyThermalLotusAdapter
from models.anythermal_lotus_adapter_v2 import AnyThermalLotusAdapterV2
from models.anythermal_lotus_adapter_v2_2 import AnyThermalLotusAdapterV22
from models.anythermal_lotus_adapter_v2_3 import AnyThermalLotusAdapterV23
from models.anythermal_lotus_conditioner import AnyThermalDirectConditioner
from models.anythermal_lotus_model import extract_anythermal_feature_pyramid
from models.anythermal_lotus_v2 import thermal_to_lotus_input
from pipeline import LotusGPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument(
        "--caption-mode",
        choices=("empty", "correct", "hard-wrong", "fixed"),
        default="empty",
    )
    parser.add_argument(
        "--fixed-caption",
        default=(
            "Music theory and abstract algebra are both studied by many "
            "curious students around the world."
        ),
        help=(
            "Scene-free neutral text used for every sample in --caption-mode fixed; "
            "separates the token-presence tax from object-layout priors."
        ),
    )
    parser.add_argument(
        "--caption-transform",
        choices=("none", "objects-only", "no-positions"),
        default="none",
        help=(
            "Rewrite captions before CLIP encoding: 'no-positions' strips the "
            "parenthesized lateral tags but keeps Near/Middle/Far bands; "
            "'objects-only' additionally strips the bands, leaving a flat object list."
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
            # MS2 ships thermal-view and RGB-view GT and mixing them is a protocol
            # violation (frozen conclusion 15). Datasets with a single registered
            # depth (RGBDT500) declare themselves and are exempt.
            if str(row.get("dataset", "MS2")).upper() == "MS2" and "/thr/" not in depth.replace("\\", "/"):
                raise ValueError(f"Manifest row {row.get('id')} lacks thermal-view GT")
            rows.append({
                "id": row["id"],
                "thermal_path": row["thermal_path"],
                "thermal_depth_path": depth,
                "caption": str(row.get("caption", "")),
            })
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


_LATERAL_TAG_RE = re.compile(r"\s*\([^)]*\)")
_DEPTH_BAND_RE = re.compile(r"\b(near|middle|far)\s*:\s*", re.IGNORECASE)


def transform_caption(caption: str, transform: str) -> str:
    """P3 spatial-caption ablations: peel the spatial scaffolding off a caption.

    'no-positions' keeps the VLM's depth-band guesses (hypothesis A carrier),
    'objects-only' keeps nothing but the object mentions.
    """
    if transform == "none":
        return caption
    text = _LATERAL_TAG_RE.sub("", caption)
    if transform == "no-positions":
        return re.sub(r"\s{2,}", " ", text).strip()
    text = _DEPTH_BAND_RE.sub(" ", text)
    items = [
        token.strip(" .,")
        for sentence in text.split(".")
        for token in sentence.split(",")
    ]
    items = [token for token in items if token]
    return ", ".join(items) + "." if items else ""


def choose_uniform(rows, count: int):
    if count <= 0 or count >= len(rows):
        return rows
    return [rows[int(i)] for i in np.linspace(0, len(rows) - 1, count, dtype=int)]


def task_embedding(device, dtype):
    task = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
    return torch.cat([torch.sin(task), torch.cos(task)], dim=-1)


@dataclass
class TrainedPipelineBundle:
    anythermal: AnyThermalEncoder
    adapter: torch.nn.Module
    lotus: LotusGPipeline
    ms2_root: Path
    seeds: dict[str, int]
    prompts: dict[str, str]


def generate_trained_prediction(input_thermal, bundle, image_path=None, dataset_name=None):
    del input_thermal, dataset_name
    if image_path is None:
        raise ValueError("Lotus evaluator did not provide image_path")
    thermal_path = bundle.ms2_root / image_path
    features, info, _ = extract_anythermal_feature_pyramid(bundle.anythermal, thermal_path)
    height, width = map(int, info.original_shape[:2])
    lotus = bundle.lotus
    device = lotus._execution_device
    dtype = lotus.unet.dtype
    target = (height // int(lotus.vae_scale_factor), width // int(lotus.vae_scale_factor))
    adapter_parameter = next(bundle.adapter.parameters(), None)
    adapter_dtype = dtype if adapter_parameter is None else adapter_parameter.dtype
    adapter_features = [feature.to(device=device, dtype=adapter_dtype) for feature in features]
    if getattr(bundle.adapter, "requires_thermal_input", False):
        audited_thermal = thermal_to_lotus_input(thermal_path, processing_res=0)
        condition = bundle.adapter(
            adapter_features,
            audited_thermal.tensor.to(device=device, dtype=adapter_dtype),
            target_size=target,
        ).to(dtype=dtype)
    else:
        condition = bundle.adapter(adapter_features, target_size=target).to(dtype=dtype)
    generator = torch.Generator(device=device).manual_seed(bundle.seeds[image_path])
    noise = torch.randn(
        (1, int(lotus.unet.config.in_channels) // 2, *target),
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
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest = args.manifest.resolve()
    checkpoint_path = args.checkpoint.resolve()
    ms2_root = args.ms2_root.resolve()
    selected = choose_uniform(read_manifest(manifest), args.max_samples)
    filename_list = output / "official_ms2_filename_list.txt"
    filename_list.write_text(
        "".join(f"{row['thermal_path']} {row['thermal_depth_path']}\n" for row in selected),
        encoding="utf-8",
    )
    config = output / "official_ms2_dataset.yaml"
    config.write_text(
        "\n".join([
            "name: ms2_thermal",
            "disp_name: MS2 thermal-view filtered LiDAR",
            "dir: .",
            f"filenames: {filename_list.as_posix()}",
            f"depth_scale: {args.depth_scale}",
            f"min_depth: {args.min_depth}",
            f"max_depth: {args.max_depth}",
            "processing_res: null",
            "",
        ]),
        encoding="utf-8",
    )
    (output / "selected_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )

    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    anythermal = AnyThermalEncoder(
        model_path=args.anythermal_model_path,
        device="cuda",
        local_files_only=args.local_files_only,
    )
    lotus = LotusGPipeline.from_pretrained(
        args.lotus_model_path,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_settings = checkpoint.get("settings", {})
    adapter_architecture = checkpoint.get(
        "adapter_architecture",
        checkpoint_settings.get("adapter_architecture", "v1_shallow"),
    )
    if checkpoint.get("train_mode") == "unet_only":
        adapter = AnyThermalDirectConditioner().to(device=device, dtype=dtype)
    else:
        if adapter_architecture == "v2_3_thermal_detail_skip":
            adapter = AnyThermalLotusAdapterV23().to(device=device, dtype=dtype)
        elif adapter_architecture == "v2_2_progressive_residual_decoder":
            adapter = AnyThermalLotusAdapterV22().to(device=device, dtype=dtype)
        elif adapter_architecture == "v2_1_spatial_decoder":
            adapter = AnyThermalLotusAdapterV2().to(device=device, dtype=dtype)
        else:
            use_output_group_norm = bool(
                checkpoint_settings.get("use_output_group_norm", True)
            )
            adapter = AnyThermalLotusAdapter(
                use_output_group_norm=use_output_group_norm,
            ).to(device=device, dtype=dtype)
    conditioner_state = checkpoint.get(
        "conditioner_state_dict",
        checkpoint.get("adapter_state_dict", checkpoint.get("adapter")),
    )
    if conditioner_state is None:
        raise RuntimeError(
            "Checkpoint has no conditioner_state_dict, adapter_state_dict, or V2 adapter state"
        )
    adapter.load_state_dict(conditioner_state, strict=True)
    if "lotus_unet_state_dict" in checkpoint:
        lotus.unet.load_state_dict(checkpoint["lotus_unet_state_dict"], strict=True)
    for module in (adapter, lotus.vae, lotus.text_encoder, lotus.unet):
        module.requires_grad_(False).eval()
    if args.caption_mode in ("correct", "hard-wrong"):
        empty_caption_ids = [row["id"] for row in selected if not row["caption"].strip()]
        if empty_caption_ids:
            raise RuntimeError(
                f"{args.caption_mode}-caption mode found empty captions: {empty_caption_ids[:5]}"
            )
    if args.caption_mode == "correct":
        prompts = {row["thermal_path"]: row["caption"] for row in selected}
    elif args.caption_mode == "hard-wrong":
        if len(selected) < 2:
            raise RuntimeError("hard-wrong caption mode needs at least 2 samples.")
        # Deterministic derangement: cyclic shift along a seeded shuffle, so no
        # sample keeps its own caption and the assignment is reproducible.
        order = list(range(len(selected)))
        random.Random(args.seed + 777).shuffle(order)
        prompts = {}
        for position, sample_index in enumerate(order):
            donor_index = order[(position + 1) % len(order)]
            prompts[selected[sample_index]["thermal_path"]] = selected[donor_index]["caption"]
    elif args.caption_mode == "fixed":
        if not args.fixed_caption.strip():
            raise SystemExit("--caption-mode fixed needs a non-empty --fixed-caption")
        prompts = {row["thermal_path"]: args.fixed_caption for row in selected}
    else:
        prompts = {row["thermal_path"]: "" for row in selected}
    if args.caption_transform != "none":
        if args.caption_mode in ("empty", "fixed"):
            raise SystemExit("--caption-transform requires --caption-mode correct/hard-wrong")
        prompts = {
            path: transform_caption(text, args.caption_transform)
            for path, text in prompts.items()
        }
        (output / "applied_prompts.jsonl").write_text(
            "".join(
                json.dumps(
                    {"thermal_path": row["thermal_path"], "prompt": prompts[row["thermal_path"]]},
                    ensure_ascii=False,
                )
                + "\n"
                for row in selected
            ),
            encoding="utf-8",
        )
    seeds = {row["thermal_path"]: args.seed + index for index, row in enumerate(selected)}
    bundle = TrainedPipelineBundle(anythermal, adapter, lotus, ms2_root, seeds, prompts)

    gen_prediction = generate_trained_prediction
    if args.save_raw_pred:
        raw_dir = output / "raw_predictions"
        raw_dir.mkdir(exist_ok=True)
        id_by_thermal_path = {row["thermal_path"]: row["id"] for row in selected}

        def gen_prediction(input_thermal, pipeline, image_path=None, dataset_name=None):
            disparity = generate_trained_prediction(input_thermal, pipeline, image_path, dataset_name)
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
    metadata = {
        "evaluator": "lotus/evaluation/evaluation.py::evaluation_depth",
        "visualizer": "lotus/utils/image_utils.py::colorize_depth_map",
        "route": str(
            checkpoint.get(
                "train_mode",
                checkpoint.get("format", "trained-adapter"),
            )
        ),
        "metric_scope": "official upstream Lotus depth-quality metrics",
        "distillation_diagnostics_included": False,
        "training_caption_mode": str(checkpoint.get("caption_mode", "disabled")),
        "caption_mode": args.caption_mode,
        "caption_transform": args.caption_transform,
        "fixed_caption": args.fixed_caption if args.caption_mode == "fixed" else None,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "adapter_use_output_group_norm": (
            False
            if checkpoint.get("train_mode") == "unet_only"
            or adapter_architecture
            in {
                "direct_zero_parameter",
                "v2_1_spatial_decoder",
                "v2_2_progressive_residual_decoder",
                "v2_3_thermal_detail_skip",
            }
            else bool(checkpoint.get("settings", {}).get("use_output_group_norm", True))
        ),
        "adapter_architecture": checkpoint.get(
            "adapter_architecture",
            checkpoint.get("settings", {}).get("adapter_architecture", "v1_shallow"),
        ),
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "sample_count": len(selected),
        "alignment": "least_square_disparity",
        "raw_predictions_saved": bool(args.save_raw_pred),
        "dtype": args.dtype,
        "metrics": tracker.result(),
    }
    (output / "official_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
