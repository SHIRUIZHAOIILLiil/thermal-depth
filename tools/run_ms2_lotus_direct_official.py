"""Run the Direct AnyThermal->Lotus-G route through the upstream Lotus evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from models.anythermal_lotus_bridge import AnyThermalLotusBridge
from pipeline import LotusGPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--caption-mode", choices=("empty", "correct"), default="empty")
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
            if not depth or "/thr/" not in depth.replace("\\", "/"):
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


def choose_uniform(rows, count: int):
    if count <= 0 or count >= len(rows):
        return rows
    return [rows[int(i)] for i in np.linspace(0, len(rows) - 1, count, dtype=int)]


def task_embedding(device, dtype):
    task = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
    return torch.cat([torch.sin(task), torch.cos(task)], dim=-1)


@dataclass
class DirectPipelineBundle:
    anythermal: AnyThermalEncoder
    bridge: AnyThermalLotusBridge
    lotus: LotusGPipeline
    ms2_root: Path
    seeds: dict[str, int]
    prompts: dict[str, str]


def generate_direct_prediction(input_thermal, bundle, image_path=None, dataset_name=None):
    del input_thermal, dataset_name
    if image_path is None:
        raise ValueError("Lotus evaluator did not provide image_path")
    thermal_path = bundle.ms2_root / image_path
    encoded = bundle.anythermal.encode(thermal_path)
    spatial = encoded["spatial_features"]
    height, width = map(int, encoded["original_shape"][:2])
    lotus = bundle.lotus
    device = lotus._execution_device
    dtype = lotus.unet.dtype
    target = (height // int(lotus.vae_scale_factor), width // int(lotus.vae_scale_factor))
    condition = bundle.bridge(
        spatial.to(device=device, dtype=dtype),
        target_size=target,
        output_channels=int(lotus.unet.config.in_channels) // 2,
    )
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
    selected_jsonl = output / "selected_manifest.jsonl"
    selected_jsonl.write_text(
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
    for module in (lotus.vae, lotus.text_encoder, lotus.unet):
        module.requires_grad_(False).eval()
    bridge = AnyThermalLotusBridge().to(device).eval()
    if sum(parameter.numel() for parameter in bridge.parameters()) != 0:
        raise RuntimeError("Direct bridge must remain zero-parameter")
    seeds = {row["thermal_path"]: args.seed + index for index, row in enumerate(selected)}
    if args.caption_mode == "correct":
        empty_caption_ids = [row["id"] for row in selected if not row["caption"].strip()]
        if empty_caption_ids:
            raise RuntimeError(f"Correct-caption mode found empty captions: {empty_caption_ids[:5]}")
        prompts = {row["thermal_path"]: row["caption"] for row in selected}
    else:
        prompts = {row["thermal_path"]: "" for row in selected}
    bundle = DirectPipelineBundle(anythermal, bridge, lotus, ms2_root, seeds, prompts)

    gen_prediction = generate_direct_prediction
    if args.save_raw_pred:
        raw_dir = output / "raw_predictions"
        raw_dir.mkdir(exist_ok=True)
        id_by_thermal_path = {row["thermal_path"]: row["id"] for row in selected}

        def gen_prediction(input_thermal, pipeline, image_path=None, dataset_name=None):
            disparity = generate_direct_prediction(input_thermal, pipeline, image_path, dataset_name)
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
        "route": "direct-zero-parameter-AnyThermal-to-Lotus-G",
        "metric_scope": "official upstream Lotus depth-quality metrics",
        "distillation_diagnostics_included": False,
        "caption_mode": args.caption_mode,
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "sample_count": len(selected),
        "alignment": "least_square_disparity",
        "raw_predictions_saved": bool(args.save_raw_pred),
        "metrics": tracker.result(),
    }
    (output / "official_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
