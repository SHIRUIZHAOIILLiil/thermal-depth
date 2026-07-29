"""Paired latent-loss audit for three MS2 thermal conditioning routes.

This is a read-only diagnostic, not training and not a depth-metric evaluator.
Every route uses the same samples, target depth latents, timestep, initial noise,
empty text embedding, and latent validity mask.

Routes:
1. thermal_vae_pretrained_unet
2. anythermal_direct_pretrained_unet
3. trained_adapter_trained_unet
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = ROOT / "lotus"
for path in (ROOT, LOTUS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.anythermal_encoder import AnyThermalEncoder
from models.anythermal_lotus_adapter import AnyThermalLotusAdapter
from models.anythermal_lotus_conditioner import AnyThermalDirectConditioner
from models.anythermal_lotus_model import (
    extract_anythermal_feature_pyramid,
    latent_valid_mask,
)
from pipeline import LotusGPipeline
from tools.train_ms2_adapter_v0 import MS2AdapterDataset, collate_samples


ROUTES = (
    "thermal_vae_pretrained_unet",
    "anythermal_direct_pretrained_unet",
    "trained_adapter_trained_unet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_uniform(dataset: MS2AdapterDataset, count: int) -> None:
    if count <= 0 or count >= len(dataset):
        return
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int)
    dataset.samples = [dataset.samples[int(index)] for index in indices]
    dataset.sample_ids = [sample["id"] for sample in dataset.samples]
    dataset._id_to_index = {sample_id: index for index, sample_id in enumerate(dataset.sample_ids)}


def empty_prompt(pipeline: LotusGPipeline, batch_size: int, device: torch.device) -> torch.Tensor:
    text_inputs = pipeline.tokenizer(
        [""] * batch_size,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        return pipeline.text_encoder(text_inputs.input_ids.to(device), return_dict=False)[0]


def task_embedding(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    task = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype).repeat(batch_size, 1)
    return torch.cat([torch.sin(task), torch.cos(task)], dim=-1)


def seeded_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


@torch.no_grad()
def target_and_noise(
    pipeline: LotusGPipeline,
    batch: Dict,
    *,
    batch_index: int,
    seed: int,
    timestep: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = pipeline.unet.dtype
    depth_values = batch["depth_values"].to(device=device, dtype=dtype)
    target_dist = pipeline.vae.encode(depth_values).latent_dist
    target = target_dist.sample(
        generator=seeded_generator(device, seed + 1_000_000 + batch_index)
    ) * pipeline.vae.config.scaling_factor
    noise = torch.randn(
        target.shape,
        generator=seeded_generator(device, seed + 2_000_000 + batch_index),
        device=device,
        dtype=dtype,
    )
    timesteps = torch.full(
        (target.shape[0],), int(timestep), device=device, dtype=torch.long
    )
    noisy = pipeline.scheduler.add_noise(target, noise, timesteps)
    mask = latent_valid_mask(
        batch["valid_mask"].to(device),
        target,
        vae_scale_factor=int(pipeline.vae_scale_factor),
    )
    if not bool(mask.any()):
        raise RuntimeError(f"Batch {batch_index} has no valid latent pixels")
    return target, noisy, timesteps, mask


def thermal_tensor(paths: Sequence[str], device: torch.device, dtype: torch.dtype):
    tensors = []
    diagnostics = []
    for path in paths:
        with Image.open(path) as image:
            raw = np.array(image, copy=True)
        converted = AnyThermalEncoder._array_to_uint8(raw)
        if converted.ndim == 3:
            converted = converted[..., 0]
        if converted.ndim != 2:
            raise RuntimeError(f"Unexpected thermal shape after conversion: {converted.shape}")
        diagnostics.append({
            "path": str(path),
            "raw_dtype": str(raw.dtype),
            "raw_min": float(np.min(raw)),
            "raw_max": float(np.max(raw)),
            "uint8_min": int(converted.min()),
            "uint8_max": int(converted.max()),
            "uint8_std": float(converted.std()),
        })
        rgb = np.repeat(converted[..., None], 3, axis=-1)
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()
        tensors.append(tensor / 127.5 - 1.0)
    return torch.stack(tensors).to(device=device, dtype=dtype), diagnostics


@torch.no_grad()
def thermal_vae_condition(
    pipeline: LotusGPipeline,
    paths: Sequence[str],
    *,
    batch_index: int,
    seed: int,
    device: torch.device,
):
    images, diagnostics = thermal_tensor(paths, device, pipeline.vae.dtype)
    dist = pipeline.vae.encode(images).latent_dist
    condition = dist.sample(
        generator=seeded_generator(device, seed + 3_000_000 + batch_index)
    ) * pipeline.vae.config.scaling_factor
    return condition.to(dtype=pipeline.unet.dtype), diagnostics


@torch.no_grad()
def anythermal_features(
    encoder: AnyThermalEncoder,
    paths: Sequence[str],
    device: torch.device,
) -> List[torch.Tensor]:
    per_sample = []
    for path in paths:
        features, _, _ = extract_anythermal_feature_pyramid(encoder, Path(path))
        per_sample.append([feature.detach().cpu() for feature in features])
    return [
        torch.cat([sample[level] for sample in per_sample], dim=0).to(device)
        for level in range(len(per_sample[0]))
    ]


@torch.no_grad()
def predict_latent(
    pipeline: LotusGPipeline,
    condition: torch.Tensor,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    prompt: torch.Tensor,
) -> torch.Tensor:
    dtype = pipeline.unet.dtype
    sample = torch.cat(
        [condition.to(dtype=dtype), noisy.to(dtype=dtype)], dim=1
    )
    return pipeline.unet(
        sample,
        timesteps,
        encoder_hidden_states=prompt.to(dtype=dtype),
        class_labels=task_embedding(sample.shape[0], sample.device, dtype),
        return_dict=False,
    )[0]


def record_losses(
    rows: List[Dict],
    route: str,
    batch: Dict,
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, int]:
    squared = (prediction.float() - target.float()).square()
    total_sum = 0.0
    total_count = 0
    for index, sample_id in enumerate(batch["ids"]):
        selected = squared[index][mask[index]]
        if selected.numel() == 0:
            raise RuntimeError(f"No valid latent elements for sample {sample_id}")
        value = float(selected.mean().cpu())
        count = int(selected.numel())
        rows.append({
            "sample_id": sample_id,
            "route": route,
            "latent_mse": value,
            "valid_latent_elements": count,
        })
        total_sum += float(selected.sum().cpu())
        total_count += count
    return total_sum, total_count


def summarize(rows: Sequence[Dict]) -> Dict:
    result = {}
    for route in ROUTES:
        selected = [row for row in rows if row["route"] == route]
        values = np.asarray([row["latent_mse"] for row in selected], dtype=np.float64)
        weighted_sum = sum(
            row["latent_mse"] * row["valid_latent_elements"] for row in selected
        )
        weighted_count = sum(row["valid_latent_elements"] for row in selected)
        result[route] = {
            "sample_count": len(selected),
            "imagewise_mean": float(values.mean()),
            "imagewise_std": float(values.std()),
            "imagewise_median": float(np.median(values)),
            "imagewise_min": float(values.min()),
            "imagewise_max": float(values.max()),
            "global_valid_latent_mse": float(weighted_sum / weighted_count),
            "valid_latent_elements": int(weighted_count),
        }
    base = result["anythermal_direct_pretrained_unet"]["imagewise_mean"]
    for route in ROUTES:
        value = result[route]["imagewise_mean"]
        result[route]["relative_to_anythermal_direct"] = float(value / base)
        result[route]["improvement_vs_anythermal_direct"] = float(base - value)
    return result


def write_csv(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "route", "latent_mse", "valid_latent_elements"),
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0 <= args.timestep <= 999:
        raise ValueError("timestep must be in [0, 999]")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest = args.manifest.resolve()
    checkpoint_path = args.trained_checkpoint.resolve()
    dataset = MS2AdapterDataset(
        manifest=manifest,
        ms2_root=args.ms2_root.resolve(),
        max_samples=None,
    )
    original_count = len(dataset)
    select_uniform(dataset, args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_samples,
    )

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this audit")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    pipeline = LotusGPipeline.from_pretrained(
        args.lotus_model_path,
        torch_dtype=torch.float32,
        local_files_only=args.local_files_only,
    ).to(device)
    anythermal = AnyThermalEncoder(
        model_path=args.anythermal_model_path,
        device="cuda",
        local_files_only=args.local_files_only,
    )
    for module in (pipeline.vae, pipeline.text_encoder, pipeline.unet, anythermal.model):
        module.requires_grad_(False).eval()
    direct = AnyThermalDirectConditioner().to(device=device, dtype=torch.float32).eval()

    rows: List[Dict] = []
    thermal_diagnostics = []
    print(f"Base routes: {len(dataset)} uniformly selected samples")
    for batch_index, batch in enumerate(loader):
        target, noisy, timesteps, mask = target_and_noise(
            pipeline,
            batch,
            batch_index=batch_index,
            seed=args.seed,
            timestep=args.timestep,
            device=device,
        )
        prompt = empty_prompt(pipeline, target.shape[0], device)
        thermal_condition, diagnostics = thermal_vae_condition(
            pipeline,
            batch["thermal_paths"],
            batch_index=batch_index,
            seed=args.seed,
            device=device,
        )
        if len(thermal_diagnostics) < 8:
            thermal_diagnostics.extend(diagnostics[: 8 - len(thermal_diagnostics)])
        thermal_pred = predict_latent(
            pipeline, thermal_condition, noisy, timesteps, prompt
        )
        record_losses(
            rows,
            "thermal_vae_pretrained_unet",
            batch,
            thermal_pred,
            target,
            mask,
        )

        features = anythermal_features(anythermal, batch["thermal_paths"], device)
        direct_condition = direct(
            [feature.to(dtype=torch.float32) for feature in features],
            target_size=tuple(target.shape[-2:]),
        )
        direct_pred = predict_latent(
            pipeline, direct_condition, noisy, timesteps, prompt
        )
        record_losses(
            rows,
            "anythermal_direct_pretrained_unet",
            batch,
            direct_pred,
            target,
            mask,
        )
        print(f"base {min((batch_index + 1) * args.batch_size, len(dataset))}/{len(dataset)}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("train_mode") != "adapter_unet":
        raise RuntimeError(
            "trained-checkpoint must be an adapter_unet checkpoint, got "
            f"{checkpoint.get('train_mode')!r}"
        )
    adapter = AnyThermalLotusAdapter().to(device=device, dtype=torch.float32).eval()
    conditioner_state = checkpoint.get(
        "conditioner_state_dict", checkpoint.get("adapter_state_dict")
    )
    if conditioner_state is None or "lotus_unet_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint lacks trained Adapter or Lotus U-Net state")
    adapter.load_state_dict(conditioner_state, strict=True)
    pipeline.unet.load_state_dict(checkpoint["lotus_unet_state_dict"], strict=True)
    adapter.requires_grad_(False)
    pipeline.unet.requires_grad_(False).eval()

    print("Trained joint route")
    for batch_index, batch in enumerate(loader):
        target, noisy, timesteps, mask = target_and_noise(
            pipeline,
            batch,
            batch_index=batch_index,
            seed=args.seed,
            timestep=args.timestep,
            device=device,
        )
        prompt = empty_prompt(pipeline, target.shape[0], device)
        features = anythermal_features(anythermal, batch["thermal_paths"], device)
        condition = adapter(
            [feature.to(dtype=torch.float32) for feature in features],
            target_size=tuple(target.shape[-2:]),
        )
        prediction = predict_latent(
            pipeline, condition, noisy, timesteps, prompt
        )
        record_losses(
            rows,
            "trained_adapter_trained_unet",
            batch,
            prediction,
            target,
            mask,
        )
        print(f"trained {min((batch_index + 1) * args.batch_size, len(dataset))}/{len(dataset)}")

    counts = {route: sum(row["route"] == route for row in rows) for route in ROUTES}
    if len(set(counts.values())) != 1 or next(iter(counts.values())) != len(dataset):
        raise RuntimeError(f"Route sample-count mismatch: {counts}")
    write_csv(output / "per_sample_loss.csv", rows)
    summary = {
        "protocol": {
            "manifest": str(manifest),
            "manifest_sha256": sha256(manifest),
            "manifest_sample_count": original_count,
            "selected_sample_count": len(dataset),
            "selection": "uniform_over_manifest_order",
            "selected_ids": list(dataset.sample_ids),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
            "lotus_model": args.lotus_model_path,
            "anythermal_model": args.anythermal_model_path,
            "timestep": int(args.timestep),
            "seed": int(args.seed),
            "caption": "empty",
            "target": "legacy per-image-normalized semi-dense GT VAE latent",
            "mask": "same max-pooled GT-valid latent mask for every route",
            "dtype": "float32",
            "training": False,
        },
        "routes": summarize(rows),
        "thermal_conversion_audit": thermal_diagnostics,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary["routes"], indent=2, ensure_ascii=False))
    print(output / "summary.json")


if __name__ == "__main__":
    main()
