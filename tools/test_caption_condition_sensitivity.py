"""Check whether Lotus-G reacts to real MS2 captions before caption training.

This is a forward-only diagnostic. It loads the Adapter-only thermal baseline,
keeps every model component frozen, and compares four text-conditioning modes
on the same fixed validation samples, noise, timestep, and random seed:

disabled -> the current wrapper behavior, prompt_embeds=None
empty    -> explicit max-length empty CLIP prompt
real     -> InternVL3-8B manifest caption
shuffled -> deterministic mismatched manifest caption
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = REPO_ROOT / "tools"
for path in (REPO_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import overfit_32_anythermal_lotus as overfit  # noqa: E402
from train_ms2_adapter_v0 import (  # noqa: E402
    DEFAULT_VAL_MANIFEST,
    MS2AdapterDataset,
    collate_samples,
    configure_trainability,
    decoded_depth,
    extract_batch_features,
    load_adapter_initialization,
    load_lotus,
    make_noise_for_batch,
    metric_values,
    normalized_gt_from_depth_values,
    normalize_prediction,
    save_image,
    scale_shift_align,
    set_seed,
)


MODES = ("disabled", "empty", "real", "shuffled")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-manifest", default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--ms2-root", default="/mnt/e/dataset/ms2")
    parser.add_argument(
        "--adapter-checkpoint",
        default="outputs/adapter_v0_thermal_only_short_run_v2/checkpoint_best.pt",
    )
    parser.add_argument(
        "--fixed-samples-json",
        default="outputs/adapter_v0_thermal_only_short_run_v2/fixed_val_samples.json",
    )
    parser.add_argument("--output-dir", default="outputs/adapter_v0_caption_sensitivity")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--anythermal-revision", default=None)
    parser.add_argument(
        "--lotus-model-path",
        default="jingheya/lotus-depth-g-v2-1-disparity",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def safe_id(text: str) -> str:
    return text.replace("/", "_").replace("\\", "_").replace(":", "_")


def jsonable_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    data = tensor.detach().float().cpu()
    return {
        "shape": list(data.shape),
        "mean": float(data.mean()),
        "std": float(data.std(unbiased=False)),
        "min": float(data.min()),
        "max": float(data.max()),
        "finite": bool(torch.isfinite(data).all()),
    }


def extended_metric_values(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    base = metric_values(pred, gt, valid)
    if not valid.any():
        base.update({"mae": 0.0, "p95": 0.0, "p99": 0.0})
        return base
    err = np.abs(pred[valid].astype(np.float64) - gt[valid].astype(np.float64))
    base.update(
        {
            "mae": float(np.mean(err)),
            "p95": float(np.percentile(err, 95.0)),
            "p99": float(np.percentile(err, 99.0)),
        }
    )
    return base


def average_records(records: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {}
    keys = sorted({key for item in records for key in item})
    return {
        key: float(np.mean([item[key] for item in records if key in item]))
        for key in keys
    }


def load_fixed_ids(dataset: MS2AdapterDataset, fixed_json: Path, max_samples: int) -> List[str]:
    if fixed_json.is_file():
        with fixed_json.open("r", encoding="utf-8") as handle:
            fixed_ids = json.load(handle)
        fixed_ids = [sample_id for sample_id in fixed_ids if sample_id in dataset.sample_ids]
    else:
        fixed_ids = []
    if not fixed_ids:
        fixed_ids = dataset.sample_ids[:max_samples]
    return fixed_ids[:max_samples]


def encode_prompts(model, captions: Sequence[str]) -> torch.Tensor:
    text_encoder = model.lotus.text_encoder
    tokenizer = model.lotus.tokenizer
    device = next(text_encoder.parameters()).device
    with torch.no_grad():
        text_inputs = tokenizer(
            list(captions),
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        return text_encoder(input_ids, return_dict=False)[0]


def prompt_embeds_for_mode(
    model,
    mode: str,
    captions: Sequence[str],
    shuffled_captions: Sequence[str],
) -> Optional[torch.Tensor]:
    if mode == "disabled":
        return None
    if mode == "empty":
        return encode_prompts(model, [""] * len(captions))
    if mode == "real":
        return encode_prompts(model, captions)
    if mode == "shuffled":
        return encode_prompts(model, shuffled_captions)
    raise ValueError(f"Unsupported caption mode: {mode}")


def reset_torch_rng(seed: int, device: torch.device, batch_marker: int) -> None:
    torch.manual_seed(seed + batch_marker)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + batch_marker)


def run_mode_forward(
    *,
    model,
    batch: Dict[str, Any],
    mode: str,
    shuffled_caption_by_id: Dict[str, str],
    device: torch.device,
    seed: int,
    timestep: int,
    batch_marker: int,
) -> Dict[str, Any]:
    features = extract_batch_features(model, batch["thermal_paths"], device=device)
    depth_values = batch["depth_values"].to(device)
    valid_mask = batch["valid_mask"].to(device)
    with torch.no_grad():
        target_latents = model.encode_depth_latents(depth_values)
    latent_size = tuple(target_latents.shape[-2:])
    unet_dtype = next(model.lotus.unet.parameters()).dtype
    noise = make_noise_for_batch(
        batch_size=depth_values.shape[0],
        latent_size=latent_size,
        global_step=0,
        batch_indices=batch["indices"],
        seed=seed,
        device=device,
        dtype=unet_dtype,
    )
    timesteps = torch.full(
        (depth_values.shape[0],),
        timestep,
        device=device,
        dtype=torch.long,
    )
    shuffled = [shuffled_caption_by_id[sample_id] for sample_id in batch["ids"]]
    prompt_embeds = prompt_embeds_for_mode(
        model,
        mode,
        batch["captions"],
        shuffled,
    )
    reset_torch_rng(seed, device, batch_marker)
    return model(
        features=features,
        depth_values=depth_values,
        valid_mask=valid_mask,
        timesteps=timesteps,
        noise=noise,
        prompt_embeds=prompt_embeds,
        return_decoded=True,
    )


def array_to_gray_image(array: np.ndarray, valid: Optional[np.ndarray] = None) -> Image.Image:
    arr = array.astype(np.float32)
    if valid is not None and valid.any():
        vals = arr[valid]
    else:
        vals = arr.reshape(-1)
    lo = float(np.nanpercentile(vals, 1.0)) if vals.size else 0.0
    hi = float(np.nanpercentile(vals, 99.0)) if vals.size else 1.0
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    if valid is not None:
        norm = np.where(valid, norm, 0.0)
    return Image.fromarray((norm * 255.0).astype(np.uint8)).convert("RGB")


def make_panel(items: Sequence[Tuple[str, np.ndarray, Optional[np.ndarray]]], path: Path) -> None:
    if not items:
        return
    images = [array_to_gray_image(array, valid).resize((240, 96)) for _, array, valid in items]
    label_h = 22
    panel = Image.new("RGB", (240 * len(images), 96 + label_h), "white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    for idx, ((label, _, _), image) in enumerate(zip(items, images)):
        x = idx * 240
        panel.paste(image, (x, label_h))
        draw.text((x + 6, 5), label, fill=(0, 0, 0), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def save_sample_outputs(
    *,
    sample_dir: Path,
    sample: Dict[str, Any],
    raw_depth: np.ndarray,
    gt_norm: np.ndarray,
    valid: np.ndarray,
    thermal: np.ndarray,
    per_mode_pred: Dict[str, np.ndarray],
    per_mode_aligned: Dict[str, np.ndarray],
    per_mode_model_pred: Dict[str, torch.Tensor],
    per_mode_prompt: Dict[str, torch.Tensor],
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_image(thermal, sample_dir / "thermal.png")
    save_image(raw_depth, sample_dir / "gt_depth.png", valid)
    Image.fromarray((valid.astype(np.uint8) * 255)).save(sample_dir / "valid_mask.png")
    np.save(sample_dir / "thermal.npy", thermal.astype(np.float32))
    np.save(sample_dir / "gt_depth.npy", raw_depth.astype(np.float32))
    np.save(sample_dir / "gt_depth_normalized.npy", gt_norm.astype(np.float32))
    np.save(sample_dir / "valid_mask.npy", valid.astype(np.uint8))

    panel_items: List[Tuple[str, np.ndarray, Optional[np.ndarray]]] = [
        ("thermal", thermal, None),
        ("gt", raw_depth, valid),
    ]
    for mode in MODES:
        pred = per_mode_pred[mode]
        aligned = per_mode_aligned[mode]
        save_image(pred, sample_dir / f"pred_{mode}.png", valid)
        save_image(aligned, sample_dir / f"pred_{mode}_aligned.png", valid)
        np.save(sample_dir / f"pred_{mode}.npy", pred.astype(np.float32))
        np.save(sample_dir / f"pred_{mode}_aligned.npy", aligned.astype(np.float32))
        torch.save(per_mode_model_pred[mode].detach().cpu(), sample_dir / f"model_pred_{mode}.pt")
        torch.save(per_mode_prompt[mode].detach().cpu(), sample_dir / f"text_embeds_{mode}.pt")
        panel_items.append((mode, aligned, valid))

    make_panel(panel_items, sample_dir / "comparison_panel.png")
    with (sample_dir / "sample.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "id": sample["id"],
                "image_id": sample["image_id"],
                "sequence": sample["sequence"],
                "split": sample["split"],
                "caption": sample["caption"],
                "thermal_path": sample["thermal_path"],
                "depth_path": sample["depth_path"],
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    dataset = MS2AdapterDataset(
        manifest=Path(args.val_manifest),
        ms2_root=Path(args.ms2_root),
        max_samples=None,
    )
    fixed_ids = load_fixed_ids(dataset, Path(args.fixed_samples_json), args.max_samples)
    samples = [dataset.get_by_id(sample_id) for sample_id in fixed_ids]
    loader = DataLoader(
        samples,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_samples,
    )
    shuffled_captions = [sample["caption"] for sample in samples[1:] + samples[:1]]
    shuffled_caption_by_id = {
        sample["id"]: caption for sample, caption in zip(samples, shuffled_captions)
    }

    print("Loading frozen Adapter-only Lotus baseline...", flush=True)
    args.train_mode = "adapter_only"
    model = load_lotus(args, device)
    configure_trainability(model, "adapter_only")
    load_info = load_adapter_initialization(Path(args.adapter_checkpoint), model, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    records_by_mode: Dict[str, List[Dict[str, float]]] = {mode: [] for mode in MODES}
    text_stats_by_mode: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in MODES}
    adapter_stats: List[Dict[str, Any]] = []
    pairwise_records: List[Dict[str, Any]] = []
    all_caption_texts: List[Dict[str, str]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            outputs_by_mode: Dict[str, Dict[str, Any]] = {}
            for mode in MODES:
                outputs = run_mode_forward(
                    model=model,
                    batch=batch,
                    mode=mode,
                    shuffled_caption_by_id=shuffled_caption_by_id,
                    device=device,
                    seed=args.seed,
                    timestep=args.timestep,
                    batch_marker=batch_idx,
                )
                outputs_by_mode[mode] = outputs
                text_stats_by_mode[mode].append(jsonable_stats(outputs["prompt_embeds"]))

            adapter = outputs_by_mode["disabled"]["condition_latent"].detach().float().cpu()
            adapter_stats.append(jsonable_stats(adapter))
            gt_norm = normalized_gt_from_depth_values(batch["depth_values"])
            valid_batch = batch["valid_mask"].numpy().astype(bool)[:, 0]
            pred_by_mode = {mode: decoded_depth(outputs_by_mode[mode]) for mode in MODES}

            for local_idx, sample_id in enumerate(batch["ids"]):
                sample = {
                    "id": sample_id,
                    "image_id": batch["image_ids"][local_idx],
                    "sequence": batch["sequences"][local_idx],
                    "split": batch["splits"][local_idx],
                    "caption": batch["captions"][local_idx],
                    "thermal_path": batch["thermal_paths"][local_idx],
                    "depth_path": batch["depth_paths"][local_idx],
                }
                valid = valid_batch[local_idx]
                per_mode_pred: Dict[str, np.ndarray] = {}
                per_mode_aligned: Dict[str, np.ndarray] = {}
                per_mode_model_pred: Dict[str, torch.Tensor] = {}
                per_mode_prompt: Dict[str, torch.Tensor] = {}
                for mode in MODES:
                    pred = pred_by_mode[mode][local_idx]
                    aligned = scale_shift_align(pred, gt_norm[local_idx], valid)
                    pred_norm = normalize_prediction(pred, valid)
                    raw_metrics = extended_metric_values(pred_norm, gt_norm[local_idx], valid)
                    aligned_metrics = extended_metric_values(aligned, gt_norm[local_idx], valid)
                    records_by_mode[mode].append(
                        {
                            "loss": float(outputs_by_mode[mode]["loss"].detach().cpu()),
                            **{f"raw_{key}": value for key, value in raw_metrics.items()},
                            **{f"aligned_{key}": value for key, value in aligned_metrics.items()},
                        }
                    )
                    per_mode_pred[mode] = pred
                    per_mode_aligned[mode] = aligned
                    per_mode_model_pred[mode] = outputs_by_mode[mode]["model_pred"][local_idx]
                    per_mode_prompt[mode] = outputs_by_mode[mode]["prompt_embeds"][local_idx]

                real_empty = np.abs(per_mode_aligned["real"] - per_mode_aligned["empty"])[valid]
                real_shuffled = np.abs(per_mode_aligned["real"] - per_mode_aligned["shuffled"])[valid]
                disabled_empty = np.abs(per_mode_aligned["disabled"] - per_mode_aligned["empty"])[valid]
                model_real_empty = torch.mean(
                    torch.abs(per_mode_model_pred["real"].float() - per_mode_model_pred["empty"].float())
                )
                model_real_shuffled = torch.mean(
                    torch.abs(per_mode_model_pred["real"].float() - per_mode_model_pred["shuffled"].float())
                )
                pairwise_records.append(
                    {
                        "sample_id": sample_id,
                        "aligned_l1_real_empty": float(real_empty.mean()) if real_empty.size else 0.0,
                        "aligned_l1_real_shuffled": float(real_shuffled.mean()) if real_shuffled.size else 0.0,
                        "aligned_l1_disabled_empty": float(disabled_empty.mean()) if disabled_empty.size else 0.0,
                        "model_pred_l1_real_empty": float(model_real_empty.cpu()),
                        "model_pred_l1_real_shuffled": float(model_real_shuffled.cpu()),
                    }
                )
                all_caption_texts.append(
                    {
                        "sample_id": sample_id,
                        "real_caption": batch["captions"][local_idx],
                        "shuffled_caption": shuffled_caption_by_id[sample_id],
                    }
                )
                save_sample_outputs(
                    sample_dir=output_dir / "samples" / f"{len(pairwise_records)-1:02d}_{safe_id(sample_id)}",
                    sample=sample,
                    raw_depth=batch["raw_depth"][local_idx],
                    gt_norm=gt_norm[local_idx],
                    valid=valid,
                    thermal=overfit.thermal_vis(Path(batch["thermal_paths"][local_idx])),
                    per_mode_pred=per_mode_pred,
                    per_mode_aligned=per_mode_aligned,
                    per_mode_model_pred=per_mode_model_pred,
                    per_mode_prompt=per_mode_prompt,
                )
            print(f"Processed batch {batch_idx + 1}/{len(loader)}", flush=True)

    mode_summary = {
        mode: average_records(records)
        for mode, records in records_by_mode.items()
    }
    text_summary = {
        mode: {
            key: float(np.mean([record[key] for record in records]))
            if key != "shape" and records
            else (records[0]["shape"] if records else [])
            for key in (records[0].keys() if records else [])
        }
        for mode, records in text_stats_by_mode.items()
    }
    pairwise_summary = average_records(
        [
            {key: value for key, value in record.items() if key != "sample_id"}
            for record in pairwise_records
        ]
    )
    adapter_summary = {
        key: float(np.mean([record[key] for record in adapter_stats]))
        if key != "shape"
        else adapter_stats[0]["shape"]
        for key in adapter_stats[0]
    }

    summary = {
        "caption_interface_doc": "docs/lotus_caption_condition_interface.md",
        "output_dir": str(output_dir),
        "adapter_checkpoint": load_info,
        "fixed_sample_ids": fixed_ids,
        "seed": args.seed,
        "timestep": args.timestep,
        "batch_size": args.batch_size,
        "caption_modes": list(MODES),
        "mode_metrics": mode_summary,
        "pairwise_summary": pairwise_summary,
        "text_embedding_stats": text_summary,
        "adapter_output_stats": adapter_summary,
        "caption_effect_detected": bool(
            pairwise_summary.get("model_pred_l1_real_empty", 0.0) > 1e-6
            or pairwise_summary.get("model_pred_l1_real_shuffled", 0.0) > 1e-6
        ),
        "notes": [
            "disabled uses the current wrapper prompt_embeds=None behavior.",
            "empty uses explicit max-length CLIP tokenization for empty strings.",
            "All modes use the same fixed samples, timestep, generated noise, and restored RNG seed.",
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with (output_dir / "pairwise_differences.json").open("w", encoding="utf-8") as handle:
        json.dump(pairwise_records, handle, indent=2, ensure_ascii=False)
    with (output_dir / "caption_texts.json").open("w", encoding="utf-8") as handle:
        json.dump(all_caption_texts, handle, indent=2, ensure_ascii=False)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
