"""Minimal RGB caption-grounding audit for the original Lotus-G pipeline.

This audit intentionally excludes AnyThermal, thermal images, adapters, and
training.  It evaluates whether the released RGB Lotus-G checkpoint reacts to
correct captions under controlled inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = REPO_ROOT / "lotus"
for search_path in (REPO_ROOT, LOTUS_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from pipeline import LotusGPipeline  # noqa: E402


TEXT_MODES = ("correct", "empty", "generic", "hard_wrong")
GENERIC_CAPTION = "A driving scene."
DEFAULT_MANIFEST = Path(
    "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/"
    "ms2_fixed_sequence_internvl3_8b_filtered_caption_val.jsonl"
)
DEFAULT_MS2_ROOT = Path("/mnt/e/dataset/ms2")
DEFAULT_OUTPUT = Path("/mnt/e/project/Iris/outputs/iris_rgb_grounding_audit")
DEFAULT_MODEL = "jingheya/lotus-depth-g-v2-1-disparity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ms2-root", type=Path, default=DEFAULT_MS2_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--processing-res", type=int, default=768)
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--num-inference-steps", type=int, default=1)
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-attention", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def caption_hash(caption: str) -> str:
    return hashlib.sha256(caption.encode("utf-8")).hexdigest()[:16]


def swap_relations(text: str) -> Tuple[str, List[str]]:
    pairs = [
        ("foreground", "background"),
        ("left", "right"),
        ("near", "far"),
        ("nearer", "farther"),
        ("nearest", "farthest"),
        ("closest", "farthest"),
        ("in front of", "behind"),
    ]
    result = text
    applied: List[str] = []
    for pair_index, (left, right) in enumerate(pairs):
        left_pattern = re.compile(rf"\b{re.escape(left)}\b", flags=re.IGNORECASE)
        right_pattern = re.compile(rf"\b{re.escape(right)}\b", flags=re.IGNORECASE)
        if not left_pattern.search(result) and not right_pattern.search(result):
            continue
        left_marker = f"__REL_LEFT_{pair_index}__"
        right_marker = f"__REL_RIGHT_{pair_index}__"
        result = left_pattern.sub(left_marker, result)
        result = right_pattern.sub(right_marker, result)
        result = result.replace(left_marker, right).replace(right_marker, left)
        applied.append(f"{left}<->{right}")
    if not applied:
        result = (
            text.rstrip(" .")
            + ". There are no vehicles, pedestrians, buildings, road, or trees in this scene."
        )
        applied.append("object_presence->explicit_absence")
    return result, applied


def select_samples(manifest: Path, root: Path, count: int) -> List[Dict[str, Any]]:
    if not 16 <= count <= 32:
        raise ValueError("--num-samples must be in [16, 32]")
    readable: List[Dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            rgb_value = row.get("rgb_path")
            depth_value = row.get("rgb_depth_path") or row.get("depth_path")
            caption = str(row.get("caption") or "").strip()
            if not rgb_value or not depth_value or not caption:
                continue
            rgb_path = resolve(root, str(rgb_value))
            depth_path = resolve(root, str(depth_value))
            if not rgb_path.is_file() or not depth_path.is_file():
                continue
            item = dict(row)
            item["manifest_line"] = line_number
            item["rgb_resolved"] = str(rgb_path)
            item["depth_resolved"] = str(depth_path)
            readable.append(item)
    if len(readable) < count:
        raise RuntimeError(f"Only {len(readable)} readable RGB/depth/caption samples found")

    # Uniform temporal coverage, then repair any rare duplicate-caption choice.
    positions = np.linspace(0, len(readable) - 1, count).round().astype(int).tolist()
    selected: List[Dict[str, Any]] = []
    seen_captions: set[str] = set()
    for position in positions:
        candidate_position = position
        while (
            readable[candidate_position]["caption"] in seen_captions
            and candidate_position + 1 < len(readable)
        ):
            candidate_position += 1
        item = readable[candidate_position]
        seen_captions.add(item["caption"])
        selected.append(item)

    for ordinal, item in enumerate(selected):
        wrong, operations = swap_relations(str(item["caption"]))
        item["selection_ordinal"] = ordinal
        item["selection_index_in_readable_manifest"] = positions[ordinal]
        item["captions"] = {
            "correct": str(item["caption"]),
            "empty": "",
            "generic": GENERIC_CAPTION,
            "hard_wrong": wrong,
        }
        item["hard_wrong_operations"] = operations
        item["caption_sha256_16"] = caption_hash(str(item["caption"]))
    return selected


def component_checkpoint_shapes(path: Path) -> Dict[str, Tuple[int, ...]]:
    result: Dict[str, Tuple[int, ...]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            result[key] = tuple(int(value) for value in handle.get_slice(key).get_shape())
    return result


def audit_component(module: torch.nn.Module, weight_path: Path) -> Dict[str, Any]:
    model_shapes = {key: tuple(value.shape) for key, value in module.state_dict().items()}
    checkpoint_shapes = component_checkpoint_shapes(weight_path)
    model_keys = set(model_shapes)
    checkpoint_keys = set(checkpoint_shapes)
    shape_mismatches = [
        {
            "parameter": key,
            "model_shape": list(model_shapes[key]),
            "checkpoint_shape": list(checkpoint_shapes[key]),
        }
        for key in sorted(model_keys & checkpoint_keys)
        if model_shapes[key] != checkpoint_shapes[key]
    ]
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    return {
        "weight_file": str(weight_path),
        "weight_file_size_bytes": weight_path.stat().st_size,
        "model_parameter_and_buffer_key_count": len(model_shapes),
        "checkpoint_key_count": len(checkpoint_shapes),
        "missing_parameters": missing,
        "unexpected_parameters": unexpected,
        "shape_mismatch_parameters": shape_mismatches,
        "strict_shape_and_key_compatible": not missing and not unexpected and not shape_mismatches,
    }


class CrossAttentionInvocationAudit:
    def __init__(self) -> None:
        self.records: Dict[str, Dict[str, Any]] = {}
        self.handles: List[Any] = []

    def attach(self, unet: torch.nn.Module) -> None:
        for name, module in unet.named_modules():
            if not name.endswith("attn2"):
                continue

            def hook(_module: torch.nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any], *, layer=name) -> None:
                encoder = kwargs.get("encoder_hidden_states")
                hidden = args[0] if args and isinstance(args[0], torch.Tensor) else kwargs.get("hidden_states")
                record = self.records.setdefault(
                    layer,
                    {
                        "call_count": 0,
                        "hidden_state_shapes": [],
                        "encoder_hidden_state_shapes": [],
                    },
                )
                record["call_count"] += 1
                if isinstance(hidden, torch.Tensor):
                    shape = list(hidden.shape)
                    if shape not in record["hidden_state_shapes"]:
                        record["hidden_state_shapes"].append(shape)
                if isinstance(encoder, torch.Tensor):
                    shape = list(encoder.shape)
                    if shape not in record["encoder_hidden_state_shapes"]:
                        record["encoder_hidden_state_shapes"].append(shape)

            self.handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class AttentionStore:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def clear(self) -> None:
        self.records.clear()


class RecordingCrossAttnProcessor:
    """Diffusers 0.28 AttnProcessor with cross-attention probabilities retained."""

    def __init__(self, layer_name: str, store: AttentionStore) -> None:
        self.layer_name = layer_name
        self.store = store

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if is_cross:
            averaged = attention_probs.view(
                batch_size, attn.heads, attention_probs.shape[-2], attention_probs.shape[-1]
            ).mean(dim=1)
            self.store.records.append(
                {
                    "layer": self.layer_name,
                    "attention": averaged[0].detach().to(device="cpu", dtype=torch.float16),
                }
            )
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def install_attention_capture(unet: Any, store: AttentionStore) -> Dict[str, Any]:
    original = dict(unet.attn_processors)
    replacements: Dict[str, Any] = {}
    for name, processor in original.items():
        if name.endswith("attn2.processor"):
            replacements[name] = RecordingCrossAttnProcessor(name, store)
        else:
            replacements[name] = processor
    unet.set_attn_processor(dict(replacements))
    return original


def infer_spatial_shape(query_length: int, latent_hw: Tuple[int, int]) -> Tuple[int, int]:
    latent_h, latent_w = latent_hw
    for factor in (1, 2, 4, 8, 16):
        height = math.ceil(latent_h / factor)
        width = math.ceil(latent_w / factor)
        if height * width == query_length:
            return height, width
    best: Tuple[float, int, int] | None = None
    target_ratio = latent_w / max(latent_h, 1)
    for height in range(1, int(math.sqrt(query_length)) + 1):
        if query_length % height:
            continue
        width = query_length // height
        score = abs((width / height) - target_ratio)
        if best is None or score < best[0]:
            best = (score, height, width)
    if best is None:
        raise RuntimeError(f"Cannot factor attention query length {query_length}")
    return best[1], best[2]


def normalize_heatmap(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    lo = float(np.nanmin(array))
    hi = float(np.nanmax(array))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - lo) / (hi - lo), 0.0, 1.0)


def spectral_rgb(array01: np.ndarray) -> np.ndarray:
    import matplotlib

    return (matplotlib.colormaps["Spectral_r"](np.clip(array01, 0, 1))[..., :3] * 255).astype(np.uint8)


def heat_rgb(array01: np.ndarray) -> np.ndarray:
    import matplotlib

    return (matplotlib.colormaps["inferno"](np.clip(array01, 0, 1))[..., :3] * 255).astype(np.uint8)


def save_array_visual(array: np.ndarray, path: Path, *, kind: str = "spectral") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    norm = normalize_heatmap(array)
    rgb = spectral_rgb(norm) if kind == "spectral" else heat_rgb(norm)
    Image.fromarray(rgb).save(path)


def content_token_indices(tokenizer: Any, token_ids: Sequence[int]) -> List[int]:
    special = {
        value
        for value in (
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            tokenizer.unk_token_id,
        )
        if value is not None
    }
    return [index for index, token_id in enumerate(token_ids) if token_id not in special]


def aggregate_attention(
    records: Sequence[Mapping[str, Any]],
    token_ids: Sequence[int],
    tokenizer: Any,
    original_hw: Tuple[int, int],
    processed_hw: Tuple[int, int],
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, float]]:
    original_h, original_w = original_hw
    latent_hw = (math.ceil(processed_hw[0] / 8), math.ceil(processed_hw[1] / 8))
    per_token: MutableMapping[int, List[np.ndarray]] = defaultdict(list)
    layer_shapes: List[Dict[str, Any]] = []
    for record in records:
        attention = record["attention"].float()
        query_length, token_count = attention.shape
        height, width = infer_spatial_shape(query_length, latent_hw)
        layer_shapes.append(
            {
                "layer": str(record["layer"]),
                "attention_shape": [query_length, token_count],
                "inferred_spatial_shape": [height, width],
            }
        )
        usable_tokens = min(token_count, len(token_ids))
        maps = attention[:, :usable_tokens].transpose(0, 1).reshape(usable_tokens, 1, height, width)
        resized = F.interpolate(
            maps,
            size=(original_h, original_w),
            mode="bilinear",
            align_corners=False,
        )[:, 0].numpy()
        for token_index in range(usable_tokens):
            per_token[token_index].append(resized[token_index])
    averaged = {
        index: np.mean(maps, axis=0).astype(np.float32)
        for index, maps in per_token.items()
        if maps
    }
    content_indices = [index for index in content_token_indices(tokenizer, token_ids) if index in averaged]
    if content_indices:
        aggregate = np.sum([averaged[index] for index in content_indices], axis=0).astype(np.float32)
    else:
        aggregate = np.zeros((original_h, original_w), dtype=np.float32)

    tokens = tokenizer.convert_ids_to_tokens(list(token_ids))
    ranked_indices = sorted(
        content_indices,
        key=lambda index: float(np.std(normalize_heatmap(averaged[index]))),
        reverse=True,
    )[:8]
    token_maps = [
        {
            "index": index,
            "token_id": int(token_ids[index]),
            "token": str(tokens[index]),
            "map": averaged[index],
            "spatial_std": float(np.std(normalize_heatmap(averaged[index]))),
        }
        for index in ranked_indices
    ]

    positive = aggregate - float(aggregate.min())
    mass = positive / max(float(positive.sum()), 1e-12)
    flat = np.sort(mass.reshape(-1))[::-1]
    top_count = max(1, int(round(0.10 * flat.size)))
    entropy = -float(np.sum(mass * np.log(np.maximum(mass, 1e-12))))
    entropy_normalized = entropy / max(math.log(max(mass.size, 2)), 1e-12)
    yy, xx = np.mgrid[0:original_h, 0:original_w]
    stats = {
        "normalized_entropy": float(entropy_normalized),
        "top_10_percent_mass": float(flat[:top_count].sum()),
        "center_of_mass_x_fraction": float((mass * xx).sum() / max(original_w - 1, 1)),
        "center_of_mass_y_fraction": float((mass * yy).sum() / max(original_h - 1, 1)),
        "captured_cross_attention_call_count": len(records),
        "content_token_count": len(content_indices),
        "layer_shapes": layer_shapes,
    }
    return aggregate, token_maps, stats


def load_rgb(path: Path) -> Tuple[torch.Tensor, np.ndarray]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float()
    return tensor / 127.5 - 1.0, rgb


def load_depth(path: Path, depth_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        raw = np.asarray(image)
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError(f"Expected 2-D uint16 depth PNG, got {raw.shape} {raw.dtype} at {path}")
    depth_m = raw.astype(np.float32) / float(depth_scale)
    valid = np.isfinite(depth_m) & (depth_m > 0.1)
    return depth_m, valid


def processed_shape(original_hw: Tuple[int, int], max_edge: int) -> Tuple[int, int]:
    height, width = original_hw
    factor = min(max_edge / width, max_edge / height)
    return int(height * factor), int(width * factor)


def task_embedding(device: torch.device) -> torch.Tensor:
    task = torch.tensor([1.0, 0.0], device=device).unsqueeze(0)
    return torch.cat([torch.sin(task), torch.cos(task)], dim=-1).repeat(1, 1)


def run_prediction(
    pipeline: LotusGPipeline,
    rgb_tensor: torch.Tensor,
    prompt: str,
    *,
    seed: int,
    timestep: int,
    num_inference_steps: int,
    processing_res: int,
    dtype: torch.dtype,
    device: torch.device,
) -> np.ndarray:
    seed_everything(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    autocast_enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=dtype if autocast_enabled else torch.float32,
        enabled=autocast_enabled,
    ):
        result = pipeline(
            rgb_in=rgb_tensor.to(device),
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="np",
            timesteps=[timestep],
            task_emb=task_embedding(device),
            processing_res=processing_res,
            match_input_res=True,
            resample_method="bilinear",
        ).images[0]
    return np.asarray(result, dtype=np.float32).mean(axis=-1)


def minmax_valid(array: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.zeros_like(array, dtype=np.float32)
    values = array[valid].astype(np.float64)
    if values.size:
        lo, hi = float(values.min()), float(values.max())
        if hi > lo:
            output = np.clip((array - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return output


def align_disparity_to_metric_depth(
    pred_disparity: np.ndarray,
    gt_depth_m: np.ndarray,
    valid: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    gt_disparity = np.zeros_like(gt_depth_m, dtype=np.float64)
    gt_disparity[valid] = 1.0 / gt_depth_m[valid]
    fit_mask = valid & np.isfinite(pred_disparity) & (pred_disparity > 0)
    x = pred_disparity[fit_mask].astype(np.float64)
    y = gt_disparity[fit_mask].astype(np.float64)
    if x.size < 2 or float(np.var(x)) <= 0:
        scale, shift = 1.0, 0.0
    else:
        design = np.stack([x, np.ones_like(x)], axis=1)
        scale, shift = np.linalg.lstsq(design, y, rcond=None)[0]
        scale, shift = float(scale), float(shift)
    aligned_disparity = np.clip(pred_disparity.astype(np.float64) * scale + shift, 1e-3, None)
    depth = (1.0 / aligned_disparity).astype(np.float32)
    if valid.any():
        depth = np.clip(depth, float(gt_depth_m[valid].min()), float(gt_depth_m[valid].max()))
    return depth, scale, shift


def metrics(
    pred_disparity: np.ndarray,
    aligned_depth_m: np.ndarray,
    gt_depth_m: np.ndarray,
    valid: np.ndarray,
) -> Dict[str, float]:
    gt_disparity = np.zeros_like(gt_depth_m, dtype=np.float32)
    gt_disparity[valid] = 1.0 / gt_depth_m[valid]
    pred_norm = minmax_valid(pred_disparity, valid)
    gt_norm = minmax_valid(gt_disparity, valid)
    p = aligned_depth_m[valid].astype(np.float64)
    g = gt_depth_m[valid].astype(np.float64)
    if p.size == 0:
        return {"norm_mse": float("nan"), "absrel": float("nan"), "rmse": float("nan"), "delta1": float("nan")}
    norm_mse = float(np.mean((pred_norm[valid] - gt_norm[valid]) ** 2))
    absrel = float(np.mean(np.abs(p - g) / np.maximum(g, 1e-6)))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    ratio = np.maximum(p / np.maximum(g, 1e-6), g / np.maximum(p, 1e-6))
    delta1 = float(np.mean(ratio < 1.25))
    return {"norm_mse": norm_mse, "absrel": absrel, "rmse": rmse, "delta1": delta1}


def pairwise_response(reference: np.ndarray, other: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    ref = reference[valid].astype(np.float64)
    alt = other[valid].astype(np.float64)
    diff = ref - alt
    mse = float(np.mean(diff**2)) if diff.size else 0.0
    mean_shift = float(np.mean(diff)) if diff.size else 0.0
    residual = diff - mean_shift
    residual_mse = float(np.mean(residual**2)) if residual.size else 0.0
    ref_std = float(np.std(ref)) if ref.size else 0.0

    if alt.size >= 2 and float(np.var(alt)) > 0:
        design = np.stack([alt, np.ones_like(alt)], axis=1)
        scale, shift = np.linalg.lstsq(design, ref, rcond=None)[0]
        affine_residual_mse = float(np.mean((ref - (alt * scale + shift)) ** 2))
    else:
        scale, shift, affine_residual_mse = 1.0, 0.0, mse
    return {
        "mean_signed_shift": mean_shift,
        "difference_rmse": math.sqrt(max(mse, 0.0)),
        "difference_mae": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "relative_difference_rmse_to_reference_std": math.sqrt(max(mse, 0.0)) / max(ref_std, 1e-12),
        "global_shift_energy_fraction": (mean_shift**2) / max(mse, 1e-24),
        "spatial_variation_energy_fraction": residual_mse / max(mse, 1e-24),
        "affine_fit_scale_other_to_reference": float(scale),
        "affine_fit_shift_other_to_reference": float(shift),
        "affine_explained_difference_fraction": 1.0 - affine_residual_mse / max(mse, 1e-24),
    }


def metric_winner(correct: float, comparison: float, metric_name: str, tolerance: float = 1e-12) -> bool:
    if metric_name == "delta1":
        return correct > comparison + tolerance
    return correct < comparison - tolerance


def average_dicts(items: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    return {
        key: float(np.mean([item[key] for item in items if key in item and np.isfinite(item[key])]))
        for key in keys
    }


def labelled_panel(images: Sequence[Tuple[str, Image.Image]], path: Path, tile_size: Tuple[int, int] = (360, 120)) -> None:
    if not images:
        return
    tile_w, tile_h = tile_size
    label_h = 22
    panel = Image.new("RGB", (tile_w * len(images), tile_h + label_h), "white")
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    for index, (label, image) in enumerate(images):
        x = index * tile_w
        panel.paste(image.convert("RGB").resize((tile_w, tile_h)), (x, label_h))
        draw.text((x + 5, 5), label, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(path)


def save_attention_outputs(
    output_dir: Path,
    sample_id: str,
    rgb: np.ndarray,
    aggregate: np.ndarray,
    token_maps: Sequence[Mapping[str, Any]],
    stats: Mapping[str, Any],
) -> None:
    sample_dir = output_dir / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    np.save(sample_dir / "correct_caption_aggregate.npy", aggregate.astype(np.float32))
    heat = heat_rgb(normalize_heatmap(aggregate))
    Image.fromarray(heat).save(sample_dir / "correct_caption_aggregate.png")
    overlay = (0.55 * rgb.astype(np.float32) + 0.45 * heat.astype(np.float32)).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(sample_dir / "correct_caption_overlay.png")
    token_metadata = []
    for rank, item in enumerate(token_maps):
        token = str(item["token"])
        safe_token = re.sub(r"[^A-Za-z0-9_-]+", "_", token).strip("_") or "token"
        stem = f"{rank:02d}_idx{item['index']:02d}_{safe_token[:30]}"
        token_map = np.asarray(item["map"], dtype=np.float32)
        np.save(sample_dir / f"{stem}.npy", token_map)
        Image.fromarray(heat_rgb(normalize_heatmap(token_map))).save(sample_dir / f"{stem}.png")
        token_metadata.append({key: value for key, value in item.items() if key != "map"})
    write_json(sample_dir / "attention_stats.json", {**dict(stats), "ranked_token_maps": token_metadata})


def build_load_report(
    pipeline: LotusGPipeline,
    model_path: str,
    snapshot_path: Path,
    load_success: bool,
    load_error: str | None,
) -> Dict[str, Any]:
    components: Dict[str, Any] = {}
    if load_success:
        components["unet"] = audit_component(
            pipeline.unet, snapshot_path / "unet" / "diffusion_pytorch_model.safetensors"
        )
        components["vae"] = audit_component(
            pipeline.vae, snapshot_path / "vae" / "diffusion_pytorch_model.safetensors"
        )
        components["text_encoder"] = audit_component(
            pipeline.text_encoder, snapshot_path / "text_encoder" / "model.safetensors"
        )
    all_missing = [f"{name}.{key}" for name, info in components.items() for key in info["missing_parameters"]]
    all_unexpected = [f"{name}.{key}" for name, info in components.items() for key in info["unexpected_parameters"]]
    all_mismatches = [
        {"component": name, **entry}
        for name, info in components.items()
        for entry in info["shape_mismatch_parameters"]
    ]
    return {
        "requested_model": model_path,
        "resolved_snapshot": str(snapshot_path),
        "snapshot_revision": snapshot_path.name,
        "local_files_only": True,
        "pipeline_class": type(pipeline).__name__ if load_success else None,
        "pipeline_load_success": load_success,
        "pipeline_load_error": load_error,
        "components": components,
        "missing_parameters": all_missing,
        "unexpected_parameters": all_unexpected,
        "shape_mismatch_parameters": all_mismatches,
        "strict_shape_and_key_compatible": load_success and not all_missing and not all_unexpected and not all_mismatches,
    }


def comparison_summary(per_sample: Sequence[Mapping[str, Any]], comparison: str) -> Dict[str, Any]:
    metric_names = ("norm_mse", "absrel", "rmse", "delta1")
    win_rates: Dict[str, float] = {}
    mean_advantages: Dict[str, float] = {}
    for metric_name in metric_names:
        wins = [
            metric_winner(
                float(row[f"correct_{metric_name}"]),
                float(row[f"{comparison}_{metric_name}"]),
                metric_name,
            )
            for row in per_sample
        ]
        win_rates[metric_name] = float(np.mean(wins))
        if metric_name == "delta1":
            advantages = [
                float(row[f"correct_{metric_name}"]) - float(row[f"{comparison}_{metric_name}"])
                for row in per_sample
            ]
        else:
            advantages = [
                float(row[f"{comparison}_{metric_name}"]) - float(row[f"correct_{metric_name}"])
                for row in per_sample
            ]
        mean_advantages[metric_name] = float(np.mean(advantages))
    per_sample_majority = []
    for row in per_sample:
        win_count = sum(
            metric_winner(
                float(row[f"correct_{metric_name}"]),
                float(row[f"{comparison}_{metric_name}"]),
                metric_name,
            )
            for metric_name in metric_names
        )
        per_sample_majority.append(win_count >= 3)
    return {
        "correct_metric_win_rates": win_rates,
        "correct_mean_metric_advantage": mean_advantages,
        "correct_wins_at_least_3_of_4_metrics_sample_ratio": float(np.mean(per_sample_majority)),
    }


def write_report(
    path: Path,
    config: Mapping[str, Any],
    load_report: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    verdict = summary["verdict"]
    modes = summary["mode_metrics_mean"]
    comparisons = summary["comparisons"]
    response = summary["prediction_response_correct_vs_hard_wrong"]
    attention = summary.get("attention_summary", {})
    lines = [
        "# Iris/Lotus RGB caption grounding audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        "This is a forward-only audit of the original RGB Lotus-G pipeline. It uses RGB images, the released VAE, "
        "the released CLIP text encoder, and the released Lotus-G checkpoint. It does not use AnyThermal, thermal "
        "images, SP-DiT, adapters, or training.",
        "",
        "## Checkpoint and text path",
        "",
        f"- Model: `{load_report['requested_model']}`",
        f"- Snapshot: `{load_report['snapshot_revision']}`",
        f"- Strict checkpoint key/shape compatible: `{load_report['strict_shape_and_key_compatible']}`",
        f"- Missing parameters: `{len(load_report['missing_parameters'])}`",
        f"- Unexpected parameters: `{len(load_report['unexpected_parameters'])}`",
        f"- Shape mismatches: `{len(load_report['shape_mismatch_parameters'])}`",
        f"- Text encoder frozen for audit inference: `{load_report['text_path_audit']['text_encoder_frozen_during_audit']}`",
        f"- Text representation shape: `{load_report['text_path_audit']['observed_prompt_embedding_shapes']}`",
        f"- Cross-attention layers actually invoked with `encoder_hidden_states`: "
        f"`{load_report['text_path_audit']['cross_attention_layer_count_with_text']}`",
        "",
        "The CLIP output is a sequence of per-token hidden states, not a pooled sentence vector, and the observed "
        "tensor is passed to U-Net `attn2` modules as `encoder_hidden_states`.",
        "",
        "## Protocol",
        "",
        f"- Fixed RGB samples: {config['num_samples']}",
        f"- Seed: {config['seed']} (reset before every caption condition for each sample)",
        f"- Scheduler: `{config['scheduler_class']}`",
        f"- Inference steps: {config['num_inference_steps']}; timestep: {config['timestep']}",
        f"- Processing resolution: {config['processing_res']}; normalization: `{config['rgb_normalization']}`",
        "- Conditions: correct, empty, generic (`A driving scene.`), hard_wrong (deterministic spatial-relation swaps).",
        "- Norm-MSE compares per-image min-max-normalized predicted disparity with normalized GT disparity.",
        "- AbsRel/RMSE/delta1 use the official Lotus disparity-space least-squares alignment followed by conversion "
        "to metric depth. MS2 uint16 depth is divided by 256; RMSE is in metres.",
        "",
        "## Mean metrics",
        "",
        "| mode | Norm-MSE ↓ | AbsRel ↓ | RMSE m ↓ | δ1 ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in TEXT_MODES:
        metric = modes[mode]
        lines.append(
            f"| {mode} | {metric['norm_mse']:.8f} | {metric['absrel']:.8f} | "
            f"{metric['rmse']:.8f} | {metric['delta1']:.8f} |"
        )
    lines.extend(["", "## Caption comparisons", ""])
    for name in ("hard_wrong", "empty", "generic"):
        item = comparisons[f"correct_vs_{name}"]
        rates = item["correct_metric_win_rates"]
        lines.append(
            f"- correct vs {name}: win rates Norm-MSE={rates['norm_mse']:.3f}, "
            f"AbsRel={rates['absrel']:.3f}, RMSE={rates['rmse']:.3f}, δ1={rates['delta1']:.3f}; "
            f"3-of-4 sample win ratio={item['correct_wins_at_least_3_of_4_metrics_sample_ratio']:.3f}."
        )
    lines.extend(
        [
            "",
            "## Spatial response",
            "",
            f"- Correct-vs-hard_wrong disparity difference RMSE: "
            f"{response.get('difference_rmse', float('nan')):.8e}",
            f"- Relative difference RMSE / prediction std: "
            f"{response.get('relative_difference_rmse_to_reference_std', float('nan')):.8e}",
            f"- Global-shift energy fraction: {response.get('global_shift_energy_fraction', float('nan')):.4f}",
            f"- Spatial-variation energy fraction: {response.get('spatial_variation_energy_fraction', float('nan')):.4f}",
            f"- Affine-explained difference fraction: "
            f"{response.get('affine_explained_difference_fraction', float('nan')):.4f}",
            "",
        ]
    )
    if attention:
        lines.extend(
            [
                "## Cross-attention maps",
                "",
                f"- Samples captured: {attention.get('sample_count', 0)}",
                f"- Mean normalized spatial entropy: {attention.get('normalized_entropy', float('nan')):.4f}",
                f"- Mean mass in top 10% pixels: {attention.get('top_10_percent_mass', float('nan')):.4f}",
                "- Maps and RGB overlays are saved under `attention_maps/`. Token maps are diagnostic; because this "
                "audit has no object masks, object-level localization is judged conservatively from overlays rather "
                "than claimed from entropy alone.",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision rule",
            "",
            summary["decision_rule_text"],
            "",
            "## Final acceptance conclusion",
            "",
            f"**{verdict['label']}**",
            "",
            verdict["text"],
            "",
            "Decision evidence: " + "; ".join(verdict["reasons"]),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in (
        "predictions",
        "difference_maps",
        "attention_maps",
        "visualizations",
    ):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    samples = select_samples(args.manifest.resolve(), args.ms2_root.resolve(), args.num_samples)

    config: Dict[str, Any] = {
        "audit_name": "iris_rgb_grounding_audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "scope": {
            "rgb_only": True,
            "uses_anythermal": False,
            "uses_thermal_images": False,
            "uses_sp_dit": False,
            "trains_model": False,
        },
        "manifest": str(args.manifest.resolve()),
        "ms2_root": str(args.ms2_root.resolve()),
        "model_path": args.model_path,
        "model_mode": "Lotus-G disparity",
        "num_samples": args.num_samples,
        "seed": args.seed,
        "seed_policy": "reset identical global CUDA/CPU RNG and per-call torch.Generator before every caption condition",
        "processing_res": args.processing_res,
        "timestep": args.timestep,
        "num_inference_steps": args.num_inference_steps,
        "dtype": args.dtype,
        "device": str(device),
        "rgb_normalization": "uint8 [0,255] -> float32 [-1,1] via x/127.5-1; original Lotus resize_max_res bilinear antialias",
        "depth_scale_uint16_per_metre": args.depth_scale,
        "text_modes": list(TEXT_MODES),
        "generic_caption": GENERIC_CAPTION,
        "attention_capture_requested": not args.no_attention,
        "sample_selection": "16 uniformly spaced readable val-manifest records, with duplicate-caption forward repair",
        "samples": [
            {
                "id": item["id"],
                "manifest_line": item["manifest_line"],
                "selection_index_in_readable_manifest": item["selection_index_in_readable_manifest"],
                "rgb_path": item["rgb_resolved"],
                "depth_path": item["depth_resolved"],
                "captions": item["captions"],
                "hard_wrong_operations": item["hard_wrong_operations"],
                "caption_sha256_16": item["caption_sha256_16"],
            }
            for item in samples
        ],
    }
    write_json(output_dir / "config.json", config)

    load_report: Dict[str, Any]
    pipeline: LotusGPipeline
    try:
        from huggingface_hub import snapshot_download

        snapshot_path = Path(snapshot_download(args.model_path, local_files_only=True)).resolve()
        pipeline = LotusGPipeline.from_pretrained(
            str(snapshot_path),
            torch_dtype=dtype,
            local_files_only=True,
        )
        load_report = build_load_report(pipeline, args.model_path, snapshot_path, True, None)
    except Exception as exc:
        load_report = {
            "requested_model": args.model_path,
            "pipeline_load_success": False,
            "pipeline_load_error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "missing_parameters": [],
            "unexpected_parameters": [],
            "shape_mismatch_parameters": [],
        }
        write_json(output_dir / "checkpoint_load_report.json", load_report)
        raise

    text_encoder_requires_grad_before = any(
        parameter.requires_grad for parameter in pipeline.text_encoder.parameters()
    )
    pipeline.vae.eval()
    pipeline.text_encoder.eval()
    pipeline.unet.eval()
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)
    pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    config["scheduler_class"] = type(pipeline.scheduler).__name__
    config["scheduler_config"] = dict(pipeline.scheduler.config)
    write_json(output_dir / "config.json", config)

    invocation_audit = CrossAttentionInvocationAudit()
    invocation_audit.attach(pipeline.unet)
    observed_prompt_shapes: set[Tuple[int, ...]] = set()
    per_sample_rows: List[Dict[str, Any]] = []
    metrics_by_mode: Dict[str, List[Dict[str, float]]] = {mode: [] for mode in TEXT_MODES}
    response_records: Dict[str, List[Dict[str, float]]] = {
        "empty": [],
        "generic": [],
        "hard_wrong": [],
    }
    attention_stats: List[Dict[str, float]] = []
    overview_panels: List[Tuple[str, Image.Image]] = []

    for sample_index, sample in enumerate(samples):
        sample_id = str(sample["id"])
        rgb_path = Path(sample["rgb_resolved"])
        depth_path = Path(sample["depth_resolved"])
        rgb_tensor, rgb = load_rgb(rgb_path)
        gt_depth_m, valid = load_depth(depth_path, args.depth_scale)
        sample_seed = args.seed + sample_index

        prediction_dir = output_dir / "predictions" / sample_id
        prediction_dir.mkdir(parents=True, exist_ok=True)
        np.save(prediction_dir / "gt_depth_m.npy", gt_depth_m.astype(np.float32))
        Image.fromarray(rgb).save(prediction_dir / "rgb.png")
        Image.fromarray((valid.astype(np.uint8) * 255)).save(prediction_dir / "valid_mask.png")

        predictions: Dict[str, np.ndarray] = {}
        aligned_depths: Dict[str, np.ndarray] = {}
        sample_metrics: Dict[str, Dict[str, float]] = {}
        alignments: Dict[str, Dict[str, float]] = {}
        for mode in TEXT_MODES:
            prompt = sample["captions"][mode]
            with torch.no_grad():
                embeds, _ = pipeline.encode_prompt(
                    prompt,
                    device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
            observed_prompt_shapes.add(tuple(int(value) for value in embeds.shape))
            pred = run_prediction(
                pipeline,
                rgb_tensor,
                prompt,
                seed=sample_seed,
                timestep=args.timestep,
                num_inference_steps=args.num_inference_steps,
                processing_res=args.processing_res,
                dtype=dtype,
                device=device,
            )
            aligned_depth, scale, shift = align_disparity_to_metric_depth(pred, gt_depth_m, valid)
            metric = metrics(pred, aligned_depth, gt_depth_m, valid)
            predictions[mode] = pred
            aligned_depths[mode] = aligned_depth
            sample_metrics[mode] = metric
            metrics_by_mode[mode].append(metric)
            alignments[mode] = {"disparity_scale": scale, "disparity_shift": shift}
            np.save(prediction_dir / f"{mode}_disparity.npy", pred.astype(np.float32))
            np.save(prediction_dir / f"{mode}_aligned_depth_m.npy", aligned_depth.astype(np.float32))
            save_array_visual(pred, prediction_dir / f"{mode}_disparity.png")
            save_array_visual(aligned_depth, prediction_dir / f"{mode}_aligned_depth_m.png")

        invocation_audit.close()
        for comparison in ("empty", "generic", "hard_wrong"):
            response_records[comparison].append(
                pairwise_response(predictions["correct"], predictions[comparison], valid)
            )

        difference = predictions["correct"] - predictions["hard_wrong"]
        difference_dir = output_dir / "difference_maps" / sample_id
        difference_dir.mkdir(parents=True, exist_ok=True)
        np.save(difference_dir / "correct_minus_hard_wrong.npy", difference.astype(np.float32))
        np.save(difference_dir / "abs_correct_minus_hard_wrong.npy", np.abs(difference).astype(np.float32))
        save_array_visual(
            np.abs(difference), difference_dir / "abs_correct_minus_hard_wrong.png", kind="heat"
        )
        signed_scale = float(np.percentile(np.abs(difference[valid]), 99)) if valid.any() else 1.0
        signed_norm = np.clip(difference / max(signed_scale, 1e-12) * 0.5 + 0.5, 0, 1)
        Image.fromarray(spectral_rgb(signed_norm)).save(difference_dir / "signed_correct_minus_hard_wrong.png")

        row: Dict[str, Any] = {
            "sample_index": sample_index,
            "sample_id": sample_id,
            "manifest_line": sample["manifest_line"],
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "sample_seed": sample_seed,
            "valid_fraction": float(valid.mean()),
            "correct_caption": sample["captions"]["correct"],
            "hard_wrong_caption": sample["captions"]["hard_wrong"],
            "hard_wrong_operations": "|".join(sample["hard_wrong_operations"]),
        }
        for mode in TEXT_MODES:
            for metric_name, value in sample_metrics[mode].items():
                row[f"{mode}_{metric_name}"] = value
            row[f"{mode}_disparity_alignment_scale"] = alignments[mode]["disparity_scale"]
            row[f"{mode}_disparity_alignment_shift"] = alignments[mode]["disparity_shift"]
        for comparison in ("empty", "generic", "hard_wrong"):
            response = response_records[comparison][-1]
            for name, value in response.items():
                row[f"correct_vs_{comparison}_{name}"] = value
        for metric_name in ("norm_mse", "absrel", "rmse", "delta1"):
            row[f"correct_beats_hard_wrong_{metric_name}"] = metric_winner(
                sample_metrics["correct"][metric_name],
                sample_metrics["hard_wrong"][metric_name],
                metric_name,
            )
        per_sample_rows.append(row)

        visual_items: List[Tuple[str, Image.Image]] = [
            ("RGB", Image.fromarray(rgb)),
            ("GT depth", Image.fromarray(spectral_rgb(normalize_heatmap(gt_depth_m)))),
        ]
        for mode in TEXT_MODES:
            visual_items.append(
                (mode, Image.fromarray(spectral_rgb(normalize_heatmap(aligned_depths[mode]))))
            )
        visual_items.append(("|correct-wrong|", Image.fromarray(heat_rgb(normalize_heatmap(np.abs(difference))))))
        panel_path = output_dir / "visualizations" / f"{sample_index:02d}_{sample_id}.png"
        labelled_panel(visual_items, panel_path, tile_size=(245, 77))
        overview_panels.append((f"{sample_index:02d}", Image.open(panel_path).copy()))
        print(f"[{sample_index + 1}/{len(samples)}] predictions complete: {sample_id}", flush=True)

    load_report["text_path_audit"] = {
        "text_encoder_class": type(pipeline.text_encoder).__name__,
        "tokenizer_class": type(pipeline.tokenizer).__name__,
        "text_encoder_hidden_size": int(pipeline.text_encoder.config.hidden_size),
        "tokenizer_model_max_length": int(pipeline.tokenizer.model_max_length),
        "text_encoder_requires_grad_any_immediately_after_from_pretrained": text_encoder_requires_grad_before,
        "text_encoder_frozen_during_audit": not any(
            parameter.requires_grad for parameter in pipeline.text_encoder.parameters()
        ),
        "original_training_code_freezes_text_encoder": True,
        "uses_per_token_hidden_states": True,
        "observed_prompt_embedding_shapes": [list(shape) for shape in sorted(observed_prompt_shapes)],
        "caption_enters_diffusion_cross_attention": bool(invocation_audit.records),
        "cross_attention_layer_count_with_text": len(invocation_audit.records),
        "cross_attention_invocations": invocation_audit.records,
        "static_data_flow": "CLIPTextModel(...)[0] -> prompt_embeds -> UNet encoder_hidden_states -> attn2",
    }
    write_json(output_dir / "checkpoint_load_report.json", load_report)

    # Attention extraction is a separate diagnostic pass. Main predictions above use
    # the checkpoint's original attention processors.
    if not args.no_attention:
        store = AttentionStore()
        original_processors = install_attention_capture(pipeline.unet, store)
        try:
            for sample_index, sample in enumerate(samples):
                sample_id = str(sample["id"])
                rgb_tensor, rgb = load_rgb(Path(sample["rgb_resolved"]))
                original_hw = (rgb.shape[0], rgb.shape[1])
                processed_hw = processed_shape(original_hw, args.processing_res)
                store.clear()
                _ = run_prediction(
                    pipeline,
                    rgb_tensor,
                    sample["captions"]["correct"],
                    seed=args.seed + sample_index,
                    timestep=args.timestep,
                    num_inference_steps=args.num_inference_steps,
                    processing_res=args.processing_res,
                    dtype=dtype,
                    device=device,
                )
                tokenized = pipeline.tokenizer(
                    sample["captions"]["correct"],
                    padding="do_not_pad",
                    max_length=pipeline.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                token_ids = tokenized.input_ids[0].tolist()
                aggregate, token_maps, stats = aggregate_attention(
                    store.records,
                    token_ids,
                    pipeline.tokenizer,
                    original_hw,
                    processed_hw,
                )
                numeric_stats = {
                    key: float(value)
                    for key, value in stats.items()
                    if isinstance(value, (float, int)) and key != "captured_cross_attention_call_count"
                }
                attention_stats.append(numeric_stats)
                save_attention_outputs(
                    output_dir / "attention_maps",
                    sample_id,
                    rgb,
                    aggregate,
                    token_maps,
                    stats,
                )
                print(f"[{sample_index + 1}/{len(samples)}] attention captured: {sample_id}", flush=True)
        finally:
            pipeline.unet.set_attn_processor(dict(original_processors))

    csv_path = output_dir / "per_sample_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_sample_rows)

    mode_means = {mode: average_dicts(items) for mode, items in metrics_by_mode.items()}
    comparisons = {
        f"correct_vs_{comparison}": comparison_summary(per_sample_rows, comparison)
        for comparison in ("hard_wrong", "empty", "generic")
    }
    response_summary = {
        comparison: average_dicts(records)
        for comparison, records in response_records.items()
    }
    attention_summary: Dict[str, Any] = average_dicts(attention_stats) if attention_stats else {}
    if attention_summary:
        attention_summary["sample_count"] = len(attention_stats)

    hard = comparisons["correct_vs_hard_wrong"]
    empty = comparisons["correct_vs_empty"]
    generic = comparisons["correct_vs_generic"]
    hard_rates = hard["correct_metric_win_rates"]
    hard_adv = hard["correct_mean_metric_advantage"]
    better_hard = (
        hard_rates["norm_mse"] >= 0.625
        and hard_adv["norm_mse"] > 0
        and sum(rate >= 0.625 for rate in hard_rates.values()) >= 3
        and hard["correct_wins_at_least_3_of_4_metrics_sample_ratio"] >= 0.625
    )
    better_empty_generic = (
        empty["correct_mean_metric_advantage"]["norm_mse"] > 0
        and generic["correct_mean_metric_advantage"]["norm_mse"] > 0
        and sum(value > 0 for value in empty["correct_mean_metric_advantage"].values()) >= 3
        and sum(value > 0 for value in generic["correct_mean_metric_advantage"].values()) >= 3
    )
    response = response_summary["hard_wrong"]
    spatial_response = (
        response.get("relative_difference_rmse_to_reference_std", 0.0) >= 1e-3
        and response.get("spatial_variation_energy_fraction", 0.0) >= 0.5
    )
    attention_available = bool(attention_summary)
    attention_concentrated = (
        attention_available
        and attention_summary.get("top_10_percent_mass", 0.0) >= 0.15
        and attention_summary.get("normalized_entropy", 1.0) <= 0.98
    )
    verdict_a = better_hard and better_empty_generic and spatial_response and attention_concentrated
    if verdict_a:
        verdict = {
            "label": "A. 原始 RGB Iris/Lotus 能区分 correct 与 hard_wrong，且文本响应具有一定空间定位能力，可以作为 teacher。",
            "text": "The released RGB Lotus-G checkpoint passes the pre-registered separation and spatial-response criteria.",
            "reasons": [
                "correct consistently beats hard_wrong",
                "correct also beats empty and generic",
                "caption-induced changes retain spatial variation after removing global shift",
                "cross-attention maps satisfy the conservative concentration threshold",
            ],
        }
    else:
        failed = []
        if not better_hard:
            failed.append("correct did not stably beat hard_wrong")
        if not better_empty_generic:
            failed.append("correct did not consistently beat both empty and generic")
        if not spatial_response:
            failed.append("caption response was too small or dominated by non-local/global effects")
        if not attention_concentrated:
            failed.append("attention did not meet the conservative spatial concentration criterion")
        verdict = {
            "label": "B. 原始 RGB Iris/Lotus 不能稳定区分 correct、empty 和 hard_wrong，当前复现不适合作为 teacher，需要先修复 Iris 文本路径。",
            "text": "The current reproduction fails at least one pre-registered teacher-readiness criterion.",
            "reasons": failed,
        }

    summary = {
        "num_samples": len(samples),
        "mode_metrics_mean": mode_means,
        "comparisons": comparisons,
        "prediction_response": response_summary,
        "prediction_response_correct_vs_hard_wrong": response_summary["hard_wrong"],
        "attention_summary": attention_summary,
        "decision_criteria": {
            "correct_stably_better_than_hard_wrong": better_hard,
            "correct_better_than_empty_and_generic": better_empty_generic,
            "nontrivial_spatial_response": spatial_response,
            "attention_spatially_concentrated": attention_concentrated,
        },
        "decision_rule_text": (
            "A requires: correct Norm-MSE win rate against hard_wrong >= 0.625 with positive mean advantage; "
            "at least three metrics and at least 62.5% of samples winning 3/4 metrics; correct mean advantage "
            "against both empty and generic on Norm-MSE and at least three metrics; relative caption response "
            ">= 1e-3 with >= 50% spatial-variation energy; and attention top-10% mass >= 0.15 with normalized "
            "entropy <= 0.98. Otherwise the conclusion is B."
        ),
        "verdict": verdict,
    }
    write_json(output_dir / "metrics_summary.json", summary)
    write_report(output_dir / "audit_report.md", config, load_report, summary)

    # A compact overview of all per-sample panels.
    if overview_panels:
        target_width = 1600
        rows: List[Image.Image] = []
        for label, panel in overview_panels:
            ratio = target_width / panel.width
            rows.append(panel.resize((target_width, max(1, int(panel.height * ratio)))))
        canvas = Image.new("RGB", (target_width, sum(row.height for row in rows)), "white")
        y = 0
        for row in rows:
            canvas.paste(row, (0, y))
            y += row.height
        canvas.save(output_dir / "visualizations" / "overview.png")

    print(json.dumps({"output_dir": str(output_dir), "verdict": verdict["label"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
