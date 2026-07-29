"""Train Adapter V0 for the MS2 thermal-only Lotus-G baseline.

This script keeps the verified Adapter V0 stack intact:

thermal -> frozen AnyThermal -> trainable Adapter -> frozen Lotus-G -> depth

Caption fields are read from the manifest for bookkeeping only; they are not
sent to the model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


try:

    from tqdm.auto import tqdm

except Exception:  # pragma: no cover - progress is optional

    tqdm = None



REPO_ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = REPO_ROOT / "lotus"
TOOLS_ROOT = REPO_ROOT / "tools"
for path in (REPO_ROOT, LOTUS_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from diffusers import DDPMScheduler  # noqa: E402

import overfit_32_anythermal_lotus as overfit  # noqa: E402
from models.anythermal_encoder import AnyThermalEncoder  # noqa: E402
from models.anythermal_lotus_adapter import AnyThermalLotusAdapter  # noqa: E402
from models.anythermal_lotus_conditioner import AnyThermalDirectConditioner  # noqa: E402
from models.anythermal_lotus_model import AnyThermalLotusModel  # noqa: E402
from pipeline import LotusGPipeline  # noqa: E402


DEFAULT_MANIFEST_DIR = (
    "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b"
)
DEFAULT_TRAIN_MANIFEST = (
    f"{DEFAULT_MANIFEST_DIR}/ms2_fixed_sequence_internvl3_8b_filtered_caption_train.jsonl"
)
DEFAULT_VAL_MANIFEST = (
    f"{DEFAULT_MANIFEST_DIR}/ms2_fixed_sequence_internvl3_8b_filtered_caption_val.jsonl"
)
EXPERIMENT_NAME = "adapter_v0_thermal_only_frozen_lotus"
JOINT_EXPERIMENT_NAME = "adapter_v0_unet_joint_no_caption"
UNET_ONLY_EXPERIMENT_NAME = "direct_bridge_lotus_unet_only_no_caption"
CAPTION_ATTN_EXPERIMENT_NAME = "adapter_v0_caption_attn"
ADAPTER_CAPTION_ATTN_EXPERIMENT_NAME = "adapter_v0_adapter_caption_attn"
ADAPTER_FULL_CAPTION_ATTN_EXPERIMENT_NAME = "adapter_v0_full_caption_attn"
TEXT_CROSS_ATTENTION_MARKER = ".attn2."
TEXT_CROSS_ATTENTION_KV_PARAM_MARKERS = (".attn2.to_k.", ".attn2.to_v.")
TEXT_CROSS_ATTENTION_FULL_PARAM_MARKERS = (
    ".attn2.to_q.",
    ".attn2.to_k.",
    ".attn2.to_v.",
    ".attn2.to_out.",
)
TRANSFORMER_BLOCK_MARKER = ".transformer_blocks."


class MS2AdapterDataset(Dataset):
    """Manifest-backed MS2 dataset using the verified overfit preprocessing."""

    def __init__(
        self,
        *,
        manifest: Path,
        ms2_root: Path,
        max_samples: Optional[int] = None,
    ) -> None:
        limit = max_samples if max_samples is not None else 10**12
        self.samples = overfit.load_manifest(manifest, ms2_root, limit)
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise ValueError(f"No readable samples found in manifest: {manifest}")
        self.sample_ids = [sample["id"] for sample in self.samples]
        self._id_to_index = {sample_id: index for index, sample_id in enumerate(self.sample_ids)}

    def __len__(self) -> int:
        return len(self.samples)

    def get_by_id(self, sample_id: str) -> Dict[str, Any]:
        return self[self._id_to_index[sample_id]]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        depth, valid_mask, raw_depth, valid_np, depth_stats = overfit.load_depth(
            Path(sample["depth"])
        )
        return {
            "index": index,
            "id": sample["id"],
            "image_id": sample["image_id"],
            "sequence": sample["sequence"],
            "split": sample["split"],
            "caption": sample["caption"],
            "thermal_path": sample["thermal"],
            "depth_path": sample["depth"],
            "depth_values": depth,
            "valid_mask": valid_mask,
            "raw_depth": raw_depth,
            "valid_mask_np": valid_np,
            "depth_stats": depth_stats,
        }


def collate_samples(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "indices": [item["index"] for item in items],
        "ids": [item["id"] for item in items],
        "image_ids": [item["image_id"] for item in items],
        "sequences": [item["sequence"] for item in items],
        "splits": [item["split"] for item in items],
        "captions": [item["caption"] for item in items],
        "thermal_paths": [item["thermal_path"] for item in items],
        "depth_paths": [item["depth_path"] for item in items],
        "depth_values": torch.stack([item["depth_values"] for item in items], dim=0),
        "valid_mask": torch.stack([item["valid_mask"] for item in items], dim=0),
        "raw_depth": [item["raw_depth"] for item in items],
        "valid_mask_np": [item["valid_mask_np"] for item in items],
        "depth_stats": [item["depth_stats"] for item in items],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--ms2-root", default="/mnt/e/dataset/ms2")
    parser.add_argument("--output-dir", default="outputs/adapter_v0_thermal_only_short_run")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-mode", choices=("adapter_only", "unet_only", "adapter_unet", "caption_attn", "adapter_caption_attn", "adapter_full_caption_attn"), default="adapter_only")
    parser.add_argument("--adapter-lr", type=float, default=None)
    parser.add_argument("--lotus-unet-lr", type=float, default=1e-6)
    parser.add_argument("--text-attn-lr", type=float, default=1e-6)
    parser.add_argument("--caption-mode", choices=("disabled", "empty", "real", "shuffled"), default="disabled")
    parser.add_argument("--caption-dropout-prob", type=float, default=0.0)
    parser.add_argument("--save-caption-comparisons", action="store_true")
    parser.add_argument("--init-adapter-checkpoint", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--visualization-interval", type=int, default=100)
    parser.add_argument("--save-steps", default=None, help="Comma-separated exact steps for checkpoint/visual saves, e.g. 0,100,250,500")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--mixed-precision",
        choices=("no", "fp16", "bf16"),
        default="no",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--anythermal-revision", default=None)
    parser.add_argument(
        "--lotus-model-path",
        default="jingheya/lotus-depth-g-v2-1-disparity",
    )
    parser.add_argument("--prediction-type", default="sample")
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--val-batch-limit", type=int, default=None)
    parser.add_argument("--num-fixed-val-samples", type=int, default=8)
    parser.add_argument("--fixed-val-samples-json", default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", choices=("none", "constant", "cosine"), default="constant")
    parser.add_argument("--append-logs", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")

    return parser.parse_args()



def parse_step_set(text: Optional[str]) -> set[int]:
    if not text:
        return set()
    steps = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 0:
            raise ValueError(f"Save steps must be non-negative, got {value}")
        steps.add(value)
    return steps
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, mixed_precision: str):
    if mixed_precision == "no" or device.type != "cuda":
        return torch.amp.autocast(device_type=device.type, enabled=False)
    dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def module_has_grad(module: torch.nn.Module) -> bool:
    return any(parameter.grad is not None for parameter in module.parameters())


def parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def trainable_parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def text_cross_attention_param_markers(train_mode: str = "caption_attn") -> Tuple[str, ...]:
    if train_mode == "adapter_full_caption_attn":
        return TEXT_CROSS_ATTENTION_FULL_PARAM_MARKERS
    return TEXT_CROSS_ATTENTION_KV_PARAM_MARKERS


def is_text_cross_attention_parameter(name: str, train_mode: str = "caption_attn") -> bool:
    markers = text_cross_attention_param_markers(train_mode)
    return TRANSFORMER_BLOCK_MARKER in name and any(marker in name for marker in markers)


def text_cross_attention_named_parameters(
    model: AnyThermalLotusModel,
    train_mode: str = "caption_attn",
) -> List[Tuple[str, torch.nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in model.lotus.unet.named_parameters()
        if is_text_cross_attention_parameter(name, train_mode)
    ]


def text_cross_attention_parameters(
    model: AnyThermalLotusModel,
    train_mode: str = "caption_attn",
) -> List[torch.nn.Parameter]:
    return [parameter for _, parameter in text_cross_attention_named_parameters(model, train_mode)]


def text_cross_attention_state_dict(
    model: AnyThermalLotusModel,
    train_mode: str = "caption_attn",
) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in text_cross_attention_named_parameters(model, train_mode)
    }


def load_text_cross_attention_state_dict(
    model: AnyThermalLotusModel,
    state_dict: Dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    current = dict(text_cross_attention_named_parameters(model, "adapter_full_caption_attn"))
    unexpected = sorted(set(state_dict) - set(current))
    if unexpected:
        raise RuntimeError(f"Unexpected text cross-attention checkpoint keys: {unexpected[:5]}")
    with torch.no_grad():
        for name, loaded_tensor in state_dict.items():
            parameter = current[name]
            loaded = loaded_tensor.to(device=device, dtype=parameter.dtype)
            if loaded.shape != parameter.shape:
                raise RuntimeError(f"Shape mismatch for {name}: {tuple(loaded.shape)} vs {tuple(parameter.shape)}")
            parameter.copy_(loaded)


def module_has_unexpected_unet_grad(model: AnyThermalLotusModel, train_mode: str) -> List[str]:
    return [
        name
        for name, parameter in model.lotus.unet.named_parameters()
        if parameter.grad is not None and not is_text_cross_attention_parameter(name, train_mode)
    ]

def experiment_name_for_mode(train_mode: str) -> str:
    if train_mode == "unet_only":
        return UNET_ONLY_EXPERIMENT_NAME
    if train_mode == "adapter_unet":
        return JOINT_EXPERIMENT_NAME
    if train_mode == "caption_attn":
        return CAPTION_ATTN_EXPERIMENT_NAME
    if train_mode == "adapter_caption_attn":
        return ADAPTER_CAPTION_ATTN_EXPERIMENT_NAME
    return EXPERIMENT_NAME



def all_named_parameters_for_audit(model: AnyThermalLotusModel) -> List[Tuple[str, torch.nn.Parameter]]:
    records: List[Tuple[str, torch.nn.Parameter]] = []
    records.extend((f"adapter.{name}", parameter) for name, parameter in model.adapter.named_parameters())
    records.extend((f"lotus.unet.{name}", parameter) for name, parameter in model.lotus.unet.named_parameters())
    records.extend((f"lotus.vae.{name}", parameter) for name, parameter in model.lotus.vae.named_parameters())
    records.extend((f"lotus.text_encoder.{name}", parameter) for name, parameter in model.lotus.text_encoder.named_parameters())
    return records
def trainable_module_records(model: AnyThermalLotusModel) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for name, parameter in all_named_parameters_for_audit(model):
        if not parameter.requires_grad:
            continue
        group = "other"
        if name.startswith("adapter."):
            group = "adapter"
        elif name.startswith("lotus.unet.") and is_text_cross_attention_parameter(name[len("lotus.unet."):], "adapter_full_caption_attn"):
            group = "text_cross_attention"
        elif name.startswith("lotus.unet."):
            group = "lotus_unet"
        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        records.append(
            {
                "module_name": module_name,
                "parameter_name": name,
                "parameter_shape": list(parameter.shape),
                "parameter_count": int(parameter.numel()),
                "group": group,
                "is_text_cross_attention": group == "text_cross_attention",
            }
        )
    return records


def save_trainable_modules(model: AnyThermalLotusModel, output_dir: Path, train_mode: str) -> None:
    records = trainable_module_records(model)
    payload = {
        "train_mode": train_mode,
        "total_trainable_params": int(sum(item["parameter_count"] for item in records)),
        "text_cross_attention_selector": {
            "requires_transformer_block_marker": TRANSFORMER_BLOCK_MARKER,
            "requires_cross_attention_marker": TEXT_CROSS_ATTENTION_MARKER,
            "selected_parameter_markers": list(text_cross_attention_param_markers(train_mode)),
            "verified_by": "docs/lotus_caption_condition_interface.md",
        },
        "parameters": records,
    }
    with (output_dir / "trainable_modules.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def grad_norm(parameters) -> float:
    total = 0.0
    has_grad = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        if not torch.isfinite(grad).all():
            return float("nan")
        total += float(grad.pow(2).sum().cpu())
        has_grad = True
    return float(total ** 0.5) if has_grad else 0.0


def trainable_parameters_for_mode(model: AnyThermalLotusModel, train_mode: str) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    if train_mode in {"adapter_only", "adapter_unet", "adapter_caption_attn", "adapter_full_caption_attn"}:
        params.extend(parameter for parameter in model.adapter.parameters() if parameter.requires_grad)
    if train_mode in {"unet_only", "adapter_unet"}:
        params.extend(parameter for parameter in model.lotus.unet.parameters() if parameter.requires_grad)
    if train_mode in {"caption_attn", "adapter_caption_attn", "adapter_full_caption_attn"}:
        params.extend(parameter for parameter in text_cross_attention_parameters(model, train_mode) if parameter.requires_grad)
    return params


def configure_trainability(model: AnyThermalLotusModel, train_mode: str) -> None:
    if train_mode not in {"adapter_only", "unet_only", "adapter_unet", "caption_attn", "adapter_caption_attn", "adapter_full_caption_attn"}:
        raise ValueError(f"Unsupported train mode: {train_mode}")

    model.anythermal_encoder.model.eval().requires_grad_(False)
    model.lotus.vae.eval().requires_grad_(False)
    model.lotus.text_encoder.eval().requires_grad_(False)

    train_adapter = train_mode in {"adapter_only", "adapter_unet", "adapter_caption_attn", "adapter_full_caption_attn"}
    model.adapter.train(train_adapter).requires_grad_(train_adapter)

    model.lotus.unet.requires_grad_(False)
    if train_mode in {"unet_only", "adapter_unet"}:
        model.lotus.unet.train().requires_grad_(True)
    elif train_mode in {"caption_attn", "adapter_caption_attn", "adapter_full_caption_attn"}:
        model.lotus.unet.train()
        selected = text_cross_attention_named_parameters(model, train_mode)
        if not selected:
            raise RuntimeError("No Lotus-G text cross-attention parameters matched the selector.")
        for _, parameter in selected:
            parameter.requires_grad_(True)
    else:
        model.lotus.unet.eval()


def set_training_modes(model: AnyThermalLotusModel, train_mode: str) -> None:
    configure_trainability(model, train_mode)

def trainability_summary(model: AnyThermalLotusModel, train_mode: str) -> Dict[str, Any]:
    modules = {
        "anythermal": model.anythermal_encoder.model,
        "adapter": model.adapter,
        "lotus_unet": model.lotus.unet,
        "vae": model.lotus.vae,
        "text_encoder": model.lotus.text_encoder,
    }
    summary = {
        name: {
            "total_params": parameter_count(module),
            "trainable_params": trainable_parameter_count(module),
            "requires_grad_any": any(parameter.requires_grad for parameter in module.parameters()),
            "training": bool(module.training),
        }
        for name, module in modules.items()
    }
    summary["train_mode"] = train_mode
    return summary


def trainability_check(model: AnyThermalLotusModel, train_mode: str) -> None:
    allowed_prefixes: List[str] = []
    if train_mode in {"adapter_only", "adapter_unet", "adapter_caption_attn", "adapter_full_caption_attn"}:
        allowed_prefixes.append("adapter.")
    if train_mode in {"unet_only", "adapter_unet"}:
        allowed_prefixes.append("lotus.unet.")
    if train_mode in {"caption_attn", "adapter_caption_attn", "adapter_full_caption_attn"}:
        allowed_prefixes.append("lotus.unet.")

    trainable = [name for name, parameter in all_named_parameters_for_audit(model) if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found.")
    unexpected = [name for name in trainable if not any(name.startswith(prefix) for prefix in allowed_prefixes)]
    if unexpected:
        raise RuntimeError(f"Unexpected trainable parameters for {train_mode}: {unexpected[:20]}")

    if train_mode == "adapter_only" and trainable_parameter_count(model.lotus.unet) != 0:
        raise RuntimeError("Lotus U-Net must be frozen in adapter_only mode.")
    if train_mode == "caption_attn" and trainable_parameter_count(model.adapter) != 0:
        raise RuntimeError("Adapter must be frozen in caption_attn mode.")
    if train_mode in {"caption_attn", "adapter_caption_attn", "adapter_full_caption_attn"}:
        bad_unet = [
            name
            for name, parameter in model.lotus.unet.named_parameters()
            if parameter.requires_grad and not is_text_cross_attention_parameter(name, train_mode)
        ]
        if bad_unet:
            raise RuntimeError(f"Non-text-cross-attention U-Net parameters are trainable: {bad_unet[:20]}")
        if not any(parameter.requires_grad for parameter in text_cross_attention_parameters(model, train_mode)):
            raise RuntimeError("Text cross-attention parameters must be trainable.")
    if train_mode in {"unet_only", "adapter_unet"} and trainable_parameter_count(model.lotus.unet) == 0:
        raise RuntimeError(f"Lotus U-Net must be trainable in {train_mode} mode.")
    if train_mode == "unet_only":
        if parameter_count(model.adapter) != 0:
            raise RuntimeError("unet_only must use the zero-parameter Direct conditioner.")
        if trainable_parameter_count(model.adapter) != 0:
            raise RuntimeError("Direct conditioner must have no trainable parameters.")

    frozen_grad_modules = {
        "anythermal": model.anythermal_encoder.model,
        "vae": model.lotus.vae,
        "text_encoder": model.lotus.text_encoder,
    }
    if train_mode == "adapter_only":
        frozen_grad_modules["lotus_unet"] = model.lotus.unet
    if train_mode in {"caption_attn", "unet_only"}:
        frozen_grad_modules["adapter"] = model.adapter
    leaked = [name for name, module in frozen_grad_modules.items() if module_has_grad(module)]
    if leaked:
        raise RuntimeError(f"Frozen modules already have gradients: {leaked}")
def extract_batch_features(
    model: AnyThermalLotusModel,
    thermal_paths: Sequence[str],
    *,
    device: torch.device,
) -> List[torch.Tensor]:
    features_per_sample = []
    for thermal_path in thermal_paths:
        features, _, _ = model.extract_features(Path(thermal_path))
        features_per_sample.append([feature.detach().cpu() for feature in features])
    levels = len(features_per_sample[0])
    return [
        torch.cat([sample[level] for sample in features_per_sample], dim=0).to(device)
        for level in range(levels)
    ]


def make_noise_for_batch(
    *,
    batch_size: int,
    latent_size: Sequence[int],
    global_step: int,
    batch_indices: Sequence[int],
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    marker = sum(int(index) for index in batch_indices) + len(batch_indices)
    return overfit.make_noise(
        batch_size,
        tuple(latent_size),
        global_step,
        marker,
        seed,
        device,
        dtype,
    )



def shuffled_captions_for_batch(captions: Sequence[str]) -> List[str]:
    if len(captions) < 2:
        raise ValueError("caption_mode=shuffled requires batch_size >= 2 to create mismatched captions.")
    return list(captions[1:]) + [captions[0]]


def validate_caption_texts(captions: Sequence[str], *, mode: str) -> None:
    if mode not in {"real", "shuffled"}:
        return
    bad = [index for index, caption in enumerate(captions) if not str(caption).strip()]
    if bad:
        raise RuntimeError(f"caption_mode={mode} requires non-empty captions; empty batch positions: {bad[:10]}")


def encode_prompt_batch(model: AnyThermalLotusModel, captions: Sequence[str]) -> torch.Tensor:
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
        prompt_embeds = text_encoder(input_ids, return_dict=False)[0]
    if prompt_embeds.ndim != 3 or prompt_embeds.shape[1] != tokenizer.model_max_length:
        raise RuntimeError(f"Unexpected text embedding shape: {tuple(prompt_embeds.shape)}")
    if not bool(torch.isfinite(prompt_embeds).all()):
        raise RuntimeError("Text embeddings contain NaN or Inf.")
    return prompt_embeds


def prompt_embeds_for_batch(
    model: AnyThermalLotusModel,
    batch: Dict[str, Any],
    *,
    caption_mode: str,
    caption_dropout_prob: float = 0.0,
    dropout_seed: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
    captions = [str(caption) for caption in batch["captions"]]
    validate_caption_texts(captions, mode=caption_mode)
    info: Dict[str, Any] = {
        "caption_mode": caption_mode,
        "caption_dropout_prob": float(caption_dropout_prob),
        "num_samples": len(captions),
        "num_empty": 0,
        "num_real": 0,
        "num_shuffled": 0,
    }
    if caption_mode == "disabled":
        info["num_empty"] = len(captions)
        return None, info
    if caption_mode == "empty":
        info["num_empty"] = len(captions)
        return encode_prompt_batch(model, [""] * len(captions)), info
    if caption_mode == "real":
        used = list(captions)
        if caption_dropout_prob > 0.0:
            if not 0.0 <= caption_dropout_prob <= 1.0:
                raise ValueError(f"caption_dropout_prob must be in [0,1], got {caption_dropout_prob}")
            rng = random.Random(dropout_seed)
            for index in range(len(used)):
                if rng.random() < caption_dropout_prob:
                    used[index] = ""
        info["num_empty"] = int(sum(1 for caption in used if not caption.strip()))
        info["num_real"] = int(len(used) - info["num_empty"])
        return encode_prompt_batch(model, used), info
    if caption_mode == "shuffled":
        shuffled = shuffled_captions_for_batch(captions)
        info["num_shuffled"] = len(shuffled)
        return encode_prompt_batch(model, shuffled), info
    raise ValueError(f"Unsupported caption mode: {caption_mode}")
def forward_batch(
    model: AnyThermalLotusModel,
    batch: Dict[str, Any],
    *,
    device: torch.device,
    global_step: int,
    seed: int,
    timestep: int,
    return_decoded: bool,
    caption_mode: str = "disabled",
    caption_dropout_prob: float = 0.0,
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
        global_step=global_step,
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
    dropout_seed = seed + global_step * 1_000_003 + sum(int(index) for index in batch["indices"])
    prompt_embeds, caption_info = prompt_embeds_for_batch(
        model,
        batch,
        caption_mode=caption_mode,
        caption_dropout_prob=caption_dropout_prob,
        dropout_seed=dropout_seed,
    )
    outputs = model(
        features=features,
        depth_values=depth_values,
        valid_mask=valid_mask,
        timesteps=timesteps,
        noise=noise,
        prompt_embeds=prompt_embeds,
        return_decoded=return_decoded,
    )
    outputs["caption_info"] = caption_info
    return outputs
def normalized_gt_from_depth_values(depth_values: torch.Tensor) -> np.ndarray:
    values = depth_values.detach().float().cpu().numpy()
    return np.clip((values[:, 0] + 1.0) * 0.5, 0.0, 1.0)


def normalize_prediction(pred: np.ndarray, valid: np.ndarray) -> np.ndarray:
    pred = pred.astype(np.float32)
    output = np.zeros_like(pred, dtype=np.float32)
    if valid.any():
        vals = pred[valid]
        lo = float(vals.min())
        hi = float(vals.max())
        if hi > lo:
            output = np.clip((pred - lo) / (hi - lo), 0.0, 1.0)
    return output


def scale_shift_align(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = pred[valid].reshape(-1).astype(np.float64)
    y = gt[valid].reshape(-1).astype(np.float64)
    if x.size < 2 or float(x.var()) == 0.0:
        return pred.astype(np.float32)
    A = np.stack([x, np.ones_like(x)], axis=1)
    scale, shift = np.linalg.lstsq(A, y, rcond=None)[0]
    return (pred * float(scale) + float(shift)).astype(np.float32)


def metric_values(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    if not valid.any():
        return {
            "corr": 0.0,
            "rmse": 0.0,
            "absrel": 0.0,
            "mae": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    p = pred[valid].astype(np.float64)
    g = gt[valid].astype(np.float64)
    if p.size < 2 or float(p.std()) == 0.0 or float(g.std()) == 0.0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(p, g)[0, 1])
    abs_error = np.abs(p - g)
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    absrel = float(np.mean(abs_error / np.maximum(np.abs(g), 1e-3)))
    return {
        "corr": corr,
        "rmse": rmse,
        "absrel": absrel,
        "mae": float(np.mean(abs_error)),
        "p90": float(np.percentile(abs_error, 90.0)),
        "p95": float(np.percentile(abs_error, 95.0)),
        "p99": float(np.percentile(abs_error, 99.0)),
    }

def decoded_depth(outputs: Dict[str, Any]) -> Optional[np.ndarray]:
    decoded = outputs.get("decoded")
    if decoded is None:
        return None
    return decoded.detach().float().cpu().mean(dim=1).numpy()


def validation_metrics_for_batch(
    outputs: Dict[str, Any],
    batch: Dict[str, Any],
) -> Dict[str, float]:
    pred_depth = decoded_depth(outputs)
    if pred_depth is None:
        return {}
    gt = normalized_gt_from_depth_values(batch["depth_values"])
    valid = batch["valid_mask"].numpy().astype(bool)[:, 0]
    raw_metrics = []
    aligned_metrics = []
    for index in range(pred_depth.shape[0]):
        pred_norm = normalize_prediction(pred_depth[index], valid[index])
        raw_metrics.append(metric_values(pred_norm, gt[index], valid[index]))
        aligned = scale_shift_align(pred_depth[index], gt[index], valid[index])
        aligned_metrics.append(metric_values(aligned, gt[index], valid[index]))
    keys = sorted({key for item in raw_metrics + aligned_metrics for key in item})
    metrics: Dict[str, float] = {}
    for key in keys:
        metrics[key] = float(np.mean([item[key] for item in raw_metrics]))
        metrics[f"aligned_{key}"] = float(np.mean([item[key] for item in aligned_metrics]))
    return metrics

def average_dicts(items: Sequence[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({key for item in items for key in item})
    return {key: float(np.mean([item[key] for item in items if key in item])) for key in keys}



def write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

def make_progress(iterable, *, total: Optional[int], desc: str, disable: bool):
    if disable or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=False)


def console_log(message: str) -> None:
    print(message, flush=True)


def gpu_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))


def module_summary(
    model: AnyThermalLotusModel,
    train_mode: str = "adapter_only",
    caption_mode: str = "disabled",
) -> Dict[str, Any]:
    experiment_name = experiment_name_for_mode(train_mode)
    if caption_mode != "disabled":
        experiment_name = f"{experiment_name}_{caption_mode}_caption"
    conditioner_type = getattr(model.adapter, "route_name", "learned_adapter_v0")
    return {
        "experiment_name": experiment_name,
        "train_mode": train_mode,
        "anythermal_blocks": [8, 9, 10, 11],
        "anythermal_feature_shape": ["B", 768, 18, 45],
        "conditioner_type": conditioner_type,
        "conditioner_output_shape": ["B", 4, 32, 80],
        "adapter_output_shape": ["B", 4, 32, 80],
        "lotus_unet_input_shape": ["B", 8, 32, 80],
        "caption_enabled": caption_mode != "disabled",
        "caption_mode": caption_mode,
        "modules": trainability_summary(model, train_mode),
        "adapter_parameter_count": parameter_count(model.adapter),
        "adapter_trainable_parameter_count": trainable_parameter_count(model.adapter),
        "conditioner_parameter_count": parameter_count(model.adapter),
        "conditioner_trainable_parameter_count": trainable_parameter_count(model.adapter),
        "lotus_unet_parameter_count": parameter_count(model.lotus.unet),
        "lotus_unet_trainable_parameter_count": trainable_parameter_count(model.lotus.unet),
        "text_cross_attention_parameter_count": int(sum(parameter.numel() for parameter in text_cross_attention_parameters(model, train_mode))),
        "text_cross_attention_trainable_parameter_count": int(sum(parameter.numel() for parameter in text_cross_attention_parameters(model, train_mode) if parameter.requires_grad)),
        "anythermal_frozen": trainable_parameter_count(model.anythermal_encoder.model) == 0,
        "vae_frozen": trainable_parameter_count(model.lotus.vae) == 0,
        "text_encoder_frozen": trainable_parameter_count(model.lotus.text_encoder) == 0,
        "lotus_unet_frozen": trainable_parameter_count(model.lotus.unet) == 0,
    }

def save_image(array: np.ndarray, path: Path, valid: Optional[np.ndarray] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    Image.fromarray((norm * 255.0).astype(np.uint8)).save(path)


@torch.no_grad()
def save_val_visuals(
    model: AnyThermalLotusModel,
    dataset: MS2AdapterDataset,
    sample_ids: Sequence[str],
    output_dir: Path,
    step: int,
    device: torch.device,
    seed: int,
    timestep: int,
    caption_mode: str = "disabled",
) -> None:
    step_dir = output_dir / "val_visuals" / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    samples = [dataset.get_by_id(sample_id) for sample_id in sample_ids]
    batch = collate_samples(samples)
    outputs = forward_batch(
        model,
        batch,
        device=device,
        global_step=step,
        seed=seed,
        timestep=timestep,
        return_decoded=True,
        caption_mode=caption_mode,
    )
    pred = decoded_depth(outputs)
    adapter = outputs["condition_latent"].detach().float().cpu().numpy()
    gt = batch["raw_depth"]
    gt_norm = normalized_gt_from_depth_values(batch["depth_values"])
    valid = [mask.astype(bool) for mask in batch["valid_mask_np"]]
    thermal = [overfit.thermal_vis(Path(path)) for path in batch["thermal_paths"]]
    stats = []
    for idx, sample in enumerate(samples):
        safe_id = sample["id"].replace("/", "_").replace("\\", "_").replace(":", "_")
        sample_dir = step_dir / f"{idx:02d}_{safe_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_image(thermal[idx], sample_dir / "thermal.png")
        save_image(gt[idx], sample_dir / "gt_depth.png", valid[idx])
        Image.fromarray((valid[idx].astype(np.uint8) * 255)).save(sample_dir / "valid_mask.png")
        if pred is not None:
            aligned_pred = scale_shift_align(pred[idx], gt_norm[idx], valid[idx])
            save_image(pred[idx], sample_dir / "pred_depth.png", valid[idx])
            save_image(aligned_pred, sample_dir / "pred_depth_aligned.png", valid[idx])
            np.save(sample_dir / "pred_depth.npy", pred[idx])
            np.save(sample_dir / "pred_depth_aligned.npy", aligned_pred)
        np.save(sample_dir / "thermal.npy", thermal[idx])
        np.save(sample_dir / "gt_depth.npy", gt[idx])
        np.save(sample_dir / "valid_mask.npy", valid[idx].astype(np.uint8))
        np.save(sample_dir / "adapter_output.npy", adapter[idx])
        torch.save(
            {
                "thermal_path": sample["thermal_path"],
                "gt_depth": batch["depth_values"][idx].cpu(),
                "raw_depth": torch.from_numpy(gt[idx].astype(np.float32)),
                "valid_mask": batch["valid_mask"][idx].cpu(),
                "adapter_output": outputs["condition_latent"][idx].detach().cpu(),
                "pred_depth": None if pred is None else torch.from_numpy(pred[idx]),
                "pred_depth_aligned": None if pred is None else torch.from_numpy(scale_shift_align(pred[idx], gt_norm[idx], valid[idx])),
                "sample": sample,
            },
            sample_dir / "tensors.pt",
        )
        stats.append({
            "sample_id": sample["id"],
            "adapter_mean": float(adapter[idx].mean()),
            "adapter_std": float(adapter[idx].std()),
            "adapter_min": float(adapter[idx].min()),
            "adapter_max": float(adapter[idx].max()),
        })
    with (step_dir / "adapter_output_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)

@torch.no_grad()
def save_caption_comparison_visuals(
    model: AnyThermalLotusModel,
    dataset: MS2AdapterDataset,
    sample_ids: Sequence[str],
    output_dir: Path,
    step: int,
    device: torch.device,
    seed: int,
    timestep: int,
) -> None:
    if len(sample_ids) < 2:
        raise ValueError("Caption comparison needs at least two fixed samples for shuffled captions.")
    step_dir = output_dir / "caption_comparison" / f"step_{step:06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    samples = [dataset.get_by_id(sample_id) for sample_id in sample_ids]
    batch = collate_samples(samples)
    modes = ("real", "empty", "shuffled")
    outputs_by_mode = {
        mode: forward_batch(
            model,
            batch,
            device=device,
            global_step=step,
            seed=seed,
            timestep=timestep,
            return_decoded=True,
            caption_mode=mode,
            caption_dropout_prob=0.0,
        )
        for mode in modes
    }
    preds = {mode: decoded_depth(outputs_by_mode[mode]) for mode in modes}
    gt_norm = normalized_gt_from_depth_values(batch["depth_values"])
    valid = [mask.astype(bool) for mask in batch["valid_mask_np"]]
    thermal = [overfit.thermal_vis(Path(path)) for path in batch["thermal_paths"]]
    summaries: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        safe_id = sample["id"].replace("/", "_").replace("\\", "_").replace(":", "_")
        sample_dir = step_dir / f"{idx:02d}_{safe_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_image(thermal[idx], sample_dir / "thermal.png")
        save_image(batch["raw_depth"][idx], sample_dir / "gt_depth.png", valid[idx])
        Image.fromarray((valid[idx].astype(np.uint8) * 255)).save(sample_dir / "valid_mask.png")
        np.save(sample_dir / "thermal.npy", thermal[idx].astype(np.float32))
        np.save(sample_dir / "gt_depth.npy", batch["raw_depth"][idx].astype(np.float32))
        np.save(sample_dir / "gt_depth_normalized.npy", gt_norm[idx].astype(np.float32))
        np.save(sample_dir / "valid_mask.npy", valid[idx].astype(np.uint8))
        with (sample_dir / "caption.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "sample_id": sample["id"],
                    "caption_real": sample["caption"],
                    "caption_empty": "",
                    "caption_shuffled": batch["captions"][(idx + 1) % len(samples)],
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        aligned_by_mode: Dict[str, np.ndarray] = {}
        metric_by_mode: Dict[str, Dict[str, float]] = {}
        for mode in modes:
            pred = preds[mode][idx]
            aligned = scale_shift_align(pred, gt_norm[idx], valid[idx])
            error = np.abs(aligned - gt_norm[idx]).astype(np.float32)
            aligned_by_mode[mode] = aligned
            metric_by_mode[mode] = metric_values(aligned, gt_norm[idx], valid[idx])
            save_image(pred, sample_dir / f"pred_{mode}.png", valid[idx])
            save_image(aligned, sample_dir / f"pred_{mode}_aligned.png", valid[idx])
            save_image(error, sample_dir / f"error_{mode}.png", valid[idx])
            np.save(sample_dir / f"pred_{mode}.npy", pred.astype(np.float32))
            np.save(sample_dir / f"pred_{mode}_aligned.npy", aligned.astype(np.float32))
            np.save(sample_dir / f"error_{mode}.npy", error)
        real_empty_l1 = float(np.mean(np.abs(aligned_by_mode["real"] - aligned_by_mode["empty"])[valid[idx]]))
        real_shuffled_l1 = float(np.mean(np.abs(aligned_by_mode["real"] - aligned_by_mode["shuffled"])[valid[idx]]))
        summaries.append(
            {
                "sample_id": sample["id"],
                "metrics": metric_by_mode,
                "real_vs_empty_aligned_l1": real_empty_l1,
                "real_vs_shuffled_aligned_l1": real_shuffled_l1,
            }
        )
    aggregate: Dict[str, Any] = {"step": step, "samples": summaries, "modes": list(modes)}
    for mode in modes:
        mode_metrics = [item["metrics"][mode] for item in summaries]
        aggregate[f"{mode}_metrics_mean"] = average_dicts(mode_metrics)
    aggregate["real_vs_empty_aligned_l1_mean"] = float(np.mean([item["real_vs_empty_aligned_l1"] for item in summaries]))
    aggregate["real_vs_shuffled_aligned_l1_mean"] = float(np.mean([item["real_vs_shuffled_aligned_l1"] for item in summaries]))
    with (step_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, ensure_ascii=False)

@torch.no_grad()
def validate(
    model: AnyThermalLotusModel,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int],
    seed: int,
    timestep: int,
    show_progress: bool = True,
    progress_desc: str = "val",
    disable_progress: bool = False,
    caption_mode: str = "disabled",
) -> Dict[str, float]:
    model.eval()
    records: List[Dict[str, float]] = []
    total = len(loader) if max_batches is None else min(max_batches, len(loader))
    iterator = make_progress(
        loader,
        total=total,
        desc=progress_desc,
        disable=(disable_progress or not show_progress),
    )
    for batch_idx, batch in enumerate(iterator):
        if max_batches is not None and batch_idx >= max_batches:
            break
        outputs = forward_batch(
            model,
            batch,
            device=device,
            global_step=batch_idx + 1,
            seed=seed,
            timestep=timestep,
            return_decoded=True,
            caption_mode=caption_mode,
        )
        adapter = outputs["condition_latent"].detach().float()
        record = {
            "val_loss": float(outputs["loss"].detach().cpu()),
            "conditioner_type": getattr(model.adapter, "route_name", "learned_adapter_v0"),
            "conditioner_output_mean": float(adapter.mean().cpu()),
            "conditioner_output_std": float(adapter.std(unbiased=False).cpu()),
            "adapter_output_mean": float(adapter.mean().cpu()),
            "adapter_output_std": float(adapter.std(unbiased=False).cpu()),
        }
        record.update(validation_metrics_for_batch(outputs, batch))
        records.append(record)
    return average_dicts(records) if records else {"val_loss": float("nan")}
def checkpoint_payload(
    model: AnyThermalLotusModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    global_step: int,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
    fixed_val_ids: Sequence[str],
    *,
    include_optimizer: bool = True,
) -> Dict[str, Any]:
    payload = {
        "conditioner_state_dict": model.adapter.state_dict(),
        "conditioner_type": getattr(model.adapter, "route_name", "learned_adapter_v0"),
        "optimizer_state_dict": optimizer.state_dict() if include_optimizer else None,
        "scheduler_state_dict": None if scheduler is None or not include_optimizer else scheduler.state_dict(),
        "resume_capable": bool(include_optimizer),
        "global_step": global_step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "best_aligned_rmse": float(getattr(args, "best_aligned_rmse", float("inf"))),
        "train_mode": args.train_mode,
        "adapter_lr": effective_adapter_lr(args),
        "lotus_unet_lr": float(args.lotus_unet_lr),
        "text_attn_lr": float(args.text_attn_lr),
        "caption_mode": args.caption_mode,
        "config": vars(args),
        "random_seed": args.seed,
        "fixed_val_ids": list(fixed_val_ids),
        "manifest_paths": {
            "train": str(args.train_manifest),
            "val": str(args.val_manifest),
        },
        "model_summary": module_summary(model, args.train_mode, args.caption_mode),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if args.train_mode != "unet_only":
        payload["adapter_state_dict"] = model.adapter.state_dict()
    if args.train_mode in {"unet_only", "adapter_unet"}:
        payload["lotus_unet_state_dict"] = model.lotus.unet.state_dict()
    if args.train_mode in {"caption_attn", "adapter_caption_attn", "adapter_full_caption_attn"}:
        payload["lotus_text_cross_attention_state_dict"] = text_cross_attention_state_dict(model, args.train_mode)
    return payload

def save_checkpoint(
    model: AnyThermalLotusModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    output_dir: Path,
    global_step: int,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
    fixed_val_ids: Sequence[str],
    name: str,
    *,
    include_optimizer: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = checkpoint_payload(
        model,
        optimizer,
        scheduler,
        global_step,
        epoch,
        best_val_loss,
        args,
        fixed_val_ids,
        include_optimizer=include_optimizer,
    )
    try:
        torch.save(payload, tmp_path)
        try:
            tmp_path.replace(path)
            return path
        except OSError as exc:
            fallback = path.with_name(
                f"{path.stem}_step_{global_step:06d}_{uuid.uuid4().hex[:8]}{path.suffix}"
            )
            tmp_path.replace(fallback)
            console_log(
                f"WARNING: could not replace checkpoint {path}: {exc}. "
                f"Saved fallback checkpoint to {fallback}."
            )
            return fallback
    except Exception as exc:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        finally:
            raise RuntimeError(
                f"Failed to save checkpoint {path}. Check disk space, output-dir permissions, "
                "and whether another process is writing the same checkpoint."
            ) from exc

def restore_rng_state(checkpoint: Dict[str, Any]) -> None:
    torch_state = checkpoint.get("torch_rng_state")
    if isinstance(torch_state, torch.Tensor):
        try:
            torch.set_rng_state(torch_state.detach().cpu().to(torch.uint8))
        except Exception as exc:
            console_log(f"WARNING: could not restore CPU RNG state: {exc}")
    elif torch_state is not None:
        console_log(f"WARNING: skipped CPU RNG state with unsupported type {type(torch_state).__name__}")

    cuda_states = checkpoint.get("cuda_rng_state_all")
    if torch.cuda.is_available() and cuda_states is not None:
        try:
            torch.cuda.set_rng_state_all([
                state.detach().cpu().to(torch.uint8) if isinstance(state, torch.Tensor) else state
                for state in cuda_states
            ])
        except Exception as exc:
            console_log(f"WARNING: could not restore CUDA RNG state: {exc}")

    if "numpy_rng_state" in checkpoint:
        try:
            np.random.set_state(checkpoint["numpy_rng_state"])
        except Exception as exc:
            console_log(f"WARNING: could not restore NumPy RNG state: {exc}")
    if "python_rng_state" in checkpoint:
        try:
            random.setstate(checkpoint["python_rng_state"])
        except Exception as exc:
            console_log(f"WARNING: could not restore Python RNG state: {exc}")

def verify_adapter_matches_checkpoint(model: AnyThermalLotusModel, checkpoint_adapter: Dict[str, torch.Tensor]) -> None:
    current = model.adapter.state_dict()
    missing = sorted(set(current) - set(checkpoint_adapter))
    unexpected = sorted(set(checkpoint_adapter) - set(current))
    if missing or unexpected:
        raise RuntimeError(f"Adapter checkpoint key mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    mismatched = []
    for key, value in current.items():
        loaded = checkpoint_adapter[key].to(device=value.device, dtype=value.dtype)
        if value.shape != loaded.shape or not torch.equal(value, loaded):
            mismatched.append(key)
            if len(mismatched) >= 5:
                break
    if mismatched:
        raise RuntimeError(f"Adapter parameters do not match loaded checkpoint: {mismatched}")


def load_adapter_initialization(path: Path, model: AnyThermalLotusModel, device: torch.device) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("adapter_state_dict", checkpoint)
    model.adapter.load_state_dict(state, strict=True)
    verify_adapter_matches_checkpoint(model, state)
    return {
        "path": str(path),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)) if isinstance(checkpoint, dict) else -1,
        "checkpoint_best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))) if isinstance(checkpoint, dict) else float("nan"),
        "loaded": True,
    }


def load_checkpoint(
    path: Path,
    model: AnyThermalLotusModel,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    device: torch.device,
) -> Tuple[int, int, float, float, List[str]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    conditioner_state = checkpoint.get("conditioner_state_dict", checkpoint.get("adapter_state_dict"))
    if conditioner_state is None:
        raise RuntimeError("Checkpoint has no conditioner_state_dict or adapter_state_dict")
    model.adapter.load_state_dict(conditioner_state, strict=True)
    if "lotus_unet_state_dict" in checkpoint:
        model.lotus.unet.load_state_dict(checkpoint["lotus_unet_state_dict"], strict=True)
    if "lotus_text_cross_attention_state_dict" in checkpoint:
        load_text_cross_attention_state_dict(model, checkpoint["lotus_text_cross_attention_state_dict"], device)
    if checkpoint.get("optimizer_state_dict") is None:
        raise RuntimeError(
            "This is a model-selection checkpoint without optimizer state; "
            "resume from checkpoint_latest.pt instead."
        )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    restore_rng_state(checkpoint)
    return (
        int(checkpoint.get("global_step", 0)),
        int(checkpoint.get("epoch", 0)),
        float(checkpoint.get("best_val_loss", float("inf"))),
        float(checkpoint.get("best_aligned_rmse", float("inf"))),
        list(checkpoint.get("fixed_val_ids", [])),
    )

def effective_adapter_lr(args: argparse.Namespace) -> float:
    return float(args.adapter_lr if args.adapter_lr is not None else args.learning_rate)


def make_optimizer(args: argparse.Namespace, model: AnyThermalLotusModel) -> torch.optim.Optimizer:
    groups: List[Dict[str, Any]] = []
    if args.train_mode in {"adapter_only", "adapter_unet", "adapter_caption_attn", "adapter_full_caption_attn"}:
        groups.append(
            {
                "params": [parameter for parameter in model.adapter.parameters() if parameter.requires_grad],
                "lr": effective_adapter_lr(args),
                "name": "adapter",
            }
        )
    if args.train_mode in {"unet_only", "adapter_unet"}:
        groups.append(
            {
                "params": [parameter for parameter in model.lotus.unet.parameters() if parameter.requires_grad],
                "lr": float(args.lotus_unet_lr),
                "name": "lotus_unet",
            }
        )
    if args.train_mode in {"caption_attn", "adapter_caption_attn", "adapter_full_caption_attn"}:
        groups.append(
            {
                "params": [parameter for parameter in text_cross_attention_parameters(model, args.train_mode) if parameter.requires_grad],
                "lr": float(args.text_attn_lr),
                "name": "text_cross_attention",
            }
        )
    for group in groups:
        if not group["params"]:
            raise RuntimeError(f"Optimizer group {group['name']} has no trainable parameters.")
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay)
def make_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "cosine":
        total = max(1, int(args.max_steps))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total)
    if args.lr_scheduler == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    raise ValueError(f"Unsupported lr scheduler: {args.lr_scheduler}")


def load_lotus(args: argparse.Namespace, device: torch.device) -> AnyThermalLotusModel:
    anythermal = AnyThermalEncoder(
        args.anythermal_model_path,
        device=str(device),
        revision=args.anythermal_revision,
        local_files_only=args.local_files_only,
    )
    lotus_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.lotus_model_path,
        subfolder="scheduler",
        local_files_only=args.local_files_only,
    )
    lotus = LotusGPipeline.from_pretrained(
        args.lotus_model_path,
        torch_dtype=lotus_dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    conditioner = (
        AnyThermalDirectConditioner().to(device)
        if args.train_mode == "unet_only"
        else AnyThermalLotusAdapter().to(device)
    )
    model = AnyThermalLotusModel(
        anythermal_encoder=anythermal,
        lotus_pipeline=lotus,
        adapter=conditioner,
        noise_scheduler=noise_scheduler,
        freeze_anythermal=True,
        freeze_lotus=True,
    ).to(device)
    configure_trainability(model, args.train_mode)
    return model

def startup_checks(
    args: argparse.Namespace,
    model: AnyThermalLotusModel,
    train_dataset: MS2AdapterDataset,
    val_dataset: MS2AdapterDataset,
    device: torch.device,
    output_dir: Path,
) -> None:
    if not Path(args.train_manifest).is_file():
        raise FileNotFoundError(args.train_manifest)
    if not Path(args.val_manifest).is_file():
        raise FileNotFoundError(args.val_manifest)
    train_ids = set(train_dataset.sample_ids)
    val_ids = set(val_dataset.sample_ids)
    overlap = train_ids.intersection(val_ids)
    if overlap:
        raise RuntimeError(f"Train/val IDs overlap; first overlap ID: {next(iter(overlap))}")

    for dataset_name, dataset in [("train", train_dataset), ("val", val_dataset)]:
        if len(dataset) == 0:
            raise RuntimeError(f"{dataset_name} dataset is empty")
        sample = dataset[0]
        if not Path(sample["thermal_path"]).is_file():
            raise FileNotFoundError(sample["thermal_path"])
        if not Path(sample["depth_path"]).is_file():
            raise FileNotFoundError(sample["depth_path"])
        if not torch.isfinite(sample["depth_values"]).all():
            raise RuntimeError(f"{dataset_name} depth has non-finite values: {sample['id']}")
        if int(sample["valid_mask"].sum()) <= 0:
            raise RuntimeError(f"{dataset_name} depth has no valid pixels: {sample['id']}")

    probe = collate_samples([train_dataset[0]])
    outputs = forward_batch(model, probe, device=device, global_step=0, seed=args.seed, timestep=args.timestep, return_decoded=False, caption_mode=args.caption_mode)
    features = extract_batch_features(model, probe["thermal_paths"], device=device)
    feature_shapes = [list(item.shape) for item in features]
    if feature_shapes != [[1, 768, 18, 45]] * 4:
        raise RuntimeError(f"Unexpected AnyThermal feature shapes: {feature_shapes}")
    adapter_shape = list(outputs["condition_latent"].shape)
    unet_shape = list(torch.cat([outputs["noisy_depth_latents"], outputs["condition_latent"]], dim=1).shape)
    if adapter_shape != [1, 4, 32, 80]:
        raise RuntimeError(f"Unexpected adapter output shape: {adapter_shape}")
    if unet_shape != [1, 8, 32, 80]:
        raise RuntimeError(f"Unexpected U-Net input shape: {unet_shape}")
    trainability_check(model, args.train_mode)

    startup_info = {
        "checks_passed": True,
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "train_manifest": str(args.train_manifest),
        "val_manifest": str(args.val_manifest),
        "ms2_root": str(args.ms2_root),
        "feature_shapes": feature_shapes,
        "conditioner_type": getattr(model.adapter, "route_name", "learned_adapter_v0"),
        "conditioner_output_shape": adapter_shape,
        "adapter_output_shape": adapter_shape,
        "lotus_unet_input_shape": unet_shape,
        "adapter_output_mean": float(outputs["condition_latent"].detach().float().mean().cpu()),
        "adapter_output_std": float(outputs["condition_latent"].detach().float().std(unbiased=False).cpu()),
        "module_summary": module_summary(model, args.train_mode, args.caption_mode),
    }
    with (output_dir / "startup_checks.json").open("w", encoding="utf-8") as handle:
        json.dump(startup_info, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    if args.resume and args.init_adapter_checkpoint:
        raise ValueError("Use either --resume or --init-adapter-checkpoint, not both.")
    if args.train_mode == "unet_only" and args.init_adapter_checkpoint:
        raise ValueError("unet_only uses the zero-parameter Direct conditioner and cannot load an Adapter checkpoint.")
    if args.train_mode in {"caption_attn", "adapter_caption_attn", "adapter_full_caption_attn"} and not args.resume and not args.init_adapter_checkpoint:
        raise ValueError("Caption experiments must initialize Adapter V0 from --init-adapter-checkpoint.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        if device.index is not None:
            torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    train_dataset = MS2AdapterDataset(
        manifest=Path(args.train_manifest),
        ms2_root=Path(args.ms2_root),
        max_samples=args.max_train_samples,
    )
    val_dataset = MS2AdapterDataset(
        manifest=Path(args.val_manifest),
        ms2_root=Path(args.ms2_root),
        max_samples=args.max_val_samples,
    )

    fixed_val_ids_path = output_dir / "fixed_val_samples.json"
    if args.fixed_val_samples_json:
        fixed_val_ids = json.loads(Path(args.fixed_val_samples_json).read_text(encoding="utf-8"))
        fixed_val_ids_path.write_text(json.dumps(fixed_val_ids, indent=2, ensure_ascii=False), encoding="utf-8")
    elif args.resume and fixed_val_ids_path.is_file():
        fixed_val_ids = json.loads(fixed_val_ids_path.read_text(encoding="utf-8"))
    else:
        fixed_val_ids = val_dataset.sample_ids[: min(args.num_fixed_val_samples, len(val_dataset))]
        fixed_val_ids_path.write_text(json.dumps(fixed_val_ids, indent=2, ensure_ascii=False), encoding="utf-8")
    missing_fixed_val_ids = [sample_id for sample_id in fixed_val_ids if sample_id not in val_dataset.sample_ids]
    if missing_fixed_val_ids:
        raise ValueError(
            "Fixed validation sample IDs are not present in the loaded val dataset. "
            f"First missing IDs: {missing_fixed_val_ids[:5]}. "
            "Increase --max-val-samples or use a matching --fixed-val-samples-json."
        )
    console_log(f"Loading models on {device}...")

    model = load_lotus(args, device)
    init_adapter_info = None
    if args.init_adapter_checkpoint and not args.resume:
        init_adapter_info = load_adapter_initialization(Path(args.init_adapter_checkpoint), model, device)
        console_log(
            "Adapter initialized from Adapter-only checkpoint: "
            f"{init_adapter_info['path']} step={init_adapter_info['checkpoint_global_step']}"
        )
    if args.train_mode == "unet_only":
        console_log("Using the zero-parameter Direct conditioner; only Lotus U-Net is trainable.")
    if args.train_mode == "adapter_unet":
        console_log("Lotus U-Net initialized from pretrained Lotus-G and set trainable.")
        console_log("Optimizer initialized for joint training with fresh parameter groups.")
    set_training_modes(model, args.train_mode)

    console_log("Models loaded. Creating optimizer and scheduler...")
    optimizer = make_optimizer(args, model)
    scheduler = make_scheduler(args, optimizer)

    start_step = 0
    start_epoch = 0
    best_val_loss = float("inf")
    best_aligned_rmse = float("inf")
    args.best_aligned_rmse = best_aligned_rmse
    if args.resume:
        start_step, start_epoch, best_val_loss, best_aligned_rmse, checkpoint_fixed_ids = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, device
        )
        args.best_aligned_rmse = best_aligned_rmse
        if checkpoint_fixed_ids:
            fixed_val_ids = checkpoint_fixed_ids
            fixed_val_ids_path.write_text(json.dumps(fixed_val_ids, indent=2, ensure_ascii=False), encoding="utf-8")

    console_log("Running startup checks...")

    startup_checks(args, model, train_dataset, val_dataset, device, output_dir)
    save_trainable_modules(model, output_dir, args.train_mode)

    console_log("Startup checks passed. Beginning training loop...")
    save_steps = parse_step_set(args.save_steps)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"args": vars(args), "module_summary": module_summary(model, args.train_mode, args.caption_mode), "init_adapter_info": init_adapter_info, "save_steps": sorted(save_steps)}, handle, indent=2, ensure_ascii=False)


    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=collate_samples,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_samples,
        persistent_workers=args.num_workers > 0,
    )

    global_step = start_step
    epoch = start_epoch
    train_metrics_path = output_dir / "train_metrics.jsonl"
    val_metrics_path = output_dir / "val_metrics.jsonl"
    if global_step == 0 and train_metrics_path.exists() and not args.append_logs:
        train_metrics_path.unlink()
    if global_step == 0 and val_metrics_path.exists() and not args.append_logs:
        val_metrics_path.unlink()

    set_training_modes(model, args.train_mode)
    optimizer.zero_grad(set_to_none=True)
    if 0 in save_steps and start_step == 0:
        console_log("Saving requested step 0 checkpoint and validation visuals...")
        save_val_visuals(model, val_dataset, fixed_val_ids, output_dir, 0, device, args.seed + 20_000, args.timestep, caption_mode=args.caption_mode)
        if args.save_caption_comparisons:
            save_caption_comparison_visuals(model, val_dataset, fixed_val_ids, output_dir, 0, device, args.seed + 30_000, args.timestep)
        save_checkpoint(model, optimizer, scheduler, output_dir, 0, epoch, best_val_loss, args, fixed_val_ids, "checkpoint_step_000000.pt")
        set_training_modes(model, args.train_mode)
    running_loss = 0.0
    micro_step = 0
    stop_training = False
    train_progress = None
    if not args.disable_progress and tqdm is not None:
        train_progress = tqdm(
            total=args.max_steps,
            initial=global_step,
            desc="train",
            dynamic_ncols=True,
        )
    else:
        console_log(f"Training progress: starting at step {global_step}/{args.max_steps}")

    while global_step < args.max_steps and not stop_training and epoch < args.num_epochs:
        epoch += 1
        for batch in train_loader:
            if global_step >= args.max_steps:
                stop_training = True
                break
            step_start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            outputs = forward_batch(model, batch, device=device, global_step=global_step + 1, seed=args.seed, timestep=args.timestep, return_decoded=False, caption_mode=args.caption_mode, caption_dropout_prob=args.caption_dropout_prob)
            loss = outputs["loss"] / max(1, args.gradient_accumulation_steps)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {global_step}: {float(loss.detach().cpu())}")
            loss.backward()
            running_loss += float(loss.detach().cpu())
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps == 0:
                adapter_grad_norm = grad_norm(model.adapter.parameters())
                lotus_unet_grad_norm = grad_norm(model.lotus.unet.parameters())
                text_attn_grad_norm = grad_norm(text_cross_attention_parameters(model, args.train_mode))

                adapter_should_train = args.train_mode in {"adapter_only", "adapter_unet", "adapter_caption_attn", "adapter_full_caption_attn"}
                text_attn_should_train = args.train_mode in {"caption_attn", "adapter_caption_attn", "adapter_full_caption_attn"}
                if adapter_should_train and not module_has_grad(model.adapter):
                    raise RuntimeError("Adapter did not receive gradients")
                if not adapter_should_train and module_has_grad(model.adapter):
                    raise RuntimeError("Frozen Adapter received gradients")
                if args.train_mode in {"unet_only", "adapter_unet"} and not module_has_grad(model.lotus.unet):
                    raise RuntimeError("Trainable Lotus U-Net did not receive gradients")
                if args.train_mode == "adapter_only" and module_has_grad(model.lotus.unet):
                    raise RuntimeError("Frozen Lotus U-Net received gradients")
                if text_attn_should_train:
                    if text_attn_grad_norm <= 0.0:
                        raise RuntimeError("Text cross-attention did not receive gradients")
                    unexpected_unet_grad = module_has_unexpected_unet_grad(model, args.train_mode)
                    if unexpected_unet_grad:
                        raise RuntimeError(f"Frozen non-text U-Net parameters received gradients: {unexpected_unet_grad[:20]}")
                if module_has_grad(model.anythermal_encoder.model):
                    raise RuntimeError("Frozen AnyThermal received gradients")
                if module_has_grad(model.lotus.vae):
                    raise RuntimeError("Frozen VAE received gradients")
                if module_has_grad(model.lotus.text_encoder):
                    raise RuntimeError("Frozen text encoder received gradients")
                if adapter_should_train and not np.isfinite(adapter_grad_norm):
                    raise RuntimeError("Adapter gradient norm is not finite")
                if args.train_mode in {"unet_only", "adapter_unet"} and not np.isfinite(lotus_unet_grad_norm):
                    raise RuntimeError("Lotus U-Net gradient norm is not finite")
                if text_attn_should_train and not np.isfinite(text_attn_grad_norm):
                    raise RuntimeError("Text cross-attention gradient norm is not finite")

                torch.nn.utils.clip_grad_norm_(trainable_parameters_for_mode(model, args.train_mode), args.max_grad_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                adapter = outputs["condition_latent"].detach().float()
                lr_by_group = {group.get("name", f"group_{idx}"): float(group["lr"]) for idx, group in enumerate(optimizer.param_groups)}
                lr = lr_by_group.get("adapter", lr_by_group.get("text_cross_attention", float(optimizer.param_groups[0]["lr"])))
                record = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "train_loss": running_loss,
                    "learning_rate": lr,
                    "adapter_lr": lr_by_group.get("adapter", 0.0),
                    "lotus_unet_lr": lr_by_group.get("lotus_unet", 0.0),
                    "text_attn_lr": lr_by_group.get("text_cross_attention", 0.0),
                    "adapter_grad_norm": float(adapter_grad_norm),
                    "lotus_unet_grad_norm": float(lotus_unet_grad_norm),
                    "text_attn_grad_norm": float(text_attn_grad_norm),
                    "caption_mode": args.caption_mode,
                    "caption_dropout_prob": args.caption_dropout_prob,
                    "caption_num_empty": int(outputs.get("caption_info", {}).get("num_empty", 0)),
                    "caption_num_real": int(outputs.get("caption_info", {}).get("num_real", 0)),
                    "conditioner_type": getattr(model.adapter, "route_name", "learned_adapter_v0"),
                    "conditioner_output_mean": float(adapter.mean().cpu()),
                    "conditioner_output_std": float(adapter.std(unbiased=False).cpu()),
                    "adapter_output_mean": float(adapter.mean().cpu()),
                    "adapter_output_std": float(adapter.std(unbiased=False).cpu()),
                    "step_time_sec": float(time.perf_counter() - step_start),
                    "gpu_memory_mb": gpu_memory_mb(device),
                }
                write_jsonl(train_metrics_path, record)
                if train_progress is not None:
                    train_progress.update(1)
                    train_progress.set_postfix(
                        loss=f"{record['train_loss']:.4f}",
                        lr=f"{lr:.2e}",
                        grad=f"{max(record['adapter_grad_norm'], record['text_attn_grad_norm'], record['lotus_unet_grad_norm']):.3f}",
                    )
                else:
                    console_log(
                        f"step {global_step}/{args.max_steps} "
                        f"loss={record['train_loss']:.6f} "
                        f"lr={lr:.2e} adapter_grad={record['adapter_grad_norm']:.4f} "
                        f"text_grad={record['text_attn_grad_norm']:.4f} "
                        f"unet_grad={record['lotus_unet_grad_norm']:.4f} "
                        f"time={record['step_time_sec']:.2f}s"
                    )
                running_loss = 0.0
                should_validate = args.validation_interval > 0 and (global_step == 1 or global_step % args.validation_interval == 0)
                should_checkpoint = global_step in save_steps or (args.checkpoint_interval > 0 and (global_step == 1 or global_step % args.checkpoint_interval == 0))
                should_visualize = global_step in save_steps or (args.visualization_interval > 0 and (global_step == 1 or global_step % args.visualization_interval == 0))

                if should_validate:
                    console_log(f"[step {global_step}] validation start...")
                    metrics = validate(
                        model=model,
                        loader=val_loader,
                        device=device,
                        max_batches=args.val_batch_limit,
                        seed=args.seed + 10_000 + global_step,
                        timestep=args.timestep,
                        progress_desc=f"val@{global_step}",
                        disable_progress=args.disable_progress,
                        caption_mode=args.caption_mode,
                    )
                    metrics.update({"global_step": global_step, "epoch": epoch})
                    write_jsonl(val_metrics_path, metrics)
                    val_loss = float(metrics.get("val_loss", float("inf")))
                    aligned_rmse = float(metrics.get("aligned_rmse", float("inf")))
                    console_log(f"[step {global_step}] validation done: val_loss={val_loss:.6f} aligned_rmse={aligned_rmse:.6f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        args.best_aligned_rmse = best_aligned_rmse
                        save_checkpoint(
                            model, optimizer, scheduler, output_dir, global_step, epoch, best_val_loss, args,
                            fixed_val_ids, "checkpoint_best_val_loss.pt", include_optimizer=False
                        )
                        if args.train_mode not in {"unet_only", "adapter_unet"}:
                            save_checkpoint(
                                model, optimizer, scheduler, output_dir, global_step, epoch, best_val_loss, args,
                                fixed_val_ids, "checkpoint_best.pt", include_optimizer=False
                            )
                    if aligned_rmse < best_aligned_rmse:
                        best_aligned_rmse = aligned_rmse
                        args.best_aligned_rmse = best_aligned_rmse
                        save_checkpoint(
                            model, optimizer, scheduler, output_dir, global_step, epoch, best_val_loss, args,
                            fixed_val_ids, "checkpoint_best_aligned_rmse.pt", include_optimizer=False
                        )
                        if args.train_mode not in {"unet_only", "adapter_unet"}:
                            save_checkpoint(
                                model, optimizer, scheduler, output_dir, global_step, epoch, best_val_loss, args,
                                fixed_val_ids, "checkpoint_best_real_aligned_rmse.pt", include_optimizer=False
                            )
                    set_training_modes(model, args.train_mode)

                if should_visualize:
                    console_log(f"[step {global_step}] saving validation visuals...")
                    save_val_visuals(model, val_dataset, fixed_val_ids, output_dir, global_step, device, args.seed + 20_000 + global_step, args.timestep, caption_mode=args.caption_mode)
                    if args.save_caption_comparisons:
                        save_caption_comparison_visuals(model, val_dataset, fixed_val_ids, output_dir, global_step, device, args.seed + 30_000 + global_step, args.timestep)
                    set_training_modes(model, args.train_mode)

                if should_checkpoint:
                    if args.train_mode not in {"unet_only", "adapter_unet"}:
                        save_checkpoint(
                            model, optimizer, scheduler, output_dir, global_step, epoch, best_val_loss, args,
                            fixed_val_ids, f"checkpoint_step_{global_step:06d}.pt"
                        )
                    save_checkpoint(
                        model, optimizer, scheduler, output_dir, global_step, epoch, best_val_loss, args,
                        fixed_val_ids, "checkpoint_latest.pt"
                    )
                if global_step >= args.max_steps:
                    stop_training = True
                    break

    if train_progress is not None:
        train_progress.close()

    final_checkpoint = None
    if global_step > 0:
        final_checkpoint = save_checkpoint(
            model,
            optimizer,
            scheduler,
            output_dir,
            global_step,
            epoch,
            best_val_loss,
            args,
            fixed_val_ids,
            "checkpoint_final.pt",
            include_optimizer=False,
        )

    final_module_summary = module_summary(model, args.train_mode, args.caption_mode)
    final_summary = {
        "experiment_name": final_module_summary["experiment_name"],
        "global_step": global_step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "best_aligned_rmse": best_aligned_rmse,
        "output_dir": str(output_dir),
        "final_checkpoint": None if final_checkpoint is None else str(final_checkpoint),
        "module_summary": final_module_summary,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(final_summary, handle, indent=2, ensure_ascii=False)
    print(json.dumps(final_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()



























