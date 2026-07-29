"""One-batch forward/backward smoke gate for Adapter V2 distillation.

This command performs no optimizer step and no training loop.  It verifies the
gradient owner, finite tensors, exact latent shape and thermal conversion for a
single MS2 Train sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.anythermal_encoder import AnyThermalEncoder  # noqa: E402
from models.anythermal_lotus_adapter_v2 import AnyThermalLotusAdapterV2  # noqa: E402
from models.anythermal_lotus_model import extract_anythermal_feature_pyramid  # noqa: E402
from models.anythermal_lotus_v2 import (  # noqa: E402
    condition_distillation_losses,
    encode_condition_latent,
    distill_condition_latent,
    thermal_to_lotus_input,
)


DEFAULT_MANIFEST = Path(
    "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/"
    "ms2_fixed_sequence_internvl3_8b_filtered_caption_train.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ms2-root", type=Path, default=Path("/mnt/e/dataset/ms2"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lotus_line_v2/smoke_1batch_v2_1_raw_mode"),
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--processing-res", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vae-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_train_sample(path: Path, index: int) -> Dict[str, Any]:
    if index < 0:
        raise ValueError("sample-index must be non-negative.")
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index != index:
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError(f"Selected row is not Train: {row.get('id')}")
            depth_path = row.get("thermal_depth_path") or row.get("depth_path")
            if not depth_path or "/thr/" not in str(depth_path).replace("\\", "/"):
                raise ValueError("Selected row does not use thermal-view GT.")
            return {
                "id": str(row["id"]),
                "thermal_path": str(row["thermal_path"]),
                "depth_path": str(depth_path),
                "split": "train",
                "manifest_index": index,
            }
    raise IndexError(f"sample-index {index} is outside the Train manifest.")


def parameter_audit(module: torch.nn.Module) -> Dict[str, Any]:
    parameters = list(module.parameters())
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    return {
        "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_count": int(
            sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        ),
        "gradient_tensor_count": len(gradients),
        "has_gradient": bool(gradients),
        "gradients_finite": bool(
            all(torch.isfinite(gradient).all() for gradient in gradients)
        ),
        "gradient_l2": float(
            torch.sqrt(
                sum((gradient.detach().float() ** 2).sum() for gradient in gradients)
            ).cpu()
        )
        if gradients
        else 0.0,
        "gradient_max_abs": float(
            max(gradient.detach().float().abs().max() for gradient in gradients).cpu()
        )
        if gradients
        else 0.0,
    }


def tensor_summary(tensor: torch.Tensor) -> Dict[str, Any]:
    values = tensor.detach().float().cpu()
    return {
        "shape": list(values.shape),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "finite": bool(torch.isfinite(values).all()),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = read_train_sample(args.train_manifest.resolve(), args.sample_index)
    thermal_path = args.ms2_root.resolve() / sample["thermal_path"]
    if not thermal_path.is_file():
        raise FileNotFoundError(f"Thermal image not found: {thermal_path}")

    from diffusers import AutoencoderKL

    vae_dtype = torch.float16 if args.vae_dtype == "fp16" else torch.float32
    vae = AutoencoderKL.from_pretrained(
        args.lotus_model_path,
        subfolder="vae",
        torch_dtype=vae_dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    vae.requires_grad_(False).eval()

    anythermal = AnyThermalEncoder(
        model_path=args.anythermal_model_path,
        device=str(device),
        local_files_only=args.local_files_only,
    )
    anythermal.model.requires_grad_(False).eval()
    adapter = AnyThermalLotusAdapterV2().to(device=device, dtype=torch.float32).train()
    adapter.requires_grad_(True)

    lotus_input = thermal_to_lotus_input(
        thermal_path,
        processing_res=args.processing_res,
    )
    teacher = encode_condition_latent(
        vae,
        lotus_input.tensor,
        posterior="mode",
    ).to(device=device, dtype=torch.float32)
    features, feature_info, anythermal_diagnostics = extract_anythermal_feature_pyramid(
        anythermal,
        thermal_path,
        enable_grad=False,
    )
    features = [feature.to(device=device, dtype=torch.float32) for feature in features]

    if anythermal_diagnostics["converted_uint8_min"] != lotus_input.diagnostics["converted_uint8_min"]:
        raise RuntimeError("AnyThermal and Lotus paths disagree on converted thermal minimum.")
    if anythermal_diagnostics["converted_uint8_max"] != lotus_input.diagnostics["converted_uint8_max"]:
        raise RuntimeError("AnyThermal and Lotus paths disagree on converted thermal maximum.")
    if abs(
        anythermal_diagnostics["converted_uint8_std"]
        - lotus_input.diagnostics["converted_uint8_std"]
    ) > 1e-6:
        raise RuntimeError("AnyThermal and Lotus paths disagree on converted thermal std.")

    result = distill_condition_latent(adapter, features, teacher)
    losses = condition_distillation_losses(result.prediction, result.target)
    if not bool(torch.isfinite(losses["total"])):
        raise RuntimeError("Distillation loss is NaN or Inf.")
    losses["total"].backward()

    adapter_audit = parameter_audit(adapter)
    anythermal_audit = parameter_audit(anythermal.model)
    vae_audit = parameter_audit(vae)
    if not adapter_audit["has_gradient"] or adapter_audit["gradient_l2"] <= 0:
        raise RuntimeError("Adapter did not receive a non-zero gradient.")
    if not adapter_audit["gradients_finite"]:
        raise RuntimeError("Adapter gradient contains NaN or Inf.")
    if anythermal_audit["has_gradient"] or vae_audit["has_gradient"]:
        raise RuntimeError("A frozen module unexpectedly received gradients.")

    summary = {
        "phase": "Adapter V2 one-batch condition-distillation smoke gate",
        "training": False,
        "optimizer_constructed": False,
        "optimizer_step": False,
        "unet_loaded": False,
        "unet_reason": "U-Net is outside the selected condition-distillation objective",
        "sample": {**sample, "thermal_path": str(thermal_path)},
        "seed": args.seed,
        "teacher_posterior": "mode",
        "thermal": lotus_input.diagnostics,
        "anythermal_thermal_diagnostics": anythermal_diagnostics,
        "feature_info": {
            "hidden_state_indices": list(feature_info.hidden_state_indices),
            "transformer_block_indices": list(feature_info.transformer_block_indices),
            "grid_size": list(feature_info.grid_size),
            "preprocessed_shape": list(feature_info.preprocessed_shape),
        },
        "features": [tensor_summary(feature) for feature in features],
        "teacher": tensor_summary(result.target),
        "prediction": tensor_summary(result.prediction),
        "distillation": result.diagnostics,
        "composite_loss": {
            key: float(value.detach().cpu()) for key, value in losses.items()
        },
        "gradient_owners": {
            "adapter": adapter_audit,
            "anythermal": anythermal_audit,
            "vae": vae_audit,
        },
        "gate_passed": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "gate_passed": True,
                "sample": sample["id"],
                "teacher_shape": summary["teacher"]["shape"],
                "prediction_shape": summary["prediction"]["shape"],
                "distillation": summary["distillation"],
                "composite_loss": summary["composite_loss"],
                "gradient_owners": summary["gradient_owners"],
            },
            indent=2,
        )
    )
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
