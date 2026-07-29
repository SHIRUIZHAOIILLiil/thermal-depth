"""Train the V2 U-Net-only route on 128 Train frames with 16 held out.

Teacher responses are cached once from Thermal-VAE mode condition through the
original pretrained Lotus U-Net.  The same U-Net is then trained as the student
using zero-parameter Direct AnyThermal condition.  No depth GT, Val/Test, V1
checkpoint, Adapter parameters, caption, or Thermal-VAE inference path is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Dict, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = ROOT / "lotus"
for path in (ROOT, LOTUS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.anythermal_encoder import AnyThermalEncoder  # noqa: E402
from models.anythermal_lotus_conditioner import AnyThermalDirectConditioner  # noqa: E402
from models.anythermal_lotus_model import extract_anythermal_feature_pyramid  # noqa: E402
from models.anythermal_lotus_v2 import encode_condition_latent, thermal_to_lotus_input  # noqa: E402
from models.anythermal_lotus_v2_4 import response_consistency_losses, seeded_noise  # noqa: E402
from tools.overfit_32_adapter_v2_distillation import batch_indices  # noqa: E402
from tools.short_128_adapter_v2_4_response_holdout import (  # noqa: E402
    read_role_manifest,
    read_source_indices,
    task_embedding,
)


DEFAULT_PROTOCOL_DIR = Path("outputs/lotus_line_v2/short_128_v2_3_holdout16")
DEFAULT_SOURCE_MANIFEST = Path(
    "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/"
    "ms2_fixed_sequence_internvl3_8b_filtered_caption_train.jsonl"
)
CHECKPOINT_FORMAT = "unet_only_v2_dense_teacher_response"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=DEFAULT_PROTOCOL_DIR / "train_128_manifest.jsonl",
    )
    parser.add_argument(
        "--holdout-manifest",
        type=Path,
        default=DEFAULT_PROTOCOL_DIR / "holdout_16_manifest.jsonl",
    )
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--ms2-root", type=Path, default=Path("/mnt/e/dataset/ms2"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lotus_line_v2/short_128_unet_only_v2_holdout16"),
    )
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--optimizer-steps", type=int, default=300)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--response-cosine-loss-weight", type=float, default=0.1)
    parser.add_argument("--response-spatial-gradient-weight", type=float, default=0.5)
    parser.add_argument("--response-multiscale-gradient-weight", type=float, default=0.5)
    parser.add_argument("--response-gradient-energy-weight", type=float, default=0.5)
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--student-dtype", choices=("fp32",), default="fp32")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_protocol(args) -> None:
    if args.micro_batch_size != 1 or args.gradient_accumulation_steps != 4:
        raise ValueError("U-Net-only V2 requires micro-batch 1 and accumulation 4.")
    if args.timestep != 999:
        raise ValueError("U-Net-only V2 must match inference timestep 999.")
    if args.optimizer_steps <= 0 or args.log_interval <= 0:
        raise ValueError("optimizer-steps and log-interval must be positive.")
    if min(
        args.learning_rate,
        args.max_grad_norm,
        args.response_cosine_loss_weight,
        args.response_spatial_gradient_weight,
        args.response_multiscale_gradient_weight,
        args.response_gradient_energy_weight,
    ) < 0:
        raise ValueError("Learning rate, clipping, and loss weights must be non-negative.")


def validate_trainability(direct, unet, frozen_modules: Sequence[torch.nn.Module]) -> None:
    if sum(parameter.numel() for parameter in direct.parameters()) != 0:
        raise RuntimeError("U-Net-only Direct conditioner must have zero parameters.")
    if not any(parameter.requires_grad for parameter in unet.parameters()):
        raise RuntimeError("Student U-Net has no trainable parameters.")
    for module in frozen_modules:
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError("A frozen U-Net-only support module is trainable.")


def cache_teacher_samples(args, rows, device, teacher_dtype):
    from pipeline import LotusGPipeline

    lotus = LotusGPipeline.from_pretrained(
        args.lotus_model_path,
        torch_dtype=teacher_dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    anythermal = AnyThermalEncoder(
        model_path=args.anythermal_model_path,
        device=str(device),
        local_files_only=args.local_files_only,
    )
    direct = AnyThermalDirectConditioner().to(device=device, dtype=torch.float32)
    for module in (lotus.vae, lotus.text_encoder, lotus.unet, anythermal.model, direct):
        module.requires_grad_(False).eval()
    prompt, _ = lotus.encode_prompt(
        prompt="",
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=None,
    )
    prompt = prompt.detach()
    cache = []
    for position, row in enumerate(rows):
        thermal_path = args.ms2_root.resolve() / row["thermal_path"]
        thermal = thermal_to_lotus_input(thermal_path, processing_res=0)
        if thermal.diagnostics["converted_uint8_std"] <= 0:
            raise RuntimeError(f"Constant thermal conversion: {row['id']}")
        teacher_condition = encode_condition_latent(
            lotus.vae, thermal.tensor, posterior="mode"
        ).to(device=device, dtype=teacher_dtype)
        features, _, diagnostics = extract_anythermal_feature_pyramid(
            anythermal, thermal_path, enable_grad=False
        )
        direct_condition = direct(
            [feature.to(device=device, dtype=torch.float32) for feature in features],
            target_size=tuple(teacher_condition.shape[-2:]),
        )
        sample_seed = args.seed + int(row["manifest_index"])
        noise = seeded_noise(
            teacher_condition.shape,
            seed=sample_seed,
            device=device,
            dtype=teacher_dtype,
            scale=float(lotus.scheduler.init_noise_sigma),
        )
        timestep = torch.full((1,), args.timestep, device=device, dtype=torch.long)
        latent_input = lotus.scheduler.scale_model_input(noise, timestep)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=teacher_dtype,
            enabled=device.type == "cuda" and teacher_dtype == torch.float16,
        ):
            teacher_response = lotus.unet(
                torch.cat([teacher_condition, latent_input], dim=1),
                timestep,
                encoder_hidden_states=prompt.to(dtype=teacher_dtype),
                class_labels=task_embedding(1, device, teacher_dtype),
                return_dict=False,
            )[0]
        cache.append(
            {
                **row,
                "direct_condition": direct_condition.detach().float().cpu(),
                "latent_input": latent_input.detach().float().cpu(),
                "teacher_response": teacher_response.detach().float().cpu(),
                "sample_seed": sample_seed,
                "thermal_std": float(diagnostics["converted_uint8_std"]),
            }
        )
        print(f"cached {position + 1:03d}/{len(rows)} {row['role']} {row['id']}", flush=True)

    lotus.vae.to("cpu")
    lotus.text_encoder.to("cpu")
    anythermal.model.to("cpu")
    lotus.unet.to(device=device, dtype=torch.float32).train().requires_grad_(True)
    prompt = prompt.to(device=device, dtype=torch.float32)
    validate_trainability(
        direct,
        lotus.unet,
        (lotus.vae, lotus.text_encoder, anythermal.model),
    )
    del anythermal
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cache, lotus, prompt, direct


def stack_microbatch(cache, index: int, device: torch.device):
    item = cache[index]
    return (
        item["direct_condition"].to(device=device, dtype=torch.float32),
        item["latent_input"].to(device=device, dtype=torch.float32),
        item["teacher_response"].to(device=device, dtype=torch.float32),
    )


def forward_losses(unet, prompt, batch, args, device):
    direct_condition, latent_input, teacher_response = batch
    timestep = torch.full(
        (direct_condition.shape[0],), args.timestep, device=device, dtype=torch.long
    )
    student_response = unet(
        torch.cat([direct_condition, latent_input], dim=1),
        timestep,
        encoder_hidden_states=prompt.repeat(direct_condition.shape[0], 1, 1),
        class_labels=task_embedding(direct_condition.shape[0], device, torch.float32),
        return_dict=False,
    )[0]
    return response_consistency_losses(
        student_response,
        teacher_response,
        cosine_weight=args.response_cosine_loss_weight,
        spatial_gradient_weight=args.response_spatial_gradient_weight,
        multiscale_gradient_weight=args.response_multiscale_gradient_weight,
        gradient_energy_weight=args.response_gradient_energy_weight,
    )


@torch.no_grad()
def evaluate(unet, prompt, cache, args, device):
    was_training = unet.training
    unet.eval()
    keys = (
        "total",
        "mse",
        "cosine",
        "spatial_gradient_loss",
        "multiscale_gradient_loss",
        "gradient_energy_loss",
        "gradient_energy_ratio",
    )
    sums = {key: 0.0 for key in keys}
    for index in range(len(cache)):
        losses = forward_losses(unet, prompt, stack_microbatch(cache, index, device), args, device)
        for key in keys:
            sums[key] += float(losses[key])
    if was_training:
        unet.train()
    return {key: value / len(cache) for key, value in sums.items()}


def evaluation_record(step, train_metrics, holdout_metrics):
    return {
        "step": step,
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"holdout_{key}": value for key, value in holdout_metrics.items()},
    }


def gate(initial, final, *, holdout: bool):
    mse_ratio = final["mse"] / initial["mse"]
    ratio_limit = 0.9 if holdout else 0.7
    cosine_minimum = 0.8 if holdout else 0.9
    passed = bool(
        math.isfinite(mse_ratio)
        and mse_ratio <= ratio_limit
        and final["cosine"] >= cosine_minimum
        and 0.5 <= final["gradient_energy_ratio"] <= 1.5
    )
    return {
        "mse_ratio": mse_ratio,
        "mse_ratio_limit": ratio_limit,
        "cosine_minimum": cosine_minimum,
        "gradient_energy_ratio_range": [0.5, 1.5],
        "passed": passed,
    }


def save_checkpoint(path, unet, args, step):
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "train_mode": "unet_only",
            "adapter_architecture": "direct_zero_parameter",
            "global_step": step,
            "settings": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "adapter": {},
            "lotus_unet_state_dict": {
                key: value.detach().cpu() for key, value in unet.state_dict().items()
            },
        },
        path,
    )


def main() -> None:
    args = parse_args()
    validate_protocol(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    teacher_dtype = torch.float16 if args.teacher_dtype == "fp16" else torch.float32
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    source_indices = read_source_indices(args.source_manifest.resolve())
    train_rows = read_role_manifest(args.train_manifest.resolve(), "train", source_indices)
    holdout_rows = read_role_manifest(args.holdout_manifest.resolve(), "holdout", source_indices)
    if len(train_rows) != 128 or len(holdout_rows) != 16:
        raise RuntimeError(
            f"Frozen protocol requires 128/16 rows, got {len(train_rows)}/{len(holdout_rows)}."
        )
    cache, lotus, prompt, direct = cache_teacher_samples(
        args, train_rows + holdout_rows, device, teacher_dtype
    )
    train_cache = cache[:128]
    holdout_cache = cache[128:]
    optimizer = torch.optim.AdamW(
        lotus.unet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    initial_train = evaluate(lotus.unet, prompt, train_cache, args, device)
    initial_holdout = evaluate(lotus.unet, prompt, holdout_cache, args, device)
    history = [evaluation_record(0, initial_train, initial_holdout)]
    print(json.dumps(history[-1]), flush=True)

    for step in range(1, args.optimizer_steps + 1):
        selected = batch_indices(
            step - 1,
            len(train_cache),
            args.gradient_accumulation_steps,
            args.seed,
        )
        optimizer.zero_grad(set_to_none=True)
        metric_sums: Dict[str, float] = {}
        for index in selected:
            losses = forward_losses(
                lotus.unet,
                prompt,
                stack_microbatch(train_cache, index, device),
                args,
                device,
            )
            (losses["total"] / len(selected)).backward()
            for key, value in losses.items():
                if value.ndim == 0:
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value.detach())
        if any(parameter.grad is not None for parameter in direct.parameters()):
            raise RuntimeError("Zero-parameter Direct conditioner unexpectedly owns gradients.")
        grad_norm = torch.nn.utils.clip_grad_norm_(lotus.unet.parameters(), args.max_grad_norm)
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError(f"Non-finite U-Net gradient at step {step}.")
        optimizer.step()
        if step % args.log_interval == 0 or step == args.optimizer_steps:
            train_metrics = evaluate(lotus.unet, prompt, train_cache, args, device)
            holdout_metrics = evaluate(lotus.unet, prompt, holdout_cache, args, device)
            history.append(evaluation_record(step, train_metrics, holdout_metrics))
            print(json.dumps(history[-1]), flush=True)

    final_train = evaluate(lotus.unet, prompt, train_cache, args, device)
    final_holdout = evaluate(lotus.unet, prompt, holdout_cache, args, device)
    train_gate = gate(initial_train, final_train, holdout=False)
    holdout_gate = gate(initial_holdout, final_holdout, holdout=True)
    gate_passed = bool(train_gate["passed"] and holdout_gate["passed"])
    checkpoint_path = output / f"checkpoint_step_{args.optimizer_steps:04d}.pt"
    save_checkpoint(checkpoint_path, lotus.unet, args, args.optimizer_steps)
    with (output / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "phase": "V2 U-Net-only dense teacher response short gate",
        "metric_scope": "Train-only diagnostics; official depth quality remains separate",
        "route": "Direct AnyThermal condition + trained Lotus U-Net",
        "teacher_cache": "Thermal-VAE mode + original pretrained frozen Lotus U-Net",
        "source_split": "Train 128 + frozen Train holdout 16",
        "uses_depth_gt": False,
        "uses_val": False,
        "uses_test": False,
        "uses_v1_checkpoint": False,
        "inference_uses_thermal_vae": False,
        "conditioner_parameters": 0,
        "unet_trainable_parameters": int(
            sum(parameter.numel() for parameter in lotus.unet.parameters() if parameter.requires_grad)
        ),
        "source_manifest_sha256": sha256(args.source_manifest.resolve()),
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "initial_train": initial_train,
        "final_train": final_train,
        "initial_holdout": initial_holdout,
        "final_holdout": final_holdout,
        "train_gate": train_gate,
        "holdout_gate": holdout_gate,
        "gate_passed": gate_passed,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "train_ids": [item["id"] for item in train_cache],
        "holdout_ids": [item["id"] for item in holdout_cache],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "initial_train": initial_train,
                "final_train": final_train,
                "initial_holdout": initial_holdout,
                "final_holdout": final_holdout,
                "train_gate": train_gate,
                "holdout_gate": holdout_gate,
                "gate_passed": gate_passed,
                "checkpoint": str(checkpoint_path),
            },
            indent=2,
        )
    )
    if not gate_passed:
        raise SystemExit("V2 U-Net-only response gate failed; do not proceed to full training.")


if __name__ == "__main__":
    main()
