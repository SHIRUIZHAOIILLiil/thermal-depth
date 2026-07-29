"""Fine-tune V2.3 on 128 Train frames with frozen-U-Net response consistency.

The original 16-frame Train holdout remains excluded from optimization.  The
Lotus VAE and U-Net are training-only frozen teachers; inference remains pure
AnyThermal + Adapter V2.3 + Lotus U-Net.
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = ROOT / "lotus"
for path in (ROOT, LOTUS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.anythermal_encoder import AnyThermalEncoder  # noqa: E402
from models.anythermal_lotus_adapter_v2_3 import AnyThermalLotusAdapterV23  # noqa: E402
from models.anythermal_lotus_model import extract_anythermal_feature_pyramid  # noqa: E402
from models.anythermal_lotus_v2 import encode_condition_latent, thermal_to_lotus_input  # noqa: E402
from models.anythermal_lotus_v2_4 import response_consistency_losses, seeded_noise  # noqa: E402
from tools import overfit_32_adapter_v2_distillation as latent_base  # noqa: E402


DEFAULT_PROTOCOL_DIR = Path("outputs/lotus_line_v2/short_128_v2_3_holdout16")
DEFAULT_SOURCE_MANIFEST = Path(
    "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/"
    "ms2_fixed_sequence_internvl3_8b_filtered_caption_train.jsonl"
)


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
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=DEFAULT_PROTOCOL_DIR / "adapter_step_1000.pt",
    )
    parser.add_argument("--ms2-root", type=Path, default=Path("/mnt/e/dataset/ms2"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lotus_line_v2/short_128_v2_4_response_holdout16"),
    )
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--condition-weight", type=float, default=1.0)
    parser.add_argument("--response-weight", type=float, default=1.0)
    parser.add_argument("--cosine-loss-weight", type=float, default=0.1)
    parser.add_argument("--channel-stats-loss-weight", type=float, default=0.1)
    parser.add_argument("--spatial-gradient-loss-weight", type=float, default=0.1)
    parser.add_argument("--multiscale-gradient-loss-weight", type=float, default=0.5)
    parser.add_argument("--response-cosine-loss-weight", type=float, default=0.1)
    parser.add_argument("--response-spatial-gradient-weight", type=float, default=0.5)
    parser.add_argument("--response-multiscale-gradient-weight", type=float, default=0.5)
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    args.adapter_architecture = "v2_3_thermal_detail_skip"
    args.use_output_group_norm = False
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source_indices(path: Path) -> Dict[str, int]:
    result = {}
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            result[str(row["id"])] = index
    return result


def read_role_manifest(
    path: Path,
    expected_role: str,
    source_indices: Dict[str, int],
) -> list[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["id"])
            if row.get("split") != "train":
                raise ValueError(f"Non-Train row in {expected_role}: {sample_id}")
            if row.get("adapter_v2_short_split") != expected_role:
                raise ValueError(f"Wrong frozen role for {sample_id}: {row.get('adapter_v2_short_split')}")
            if sample_id not in source_indices:
                raise ValueError(f"Sample missing from source manifest: {sample_id}")
            rows.append(
                {
                    "id": sample_id,
                    "thermal_path": str(row["thermal_path"]),
                    "manifest_index": source_indices[sample_id],
                    "role": expected_role,
                }
            )
    return rows


def task_embedding(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    task = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype).repeat(batch_size, 1)
    return torch.cat([torch.sin(task), torch.cos(task)], dim=-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cache_samples(args, rows, device, dtype):
    from pipeline import LotusGPipeline

    lotus = LotusGPipeline.from_pretrained(
        args.lotus_model_path,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    anythermal = AnyThermalEncoder(
        model_path=args.anythermal_model_path,
        device=str(device),
        local_files_only=args.local_files_only,
    )
    for module in (lotus.vae, lotus.text_encoder, lotus.unet, anythermal.model):
        module.requires_grad_(False).eval()
    prompt, _ = lotus.encode_prompt(
        prompt="",
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=None,
    )
    prompt = prompt.detach().to(dtype=dtype)
    cache = []
    for position, row in enumerate(rows):
        thermal_path = args.ms2_root.resolve() / row["thermal_path"]
        thermal = thermal_to_lotus_input(thermal_path, processing_res=0)
        if thermal.diagnostics["converted_uint8_std"] <= 0:
            raise RuntimeError(f"Constant thermal input: {row['id']}")
        teacher_condition = encode_condition_latent(
            lotus.vae, thermal.tensor, posterior="mode"
        ).to(device=device, dtype=dtype)
        features, _, diagnostics = extract_anythermal_feature_pyramid(
            anythermal, thermal_path, enable_grad=False
        )
        sample_seed = args.seed + int(row["manifest_index"])
        noise = seeded_noise(
            teacher_condition.shape,
            seed=sample_seed,
            device=device,
            dtype=dtype,
            scale=float(lotus.scheduler.init_noise_sigma),
        )
        timestep = torch.full((1,), args.timestep, device=device, dtype=torch.long)
        latent_input = lotus.scheduler.scale_model_input(noise, timestep)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
        ):
            teacher_response = lotus.unet(
                torch.cat([teacher_condition, latent_input], dim=1),
                timestep,
                encoder_hidden_states=prompt,
                class_labels=task_embedding(1, device, dtype),
                return_dict=False,
            )[0]
        cache.append(
            {
                **row,
                "features": [feature.detach().float().cpu() for feature in features],
                "thermal": thermal.tensor.detach().float().cpu(),
                "teacher_condition": teacher_condition.detach().float().cpu(),
                "latent_input": latent_input.detach().cpu(),
                "teacher_response": teacher_response.detach().float().cpu(),
                "sample_seed": sample_seed,
                "thermal_std": float(diagnostics["converted_uint8_std"]),
            }
        )
        print(f"cached {position + 1:03d}/{len(rows)} {row['role']} {row['id']}", flush=True)
    anythermal.model.to("cpu")
    lotus.vae.to("cpu")
    lotus.text_encoder.to("cpu")
    del anythermal
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cache, lotus, prompt


def stack_batch(cache, indices: Sequence[int], device, dtype):
    features = [
        torch.cat([cache[index]["features"][level] for index in indices]).to(device)
        for level in range(4)
    ]
    thermal = torch.cat([cache[index]["thermal"] for index in indices]).to(device)
    condition = torch.cat([cache[index]["teacher_condition"] for index in indices]).to(device)
    latent_input = torch.cat([cache[index]["latent_input"] for index in indices]).to(
        device=device, dtype=dtype
    )
    response = torch.cat([cache[index]["teacher_response"] for index in indices]).to(device)
    return features, thermal, condition, latent_input, response


def forward_losses(adapter, lotus, prompt, batch, args, device, dtype):
    features, thermal, teacher_condition, latent_input, teacher_response = batch
    student_condition = adapter(
        features, thermal, target_size=tuple(teacher_condition.shape[-2:])
    )
    batch_size = student_condition.shape[0]
    timestep = torch.full((batch_size,), args.timestep, device=device, dtype=torch.long)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        student_response = lotus.unet(
            torch.cat([student_condition.to(dtype), latent_input], dim=1),
            timestep,
            encoder_hidden_states=prompt.repeat(batch_size, 1, 1),
            class_labels=task_embedding(batch_size, device, dtype),
            return_dict=False,
        )[0]
    condition = latent_base.loss_components(student_condition, teacher_condition, args)
    response = response_consistency_losses(
        student_response,
        teacher_response,
        cosine_weight=args.response_cosine_loss_weight,
        spatial_gradient_weight=args.response_spatial_gradient_weight,
        multiscale_gradient_weight=args.response_multiscale_gradient_weight,
    )
    condition_cosine = F.cosine_similarity(
        student_condition.float().flatten(1),
        teacher_condition.float().flatten(1),
        dim=1,
        eps=1e-8,
    ).mean()
    condition_prediction_energy = latent_base.spatial_gradient_energy(student_condition).mean()
    condition_target_energy = latent_base.spatial_gradient_energy(teacher_condition).mean()
    total = args.condition_weight * condition["total"] + args.response_weight * response["total"]
    return {
        "total_loss": total,
        "condition_total": condition["total"],
        "condition_mse": condition["latent_mse"],
        "condition_cosine": condition_cosine,
        "condition_gradient_ratio": condition_prediction_energy
        / condition_target_energy.clamp_min(1e-12),
        "response_total": response["total"],
        "response_mse": response["mse"],
        "response_cosine": response["cosine"],
        "response_spatial_gradient_loss": response["spatial_gradient_loss"],
        "response_multiscale_gradient_loss": response["multiscale_gradient_loss"],
        "response_gradient_ratio": response["gradient_energy_ratio"],
    }


@torch.no_grad()
def evaluate(adapter, lotus, prompt, cache, args, device, dtype):
    was_training = adapter.training
    adapter.eval()
    sums: Dict[str, float] = {}
    count = 0
    for start in range(0, len(cache), args.batch_size):
        indices = list(range(start, min(start + args.batch_size, len(cache))))
        metrics = forward_losses(
            adapter,
            lotus,
            prompt,
            stack_batch(cache, indices, device, dtype),
            args,
            device,
            dtype,
        )
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value) * len(indices)
        count += len(indices)
    if was_training:
        adapter.train()
    return {key: value / count for key, value in sums.items()}


def evaluation_record(step, train_metrics, holdout_metrics):
    return {
        "step": step,
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"holdout_{key}": value for key, value in holdout_metrics.items()},
    }


def response_gate(initial, final, *, holdout: bool):
    ratio = final["response_mse"] / initial["response_mse"]
    ratio_limit = 0.8 if holdout else 0.7
    cosine_minimum = 0.97 if holdout else 0.98
    passed = bool(
        math.isfinite(ratio)
        and ratio <= ratio_limit
        and final["response_cosine"] >= cosine_minimum
        and 0.6 <= final["response_gradient_ratio"] <= 1.4
        and final["condition_cosine"] >= initial["condition_cosine"] - 0.01
        and final["condition_gradient_ratio"] >= 0.6
    )
    return {
        "response_mse_ratio": ratio,
        "response_mse_ratio_limit": ratio_limit,
        "response_cosine_minimum": cosine_minimum,
        "response_gradient_ratio_range": [0.6, 1.4],
        "condition_cosine_max_drop": 0.01,
        "condition_gradient_ratio_minimum": 0.6,
        "passed": passed,
    }


def save_checkpoint(path, adapter, optimizer, step, args, init_hash):
    torch.save(
        {
            "format": "adapter_v2_4_unet_response_consistency",
            "adapter_architecture": "v2_3_thermal_detail_skip",
            "global_step": step,
            "init_checkpoint_sha256": init_hash,
            "settings": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "adapter": {
                key: value.detach().cpu() for key, value in adapter.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size != 4 or args.timestep != 999:
        raise ValueError("V2.4 gate requires batch size 4 and inference timestep 999.")
    if min(
        args.condition_weight,
        args.response_weight,
        args.response_cosine_loss_weight,
        args.response_spatial_gradient_weight,
        args.response_multiscale_gradient_weight,
    ) < 0:
        raise ValueError("All V2.4 loss weights must be non-negative.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    source_indices = read_source_indices(args.source_manifest.resolve())
    train_rows = read_role_manifest(args.train_manifest.resolve(), "train", source_indices)
    holdout_rows = read_role_manifest(
        args.holdout_manifest.resolve(), "holdout", source_indices
    )
    if len(train_rows) != 128 or len(holdout_rows) != 16:
        raise RuntimeError(
            f"Frozen protocol requires 128/16 rows, got {len(train_rows)}/{len(holdout_rows)}."
        )
    cache, lotus, prompt = cache_samples(
        args, train_rows + holdout_rows, device, dtype
    )
    train_cache = cache[: len(train_rows)]
    holdout_cache = cache[len(train_rows) :]

    checkpoint_path = args.init_checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("adapter_architecture") != "v2_3_thermal_detail_skip":
        raise RuntimeError("V2.4 init checkpoint is not Adapter V2.3.")
    adapter = AnyThermalLotusAdapterV23().to(device=device, dtype=torch.float32)
    adapter.load_state_dict(checkpoint["adapter"], strict=True)
    adapter.train()
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    init_hash = sha256(checkpoint_path)
    initial_train = evaluate(adapter, lotus, prompt, train_cache, args, device, dtype)
    initial_holdout = evaluate(adapter, lotus, prompt, holdout_cache, args, device, dtype)
    history = [evaluation_record(0, initial_train, initial_holdout)]
    save_checkpoint(output / "adapter_step_0000.pt", adapter, optimizer, 0, args, init_hash)
    print(json.dumps(history[-1]), flush=True)

    for step in range(1, args.steps + 1):
        indices = latent_base.batch_indices(
            step - 1, len(train_cache), args.batch_size, args.seed
        )
        optimizer.zero_grad(set_to_none=True)
        losses = forward_losses(
            adapter,
            lotus,
            prompt,
            stack_batch(train_cache, indices, device, dtype),
            args,
            device,
            dtype,
        )
        loss = losses["total_loss"]
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite loss at step {step}.")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in adapter.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise RuntimeError(f"Missing/non-finite Adapter gradient at step {step}.")
        if any(parameter.grad is not None for parameter in lotus.unet.parameters()):
            raise RuntimeError("Frozen U-Net unexpectedly owns gradients.")
        optimizer.step()
        if step % args.log_interval == 0 or step == args.steps:
            train_metrics = evaluate(
                adapter, lotus, prompt, train_cache, args, device, dtype
            )
            holdout_metrics = evaluate(
                adapter, lotus, prompt, holdout_cache, args, device, dtype
            )
            history.append(evaluation_record(step, train_metrics, holdout_metrics))
            print(json.dumps(history[-1]), flush=True)

    final_train = evaluate(adapter, lotus, prompt, train_cache, args, device, dtype)
    final_holdout = evaluate(adapter, lotus, prompt, holdout_cache, args, device, dtype)
    final_path = output / f"adapter_step_{args.steps:04d}.pt"
    save_checkpoint(final_path, adapter, optimizer, args.steps, args, init_hash)
    train_gate = response_gate(initial_train, final_train, holdout=False)
    holdout_gate = response_gate(initial_holdout, final_holdout, holdout=True)
    gate_passed = bool(train_gate["passed"] and holdout_gate["passed"])
    with (output / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "phase": "Adapter V2.4 frozen-U-Net response-consistency short gate",
        "metric_scope": "Train-only diagnostics; official depth quality remains separate",
        "route": "AnyThermal + Adapter V2.3 + frozen Lotus U-Net",
        "teacher_training_only": "Thermal-VAE mode condition through frozen Lotus U-Net",
        "inference_uses_thermal_vae": False,
        "source_split": "Train 128 + frozen Train holdout 16",
        "uses_val": False,
        "uses_test": False,
        "v1_checkpoint_used": False,
        "train_manifest": str(args.train_manifest.resolve()),
        "holdout_manifest": str(args.holdout_manifest.resolve()),
        "init_checkpoint": str(checkpoint_path),
        "init_checkpoint_sha256": init_hash,
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
        "checkpoint": str(final_path),
        "checkpoint_sha256": sha256(final_path),
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
                "checkpoint": str(final_path),
            },
            indent=2,
        )
    )
    if not gate_passed:
        raise SystemExit("V2.4 response gate failed; do not proceed to official evaluation.")


if __name__ == "__main__":
    main()
