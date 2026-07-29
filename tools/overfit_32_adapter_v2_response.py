"""Overfit 32 Train samples using frozen Lotus U-Net response consistency."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = ROOT / "lotus"
for path in (ROOT, LOTUS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.anythermal_encoder import AnyThermalEncoder  # noqa: E402
from models.anythermal_lotus_adapter_v2 import AnyThermalLotusAdapterV2  # noqa: E402
from models.anythermal_lotus_model import extract_anythermal_feature_pyramid  # noqa: E402
from models.anythermal_lotus_v2 import (  # noqa: E402
    condition_distillation_losses,
    encode_condition_latent,
    thermal_to_lotus_input,
)
from pipeline import LotusGPipeline  # noqa: E402
from tools.overfit_32_adapter_v2_distillation import (  # noqa: E402
    batch_indices,
    read_train_rows,
    select_uniform,
)


DEFAULT_MANIFEST = Path(
    "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/"
    "ms2_fixed_sequence_internvl3_8b_filtered_caption_train.jsonl"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ms2-root", type=Path, default=Path("/mnt/e/dataset/ms2"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lotus_line_v2/overfit_32_v2_1_unet_response"),
    )
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument(
        "--holdout-samples",
        type=int,
        default=0,
        help="Selected Train samples excluded from optimization and used only for monitoring.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--response-weight", type=float, default=1.0)
    parser.add_argument("--latent-weight", type=float, default=0.1)
    parser.add_argument("--cosine-loss-weight", type=float, default=0.1)
    parser.add_argument("--channel-stats-loss-weight", type=float, default=0.1)
    parser.add_argument("--spatial-gradient-loss-weight", type=float, default=0.02)
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def task_embedding(batch_size, device, dtype):
    task = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype).repeat(batch_size, 1)
    return torch.cat([torch.sin(task), torch.cos(task)], dim=-1)


def cache_samples(args, device, dtype):
    rows = select_uniform(read_train_rows(args.train_manifest.resolve()), args.num_samples)
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
        condition = encode_condition_latent(
            lotus.vae,
            thermal.tensor,
            posterior="mode",
        ).to(device=device, dtype=dtype)
        features, _, diagnostics = extract_anythermal_feature_pyramid(
            anythermal,
            thermal_path,
            enable_grad=False,
        )
        sample_seed = args.seed + int(row["manifest_index"])
        generator = torch.Generator(device=device).manual_seed(sample_seed)
        noise = torch.randn(
            condition.shape,
            generator=generator,
            device=device,
            dtype=dtype,
        ) * lotus.scheduler.init_noise_sigma
        timestep = torch.full((1,), args.timestep, device=device, dtype=torch.long)
        latent_input = lotus.scheduler.scale_model_input(noise, timestep)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
        ):
            teacher_response = lotus.unet(
                torch.cat([condition, latent_input], dim=1),
                timestep,
                encoder_hidden_states=prompt,
                class_labels=task_embedding(1, device, dtype),
                return_dict=False,
            )[0]
        cache.append(
            {
                **row,
                "features": [feature.detach().float().cpu() for feature in features],
                "condition": condition.detach().float().cpu(),
                "latent_input": latent_input.detach().cpu(),
                "teacher_response": teacher_response.detach().float().cpu(),
                "thermal_std": float(diagnostics["converted_uint8_std"]),
                "sample_seed": sample_seed,
            }
        )
        print(f"cached {position + 1:02d}/{len(rows)} {row['id']}", flush=True)

    anythermal.model.to("cpu")
    lotus.vae.to("cpu")
    lotus.text_encoder.to("cpu")
    del anythermal
    torch.cuda.empty_cache()
    return cache, lotus, prompt


def stack_batch(cache, indices, device, dtype):
    features = [
        torch.cat([cache[index]["features"][level] for index in indices]).to(device)
        for level in range(4)
    ]
    condition = torch.cat([cache[index]["condition"] for index in indices]).to(device)
    latent_input = torch.cat([cache[index]["latent_input"] for index in indices]).to(
        device=device, dtype=dtype
    )
    teacher_response = torch.cat(
        [cache[index]["teacher_response"] for index in indices]
    ).to(device)
    return features, condition, latent_input, teacher_response


def forward_losses(adapter, lotus, prompt, batch, args, device, dtype):
    features, teacher_condition, latent_input, teacher_response = batch
    student_condition = adapter(features, target_size=tuple(teacher_condition.shape[-2:]))
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
    response_mse = F.mse_loss(student_response.float(), teacher_response.float())
    response_cosine = F.cosine_similarity(
        student_response.float().flatten(1),
        teacher_response.float().flatten(1),
        dim=1,
        eps=1e-8,
    ).mean()
    latent = condition_distillation_losses(
        student_condition,
        teacher_condition,
        cosine_weight=args.cosine_loss_weight,
        channel_stats_weight=args.channel_stats_loss_weight,
        spatial_gradient_weight=args.spatial_gradient_loss_weight,
    )
    total = args.response_weight * response_mse + args.latent_weight * latent["total"]
    return {
        "total_loss": total,
        "response_mse": response_mse,
        "response_cosine": response_cosine,
        "latent_mse": latent["latent_mse"],
        "latent_cosine": 1.0 - latent["cosine_loss"],
        "channel_stats_loss": latent["channel_stats_loss"],
        "spatial_gradient_loss": latent["spatial_gradient_loss"],
    }


@torch.no_grad()
def evaluate(adapter, lotus, prompt, cache, args, device, dtype, indices=None):
    was_training = adapter.training
    adapter.eval()
    keys = (
        "total_loss",
        "response_mse",
        "response_cosine",
        "latent_mse",
        "latent_cosine",
        "channel_stats_loss",
        "spatial_gradient_loss",
    )
    sums = {key: 0.0 for key in keys}
    count = 0
    evaluation_indices = list(range(len(cache))) if indices is None else list(indices)
    for start in range(0, len(evaluation_indices), args.batch_size):
        batch_ids = evaluation_indices[start : start + args.batch_size]
        losses = forward_losses(
            adapter,
            lotus,
            prompt,
            stack_batch(cache, batch_ids, device, dtype),
            args,
            device,
            dtype,
        )
        for key in keys:
            sums[key] += float(losses[key]) * len(batch_ids)
        count += len(batch_ids)
    if was_training:
        adapter.train()
    return {key: value / count for key, value in sums.items()}


def save_checkpoint(adapter, output, step, args):
    path = output / f"adapter_step_{step:04d}.pt"
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "format": "adapter_v2_1_unet_response_consistency",
            "adapter_architecture": "v2_1_spatial_decoder",
            "global_step": step,
            "settings": settings,
            "adapter": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
        },
        path,
    )
    return path


def split_train_holdout(total: int, holdout_count: int):
    if holdout_count < 0 or holdout_count >= total:
        raise ValueError(f"holdout-samples must be in [0,{total - 1}], got {holdout_count}.")
    if holdout_count == 0:
        return list(range(total)), []
    holdout = sorted(set(map(int, np.linspace(0, total - 1, holdout_count, dtype=int))))
    if len(holdout) != holdout_count:
        raise RuntimeError("Holdout selection produced duplicate positions.")
    holdout_set = set(holdout)
    train = [index for index in range(total) if index not in holdout_set]
    return train, holdout


def evaluation_record(step, train_metrics, holdout_metrics=None):
    record = {"step": step}
    record.update({f"train_{key}": value for key, value in train_metrics.items()})
    if holdout_metrics is not None:
        record.update({f"holdout_{key}": value for key, value in holdout_metrics.items()})
    return record


def main():
    args = parse_args()
    if args.batch_size != 4:
        raise ValueError("Response gate requires effective batch size 4.")
    if min(args.response_weight, args.latent_weight) < 0:
        raise ValueError("Loss weights must be non-negative.")
    seed_everything(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    cache, lotus, prompt = cache_samples(args, device, dtype)
    train_indices, holdout_indices = split_train_holdout(
        len(cache), args.holdout_samples
    )
    adapter = AnyThermalLotusAdapterV2().to(device=device, dtype=torch.float32).train()
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    initial_train = evaluate(
        adapter, lotus, prompt, cache, args, device, dtype, train_indices
    )
    initial_holdout = (
        evaluate(adapter, lotus, prompt, cache, args, device, dtype, holdout_indices)
        if holdout_indices
        else None
    )
    history = [evaluation_record(0, initial_train, initial_holdout)]
    save_checkpoint(adapter, output, 0, args)
    print(json.dumps(history[-1]), flush=True)

    for step in range(1, args.steps + 1):
        local_indices = batch_indices(
            step - 1, len(train_indices), args.batch_size, args.seed
        )
        indices = [train_indices[index] for index in local_indices]
        optimizer.zero_grad(set_to_none=True)
        losses = forward_losses(
            adapter,
            lotus,
            prompt,
            stack_batch(cache, indices, device, dtype),
            args,
            device,
            dtype,
        )
        loss = losses["total_loss"]
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite loss at step {step}.")
        loss.backward()
        gradients = [parameter.grad for parameter in adapter.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError(f"Missing/non-finite Adapter gradient at step {step}.")
        if any(parameter.grad is not None for parameter in lotus.unet.parameters()):
            raise RuntimeError("Frozen U-Net unexpectedly owns gradients.")
        optimizer.step()
        if step % args.log_interval == 0 or step == args.steps:
            train_metrics = evaluate(
                adapter, lotus, prompt, cache, args, device, dtype, train_indices
            )
            holdout_metrics = (
                evaluate(
                    adapter,
                    lotus,
                    prompt,
                    cache,
                    args,
                    device,
                    dtype,
                    holdout_indices,
                )
                if holdout_indices
                else None
            )
            history.append(evaluation_record(step, train_metrics, holdout_metrics))
            print(json.dumps(history[-1]), flush=True)

    final_train = evaluate(
        adapter, lotus, prompt, cache, args, device, dtype, train_indices
    )
    final_holdout = (
        evaluate(adapter, lotus, prompt, cache, args, device, dtype, holdout_indices)
        if holdout_indices
        else None
    )
    checkpoint = save_checkpoint(adapter, output, args.steps, args)
    train_response_ratio = final_train["response_mse"] / initial_train["response_mse"]
    holdout_response_ratio = (
        final_holdout["response_mse"] / initial_holdout["response_mse"]
        if final_holdout is not None and initial_holdout is not None
        else None
    )
    train_gate = bool(
        math.isfinite(train_response_ratio)
        and train_response_ratio <= 0.2
        and final_train["response_cosine"] >= 0.9
    )
    holdout_gate = bool(
        final_holdout is None
        or (
            math.isfinite(holdout_response_ratio)
            and holdout_response_ratio <= 0.7
            and final_holdout["response_cosine"] >= 0.75
        )
    )
    gate_passed = train_gate and holdout_gate
    with (output / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "phase": "Adapter V2.1 Train-only frozen-U-Net response-consistency gate",
        "metric_scope": "training diagnostics only; not official depth quality",
        "source_split": "Train only",
        "uses_val": False,
        "uses_test": False,
        "v1_checkpoint_used": False,
        "settings": {
            "num_samples": args.num_samples,
            "optimization_samples": len(train_indices),
            "holdout_samples": len(holdout_indices),
            "effective_batch_size": args.batch_size,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "response_weight": args.response_weight,
            "latent_weight": args.latent_weight,
            "timestep": args.timestep,
            "prompt": "",
            "teacher_posterior": "mode",
            "processing_res": 0,
            "adapter_architecture": "v2_1_spatial_decoder",
        },
        "initial_train": initial_train,
        "final_train": final_train,
        "initial_holdout": initial_holdout,
        "final_holdout": final_holdout,
        "train_response_mse_ratio": train_response_ratio,
        "holdout_response_mse_ratio": holdout_response_ratio,
        "train_gate_passed": train_gate,
        "holdout_gate_passed": holdout_gate,
        "gate_rule": (
            "train ratio <= 0.2 and cosine >= 0.9; "
            "holdout ratio <= 0.7 and cosine >= 0.75"
        ),
        "gate_passed": gate_passed,
        "checkpoint": str(checkpoint),
        "optimization_sample_ids": [cache[index]["id"] for index in train_indices],
        "holdout_sample_ids": [cache[index]["id"] for index in holdout_indices],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "initial_train": initial_train,
        "final_train": final_train,
        "initial_holdout": initial_holdout,
        "final_holdout": final_holdout,
        "train_response_mse_ratio": train_response_ratio,
        "holdout_response_mse_ratio": holdout_response_ratio,
        "gate_passed": gate_passed,
        "checkpoint": str(checkpoint),
    }, indent=2))
    if not gate_passed:
        raise SystemExit("Response-consistency gate failed; do not proceed.")


if __name__ == "__main__":
    main()
