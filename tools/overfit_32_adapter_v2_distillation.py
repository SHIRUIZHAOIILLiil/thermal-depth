"""Overfit fixed MS2 Train samples with Adapter V2 condition distillation.

Frozen AnyThermal features and fixed-seed frozen Lotus VAE condition latents
are cached once.  Only a freshly initialized Adapter is optimized.  No U-Net,
caption, Val/Test data, or V1 checkpoint is used.
"""

from __future__ import annotations

import argparse
import csv
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.anythermal_encoder import AnyThermalEncoder  # noqa: E402
from models.anythermal_lotus_adapter import AnyThermalLotusAdapter  # noqa: E402
from models.anythermal_lotus_adapter_v2 import AnyThermalLotusAdapterV2  # noqa: E402
from models.anythermal_lotus_adapter_v2_2 import AnyThermalLotusAdapterV22  # noqa: E402
from models.anythermal_lotus_adapter_v2_3 import AnyThermalLotusAdapterV23  # noqa: E402
from models.anythermal_lotus_model import extract_anythermal_feature_pyramid  # noqa: E402
from models.anythermal_lotus_v2 import (  # noqa: E402
    condition_distillation_losses,
    encode_condition_latent,
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
        default=Path("outputs/lotus_line_v2/overfit_1_v2_3_thermal_detail_skip"),
    )
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--processing-res", type=int, default=0)
    parser.add_argument(
        "--teacher-posterior",
        choices=("mode", "mean", "sample"),
        default="mode",
    )
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--adapter-architecture",
        choices=(
            "v1_shallow",
            "v2_1_spatial_decoder",
            "v2_2_progressive_residual_decoder",
            "v2_3_thermal_detail_skip",
        ),
        default="v2_3_thermal_detail_skip",
    )
    parser.add_argument("--cosine-loss-weight", type=float, default=0.1)
    parser.add_argument("--channel-stats-loss-weight", type=float, default=0.1)
    parser.add_argument("--spatial-gradient-loss-weight", type=float, default=0.1)
    parser.add_argument("--multiscale-gradient-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--use-output-group-norm",
        action="store_true",
        help="Disabled by default in V2 because it erases sample-dependent latent scale.",
    )
    parser.add_argument("--log-interval", type=int, default=25)
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


def read_train_rows(path: Path) -> list[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for manifest_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError(f"Non-Train sample in Train manifest: {row.get('id')}")
            rows.append(
                {
                    "id": str(row["id"]),
                    "thermal_path": str(row["thermal_path"]),
                    "manifest_index": manifest_index,
                }
            )
    return rows


def select_uniform(rows: Sequence[Dict[str, Any]], count: int) -> list[Dict[str, Any]]:
    if count <= 0 or count > len(rows):
        raise ValueError(f"num-samples must be in [1,{len(rows)}], got {count}.")
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    if len(set(map(int, indices))) != count:
        raise RuntimeError("Uniform selection produced duplicate manifest indices.")
    return [rows[int(index)] for index in indices]


def cache_samples(args: argparse.Namespace, device: torch.device):
    from diffusers import AutoencoderKL

    rows = select_uniform(read_train_rows(args.train_manifest.resolve()), args.num_samples)
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

    cache = []
    expected_feature_shapes = None
    expected_teacher_shape = None
    for position, row in enumerate(rows):
        thermal_path = args.ms2_root.resolve() / row["thermal_path"]
        if not thermal_path.is_file():
            raise FileNotFoundError(f"Thermal image not found: {thermal_path}")
        lotus_input = thermal_to_lotus_input(
            thermal_path,
            processing_res=args.processing_res,
        )
        teacher_seed = args.seed + int(row["manifest_index"])
        teacher = encode_condition_latent(
            vae,
            lotus_input.tensor,
            posterior=args.teacher_posterior,
            seed=teacher_seed if args.teacher_posterior == "sample" else None,
        ).detach().float().cpu()
        features, _, diagnostics = extract_anythermal_feature_pyramid(
            anythermal,
            thermal_path,
            enable_grad=False,
        )
        features = [feature.detach().float().cpu() for feature in features]
        feature_shapes = [tuple(feature.shape[1:]) for feature in features]
        teacher_shape = tuple(teacher.shape[1:])
        if expected_feature_shapes is None:
            expected_feature_shapes = feature_shapes
            expected_teacher_shape = teacher_shape
        if feature_shapes != expected_feature_shapes:
            raise RuntimeError(f"Inconsistent AnyThermal feature shapes: {feature_shapes}")
        if teacher_shape != expected_teacher_shape:
            raise RuntimeError(f"Inconsistent teacher shape: {teacher_shape}")
        if diagnostics["converted_uint8_std"] <= 0:
            raise RuntimeError(f"Constant thermal conversion for {row['id']}")
        cache.append(
            {
                **row,
                "position": position,
                "thermal_path_resolved": str(thermal_path),
                "teacher_seed": teacher_seed,
                "features": features,
                "thermal": lotus_input.tensor.detach().float().cpu(),
                "teacher": teacher,
                "thermal_diagnostics": lotus_input.diagnostics,
            }
        )
        print(f"cached {position + 1:02d}/{len(rows)} {row['id']}", flush=True)

    del anythermal, vae
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cache


def stack_batch(cache, indices: Sequence[int], device: torch.device):
    features = [
        torch.cat([cache[index]["features"][level] for index in indices], dim=0).to(device)
        for level in range(len(cache[0]["features"]))
    ]
    thermal = torch.cat([cache[index]["thermal"] for index in indices], dim=0).to(device)
    teacher = torch.cat([cache[index]["teacher"] for index in indices], dim=0).to(device)
    return features, thermal, teacher


def batch_indices(step: int, total: int, batch_size: int, seed: int) -> list[int]:
    batches_per_epoch = math.ceil(total / batch_size)
    epoch = step // batches_per_epoch
    batch_number = step % batches_per_epoch
    indices = list(range(total))
    random.Random(seed + epoch).shuffle(indices)
    start = batch_number * batch_size
    selected = indices[start : start + batch_size]
    if not selected:
        raise RuntimeError("Empty training batch.")
    return selected


def prediction(adapter, features, thermal, teacher):
    if getattr(adapter, "requires_thermal_input", False):
        output = adapter(features, thermal, target_size=tuple(teacher.shape[-2:]))
    else:
        output = adapter(features, target_size=tuple(teacher.shape[-2:]))
    if output.shape != teacher.shape:
        raise RuntimeError(f"Adapter/teacher shape mismatch: {output.shape} != {teacher.shape}")
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("Adapter output contains NaN/Inf.")
    return output


def make_adapter(args: argparse.Namespace) -> torch.nn.Module:
    if args.adapter_architecture == "v2_3_thermal_detail_skip":
        if args.use_output_group_norm:
            raise ValueError("V2.3 has no output GroupNorm by design.")
        return AnyThermalLotusAdapterV23()
    if args.adapter_architecture == "v2_2_progressive_residual_decoder":
        if args.use_output_group_norm:
            raise ValueError("V2.2 has no output GroupNorm by design.")
        return AnyThermalLotusAdapterV22()
    if args.adapter_architecture == "v2_1_spatial_decoder":
        if args.use_output_group_norm:
            raise ValueError("V2.1 has no output GroupNorm by design.")
        return AnyThermalLotusAdapterV2()
    return AnyThermalLotusAdapter(
        use_output_group_norm=args.use_output_group_norm,
    )


def loss_components(output, target, args: argparse.Namespace):
    losses = condition_distillation_losses(
        output,
        target,
        cosine_weight=args.cosine_loss_weight,
        channel_stats_weight=args.channel_stats_loss_weight,
        spatial_gradient_weight=args.spatial_gradient_loss_weight,
    )
    multiscale = output.new_zeros(())
    if args.adapter_architecture == "v2_3_thermal_detail_skip":
        for scale in (2, 4):
            pred_scaled = F.avg_pool2d(output.float(), scale, stride=scale)
            target_scaled = F.avg_pool2d(target.detach().float(), scale, stride=scale)
            pred_dx = pred_scaled[..., 1:] - pred_scaled[..., :-1]
            target_dx = target_scaled[..., 1:] - target_scaled[..., :-1]
            pred_dy = pred_scaled[..., 1:, :] - pred_scaled[..., :-1, :]
            target_dy = target_scaled[..., 1:, :] - target_scaled[..., :-1, :]
            multiscale = multiscale + F.l1_loss(pred_dx, target_dx) + F.l1_loss(
                pred_dy, target_dy
            )
    losses["multiscale_gradient_loss"] = multiscale
    losses["total"] = losses["total"] + args.multiscale_gradient_loss_weight * multiscale
    return losses


def spatial_gradient_energy(value: torch.Tensor) -> torch.Tensor:
    """Mean absolute horizontal/vertical latent variation per sample."""

    dx = value[..., 1:] - value[..., :-1]
    dy = value[..., 1:, :] - value[..., :-1, :]
    return dx.abs().mean(dim=(1, 2, 3)) + dy.abs().mean(dim=(1, 2, 3))


@torch.no_grad()
def evaluate(adapter, cache, device: torch.device, batch_size: int, args) -> Dict[str, Any]:
    was_training = adapter.training
    adapter.eval()
    component_sums = {
        "total_loss": 0.0,
        "mse": 0.0,
        "cosine_loss": 0.0,
        "channel_stats_loss": 0.0,
        "spatial_gradient_loss": 0.0,
        "multiscale_gradient_loss": 0.0,
    }
    cosine_sum = pearson_sum = 0.0
    prediction_gradient_sum = teacher_gradient_sum = 0.0
    count = 0
    fixed_eight = []
    for start in range(0, len(cache), batch_size):
        indices = list(range(start, min(start + batch_size, len(cache))))
        features, thermal, teacher = stack_batch(cache, indices, device)
        output = prediction(adapter, features, thermal, teacher).float()
        target = teacher.float()
        batch_count = len(indices)
        losses = loss_components(output, target, args)
        mse = losses["latent_mse"]
        cosine = F.cosine_similarity(output.flatten(1), target.flatten(1), dim=1, eps=1e-8)
        output_centered = output.flatten(1) - output.flatten(1).mean(1, keepdim=True)
        target_centered = target.flatten(1) - target.flatten(1).mean(1, keepdim=True)
        pearson = F.cosine_similarity(output_centered, target_centered, dim=1, eps=1e-8)
        prediction_gradient = spatial_gradient_energy(output)
        teacher_gradient = spatial_gradient_energy(target)
        component_sums["total_loss"] += float(losses["total"]) * batch_count
        component_sums["mse"] += float(mse) * batch_count
        component_sums["cosine_loss"] += float(losses["cosine_loss"]) * batch_count
        component_sums["channel_stats_loss"] += float(losses["channel_stats_loss"]) * batch_count
        component_sums["spatial_gradient_loss"] += float(losses["spatial_gradient_loss"]) * batch_count
        component_sums["multiscale_gradient_loss"] += float(
            losses["multiscale_gradient_loss"]
        ) * batch_count
        cosine_sum += float(cosine.sum())
        pearson_sum += float(pearson.sum())
        prediction_gradient_sum += float(prediction_gradient.sum())
        teacher_gradient_sum += float(teacher_gradient.sum())
        count += batch_count
        for local, cache_index in enumerate(indices):
            if cache_index < 8:
                fixed_eight.append(
                    {
                        "id": cache[cache_index]["id"],
                        "prediction_shape": list(output[local].shape),
                        "teacher_shape": list(target[local].shape),
                        "prediction_mean": float(output[local].mean()),
                        "prediction_std": float(output[local].std(unbiased=False)),
                        "teacher_mean": float(target[local].mean()),
                        "teacher_std": float(target[local].std(unbiased=False)),
                        "mse": float(F.mse_loss(output[local], target[local])),
                        "cosine": float(cosine[local]),
                        "pearson": float(pearson[local]),
                        "finite": bool(
                            torch.isfinite(output[local]).all()
                            and torch.isfinite(target[local]).all()
                        ),
                    }
                )
    if was_training:
        adapter.train()
    mean_prediction_gradient = prediction_gradient_sum / count
    mean_teacher_gradient = teacher_gradient_sum / count
    return {
        **{key: value / count for key, value in component_sums.items()},
        "cosine": cosine_sum / count,
        "pearson": pearson_sum / count,
        "prediction_gradient_energy": mean_prediction_gradient,
        "teacher_gradient_energy": mean_teacher_gradient,
        "gradient_energy_ratio": mean_prediction_gradient / max(mean_teacher_gradient, 1e-12),
        "fixed_eight": fixed_eight,
    }


def save_checkpoint(adapter, output_dir: Path, step: int, args: argparse.Namespace) -> Path:
    path = output_dir / f"adapter_step_{step:04d}.pt"
    settings = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    torch.save(
        {
            "format": (
                "adapter_v2_3_condition_distillation"
                if args.adapter_architecture == "v2_3_thermal_detail_skip"
                else (
                    "adapter_v2_2_condition_distillation"
                    if args.adapter_architecture == "v2_2_progressive_residual_decoder"
                    else "adapter_v2_1_condition_distillation"
                )
            ),
            "adapter_architecture": args.adapter_architecture,
            "global_step": step,
            "adapter": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
            "settings": settings,
        },
        path,
    )
    return path


def main() -> None:
    args = parse_args()
    if not (
        (args.num_samples == 1 and args.batch_size == 1)
        or (args.num_samples > 1 and args.batch_size == 4)
    ):
        raise ValueError("Use batch size 1 for the single-image diagnostic, otherwise 4.")
    if args.steps <= 0 or args.log_interval <= 0:
        raise ValueError("steps and log-interval must be positive.")
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cache = cache_samples(args, device)
    adapter = make_adapter(args).to(device=device, dtype=torch.float32).train()
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    initial = evaluate(adapter, cache, device, args.batch_size, args)
    save_checkpoint(adapter, output_dir, 0, args)
    history = [
        {"step": 0, **{key: value for key, value in initial.items() if key != "fixed_eight"}}
    ]
    print(json.dumps(history[-1]), flush=True)

    for step in range(1, args.steps + 1):
        indices = batch_indices(step - 1, len(cache), args.batch_size, args.seed)
        features, thermal, teacher = stack_batch(cache, indices, device)
        optimizer.zero_grad(set_to_none=True)
        output = prediction(adapter, features, thermal, teacher)
        losses = loss_components(output, teacher, args)
        loss = losses["total"]
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite loss at step {step}.")
        loss.backward()
        gradients = [parameter.grad for parameter in adapter.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError(f"Missing or non-finite Adapter gradient at step {step}.")
        optimizer.step()

        if step % args.log_interval == 0 or step == args.steps:
            metrics = evaluate(adapter, cache, device, args.batch_size, args)
            record = {
                "step": step,
                **{key: value for key, value in metrics.items() if key != "fixed_eight"},
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    final = evaluate(adapter, cache, device, args.batch_size, args)
    checkpoint = save_checkpoint(adapter, output_dir, args.steps, args)
    mse_ratio = final["mse"] / initial["mse"]
    cosine_gain = final["cosine"] - initial["cosine"]
    pearson_gain = final["pearson"] - initial["pearson"]
    if args.num_samples == 1:
        gate_rule = (
            "single-image capacity diagnostic: mse_ratio <= 0.05, cosine/pearson >= 0.98, "
            "and latent gradient-energy ratio in [0.8,1.2]"
        )
        gate_passed = bool(
            math.isfinite(mse_ratio)
            and mse_ratio <= 0.05
            and final["cosine"] >= 0.98
            and final["pearson"] >= 0.98
            and 0.8 <= final["gradient_energy_ratio"] <= 1.2
            and all(item["finite"] for item in final["fixed_eight"])
        )
    else:
        gate_rule = (
            "32-sample diagnostic: mse_ratio <= 0.2, cosine/pearson >= 0.9, "
            "and latent gradient-energy ratio in [0.5,1.5]"
        )
        gate_passed = bool(
            math.isfinite(mse_ratio)
            and mse_ratio <= 0.2
            and final["cosine"] >= 0.9
            and final["pearson"] >= 0.9
            and 0.5 <= final["gradient_energy_ratio"] <= 1.5
            and all(item["finite"] for item in final["fixed_eight"])
        )

    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "phase": f"Adapter V2 {args.num_samples}-sample condition-distillation overfit gate",
        "route": (
            "frozen AnyThermal + audited thermal detail + trainable Adapter; "
            "frozen VAE teacher; no U-Net"
            if args.adapter_architecture == "v2_3_thermal_detail_skip"
            else "frozen AnyThermal + trainable Adapter; frozen VAE teacher; no U-Net"
        ),
        "source_split": "Train only",
        "uses_val": False,
        "uses_test": False,
        "resume": None,
        "v1_checkpoint_used": False,
        "settings": {
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "effective_batch_size": args.batch_size,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "processing_res": args.processing_res,
            "teacher_posterior": args.teacher_posterior,
            "use_output_group_norm": args.use_output_group_norm,
            "adapter_architecture": args.adapter_architecture,
            "loss_weights": {
                "cosine": args.cosine_loss_weight,
                "channel_stats": args.channel_stats_loss_weight,
                "spatial_gradient": args.spatial_gradient_loss_weight,
                "multiscale_gradient": args.multiscale_gradient_loss_weight,
            },
        },
        "sample_ids": [item["id"] for item in cache],
        "initial": {key: value for key, value in initial.items() if key != "fixed_eight"},
        "final": {key: value for key, value in final.items() if key != "fixed_eight"},
        "mse_ratio": mse_ratio,
        "cosine_gain": cosine_gain,
        "pearson_gain": pearson_gain,
        "fixed_eight_initial": initial["fixed_eight"],
        "fixed_eight_final": final["fixed_eight"],
        "checkpoint": str(checkpoint),
        "gate_rule": gate_rule + "; official depth/vis gate remains separate",
        "gate_passed": gate_passed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "initial": summary["initial"],
                "final": summary["final"],
                "mse_ratio": mse_ratio,
                "cosine_gain": cosine_gain,
                "pearson_gain": pearson_gain,
                "gate_passed": gate_passed,
                "checkpoint": str(checkpoint),
            },
            indent=2,
        )
    )
    print(output_dir / "summary.json")
    if not gate_passed:
        raise SystemExit(
            f"{args.num_samples}-sample overfit gate did not pass; "
            "do not proceed to larger training."
        )


if __name__ == "__main__":
    main()
