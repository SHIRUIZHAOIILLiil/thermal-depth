"""Train pure AnyThermal -> Adapter V2.3 -> frozen Lotus U-Net for one epoch.

V2.4 uses a frozen Thermal-VAE condition and frozen Lotus U-Net response only
as Train-time teachers.  No depth GT, Val, Test, caption, V1 checkpoint, or
Thermal-VAE inference path is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

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
from models.anythermal_lotus_v2 import (  # noqa: E402
    condition_distillation_losses,
    encode_condition_latent,
    thermal_to_lotus_input,
)
from models.anythermal_lotus_v2_4 import (  # noqa: E402
    gradient_energy,
    response_consistency_losses,
    seeded_noise,
)


DEFAULT_MANIFEST = Path(
    "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/"
    "ms2_fixed_sequence_internvl3_8b_filtered_caption_train.jsonl"
)
SAVE_STEPS = (0, 100, 500, 1000, 2000)
CHECKPOINT_FORMAT = "adapter_v2_4_full_epoch_unet_response_consistency"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ms2-root", type=Path, default=Path("/mnt/e/dataset/ms2"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lotus_line_v2/full_train_epoch1_v2_4_response"),
    )
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
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
    parser.add_argument(
        "--caption-mode",
        choices=("empty", "correct"),
        default="empty",
        help=(
            "correct: encode each sample's manifest caption as the prompt "
            "(Arm-1 protocol; pair with the clip75 train manifest). Default "
            "empty reproduces the original V2.4 run exactly."
        ),
    )
    parser.add_argument(
        "--caption-dropout",
        type=float,
        default=0.1,
        help="With --caption-mode correct: per-visit probability of swapping in the empty prompt (Arm-1 uses 0.1).",
    )
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume only this exact V2.4 run after interruption.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_train_manifest(path: Path, root: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for manifest_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError(f"Non-Train row in Train manifest: {row.get('id')}")
            thermal_path = root / row["thermal_path"]
            if not thermal_path.is_file():
                raise FileNotFoundError(f"Missing thermal input: {thermal_path}")
            rows.append(
                {
                    "id": str(row["id"]),
                    "manifest_index": manifest_index,
                    "thermal_path": thermal_path,
                    "caption": str(row.get("caption", "")),
                }
            )
    if not rows:
        raise ValueError("Train manifest is empty.")
    return rows


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def task_embedding(batch_size, device, dtype):
    task = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype).repeat(batch_size, 1)
    return torch.cat([torch.sin(task), torch.cos(task)], dim=-1)


def frozen_audit(module):
    return {
        "parameters": int(sum(parameter.numel() for parameter in module.parameters())),
        "trainable": int(
            sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        ),
    }


def multiscale_gradient_loss(prediction, target):
    loss = prediction.new_zeros(())
    for scale in (2, 4):
        pred = F.avg_pool2d(prediction.float(), scale, stride=scale)
        ref = F.avg_pool2d(target.detach().float(), scale, stride=scale)
        loss = loss + F.l1_loss(pred[..., 1:] - pred[..., :-1], ref[..., 1:] - ref[..., :-1])
        loss = loss + F.l1_loss(pred[..., 1:, :] - pred[..., :-1, :], ref[..., 1:, :] - ref[..., :-1, :])
    return loss


def build_batch(rows, indices, anythermal, lotus, prompt, args, device, dtype):
    feature_items = []
    thermal_tensors = []
    diagnostics = []
    for index in indices:
        row = rows[index]
        thermal = thermal_to_lotus_input(row["thermal_path"], processing_res=0)
        if thermal.diagnostics["converted_uint8_std"] <= 0:
            raise RuntimeError(f"Constant thermal conversion: {row['id']}")
        features, _, anythermal_diag = extract_anythermal_feature_pyramid(
            anythermal, row["thermal_path"], enable_grad=False
        )
        feature_items.append([feature.detach().float() for feature in features])
        thermal_tensors.append(thermal.tensor)
        diagnostics.append(
            {
                "id": row["id"],
                "manifest_index": row["manifest_index"],
                "thermal": thermal.diagnostics,
                "anythermal_converted_std": anythermal_diag["converted_uint8_std"],
            }
        )
    features = [
        torch.cat([item[level] for item in feature_items], dim=0).to(device)
        for level in range(4)
    ]
    thermal_batch = torch.cat(thermal_tensors, dim=0)
    teacher_condition = encode_condition_latent(
        lotus.vae, thermal_batch, posterior="mode"
    ).to(device=device, dtype=torch.float32)
    latent_inputs = []
    for index in indices:
        row = rows[index]
        noise = seeded_noise(
            (1, *teacher_condition.shape[1:]),
            seed=args.seed + row["manifest_index"],
            device=device,
            dtype=dtype,
            scale=float(lotus.scheduler.init_noise_sigma),
        )
        timestep = torch.full((1,), args.timestep, device=device, dtype=torch.long)
        latent_inputs.append(lotus.scheduler.scale_model_input(noise, timestep))
    latent_input = torch.cat(latent_inputs, dim=0)
    timestep = torch.full((len(indices),), args.timestep, device=device, dtype=torch.long)
    unet_input = torch.cat([teacher_condition.to(dtype), latent_input], dim=1)
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
    ):
        if isinstance(prompt, list):  # mixed-length captions: per-sample forwards
            teacher_response = torch.cat([
                lotus.unet(
                    unet_input[j : j + 1],
                    timestep[:1],
                    encoder_hidden_states=prompt[j].to(dtype),
                    class_labels=task_embedding(1, device, dtype),
                    return_dict=False,
                )[0]
                for j in range(len(indices))
            ], dim=0)
        else:
            teacher_response = lotus.unet(
                unet_input,
                timestep,
                encoder_hidden_states=(
                    prompt if prompt.shape[0] == len(indices) else prompt.repeat(len(indices), 1, 1)
                ),
                class_labels=task_embedding(len(indices), device, dtype),
                return_dict=False,
            )[0]
    return (
        features,
        torch.cat(thermal_tensors, dim=0).to(device=device, dtype=torch.float32),
        teacher_condition,
        latent_input,
        teacher_response.detach().float(),
        diagnostics,
    )


def forward_losses(adapter, lotus, prompt, batch, args, device, dtype):
    features, thermal, teacher_condition, latent_input, teacher_response, _ = batch
    student_condition = adapter(
        features, thermal, target_size=tuple(teacher_condition.shape[-2:])
    )
    batch_size = student_condition.shape[0]
    timestep = torch.full((batch_size,), args.timestep, device=device, dtype=torch.long)
    student_unet_input = torch.cat([student_condition.to(dtype), latent_input], dim=1)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        if isinstance(prompt, list):  # mixed-length captions: per-sample forwards
            student_response = torch.cat([
                lotus.unet(
                    student_unet_input[j : j + 1],
                    timestep[:1],
                    encoder_hidden_states=prompt[j].to(dtype),
                    class_labels=task_embedding(1, device, dtype),
                    return_dict=False,
                )[0]
                for j in range(batch_size)
            ], dim=0)
        else:
            student_response = lotus.unet(
                student_unet_input,
                timestep,
                encoder_hidden_states=(
                    prompt if prompt.shape[0] == batch_size else prompt.repeat(batch_size, 1, 1)
                ),
                class_labels=task_embedding(batch_size, device, dtype),
                return_dict=False,
            )[0]
    condition = condition_distillation_losses(
        student_condition,
        teacher_condition,
        cosine_weight=args.cosine_loss_weight,
        channel_stats_weight=args.channel_stats_loss_weight,
        spatial_gradient_weight=args.spatial_gradient_loss_weight,
    )
    condition_multiscale = multiscale_gradient_loss(student_condition, teacher_condition)
    condition_total = (
        condition["total"] + args.multiscale_gradient_loss_weight * condition_multiscale
    )
    response = response_consistency_losses(
        student_response,
        teacher_response,
        cosine_weight=args.response_cosine_loss_weight,
        spatial_gradient_weight=args.response_spatial_gradient_weight,
        multiscale_gradient_weight=args.response_multiscale_gradient_weight,
    )
    condition_prediction_energy = gradient_energy(student_condition.float()).mean()
    condition_target_energy = gradient_energy(teacher_condition.float()).mean()
    total = args.condition_weight * condition_total + args.response_weight * response["total"]
    return {
        "total_loss": total,
        "condition_total": condition_total,
        "condition_mse": condition["latent_mse"],
        "condition_cosine": 1.0 - condition["cosine_loss"],
        "condition_spatial_gradient_loss": condition["spatial_gradient_loss"],
        "condition_multiscale_gradient_loss": condition_multiscale,
        "condition_gradient_ratio": condition_prediction_energy
        / condition_target_energy.clamp_min(1e-12),
        "response_total": response["total"],
        "response_mse": response["mse"],
        "response_cosine": response["cosine"],
        "response_spatial_gradient_loss": response["spatial_gradient_loss"],
        "response_multiscale_gradient_loss": response["multiscale_gradient_loss"],
        "response_gradient_ratio": response["gradient_energy_ratio"],
    }


def serializable_settings(args):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key != "resume"
    }


def save_checkpoint(path, adapter, optimizer, step, next_offset, permutation, args, manifest_hash):
    payload = {
        "format": CHECKPOINT_FORMAT,
        "adapter_architecture": "v2_3_thermal_detail_skip",
        "global_step": step,
        "next_sample_offset": next_offset,
        "permutation": permutation,
        "manifest_sha256": manifest_hash,
        "settings": serializable_settings(args),
        "adapter": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
        "optimizer": optimizer.state_dict(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def expected_training_shape(sample_count: int, batch_size: int):
    if sample_count <= 0 or batch_size <= 0:
        raise ValueError("sample_count and batch_size must be positive.")
    return {
        "optimizer_updates": math.ceil(sample_count / batch_size),
        "last_batch_size": sample_count % batch_size or batch_size,
    }


def validate_resume_position(step: int, offset: int, sample_count: int, batch_size: int) -> None:
    shape = expected_training_shape(sample_count, batch_size)
    if step < 0 or step > shape["optimizer_updates"]:
        raise RuntimeError(f"Resume step {step} is outside the frozen epoch.")
    expected_offset = min(step * batch_size, sample_count)
    if offset != expected_offset:
        raise RuntimeError(
            f"Resume offset {offset} does not match step {step} expected {expected_offset}."
        )


def main() -> None:
    args = parse_args()
    if args.batch_size != 4 or args.timestep != 999:
        raise ValueError("Formal V2.4 training requires batch size 4 and timestep 999.")
    if min(
        args.condition_weight,
        args.response_weight,
        args.response_cosine_loss_weight,
        args.response_spatial_gradient_weight,
        args.response_multiscale_gradient_weight,
    ) < 0:
        raise ValueError("All V2.4 loss weights must be non-negative.")
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    manifest = args.train_manifest.resolve()
    ms2_root = args.ms2_root.resolve()
    manifest_hash = sha256(manifest)
    rows = read_train_manifest(manifest, ms2_root)
    shape = expected_training_shape(len(rows), args.batch_size)
    total_updates = shape["optimizer_updates"]
    permutation = list(range(len(rows)))
    random.Random(args.seed).shuffle(permutation)
    output = args.output_dir.resolve()
    if args.resume is None:
        if output.exists() and any(output.iterdir()):
            raise SystemExit(f"Refusing non-empty output directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=True)

    frozen_config = {
        "route": "pure AnyThermal + thermal detail Adapter V2.3 + frozen Lotus U-Net",
        "training_objective": "V2.4 condition + frozen-U-Net response/gradient consistency",
        "initialization": "fresh random Adapter V2.3; no smoke checkpoint",
        "teacher_training_only": "frozen Thermal-VAE mode condition",
        "inference_uses_thermal_vae": False,
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "train_samples": len(rows),
        "epochs": 1,
        "batch_size": args.batch_size,
        "drop_last": False,
        **shape,
        "checkpoint_steps": [*SAVE_STEPS, total_updates],
        "selection_rule": "end-of-complete-epoch checkpoint; no cherry-picking",
        "uses_val": False,
        "uses_test": False,
        "v1_checkpoint_used": False,
        "caption_mode": args.caption_mode,
        "caption": (
            f"per-sample manifest caption, dropout {args.caption_dropout} (Arm-1 protocol)"
            if args.caption_mode == "correct"
            else "empty prompt throughout"
        ),
        "settings": serializable_settings(args),
    }
    config_path = output / "frozen_config.json"
    if not config_path.exists():
        config_path.write_text(
            json.dumps(frozen_config, indent=2, ensure_ascii=False), encoding="utf-8"
        )

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
    empty_prompt, _ = lotus.encode_prompt(
        prompt="", device=device, num_images_per_prompt=1, do_classifier_free_guidance=None
    )
    empty_prompt = empty_prompt.detach().to(dtype=dtype)

    if args.caption_mode == "correct":
        missing_captions = [row["id"] for row in rows if not row["caption"].strip()]
        if missing_captions:
            raise SystemExit(
                f"--caption-mode correct but {len(missing_captions)} rows lack captions "
                f"(first: {missing_captions[:3]}); use the clip75 train manifest"
            )
        caption_rng = random.Random(args.seed + 424242)

        def prompt_for(row):
            if caption_rng.random() < args.caption_dropout:
                return empty_prompt
            encoded, _ = lotus.encode_prompt(
                prompt=row["caption"], device=device,
                num_images_per_prompt=1, do_classifier_free_guidance=None,
            )
            return encoded.detach().to(dtype=dtype)
    else:
        def prompt_for(row):
            del row
            return empty_prompt
    adapter = AnyThermalLotusAdapterV23().to(device=device, dtype=torch.float32).train()
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    start_step = 0
    next_offset = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume.resolve(), map_location="cpu", weights_only=False)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise RuntimeError("Resume checkpoint is not a formal V2.4 run.")
        if checkpoint.get("manifest_sha256") != manifest_hash:
            raise RuntimeError("Resume manifest hash differs from the frozen run.")
        if checkpoint.get("permutation") != permutation:
            raise RuntimeError("Resume sample permutation differs from the frozen run.")
        if checkpoint.get("settings") != serializable_settings(args):
            raise RuntimeError("Resume settings differ from the frozen V2.4 run.")
        adapter.load_state_dict(checkpoint["adapter"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["global_step"])
        next_offset = int(checkpoint["next_sample_offset"])
        validate_resume_position(start_step, next_offset, len(rows), args.batch_size)
    if start_step == 0 and next_offset == 0:
        save_checkpoint(
            output / "adapter_step_0000.pt",
            adapter,
            optimizer,
            0,
            0,
            permutation,
            args,
            manifest_hash,
        )

    log_path = output / "training_metrics.jsonl"
    diagnostics_path = output / "thermal_audit_first8.json"
    first_diagnostics = []
    last_record = None
    start_time = time.time()
    with log_path.open("a", encoding="utf-8") as log_handle:
        for step in range(start_step + 1, total_updates + 1):
            batch_indices = permutation[next_offset : next_offset + args.batch_size]
            if not batch_indices:
                raise RuntimeError(f"Empty batch at step {step} offset {next_offset}.")
            # Lotus encode_prompt uses do_not_pad, so caption lengths vary; a
            # mixed-length batch falls back to per-sample U-Net forwards
            # (mathematically identical -- the U-Net has no cross-sample ops).
            sample_prompts = [prompt_for(rows[i]) for i in batch_indices]
            if len({p.shape[1] for p in sample_prompts}) == 1:
                prompt_batch = torch.cat(sample_prompts, dim=0)
            else:
                prompt_batch = sample_prompts
            batch = build_batch(
                rows, batch_indices, anythermal, lotus, prompt_batch, args, device, dtype
            )
            if len(first_diagnostics) < 8:
                first_diagnostics.extend(batch[-1][: 8 - len(first_diagnostics)])
                if len(first_diagnostics) == 8:
                    diagnostics_path.write_text(
                        json.dumps(first_diagnostics, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
            optimizer.zero_grad(set_to_none=True)
            losses = forward_losses(adapter, lotus, prompt_batch, batch, args, device, dtype)
            total_loss = losses["total_loss"]
            if not bool(torch.isfinite(total_loss)):
                raise RuntimeError(f"Non-finite loss at step {step}.")
            total_loss.backward()
            gradients = [
                parameter.grad for parameter in adapter.parameters() if parameter.grad is not None
            ]
            if not gradients or not all(torch.isfinite(value).all() for value in gradients):
                raise RuntimeError(f"Missing/non-finite Adapter gradient at step {step}.")
            if any(parameter.grad is not None for parameter in lotus.unet.parameters()):
                raise RuntimeError("Frozen U-Net unexpectedly owns gradients.")
            optimizer.step()
            next_offset += len(batch_indices)
            last_record = {
                "step": step,
                "samples_seen": next_offset,
                "batch_size": len(batch_indices),
                "elapsed_seconds": time.time() - start_time,
                **{key: float(value.detach().cpu()) for key, value in losses.items()},
            }
            log_handle.write(json.dumps(last_record) + "\n")
            log_handle.flush()
            if step % args.log_interval == 0 or step == 1 or step == total_updates:
                print(json.dumps(last_record), flush=True)
            if step in SAVE_STEPS or step == total_updates:
                name = "adapter_end.pt" if step == total_updates else f"adapter_step_{step:04d}.pt"
                save_checkpoint(
                    output / name,
                    adapter,
                    optimizer,
                    step,
                    next_offset,
                    permutation,
                    args,
                    manifest_hash,
                )

    if next_offset != len(rows):
        raise RuntimeError(f"Incomplete epoch: consumed {next_offset}/{len(rows)} samples.")
    if first_diagnostics or not diagnostics_path.exists():
        diagnostics_path.write_text(
            json.dumps(first_diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    summary = {
        **frozen_config,
        "completed": True,
        "global_step": total_updates,
        "samples_seen": next_offset,
        "end_checkpoint": str(output / "adapter_end.pt"),
        "end_checkpoint_sha256": sha256(output / "adapter_end.pt"),
        "elapsed_seconds": time.time() - start_time,
        "final_training_record": last_record,
        "frozen_modules": {
            "anythermal": frozen_audit(anythermal.model),
            "vae": frozen_audit(lotus.vae),
            "text_encoder": frozen_audit(lotus.text_encoder),
            "unet": frozen_audit(lotus.unet),
        },
        "adapter_trainable_parameters": int(
            sum(parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad)
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
