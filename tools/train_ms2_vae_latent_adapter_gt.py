"""Six-route line e: train a latent adapter on the frozen Thermal-VAE condition.

    thermal -> frozen Lotus VAE mode-encoding -> TRAINABLE latent adapter
    -> frozen Lotus U-Net -> disparity -> masked GT SSI

Single-variable counterpart of line f (AnyThermal features -> Adapter): the
trainable module plays the same role, only the feature source differs.  The
adapter is identity at initialization (zero-init output conv), so step 0
reproduces the zero-training thermal baseline (line c).

Distilling to the Thermal-VAE condition would be degenerate here (the teacher
target equals the adapter input), so the anchor is frozen-teacher RESPONSE
consistency: the pretrained U-Net's response to the unadapted condition.  GT
supervision is the champion-protocol masked SSI L1 decoded through an fp32
VAE copy (``--gt-decode-fp32`` recommended).  The gradient path to the adapter
runs through a frozen fp32 U-Net copy (fp16 backward underflows -- the same
lesson as the GT-decode fix).

Champion protocol otherwise: empty prompt, micro-batch 1 x accumulation 4,
seed 20260703, 1 epoch, end-of-epoch checkpoint, no Val/Test at train time.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = ROOT / "lotus"
TOOLS_ROOT = ROOT / "tools"
for path in (ROOT, LOTUS_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.anythermal_lotus_v2 import thermal_to_lotus_input  # noqa: E402
from models.anythermal_lotus_v2_4 import response_consistency_losses, seeded_noise  # noqa: E402
from models.thermal_vae_latent_adapter import ThermalVAELatentAdapter  # noqa: E402
from train_ms2_joint_gt_v3 import (  # noqa: E402
    load_gt_disparity,
    masked_ssi_l1,
    decode_to_disparity,
)
from train_ms2_thermal_vae_unet_gt import (  # noqa: E402
    expected_training_shape,
    frozen_audit,
    read_train_manifest,
    seed_everything,
    serializable_settings,
    sha256,
    task_embedding,
    validate_resume_position,
)


DEFAULT_MANIFEST = Path(
    "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/"
    "ms2_fixed_sequence_internvl3_8b_filtered_caption_train.jsonl"
)
SAVE_STEPS = (0, 100, 500, 1000, 2000)
CHECKPOINT_FORMAT = "thermal_vae_latent_adapter_full_epoch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ms2-root", type=Path, default=Path("/mnt/e/dataset/ms2"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lotus_line_v2/full_train_epoch1_vae_latent_adapter_gt"),
    )
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--adapter-hidden-channels", type=int, default=256)
    parser.add_argument("--adapter-blocks", type=int, default=6)
    parser.add_argument("--adapter-learning-rate", type=float, default=1e-4)
    parser.add_argument("--gt-loss-weight", type=float, default=5.0)
    parser.add_argument("--gt-min-depth", type=float, default=0.1)
    parser.add_argument("--gt-max-depth", type=float, default=80.0)
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--min-gt-valid-fraction", type=float, default=0.0,
                        help="Drop manifest rows whose gt_valid_fraction is below this "
                             "(RGBDT500 only; MS2 rows carry no such field and pass through).")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adapter-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--response-weight", type=float, default=1.0)
    parser.add_argument("--response-cosine-loss-weight", type=float, default=0.1)
    parser.add_argument("--response-spatial-gradient-weight", type=float, default=0.5)
    parser.add_argument("--response-multiscale-gradient-weight", type=float, default=0.5)
    parser.add_argument("--response-gradient-energy-weight", type=float, default=0.5)
    parser.add_argument(
        "--caption-mode",
        choices=("empty", "correct"),
        default="empty",
        help=(
            "correct: encode each sample's manifest caption as the prompt "
            "(Arm-1 protocol; pair with the clip75 train manifest). Default "
            "empty reproduces the original line-e run exactly."
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
    parser.add_argument("--teacher-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument(
        "--gt-decode-fp32",
        action="store_true",
        help="fp32 VAE copy for the GT-loss decode (legacy fp16 backward underflows).",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--smoke-updates", type=int, default=None)
    parser.add_argument("--overfit-steps", type=int, default=None)
    return parser.parse_args()


def validate_protocol(args) -> None:
    if args.micro_batch_size != 1 or args.gradient_accumulation_steps != 4:
        raise ValueError("Line e requires micro-batch 1 and accumulation 4 (champion protocol).")
    if args.timestep != 999:
        raise ValueError("Line e must match inference timestep 999.")
    if args.gt_loss_weight <= 0:
        raise ValueError("gt-loss-weight must be positive.")
    if args.smoke_updates is not None and args.overfit_steps is not None:
        raise ValueError("Choose either --smoke-updates or --overfit-steps, not both.")
    if args.smoke_updates is not None:
        if args.smoke_updates <= 0:
            raise ValueError("--smoke-updates must be positive.")
        if "smoke" not in args.output_dir.name:
            raise ValueError("--smoke-updates requires an output dir name containing 'smoke'.")
        if args.resume is not None:
            raise ValueError("--smoke-updates cannot be combined with --resume.")
    if args.overfit_steps is not None:
        if args.overfit_steps <= 0:
            raise ValueError("--overfit-steps must be positive.")
        if "overfit" not in args.output_dir.name:
            raise ValueError("--overfit-steps requires an output dir name containing 'overfit'.")
        if args.resume is not None:
            raise ValueError("--overfit-steps cannot be combined with --resume.")


def load_sample(row, args):
    thermal = thermal_to_lotus_input(row["thermal_path"], processing_res=0)
    if thermal.diagnostics["converted_uint8_std"] <= 0:
        raise RuntimeError(f"Constant thermal conversion: {row['id']}")
    gt_disparity, valid_mask = load_gt_disparity(
        row["depth_path"], args.gt_min_depth, args.gt_max_depth, args.depth_scale
    )
    diagnostics = {
        "id": row["id"],
        "manifest_index": row["manifest_index"],
        "thermal": thermal.diagnostics,
        "gt_valid_pixels": int(valid_mask.sum()),
    }
    return thermal.tensor, gt_disparity, valid_mask, diagnostics


def encode_condition(lotus, thermal_tensor, device):
    encoder_dtype = next(lotus.vae.encoder.parameters()).dtype
    with torch.no_grad():
        posterior = lotus.vae.encode(
            thermal_tensor.to(device=device, dtype=encoder_dtype)
        ).latent_dist
        condition = posterior.mode() * lotus.vae.config.scaling_factor
    return condition.to(dtype=torch.float32)


def forward_losses(
    row, thermal_tensor, gt_disparity, valid_mask,
    lotus, teacher_unet, frozen_unet_fp32, adapter, prompt, prompt_fp32,
    args, device, teacher_dtype, gt_vae=None,
):
    condition_original = encode_condition(lotus, thermal_tensor, device)
    condition = adapter(condition_original)
    noise = seeded_noise(
        (1, *condition.shape[1:]),
        seed=args.seed + int(row["manifest_index"]),
        device=device,
        dtype=teacher_dtype,
        scale=float(lotus.scheduler.init_noise_sigma),
    )
    timestep = torch.full((1,), args.timestep, device=device, dtype=torch.long)
    latent_input = lotus.scheduler.scale_model_input(noise, timestep).detach()
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=teacher_dtype,
        enabled=device.type == "cuda" and teacher_dtype == torch.float16,
    ):
        teacher_response = teacher_unet(
            torch.cat([condition_original.detach().to(teacher_dtype), latent_input], dim=1),
            timestep,
            encoder_hidden_states=prompt.to(dtype=teacher_dtype),
            class_labels=task_embedding(1, device, teacher_dtype),
            return_dict=False,
        )[0]
    # frozen fp32 U-Net: parameters have no grad, but the input path does
    student_response = frozen_unet_fp32(
        torch.cat([condition, latent_input.float()], dim=1),
        timestep,
        encoder_hidden_states=prompt_fp32,
        class_labels=task_embedding(1, device, torch.float32),
        return_dict=False,
    )[0]
    response = response_consistency_losses(
        student_response,
        teacher_response.detach().float(),
        cosine_weight=args.response_cosine_loss_weight,
        spatial_gradient_weight=args.response_spatial_gradient_weight,
        multiscale_gradient_weight=args.response_multiscale_gradient_weight,
        gradient_energy_weight=args.response_gradient_energy_weight,
    )
    predicted_disparity = decode_to_disparity(lotus, student_response, device, gt_vae=gt_vae)
    if predicted_disparity.shape != gt_disparity.shape[-2:]:
        raise RuntimeError("Decoded disparity does not match GT shape.")
    gt_loss, gt_abs_rel, gt_valid = masked_ssi_l1(
        predicted_disparity[None], gt_disparity, valid_mask
    )
    condition_delta = float(
        torch.mean(torch.abs(condition.detach() - condition_original.detach()))
    )
    total = args.response_weight * response["total"] + args.gt_loss_weight * gt_loss
    return {
        "total": total,
        "response_total": response["total"],
        "response_mse": response["mse"],
        "response_cosine": response["cosine"],
        "response_gradient_energy_ratio": response["gradient_energy_ratio"],
        "gt_ssi_l1": gt_loss,
        "gt_abs_rel": gt_abs_rel,
        "gt_valid_pixels": torch.tensor(float(gt_valid)),
        "condition_delta_l1": torch.tensor(condition_delta),
    }


def save_checkpoint(path, adapter, optimizer, step, next_offset, permutation, args, manifest_hash):
    payload = {
        "format": CHECKPOINT_FORMAT,
        "train_mode": "vae_latent_adapter",
        "adapter_architecture": "thermal_vae_latent_adapter_v1",
        "adapter_hidden_channels": args.adapter_hidden_channels,
        "adapter_blocks": args.adapter_blocks,
        "gt_loss_weight": args.gt_loss_weight,
        "global_step": step,
        "next_sample_offset": next_offset,
        "permutation": permutation,
        "manifest_sha256": manifest_hash,
        "settings": serializable_settings(args),
        "adapter_state_dict": {
            key: value.detach().cpu() for key, value in adapter.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    validate_protocol(args)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    teacher_dtype = torch.float16 if args.teacher_dtype == "fp16" else torch.float32
    effective_batch = args.micro_batch_size * args.gradient_accumulation_steps
    manifest = args.train_manifest.resolve()
    ms2_root = args.ms2_root.resolve()
    manifest_hash = sha256(manifest)
    rows = read_train_manifest(manifest, ms2_root, args.min_gt_valid_fraction)
    shape = expected_training_shape(len(rows), effective_batch)
    total_updates = shape["optimizer_updates"]
    if args.smoke_updates is not None:
        total_updates = min(args.smoke_updates, total_updates)
    if args.overfit_steps is not None:
        total_updates = args.overfit_steps
    import random as random_module

    permutation = list(range(len(rows)))
    random_module.Random(args.seed).shuffle(permutation)
    overfit_pool = permutation[:32]
    output = args.output_dir.resolve()
    if args.resume is None:
        if output.exists() and any(output.iterdir()):
            raise SystemExit(f"Refusing non-empty output directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=True)

    frozen_config = {
        "route": "Line e: frozen Thermal-VAE latent + TRAINABLE latent adapter + frozen Lotus U-Net + masked GT",
        "six_route_line": "e",
        "single_variable_against": "line f (AnyThermal features -> Adapter): only the feature source differs",
        "training_objective": (
            "frozen-teacher response consistency (anchor; teacher = pretrained U-Net on the "
            "UNADAPTED condition) + masked scale-shift-invariant L1"
        ),
        "condition_source": "frozen Lotus VAE mode-encoding of thermal, then trainable residual adapter",
        "adapter_identity_at_init": True,
        "caption": (
            f"per-sample manifest caption, dropout {args.caption_dropout} (Arm-1 protocol)"
            if args.caption_mode == "correct"
            else "empty prompt throughout"
        ),
        "caption_mode": args.caption_mode,
        "gt_loss_weight": args.gt_loss_weight,
        "initialization": "pretrained Lotus frozen (fp16 teacher + fp32 gradient-path copy); adapter identity",
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "train_samples": len(rows),
        "epochs": 1,
        "effective_batch_size": effective_batch,
        "micro_batch_size": args.micro_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "drop_last": False,
        **shape,
        "checkpoint_steps": [*SAVE_STEPS, shape["optimizer_updates"]],
        "selection_rule": "end-of-complete-epoch checkpoint; no cherry-picking",
        "uses_val": False,
        "uses_test": False,
        "uses_depth_gt": True,
        "smoke_updates": args.smoke_updates,
        "overfit_steps": args.overfit_steps,
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
        torch_dtype=teacher_dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    teacher_unet = lotus.unet
    for module in (lotus.vae, lotus.text_encoder, teacher_unet):
        module.requires_grad_(False).eval()
    frozen_unet_fp32 = copy.deepcopy(teacher_unet).to(device=device, dtype=torch.float32)
    frozen_unet_fp32.requires_grad_(False).eval()
    gt_vae = None
    if args.gt_decode_fp32:
        gt_vae = copy.deepcopy(lotus.vae).to(device=device, dtype=torch.float32)
        gt_vae.requires_grad_(False).eval()
        gt_vae.encoder = None
    empty_prompt, _ = lotus.encode_prompt(
        prompt="", device=device, num_images_per_prompt=1, do_classifier_free_guidance=None
    )
    empty_prompt = empty_prompt.detach()

    if args.caption_mode == "correct":
        missing_captions = [row["id"] for row in rows if not row["caption"].strip()]
        if missing_captions:
            raise SystemExit(
                f"--caption-mode correct but {len(missing_captions)} rows lack captions "
                f"(first: {missing_captions[:3]}); use the clip75 train manifest"
            )
        caption_rng = random_module.Random(args.seed + 424242)

        def prompt_for(row):
            if caption_rng.random() < args.caption_dropout:
                return empty_prompt
            encoded, _ = lotus.encode_prompt(
                prompt=row["caption"], device=device,
                num_images_per_prompt=1, do_classifier_free_guidance=None,
            )
            return encoded.detach()
    else:
        def prompt_for(row):
            del row
            return empty_prompt

    adapter = ThermalVAELatentAdapter(
        hidden_channels=args.adapter_hidden_channels, blocks=args.adapter_blocks
    ).to(device=device, dtype=torch.float32)
    adapter.train().requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [{"params": adapter.parameters(), "lr": args.adapter_learning_rate}],
        weight_decay=args.weight_decay,
    )

    start_step = 0
    next_offset = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume.resolve(), map_location="cpu", weights_only=False)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise RuntimeError("Resume checkpoint is not a formal line-e run.")
        if checkpoint.get("manifest_sha256") != manifest_hash:
            raise RuntimeError("Resume manifest hash differs from the frozen run.")
        if checkpoint.get("permutation") != permutation:
            raise RuntimeError("Resume sample permutation differs from the frozen run.")
        if checkpoint.get("settings") != serializable_settings(args):
            raise RuntimeError("Resume settings differ from the frozen line-e run.")
        adapter.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["global_step"])
        next_offset = int(checkpoint["next_sample_offset"])
        validate_resume_position(start_step, next_offset, len(rows), effective_batch)
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    formal_run = args.smoke_updates is None and args.overfit_steps is None
    if start_step == 0 and next_offset == 0 and formal_run:
        save_checkpoint(
            output / "vae_adapter_step_0000.pt",
            adapter, optimizer, 0, 0, permutation, args, manifest_hash,
        )

    log_path = output / "training_metrics.jsonl"
    diagnostics_path = output / "thermal_audit_first8.json"
    first_diagnostics = []
    last_record = None
    start_time = time.time()
    with log_path.open("a", encoding="utf-8") as log_handle:
        for step in range(start_step + 1, total_updates + 1):
            if args.overfit_steps is not None:
                base = ((step - 1) * effective_batch) % len(overfit_pool)
                batch_indices = [
                    overfit_pool[(base + offset) % len(overfit_pool)]
                    for offset in range(effective_batch)
                ]
            else:
                batch_indices = permutation[next_offset : next_offset + effective_batch]
            if not batch_indices:
                raise RuntimeError(f"Empty batch at step {step} offset {next_offset}.")
            optimizer.zero_grad(set_to_none=True)
            metric_sums = {}
            for row_index in batch_indices:
                row = rows[row_index]
                thermal_tensor, gt_disparity, valid_mask, diagnostics = load_sample(row, args)
                gt_disparity = gt_disparity.to(device)
                valid_mask = valid_mask.to(device)
                if len(first_diagnostics) < 8:
                    first_diagnostics.append(diagnostics)
                    if len(first_diagnostics) == 8:
                        diagnostics_path.write_text(
                            json.dumps(first_diagnostics, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                sample_prompt = prompt_for(row)
                losses = forward_losses(
                    row, thermal_tensor, gt_disparity, valid_mask,
                    lotus, teacher_unet, frozen_unet_fp32, adapter,
                    sample_prompt, sample_prompt.to(dtype=torch.float32),
                    args, device, teacher_dtype, gt_vae=gt_vae,
                )
                if not bool(torch.isfinite(losses["total"])):
                    raise RuntimeError(f"Non-finite loss at step {step} sample {row['id']}.")
                (losses["total"] / len(batch_indices)).backward()
                for key, value in losses.items():
                    if value.ndim == 0:
                        metric_sums[key] = metric_sums.get(key, 0.0) + float(value.detach())
            for guard_name, guard_module in (
                ("teacher U-Net", teacher_unet),
                ("frozen fp32 U-Net", frozen_unet_fp32),
                ("VAE", lotus.vae),
            ):
                if any(parameter.grad is not None for parameter in guard_module.parameters()):
                    raise RuntimeError(f"Frozen {guard_name} unexpectedly owns gradients.")
            adapter_grad_norm = torch.nn.utils.clip_grad_norm_(
                adapter.parameters(), args.adapter_max_grad_norm
            )
            if not bool(torch.isfinite(adapter_grad_norm)):
                raise RuntimeError(f"Non-finite adapter gradient at step {step}.")
            optimizer.step()
            if args.overfit_steps is None:
                next_offset += len(batch_indices)
            last_record = {
                "step": step,
                "samples_seen": next_offset,
                "batch_size": len(batch_indices),
                "elapsed_seconds": time.time() - start_time,
                "adapter_grad_norm": float(adapter_grad_norm),
                **{key: value / len(batch_indices) for key, value in metric_sums.items()},
            }
            log_handle.write(json.dumps(last_record) + "\n")
            log_handle.flush()
            if step % args.log_interval == 0 or step == 1 or step == total_updates:
                print(json.dumps(last_record), flush=True)
            if formal_run and (step in SAVE_STEPS or step == total_updates):
                name = (
                    "vae_adapter_end.pt"
                    if step == total_updates
                    else f"vae_adapter_step_{step:04d}.pt"
                )
                save_checkpoint(
                    output / name,
                    adapter, optimizer, step, next_offset,
                    permutation, args, manifest_hash,
                )

    if not formal_run:
        label = "smoke_only" if args.smoke_updates is not None else "overfit_only"
        partial = {
            **frozen_config,
            label: True,
            "completed_updates": total_updates,
            "final_training_record": last_record,
            "elapsed_seconds": time.time() - start_time,
        }
        name = "smoke_summary.json" if args.smoke_updates is not None else "overfit_summary.json"
        (output / name).write_text(json.dumps(partial, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(partial, indent=2, ensure_ascii=False))
        return

    if next_offset != len(rows):
        raise RuntimeError(f"Incomplete epoch: consumed {next_offset}/{len(rows)} samples.")
    if first_diagnostics and not diagnostics_path.exists():
        diagnostics_path.write_text(
            json.dumps(first_diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    summary = {
        **frozen_config,
        "completed": True,
        "global_step": total_updates,
        "samples_seen": next_offset,
        "end_checkpoint": str(output / "vae_adapter_end.pt"),
        "end_checkpoint_sha256": sha256(output / "vae_adapter_end.pt"),
        "elapsed_seconds": time.time() - start_time,
        "frozen_modules": {
            "vae": frozen_audit(lotus.vae),
            "text_encoder": frozen_audit(lotus.text_encoder),
            "teacher_unet": frozen_audit(teacher_unet),
            "frozen_unet_fp32": frozen_audit(frozen_unet_fp32),
        },
        "adapter_trainable_parameters": int(
            sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        ),
        "final_training_record": last_record,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
