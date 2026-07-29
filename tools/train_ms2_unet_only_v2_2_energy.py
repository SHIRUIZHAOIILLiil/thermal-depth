"""Train the U-Net-only route (Direct AnyThermal condition) for one full epoch.

Full-scale version of the gate-passing ``short_128_unet_only_v2_2_energy``
recipe.  A frozen fp16 copy of the pretrained Lotus U-Net provides teacher
responses from the Thermal-VAE mode condition; the fp32 student U-Net is
trained with zero-parameter Direct AnyThermal condition against those
responses.  No depth GT, Val, Test, caption, V1 checkpoint, Adapter
parameters, or Thermal-VAE inference path is used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

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


DEFAULT_MANIFEST = Path(
    "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/"
    "ms2_fixed_sequence_internvl3_8b_filtered_caption_train.jsonl"
)
SAVE_STEPS = (0, 100, 500, 1000, 2000)
CHECKPOINT_FORMAT = "unet_only_v2_2_energy_full_epoch_response"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ms2-root", type=Path, default=Path("/mnt/e/dataset/ms2"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lotus_line_v2/full_train_epoch1_unet_only_v2_2_energy"),
    )
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
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
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume only this exact U-Net-only full run after interruption.",
    )
    parser.add_argument(
        "--smoke-updates",
        type=int,
        default=None,
        help="Run only N optimizer updates into a throwaway 'smoke' output dir.",
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


def validate_protocol(args) -> None:
    if args.micro_batch_size != 1 or args.gradient_accumulation_steps != 4:
        raise ValueError("U-Net-only V2.2-energy requires micro-batch 1 and accumulation 4.")
    if args.timestep != 999:
        raise ValueError("U-Net-only V2.2-energy must match inference timestep 999.")
    if min(
        args.learning_rate,
        args.max_grad_norm,
        args.response_cosine_loss_weight,
        args.response_spatial_gradient_weight,
        args.response_multiscale_gradient_weight,
        args.response_gradient_energy_weight,
    ) < 0:
        raise ValueError("Learning rate, clipping, and loss weights must be non-negative.")
    if args.smoke_updates is not None:
        if args.smoke_updates <= 0:
            raise ValueError("--smoke-updates must be positive.")
        if "smoke" not in args.output_dir.name:
            raise ValueError("--smoke-updates requires an output dir name containing 'smoke'.")
        if args.resume is not None:
            raise ValueError("--smoke-updates cannot be combined with --resume.")


def validate_trainability(direct, student_unet, frozen_modules) -> None:
    if sum(parameter.numel() for parameter in direct.parameters()) != 0:
        raise RuntimeError("U-Net-only Direct conditioner must have zero parameters.")
    if not any(parameter.requires_grad for parameter in student_unet.parameters()):
        raise RuntimeError("Student U-Net has no trainable parameters.")
    for module in frozen_modules:
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise RuntimeError("A frozen U-Net-only support module is trainable.")


def build_sample(row, anythermal, lotus, teacher_unet, direct, prompt, args, device, teacher_dtype):
    """Prepare Direct condition, latent input, and teacher response for one sample."""
    thermal = thermal_to_lotus_input(row["thermal_path"], processing_res=0)
    if thermal.diagnostics["converted_uint8_std"] <= 0:
        raise RuntimeError(f"Constant thermal conversion: {row['id']}")
    teacher_condition = encode_condition_latent(
        lotus.vae, thermal.tensor, posterior="mode"
    ).to(device=device, dtype=teacher_dtype)
    features, _, anythermal_diag = extract_anythermal_feature_pyramid(
        anythermal, row["thermal_path"], enable_grad=False
    )
    direct_condition = direct(
        [feature.to(device=device, dtype=torch.float32) for feature in features],
        target_size=tuple(teacher_condition.shape[-2:]),
    )
    noise = seeded_noise(
        teacher_condition.shape,
        seed=args.seed + int(row["manifest_index"]),
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
        teacher_response = teacher_unet(
            torch.cat([teacher_condition, latent_input], dim=1),
            timestep,
            encoder_hidden_states=prompt.to(dtype=teacher_dtype),
            class_labels=task_embedding(1, device, teacher_dtype),
            return_dict=False,
        )[0]
    diagnostics = {
        "id": row["id"],
        "manifest_index": row["manifest_index"],
        "thermal": thermal.diagnostics,
        "anythermal_converted_std": anythermal_diag["converted_uint8_std"],
    }
    return (
        direct_condition.detach().float(),
        latent_input.detach().float(),
        teacher_response.detach().float(),
        diagnostics,
    )


def forward_losses(student_unet, prompt_fp32, sample, args, device):
    direct_condition, latent_input, teacher_response = sample
    timestep = torch.full(
        (direct_condition.shape[0],), args.timestep, device=device, dtype=torch.long
    )
    student_response = student_unet(
        torch.cat([direct_condition, latent_input], dim=1),
        timestep,
        encoder_hidden_states=prompt_fp32.repeat(direct_condition.shape[0], 1, 1),
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


def serializable_settings(args):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in ("resume", "smoke_updates")
    }


def save_checkpoint(path, student_unet, optimizer, step, next_offset, permutation, args, manifest_hash):
    payload = {
        "format": CHECKPOINT_FORMAT,
        "train_mode": "unet_only",
        "adapter_architecture": "direct_zero_parameter",
        "global_step": step,
        "next_sample_offset": next_offset,
        "permutation": permutation,
        "manifest_sha256": manifest_hash,
        "settings": serializable_settings(args),
        "adapter": {},
        "lotus_unet_state_dict": {
            key: value.detach().cpu() for key, value in student_unet.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def expected_training_shape(sample_count: int, effective_batch: int):
    if sample_count <= 0 or effective_batch <= 0:
        raise ValueError("sample_count and effective_batch must be positive.")
    return {
        "optimizer_updates": math.ceil(sample_count / effective_batch),
        "last_batch_size": sample_count % effective_batch or effective_batch,
    }


def validate_resume_position(step: int, offset: int, sample_count: int, effective_batch: int) -> None:
    shape = expected_training_shape(sample_count, effective_batch)
    if step < 0 or step > shape["optimizer_updates"]:
        raise RuntimeError(f"Resume step {step} is outside the frozen epoch.")
    expected_offset = min(step * effective_batch, sample_count)
    if offset != expected_offset:
        raise RuntimeError(
            f"Resume offset {offset} does not match step {step} expected {expected_offset}."
        )


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
    rows = read_train_manifest(manifest, ms2_root)
    shape = expected_training_shape(len(rows), effective_batch)
    total_updates = shape["optimizer_updates"]
    if args.smoke_updates is not None:
        total_updates = min(args.smoke_updates, total_updates)
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
        "route": "Direct AnyThermal condition + trained Lotus U-Net",
        "training_objective": "V2.2-energy frozen-teacher response/gradient consistency",
        "initialization": "pretrained Lotus U-Net (fp16 load, fp32 student cast); fresh optimizer",
        "teacher_training_only": "frozen Thermal-VAE mode condition + frozen pretrained U-Net copy",
        "inference_uses_thermal_vae": False,
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
        "uses_depth_gt": False,
        "v1_checkpoint_used": False,
        "smoke_updates": args.smoke_updates,
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
    anythermal = AnyThermalEncoder(
        model_path=args.anythermal_model_path,
        device=str(device),
        local_files_only=args.local_files_only,
    )
    direct = AnyThermalDirectConditioner().to(device=device, dtype=torch.float32)
    teacher_unet = lotus.unet
    for module in (lotus.vae, lotus.text_encoder, teacher_unet, anythermal.model, direct):
        module.requires_grad_(False).eval()
    prompt, _ = lotus.encode_prompt(
        prompt="", device=device, num_images_per_prompt=1, do_classifier_free_guidance=None
    )
    prompt = prompt.detach()
    prompt_fp32 = prompt.to(dtype=torch.float32)

    # Student starts bit-identical to the frozen teacher, mirroring the passed
    # short gate which upcast the fp16-loaded pretrained U-Net to fp32.
    student_unet = copy.deepcopy(teacher_unet).to(device=device, dtype=torch.float32)
    student_unet.train().requires_grad_(True)
    validate_trainability(
        direct, student_unet, (lotus.vae, lotus.text_encoder, teacher_unet, anythermal.model)
    )
    optimizer = torch.optim.AdamW(
        student_unet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    start_step = 0
    next_offset = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume.resolve(), map_location="cpu", weights_only=False)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise RuntimeError("Resume checkpoint is not a formal U-Net-only full run.")
        if checkpoint.get("manifest_sha256") != manifest_hash:
            raise RuntimeError("Resume manifest hash differs from the frozen run.")
        if checkpoint.get("permutation") != permutation:
            raise RuntimeError("Resume sample permutation differs from the frozen run.")
        if checkpoint.get("settings") != serializable_settings(args):
            raise RuntimeError("Resume settings differ from the frozen U-Net-only run.")
        student_unet.load_state_dict(checkpoint["lotus_unet_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["global_step"])
        next_offset = int(checkpoint["next_sample_offset"])
        validate_resume_position(start_step, next_offset, len(rows), effective_batch)
    if start_step == 0 and next_offset == 0 and args.smoke_updates is None:
        save_checkpoint(
            output / "unet_step_0000.pt",
            student_unet,
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
            batch_indices = permutation[next_offset : next_offset + effective_batch]
            if not batch_indices:
                raise RuntimeError(f"Empty batch at step {step} offset {next_offset}.")
            optimizer.zero_grad(set_to_none=True)
            metric_sums = {}
            for row_index in batch_indices:
                row = rows[row_index]
                direct_condition, latent_input, teacher_response, diagnostics = build_sample(
                    row, anythermal, lotus, teacher_unet, direct, prompt, args, device, teacher_dtype
                )
                if len(first_diagnostics) < 8:
                    first_diagnostics.append(diagnostics)
                    if len(first_diagnostics) == 8:
                        diagnostics_path.write_text(
                            json.dumps(first_diagnostics, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                losses = forward_losses(
                    student_unet,
                    prompt_fp32,
                    (direct_condition, latent_input, teacher_response),
                    args,
                    device,
                )
                if not bool(torch.isfinite(losses["total"])):
                    raise RuntimeError(f"Non-finite loss at step {step} sample {row['id']}.")
                (losses["total"] / len(batch_indices)).backward()
                for key, value in losses.items():
                    if value.ndim == 0:
                        metric_sums[key] = metric_sums.get(key, 0.0) + float(value.detach())
            if any(parameter.grad is not None for parameter in direct.parameters()):
                raise RuntimeError("Zero-parameter Direct conditioner unexpectedly owns gradients.")
            if any(parameter.grad is not None for parameter in teacher_unet.parameters()):
                raise RuntimeError("Frozen teacher U-Net unexpectedly owns gradients.")
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student_unet.parameters(), args.max_grad_norm
            )
            if not bool(torch.isfinite(grad_norm)):
                raise RuntimeError(f"Non-finite U-Net gradient at step {step}.")
            optimizer.step()
            next_offset += len(batch_indices)
            last_record = {
                "step": step,
                "samples_seen": next_offset,
                "batch_size": len(batch_indices),
                "elapsed_seconds": time.time() - start_time,
                "grad_norm": float(grad_norm),
                **{
                    key: value / len(batch_indices)
                    for key, value in metric_sums.items()
                },
            }
            log_handle.write(json.dumps(last_record) + "\n")
            log_handle.flush()
            if step % args.log_interval == 0 or step == 1 or step == total_updates:
                print(json.dumps(last_record), flush=True)
            if args.smoke_updates is None and (
                step in SAVE_STEPS or step == total_updates
            ):
                name = "unet_end.pt" if step == total_updates else f"unet_step_{step:04d}.pt"
                save_checkpoint(
                    output / name,
                    student_unet,
                    optimizer,
                    step,
                    next_offset,
                    permutation,
                    args,
                    manifest_hash,
                )

    if args.smoke_updates is not None:
        smoke_summary = {
            **frozen_config,
            "smoke_only": True,
            "completed_updates": total_updates,
            "final_training_record": last_record,
            "elapsed_seconds": time.time() - start_time,
        }
        (output / "smoke_summary.json").write_text(
            json.dumps(smoke_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(smoke_summary, indent=2, ensure_ascii=False))
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
        "end_checkpoint": str(output / "unet_end.pt"),
        "end_checkpoint_sha256": sha256(output / "unet_end.pt"),
        "elapsed_seconds": time.time() - start_time,
        "final_training_record": last_record,
        "frozen_modules": {
            "anythermal": frozen_audit(anythermal.model),
            "vae": frozen_audit(lotus.vae),
            "text_encoder": frozen_audit(lotus.text_encoder),
            "teacher_unet": frozen_audit(teacher_unet),
        },
        "conditioner_parameters": 0,
        "unet_trainable_parameters": int(
            sum(
                parameter.numel()
                for parameter in student_unet.parameters()
                if parameter.requires_grad
            )
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
