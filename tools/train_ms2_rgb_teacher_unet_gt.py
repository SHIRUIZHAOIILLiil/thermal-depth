"""Train the RGB depth teacher: Lotus U-Net fine-tuned on MS2 RGB + RGB-view GT.

Same recipe as the thermal champion line (masked SSI lambda=5, 1 epoch,
lr 1e-6, micro-batch 1 x accumulation 4, frozen-teacher response term as a
light trust region, empty prompt).  Input is the native-resolution RGB frame
(cropped to a multiple of 8); GT is the RGB-view filtered LiDAR — same view,
no cross-view supervision anywhere.

Purpose: probe 2 showed zero-shot Lotus reaches only AbsRel ~0.156 on MS2
RGB.  This run measures how far domain fine-tuning lifts the teacher before
we decide whether dense cross-view distillation has profit margin.
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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = ROOT / "lotus"
for path in (ROOT, LOTUS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.anythermal_lotus_v2_4 import response_consistency_losses  # noqa: E402
from train_ms2_joint_gt_v3 import decode_to_disparity, masked_ssi_l1  # noqa: E402


def masked_log_depth_l1(prediction, gt_disparity, valid_mask, min_depth, max_depth):
    """Disparity-aligned L1 in log-depth space: range-equitable relative error.

    Alignment (scale/shift) is solved in disparity space and detached, exactly
    as in masked_ssi_l1; the penalty is then taken on log-depth so a 20% error
    at 40 m costs the same as a 20% error at 4 m.
    """
    mask = valid_mask > 0.5
    count = int(mask.sum())
    if count < 100:
        raise RuntimeError(f"GT valid pixels {count} below minimum 100.")
    pred = prediction[mask]
    gt = gt_disparity[mask]
    with torch.no_grad():
        ones = torch.ones_like(pred)
        design = torch.stack([pred, ones], dim=1)
        solution = torch.linalg.lstsq(design, gt[:, None]).solution.squeeze(1)
        scale, shift = solution[0], solution[1]
        if not bool(torch.isfinite(scale)) or not bool(torch.isfinite(shift)):
            raise RuntimeError("Non-finite scale/shift in GT alignment.")
    aligned = (scale * pred + shift).clamp(1.0 / max_depth, 1.0 / min_depth)
    pred_log_depth = -torch.log(aligned)
    gt_log_depth = -torch.log(gt.clamp_min(1.0 / max_depth))
    loss = (pred_log_depth - gt_log_depth).abs().mean()
    with torch.no_grad():
        abs_rel = ((1.0 / aligned - 1.0 / gt).abs() * gt).mean()
    return loss, abs_rel, count


DEFAULT_MANIFEST = Path(
    "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/"
    "ms2_fixed_sequence_internvl3_8b_filtered_caption_train.jsonl"
)
SAVE_STEPS = (0, 100, 500, 1000, 2000)
CHECKPOINT_FORMAT = "rgb_teacher_unet_gt_full_epoch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ms2-root", type=Path, default=Path("/mnt/e/dataset/ms2"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--gt-loss-weight", type=float, default=5.0)
    parser.add_argument(
        "--gt-loss-form",
        choices=("ssi_disparity", "log_depth"),
        default="ssi_disparity",
        help="log_depth: after disparity alignment, L1 in log-depth space "
        "(range-equitable; fixes far-field insensitivity of disparity L1).",
    )
    parser.add_argument("--response-weight", type=float, default=0.1)
    parser.add_argument("--gt-min-depth", type=float, default=0.1)
    parser.add_argument("--gt-max-depth", type=float, default=80.0)
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--unet-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--unet-max-grad-norm", type=float, default=1.0)
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
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--smoke-updates", type=int, default=None)
    parser.add_argument("--overfit-steps", type=int, default=None)
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
            rgb_path = root / row["rgb_path"]
            depth_path = root / row["rgb_depth_path"]
            if not rgb_path.is_file():
                raise FileNotFoundError(f"Missing RGB input: {rgb_path}")
            if not depth_path.is_file():
                raise FileNotFoundError(f"Missing RGB-view GT depth: {depth_path}")
            rows.append(
                {
                    "id": str(row["id"]),
                    "manifest_index": manifest_index,
                    "rgb_path": rgb_path,
                    "depth_path": depth_path,
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
        raise ValueError("RGB teacher training requires micro-batch 1 and accumulation 4.")
    if args.timestep != 999:
        raise ValueError("RGB teacher training must match inference timestep 999.")
    if args.gt_loss_weight <= 0:
        raise ValueError("gt-loss-weight must be positive.")
    if args.response_weight < 0:
        raise ValueError("response-weight must be non-negative.")
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


def load_rgb_tensor(path: Path):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    width -= width % 8
    height -= height % 8
    image = image.crop((0, 0, width, height))
    array = np.asarray(image, np.float32) / 255.0
    if float(array.std()) <= 0:
        raise RuntimeError(f"Constant RGB image: {path}")
    return torch.from_numpy(array * 2 - 1).permute(2, 0, 1)[None], height, width


def load_gt(path: Path, height: int, width: int, args):
    depth = np.asarray(Image.open(path), np.float32)[:height, :width] / args.depth_scale
    valid = (depth > args.gt_min_depth) & (depth < args.gt_max_depth) & np.isfinite(depth)
    disparity = np.zeros_like(depth)
    disparity[valid] = 1.0 / depth[valid]
    return (
        torch.from_numpy(disparity)[None],
        torch.from_numpy(valid.astype(np.float32))[None],
    )


def build_sample(row, lotus, teacher_unet, prompt, args, device, teacher_dtype):
    rgb_tensor, height, width = load_rgb_tensor(row["rgb_path"])
    with torch.no_grad():
        condition = (
            lotus.vae.encode(rgb_tensor.to(device=device, dtype=teacher_dtype)).latent_dist.mode()
            * lotus.vae.config.scaling_factor
        ).to(dtype=torch.float32)
    gt_disparity, valid_mask = load_gt(row["depth_path"], height, width, args)
    gt_disparity = gt_disparity.to(device)
    valid_mask = valid_mask.to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed + int(row["manifest_index"]))
    noise = (
        torch.randn(condition.shape, generator=generator, device=device, dtype=teacher_dtype)
        * float(lotus.scheduler.init_noise_sigma)
    )
    timestep = torch.full((1,), args.timestep, device=device, dtype=torch.long)
    latent_input = lotus.scheduler.scale_model_input(noise, timestep)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=teacher_dtype,
        enabled=device.type == "cuda" and teacher_dtype == torch.float16,
    ):
        teacher_response = teacher_unet(
            torch.cat([condition.to(teacher_dtype), latent_input], dim=1),
            timestep,
            encoder_hidden_states=prompt.to(dtype=teacher_dtype),
            class_labels=task_embedding(1, device, teacher_dtype),
            return_dict=False,
        )[0]
    diagnostics = {
        "id": row["id"],
        "manifest_index": row["manifest_index"],
        "rgb_hw": [height, width],
        "gt_valid_pixels": int(valid_mask.sum()),
    }
    return (
        condition,
        latent_input.detach().float(),
        teacher_response.detach().float(),
        gt_disparity,
        valid_mask,
        diagnostics,
    )


def forward_losses(student_unet, lotus, prompt_fp32, sample, args, device):
    condition, latent_input, teacher_response, gt_disparity, valid_mask = sample
    timestep = torch.full((1,), args.timestep, device=device, dtype=torch.long)
    student_response = student_unet(
        torch.cat([condition, latent_input], dim=1),
        timestep,
        encoder_hidden_states=prompt_fp32,
        class_labels=task_embedding(1, device, torch.float32),
        return_dict=False,
    )[0]
    response = response_consistency_losses(
        student_response,
        teacher_response,
        cosine_weight=args.response_cosine_loss_weight,
        spatial_gradient_weight=args.response_spatial_gradient_weight,
        multiscale_gradient_weight=args.response_multiscale_gradient_weight,
        gradient_energy_weight=args.response_gradient_energy_weight,
    )
    predicted_disparity = decode_to_disparity(lotus, student_response, device)
    if predicted_disparity.shape != gt_disparity.shape[-2:]:
        raise RuntimeError("Decoded disparity does not match GT shape.")
    if args.gt_loss_form == "log_depth":
        gt_loss, gt_abs_rel, gt_valid = masked_log_depth_l1(
            predicted_disparity[None], gt_disparity, valid_mask,
            args.gt_min_depth, args.gt_max_depth,
        )
    else:
        gt_loss, gt_abs_rel, gt_valid = masked_ssi_l1(
            predicted_disparity[None], gt_disparity, valid_mask
        )
    total = args.response_weight * response["total"] + args.gt_loss_weight * gt_loss
    return {
        "total": total,
        "response_total": response["total"],
        "response_mse": response["mse"],
        "response_cosine": response["cosine"],
        "gt_ssi_l1": gt_loss,
        "gt_abs_rel": gt_abs_rel,
        "gt_valid_pixels": torch.tensor(float(gt_valid)),
    }


def serializable_settings(args):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in ("resume", "smoke_updates", "overfit_steps")
    }


def save_checkpoint(path, student_unet, optimizer, step, next_offset, permutation, args, manifest_hash):
    payload = {
        "format": CHECKPOINT_FORMAT,
        "train_mode": "rgb_teacher_unet",
        "gt_loss_weight": args.gt_loss_weight,
        "response_weight": args.response_weight,
        "global_step": step,
        "next_sample_offset": next_offset,
        "permutation": permutation,
        "manifest_sha256": manifest_hash,
        "settings": serializable_settings(args),
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
    if args.overfit_steps is not None:
        total_updates = args.overfit_steps
    permutation = list(range(len(rows)))
    random.Random(args.seed).shuffle(permutation)
    overfit_pool = permutation[:32]
    output = args.output_dir.resolve()
    if args.resume is None:
        if output.exists() and any(output.iterdir()):
            raise SystemExit(f"Refusing non-empty output directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=True)

    frozen_config = {
        "route": "RGB depth teacher: Lotus U-Net fine-tuned on MS2 RGB + RGB-view GT",
        "training_objective": (
            "masked scale-shift-invariant L1 vs RGB-view LiDAR disparity "
            "+ light frozen-teacher response trust region"
        ),
        "view_consistency": "input RGB view, GT RGB view — no cross-view supervision",
        "gt_loss_weight": args.gt_loss_weight,
        "response_weight": args.response_weight,
        "initialization": "pretrained Lotus U-Net (fp16 load, fp32 student cast); fresh optimizer",
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
    prompt, _ = lotus.encode_prompt(
        prompt="", device=device, num_images_per_prompt=1, do_classifier_free_guidance=None
    )
    prompt = prompt.detach()
    prompt_fp32 = prompt.to(dtype=torch.float32)

    student_unet = copy.deepcopy(teacher_unet).to(device=device, dtype=torch.float32)
    student_unet.train().requires_grad_(True)
    optimizer = torch.optim.AdamW(
        student_unet.parameters(), lr=args.unet_learning_rate, weight_decay=args.weight_decay
    )

    start_step = 0
    next_offset = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume.resolve(), map_location="cpu", weights_only=False)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise RuntimeError("Resume checkpoint is not a formal RGB teacher run.")
        if checkpoint.get("manifest_sha256") != manifest_hash:
            raise RuntimeError("Resume manifest hash differs from the frozen run.")
        if checkpoint.get("permutation") != permutation:
            raise RuntimeError("Resume sample permutation differs from the frozen run.")
        if checkpoint.get("settings") != serializable_settings(args):
            raise RuntimeError("Resume settings differ from the frozen RGB teacher run.")
        student_unet.load_state_dict(checkpoint["lotus_unet_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["global_step"])
        next_offset = int(checkpoint["next_sample_offset"])
        validate_resume_position(start_step, next_offset, len(rows), effective_batch)
    formal_run = args.smoke_updates is None and args.overfit_steps is None
    if start_step == 0 and next_offset == 0 and formal_run:
        save_checkpoint(
            output / "rgb_teacher_step_0000.pt",
            student_unet,
            optimizer,
            0,
            0,
            permutation,
            args,
            manifest_hash,
        )

    log_path = output / "training_metrics.jsonl"
    diagnostics_path = output / "rgb_audit_first8.json"
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
                sample = build_sample(
                    row, lotus, teacher_unet, prompt, args, device, teacher_dtype
                )
                if len(first_diagnostics) < 8:
                    first_diagnostics.append(sample[-1])
                    if len(first_diagnostics) == 8:
                        diagnostics_path.write_text(
                            json.dumps(first_diagnostics, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                losses = forward_losses(
                    student_unet, lotus, prompt_fp32, sample[:-1], args, device
                )
                if not bool(torch.isfinite(losses["total"])):
                    raise RuntimeError(f"Non-finite loss at step {step} sample {row['id']}.")
                (losses["total"] / len(batch_indices)).backward()
                for key, value in losses.items():
                    if value.ndim == 0:
                        metric_sums[key] = metric_sums.get(key, 0.0) + float(value.detach())
            if any(parameter.grad is not None for parameter in teacher_unet.parameters()):
                raise RuntimeError("Frozen teacher U-Net unexpectedly owns gradients.")
            unet_grad_norm = torch.nn.utils.clip_grad_norm_(
                student_unet.parameters(), args.unet_max_grad_norm
            )
            if not bool(torch.isfinite(unet_grad_norm)):
                raise RuntimeError(f"Non-finite U-Net gradient at step {step}.")
            optimizer.step()
            if args.overfit_steps is None:
                next_offset += len(batch_indices)
            last_record = {
                "step": step,
                "samples_seen": next_offset,
                "batch_size": len(batch_indices),
                "elapsed_seconds": time.time() - start_time,
                "unet_grad_norm": float(unet_grad_norm),
                **{key: value / len(batch_indices) for key, value in metric_sums.items()},
            }
            log_handle.write(json.dumps(last_record) + "\n")
            log_handle.flush()
            if step % args.log_interval == 0 or step == 1 or step == total_updates:
                print(json.dumps(last_record), flush=True)
            if formal_run and (step in SAVE_STEPS or step == total_updates):
                name = (
                    "rgb_teacher_end.pt"
                    if step == total_updates
                    else f"rgb_teacher_step_{step:04d}.pt"
                )
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
        "end_checkpoint": str(output / "rgb_teacher_end.pt"),
        "end_checkpoint_sha256": sha256(output / "rgb_teacher_end.pt"),
        "elapsed_seconds": time.time() - start_time,
        "final_training_record": last_record,
        "frozen_modules": {
            "vae": frozen_audit(lotus.vae),
            "text_encoder": frozen_audit(lotus.text_encoder),
            "teacher_unet": frozen_audit(teacher_unet),
        },
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
