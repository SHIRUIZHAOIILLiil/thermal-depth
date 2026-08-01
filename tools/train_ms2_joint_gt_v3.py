"""P1: jointly train Adapter V2.3 + Lotus U-Net with masked GT depth supervision.

Extends the Phase H joint caption run with a third loss term: the student
U-Net's x0 output is decoded through the frozen VAE exactly as the official
evaluator does, converted to disparity, scale-shift aligned to the LiDAR GT
disparity on valid pixels (closed-form least squares, detached fit), and
penalized with a masked L1.  This gives caption content its first gradient
path toward depth accuracy.

Two arms share this script:
  --caption-training dropout  -> Arm 1 (captions + frozen 10% dropout)
  --caption-training off      -> Arm 2 (always-empty prompt control)

Distillation terms from the V2 protocol are kept unchanged as regularizers.
No Val/Test data is used; GT comes from the Train manifest's LiDAR maps only.
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
import torch.nn.functional as F
from PIL import Image

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
from models.anythermal_lotus_v2_4 import response_consistency_losses, seeded_noise  # noqa: E402


DEFAULT_MANIFEST = Path(
    "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/"
    "ms2_fixed_sequence_internvl3_8b_filtered_caption_train.jsonl"
)
SAVE_STEPS = (0, 100, 500, 1000, 2000)
CHECKPOINT_FORMAT = "adapter_unet_joint_gt_v3_full_epoch"
MIN_VALID_PIXELS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ms2-root", type=Path, default=Path("/mnt/e/dataset/ms2"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--caption-training", choices=("dropout", "off"), required=True)
    parser.add_argument("--caption-dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--gt-loss-form",
        choices=("ssi_disparity", "log_depth"),
        default="ssi_disparity",
        help="log_depth: range-equitable L1 in log-depth space after alignment.",
    )
    parser.add_argument("--gt-loss-weight", type=float, default=0.5)
    parser.add_argument("--gt-min-depth", type=float, default=0.1)
    parser.add_argument("--gt-max-depth", type=float, default=80.0)
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--min-gt-valid-fraction", type=float, default=0.0,
                        help="Drop manifest rows whose gt_valid_fraction is below this "
                             "(RGBDT500 only; MS2 rows carry no such field and pass through).")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--adapter-learning-rate", type=float, default=3e-4)
    parser.add_argument("--unet-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--unet-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--condition-weight", type=float, default=1.0)
    parser.add_argument("--response-weight", type=float, default=1.0)
    parser.add_argument("--cosine-loss-weight", type=float, default=0.1)
    parser.add_argument("--channel-stats-loss-weight", type=float, default=0.1)
    parser.add_argument("--spatial-gradient-loss-weight", type=float, default=0.1)
    parser.add_argument("--multiscale-gradient-loss-weight", type=float, default=0.5)
    parser.add_argument("--response-cosine-loss-weight", type=float, default=0.1)
    parser.add_argument("--response-spatial-gradient-weight", type=float, default=0.5)
    parser.add_argument("--response-multiscale-gradient-weight", type=float, default=0.5)
    parser.add_argument("--response-gradient-energy-weight", type=float, default=0.5)
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument(
        "--gt-decode-fp32",
        action="store_true",
        help=(
            "Decode the GT-loss path through an fp32 VAE copy. The legacy fp16 "
            "autocast decode underflows the GT gradient to ~zero (found 2026-07-11)."
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--smoke-updates",
        type=int,
        default=None,
        help="Run only N optimizer updates into a throwaway 'smoke' output dir.",
    )
    parser.add_argument(
        "--overfit-steps",
        type=int,
        default=None,
        help="Cycle the first 32 permutation samples for N updates ('overfit' dir only).",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_train_manifest(path: Path, root: Path, require_captions: bool,
                        min_gt_valid_fraction: float = 0.0):
    """Rows for training.

    Dataset-portability rules mirror ``train_ms2_thermal_vae_unet_gt.py`` so the
    two lines can train on the same manifests: RGBDT500 names the GT field
    ``depth_path`` and carries ``gt_valid_fraction``, MS2 uses
    ``thermal_depth_path`` (which must be the thermal-view GT there) and has no
    valid-fraction field, so MS2 rows pass the filter untouched.
    """
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for manifest_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError(f"Non-Train row in Train manifest: {row.get('id')}")
            depth_field = row.get("thermal_depth_path") or row.get("depth_path")
            if not depth_field:
                raise ValueError(f"Row {row.get('id')} has no depth GT path")
            # Degenerate GT (RGBDT500 frames that are almost all holes) would make
            # the masked SSI loss raise mid-epoch; drop them up front.
            valid_fraction = row.get("gt_valid_fraction")
            if valid_fraction is not None and float(valid_fraction) < min_gt_valid_fraction:
                continue
            thermal_path = root / row["thermal_path"]
            depth_path = root / depth_field
            if not thermal_path.is_file():
                raise FileNotFoundError(f"Missing thermal input: {thermal_path}")
            if not depth_path.is_file():
                raise FileNotFoundError(f"Missing GT depth: {depth_path}")
            caption = str(row.get("caption", "")).strip()
            if require_captions and not caption:
                raise ValueError(f"Empty manifest caption for Train row: {row.get('id')}")
            rows.append(
                {
                    "id": str(row["id"]),
                    "manifest_index": manifest_index,
                    "thermal_path": thermal_path,
                    "depth_path": depth_path,
                    "caption": caption,
                }
            )
    if not rows:
        raise ValueError("Train manifest is empty.")
    return rows


def caption_dropped(seed: int, manifest_index: int, probability: float) -> bool:
    """Frozen deterministic per-sample caption dropout decision (same rule as V2)."""
    return random.Random(f"{seed}:caption_dropout:{manifest_index}").random() < probability


def load_gt_disparity(path: Path, min_depth: float, max_depth: float, depth_scale: float):
    """Return (gt_disparity, valid_mask) float32 tensors shaped [1, H, W]."""
    depth = np.asarray(Image.open(path), dtype=np.float32) / depth_scale
    valid = (depth > min_depth) & (depth < max_depth) & np.isfinite(depth)
    disparity = np.zeros_like(depth)
    disparity[valid] = 1.0 / depth[valid]
    return (
        torch.from_numpy(disparity)[None],
        torch.from_numpy(valid.astype(np.float32))[None],
    )


def masked_log_depth_l1(prediction, gt_disparity, valid_mask, min_depth, max_depth):
    """Disparity-aligned L1 in log-depth space: range-equitable relative error.

    Same detached scale-shift alignment as masked_ssi_l1, but the penalty is
    taken on log-depth so equal relative errors cost the same at all ranges
    (a 20% error at 40 m contributes only ~10% of a 20% error at 4 m under
    disparity L1).
    """
    mask = valid_mask > 0.5
    count = int(mask.sum())
    if count < MIN_VALID_PIXELS:
        raise RuntimeError(f"GT valid pixels {count} below minimum {MIN_VALID_PIXELS}.")
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


def masked_ssi_l1(prediction, gt_disparity, valid_mask):
    """Scale-shift-invariant masked L1 in disparity space.

    The affine fit is solved in closed form on valid pixels and detached, so
    gradients flow only through the aligned residual — mirroring the official
    evaluator's least_square_disparity alignment.
    """
    mask = valid_mask > 0.5
    count = int(mask.sum())
    if count < MIN_VALID_PIXELS:
        raise RuntimeError(f"GT valid pixels {count} below minimum {MIN_VALID_PIXELS}.")
    pred = prediction[mask]
    gt = gt_disparity[mask]
    with torch.no_grad():
        ones = torch.ones_like(pred)
        design = torch.stack([pred, ones], dim=1)
        solution = torch.linalg.lstsq(design, gt[:, None]).solution.squeeze(1)
        scale, shift = solution[0], solution[1]
        if not bool(torch.isfinite(scale)) or not bool(torch.isfinite(shift)):
            raise RuntimeError("Non-finite scale/shift in GT alignment.")
    aligned = scale * pred + shift
    loss = (aligned - gt).abs().mean()
    with torch.no_grad():
        gt_depth = 1.0 / gt.clamp_min(1e-6)
        pred_depth = 1.0 / aligned.detach().clamp_min(1e-6)
        abs_rel = ((pred_depth - gt_depth).abs() / gt_depth).mean()
    return loss, abs_rel, count


def ssi_grad_matching(prediction, gt_disparity, valid_mask, scales: int = 4):
    """MiDaS's multi-scale gradient matching term, the companion to the data term.

    `masked_ssi_l1` scores values pointwise, which leaves blur unpunished: where
    the teacher and the input disagree -- object boundaries, above all -- a
    smooth ramp is a lower-risk answer than a sharp step in a slightly wrong
    place. AbsRel and RMSE, being pointwise averages themselves, reward that
    same hedge. This term scores the *changes*: the residual's spatial gradient
    is exactly what blur creates, so the ramp stops being free.

    Uses the same closed-form detached affine fit as `masked_ssi_l1`, so the two
    terms share one alignment convention and can be summed. Keep them adjacent;
    they must not drift apart.

    Gradients are taken only where both neighbours are valid -- across the edge
    of the valid mask, "value to no value" would otherwise read as a depth step.
    Downsampling subsamples rather than averages, which keeps the mask exact.
    """
    mask = valid_mask > 0.5
    count = int(mask.sum())
    if count < MIN_VALID_PIXELS:
        raise RuntimeError(f"GT valid pixels {count} below minimum {MIN_VALID_PIXELS}.")
    with torch.no_grad():
        pred_v, gt_v = prediction[mask], gt_disparity[mask]
        design = torch.stack([pred_v, torch.ones_like(pred_v)], dim=1)
        solution = torch.linalg.lstsq(design, gt_v[:, None]).solution.squeeze(1)
        scale, shift = solution[0], solution[1]
        if not bool(torch.isfinite(scale)) or not bool(torch.isfinite(shift)):
            raise RuntimeError("Non-finite scale/shift in GT alignment.")

    residual = (scale * prediction + shift) - gt_disparity
    residual = residual * mask

    total = residual.new_zeros(())
    pairs = 0
    level_residual, level_mask = residual, mask
    for _ in range(max(1, scales)):
        if level_residual.shape[-1] < 2 or level_residual.shape[-2] < 2:
            break
        mask_x = level_mask[..., :, :-1] & level_mask[..., :, 1:]
        mask_y = level_mask[..., :-1, :] & level_mask[..., 1:, :]
        grad_x = (level_residual[..., :, :-1] - level_residual[..., :, 1:]).abs()
        grad_y = (level_residual[..., :-1, :] - level_residual[..., 1:, :]).abs()
        here = int(mask_x.sum()) + int(mask_y.sum())
        if here:
            total = total + grad_x[mask_x].sum() + grad_y[mask_y].sum()
            pairs += here
        level_residual = level_residual[..., ::2, ::2]
        level_mask = level_mask[..., ::2, ::2]

    if not pairs:
        return residual.new_zeros(())
    return total / pairs


def decode_to_disparity(lotus, x0, device, gt_vae=None):
    """Mirror the official evaluator: decode x0, denormalize, channel-mean."""
    if gt_vae is not None:
        # fp32 decode: the fp16 backward through the VAE underflows the GT
        # gradient to exactly zero on most steps (adapter grad ~1e-4 vs the
        # 2.5-37 healthy range), leaving the GT term without training signal.
        decoded = gt_vae.decode(
            x0.float() / gt_vae.config.scaling_factor, return_dict=False
        )[0]
        return (decoded.mean(dim=1) / 2.0 + 0.5).squeeze(0)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        decoded = lotus.vae.decode(
            x0 / lotus.vae.config.scaling_factor, return_dict=False
        )[0]
    return (decoded.float().mean(dim=1) / 2.0 + 0.5).squeeze(0)


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


def validate_protocol(args) -> None:
    if args.micro_batch_size != 1 or args.gradient_accumulation_steps != 4:
        raise ValueError("Joint GT V3 requires micro-batch 1 and accumulation 4.")
    if args.timestep != 999:
        raise ValueError("Joint GT V3 must match inference timestep 999.")
    if not (0.0 <= args.caption_dropout < 1.0):
        raise ValueError("caption-dropout must be in [0, 1).")
    if args.gt_loss_weight <= 0:
        raise ValueError("gt-loss-weight must be positive (this is the point of P1).")
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


def encode_caption(lotus, text: str, device):
    with torch.no_grad():
        prompt, _ = lotus.encode_prompt(
            prompt=text,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=None,
        )
    return prompt.detach()


def build_sample(row, anythermal, lotus, teacher_unet, prompt, args, device, teacher_dtype):
    thermal = thermal_to_lotus_input(row["thermal_path"], processing_res=0)
    if thermal.diagnostics["converted_uint8_std"] <= 0:
        raise RuntimeError(f"Constant thermal conversion: {row['id']}")
    teacher_condition = encode_condition_latent(
        lotus.vae, thermal.tensor, posterior="mode"
    ).to(device=device, dtype=torch.float32)
    features, _, anythermal_diag = extract_anythermal_feature_pyramid(
        anythermal, row["thermal_path"], enable_grad=False
    )
    features = [feature.detach().float().to(device) for feature in features]
    thermal_tensor = thermal.tensor.to(device=device, dtype=torch.float32)
    gt_disparity, valid_mask = load_gt_disparity(
        row["depth_path"], args.gt_min_depth, args.gt_max_depth, args.depth_scale
    )
    gt_disparity = gt_disparity.to(device)
    valid_mask = valid_mask.to(device)
    noise = seeded_noise(
        (1, *teacher_condition.shape[1:]),
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
            torch.cat([teacher_condition.to(teacher_dtype), latent_input], dim=1),
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
        "gt_valid_pixels": int(valid_mask.sum()),
    }
    return (
        features,
        thermal_tensor,
        teacher_condition,
        latent_input.detach().float(),
        teacher_response.detach().float(),
        gt_disparity,
        valid_mask,
        diagnostics,
    )


def forward_losses(adapter, student_unet, lotus, prompt_fp32, sample, args, device, gt_vae=None):
    (
        features,
        thermal_tensor,
        teacher_condition,
        latent_input,
        teacher_response,
        gt_disparity,
        valid_mask,
    ) = sample
    student_condition = adapter(
        features, thermal_tensor, target_size=tuple(teacher_condition.shape[-2:])
    )
    timestep = torch.full(
        (student_condition.shape[0],), args.timestep, device=device, dtype=torch.long
    )
    student_response = student_unet(
        torch.cat([student_condition, latent_input], dim=1),
        timestep,
        encoder_hidden_states=prompt_fp32.repeat(student_condition.shape[0], 1, 1),
        class_labels=task_embedding(student_condition.shape[0], device, torch.float32),
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
        gradient_energy_weight=args.response_gradient_energy_weight,
    )
    predicted_disparity = decode_to_disparity(lotus, student_response, device, gt_vae=gt_vae)
    if predicted_disparity.shape != gt_disparity.shape[-2:]:
        raise RuntimeError(
            f"Decoded disparity {tuple(predicted_disparity.shape)} does not match GT "
            f"{tuple(gt_disparity.shape[-2:])}."
        )
    if args.gt_loss_form == "log_depth":
        gt_loss, gt_abs_rel, gt_valid = masked_log_depth_l1(
            predicted_disparity[None], gt_disparity, valid_mask,
            args.gt_min_depth, args.gt_max_depth,
        )
    else:
        gt_loss, gt_abs_rel, gt_valid = masked_ssi_l1(
            predicted_disparity[None], gt_disparity, valid_mask
        )
    total = (
        args.condition_weight * condition_total
        + args.response_weight * response["total"]
        + args.gt_loss_weight * gt_loss
    )
    return {
        "total": total,
        "condition_total": condition_total,
        "condition_mse": condition["latent_mse"],
        "condition_cosine": 1.0 - condition["cosine_loss"],
        "response_total": response["total"],
        "response_mse": response["mse"],
        "response_cosine": response["cosine"],
        "response_gradient_energy_ratio": response["gradient_energy_ratio"],
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


def save_checkpoint(path, adapter, student_unet, optimizer, step, next_offset, permutation, args, manifest_hash):
    payload = {
        "format": CHECKPOINT_FORMAT,
        "train_mode": "adapter_unet_joint",
        "adapter_architecture": "v2_3_thermal_detail_skip",
        "caption_training": args.caption_training,
        "caption_dropout": args.caption_dropout if args.caption_training == "dropout" else None,
        "gt_loss_weight": args.gt_loss_weight,
        "global_step": step,
        "next_sample_offset": next_offset,
        "permutation": permutation,
        "manifest_sha256": manifest_hash,
        "settings": serializable_settings(args),
        "adapter": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
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
    rows = read_train_manifest(
        manifest, ms2_root, require_captions=args.caption_training == "dropout",
        min_gt_valid_fraction=args.min_gt_valid_fraction,
    )
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")
    total_samples = len(rows) * args.epochs
    shape = expected_training_shape(total_samples, effective_batch)
    total_updates = shape["optimizer_updates"]
    if args.smoke_updates is not None:
        total_updates = min(args.smoke_updates, total_updates)
    if args.overfit_steps is not None:
        total_updates = args.overfit_steps
    permutation = list(range(len(rows)))
    random.Random(args.seed).shuffle(permutation)
    # Multi-epoch: the same frozen permutation repeats each epoch (deterministic,
    # resume-safe); epoch boundaries get their own named checkpoints.
    sample_stream = permutation * args.epochs
    epoch_end_steps = {
        math.ceil(len(rows) * epoch / effective_batch): epoch
        for epoch in range(1, args.epochs + 1)
    }
    overfit_pool = permutation[:32]
    output = args.output_dir.resolve()
    if args.resume is None:
        if output.exists() and any(output.iterdir()):
            raise SystemExit(f"Refusing non-empty output directory: {output}")
        output.mkdir(parents=True, exist_ok=True)
    else:
        output.mkdir(parents=True, exist_ok=True)

    dropout_count = (
        sum(
            1
            for row in rows
            if caption_dropped(args.seed, row["manifest_index"], args.caption_dropout)
        )
        if args.caption_training == "dropout"
        else 0
    )
    frozen_config = {
        "route": "P1: Adapter V2.3 + joint U-Net + masked GT depth supervision",
        "arm": (
            "Arm1 caption+dropout" if args.caption_training == "dropout" else "Arm2 no-caption control"
        ),
        "training_objective": (
            "V2 condition distillation + response consistency (regularizers) "
            "+ masked scale-shift-invariant L1 vs LiDAR disparity on valid pixels"
        ),
        "gt_source": "Train manifest thermal_depth_path (thermal-view filtered LiDAR)",
        "gt_alignment": "closed-form least squares scale/shift in disparity space, detached",
        "gt_loss_weight": args.gt_loss_weight,
        "decode_path": "vae.decode(x0/scaling_factor) -> denormalize -> channel mean (mirrors official evaluator)",
        "initialization": "fresh random Adapter V2.3; pretrained Lotus U-Net; fresh optimizer",
        "caption_training": args.caption_training,
        "caption_dropout_probability": (
            args.caption_dropout if args.caption_training == "dropout" else None
        ),
        "caption_dropout_samples": dropout_count,
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "train_samples": len(rows),
        "epochs": args.epochs,
        "gt_loss_form": args.gt_loss_form,
        "epoch_end_steps": sorted(epoch_end_steps),
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
    anythermal = AnyThermalEncoder(
        model_path=args.anythermal_model_path,
        device=str(device),
        local_files_only=args.local_files_only,
    )
    teacher_unet = lotus.unet
    for module in (lotus.vae, lotus.text_encoder, teacher_unet, anythermal.model):
        module.requires_grad_(False).eval()
    gt_vae = None
    if args.gt_decode_fp32:
        gt_vae = copy.deepcopy(lotus.vae).to(device=device, dtype=torch.float32)
        gt_vae.requires_grad_(False).eval()
        gt_vae.encoder = None  # decode-only copy; frees the unused encoder half
    empty_prompt = encode_caption(lotus, "", device)

    adapter = AnyThermalLotusAdapterV23().to(device=device, dtype=torch.float32).train()
    student_unet = copy.deepcopy(teacher_unet).to(device=device, dtype=torch.float32)
    student_unet.train().requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter.parameters(), "lr": args.adapter_learning_rate},
            {"params": student_unet.parameters(), "lr": args.unet_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )

    start_step = 0
    next_offset = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume.resolve(), map_location="cpu", weights_only=False)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise RuntimeError("Resume checkpoint is not a formal P1 run.")
        if checkpoint.get("manifest_sha256") != manifest_hash:
            raise RuntimeError("Resume manifest hash differs from the frozen run.")
        if checkpoint.get("permutation") != permutation:
            raise RuntimeError("Resume sample permutation differs from the frozen run.")
        if checkpoint.get("settings") != serializable_settings(args):
            raise RuntimeError("Resume settings differ from the frozen P1 run.")
        adapter.load_state_dict(checkpoint["adapter"], strict=True)
        student_unet.load_state_dict(checkpoint["lotus_unet_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["global_step"])
        next_offset = int(checkpoint["next_sample_offset"])
        validate_resume_position(start_step, next_offset, total_samples, effective_batch)
    formal_run = args.smoke_updates is None and args.overfit_steps is None
    if start_step == 0 and next_offset == 0 and formal_run:
        save_checkpoint(
            output / "gt_v3_step_0000.pt",
            adapter,
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
    caption_used_count = 0
    caption_dropped_count = 0
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
                batch_indices = sample_stream[next_offset : next_offset + effective_batch]
            if not batch_indices:
                raise RuntimeError(f"Empty batch at step {step} offset {next_offset}.")
            optimizer.zero_grad(set_to_none=True)
            metric_sums = {}
            for row_index in batch_indices:
                row = rows[row_index]
                if args.caption_training == "dropout" and not caption_dropped(
                    args.seed, row["manifest_index"], args.caption_dropout
                ):
                    prompt = encode_caption(lotus, row["caption"], device)
                    caption_used_count += 1
                else:
                    prompt = empty_prompt
                    caption_dropped_count += 1
                sample = build_sample(
                    row, anythermal, lotus, teacher_unet, prompt, args, device, teacher_dtype
                )
                if len(first_diagnostics) < 8:
                    first_diagnostics.append(sample[-1])
                    if len(first_diagnostics) == 8:
                        diagnostics_path.write_text(
                            json.dumps(first_diagnostics, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                losses = forward_losses(
                    adapter,
                    student_unet,
                    lotus,
                    prompt.to(dtype=torch.float32),
                    sample[:-1],
                    args,
                    device,
                    gt_vae=gt_vae,
                )
                if not bool(torch.isfinite(losses["total"])):
                    raise RuntimeError(f"Non-finite loss at step {step} sample {row['id']}.")
                (losses["total"] / len(batch_indices)).backward()
                for key, value in losses.items():
                    if value.ndim == 0:
                        metric_sums[key] = metric_sums.get(key, 0.0) + float(value.detach())
            if any(parameter.grad is not None for parameter in teacher_unet.parameters()):
                raise RuntimeError("Frozen teacher U-Net unexpectedly owns gradients.")
            adapter_grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), float("inf"))
            unet_grad_norm = torch.nn.utils.clip_grad_norm_(
                student_unet.parameters(), args.unet_max_grad_norm
            )
            if not bool(torch.isfinite(adapter_grad_norm)) or not bool(
                torch.isfinite(unet_grad_norm)
            ):
                raise RuntimeError(f"Non-finite gradient at step {step}.")
            optimizer.step()
            if args.overfit_steps is None:
                next_offset += len(batch_indices)
            last_record = {
                "step": step,
                "samples_seen": next_offset,
                "batch_size": len(batch_indices),
                "elapsed_seconds": time.time() - start_time,
                "adapter_grad_norm": float(adapter_grad_norm),
                "unet_grad_norm": float(unet_grad_norm),
                "captions_used": caption_used_count,
                "captions_dropped": caption_dropped_count,
                **{key: value / len(batch_indices) for key, value in metric_sums.items()},
            }
            log_handle.write(json.dumps(last_record) + "\n")
            log_handle.flush()
            if step % args.log_interval == 0 or step == 1 or step == total_updates:
                print(json.dumps(last_record), flush=True)
            if formal_run and (
                step in SAVE_STEPS or step == total_updates or step in epoch_end_steps
            ):
                if step == total_updates:
                    name = "gt_v3_end.pt"
                elif step in epoch_end_steps:
                    name = f"gt_v3_epoch{epoch_end_steps[step]}.pt"
                else:
                    name = f"gt_v3_step_{step:04d}.pt"
                save_checkpoint(
                    output / name,
                    adapter,
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
        partial_summary = {
            **frozen_config,
            label: True,
            "completed_updates": total_updates,
            "captions_used": caption_used_count,
            "captions_dropped": caption_dropped_count,
            "final_training_record": last_record,
            "elapsed_seconds": time.time() - start_time,
        }
        name = "smoke_summary.json" if args.smoke_updates is not None else "overfit_summary.json"
        (output / name).write_text(
            json.dumps(partial_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(partial_summary, indent=2, ensure_ascii=False))
        return

    if next_offset != total_samples:
        raise RuntimeError(f"Incomplete training: consumed {next_offset}/{total_samples} samples.")
    if first_diagnostics and not diagnostics_path.exists():
        diagnostics_path.write_text(
            json.dumps(first_diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    summary = {
        **frozen_config,
        "completed": True,
        "global_step": total_updates,
        "samples_seen": next_offset,
        "captions_used": caption_used_count,
        "captions_dropped": caption_dropped_count,
        "end_checkpoint": str(output / "gt_v3_end.pt"),
        "end_checkpoint_sha256": sha256(output / "gt_v3_end.pt"),
        "elapsed_seconds": time.time() - start_time,
        "final_training_record": last_record,
        "frozen_modules": {
            "anythermal": frozen_audit(anythermal.model),
            "vae": frozen_audit(lotus.vae),
            "text_encoder": frozen_audit(lotus.text_encoder),
            "teacher_unet": frozen_audit(teacher_unet),
        },
        "adapter_trainable_parameters": int(
            sum(parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad)
        ),
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
