"""Arm 6 / Arm 6-E: train Lotus on the genuine Thermal-VAE condition + GT loss.

Arm 6   (default)             frozen VAE, train U-Net only.
Arm 6-E (--train-vae-encoder) additionally unfreezes the VAE *encoder* (and
        quant_conv) with its own learning rate, giving the original route the
        "condition plasticity" that made the adapter line win.  The VAE
        *decoder* stays frozen so the latent contract on the decode side is
        untouched.

Objective: frozen-teacher response consistency (regularizer; the teacher
response follows the current condition, so it pins the U-Net mapping, not the
encoder) + masked GT SSI (lambda=5).  Empty prompt, champion protocol
(micro-batch 1 x accumulation 4, seed 20260703, 1 epoch, end-of-epoch
checkpoint, no Val/Test at train time).
"""

from __future__ import annotations

import argparse
import copy
import glob
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

from models.anythermal_lotus_v2 import thermal_to_lotus_input  # noqa: E402
from models.anythermal_lotus_v2_4 import response_consistency_losses, seeded_noise  # noqa: E402
from train_ms2_joint_gt_v3 import (  # noqa: E402
    MIN_VALID_PIXELS,
    load_gt_disparity,
    masked_ssi_l1,
    decode_to_disparity,
)


DEFAULT_MANIFEST = Path(
    "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/"
    "ms2_fixed_sequence_internvl3_8b_filtered_caption_train.jsonl"
)
SAVE_STEPS = (0, 100, 500, 1000, 2000)
CHECKPOINT_FORMAT = "thermal_vae_unet_gt_full_epoch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ms2-root", type=Path, default=Path("/mnt/e/dataset/ms2"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lotus_line_v2/full_train_epoch1_thermal_vae_unet_gt"),
    )
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--train-vae-encoder", action="store_true")
    parser.add_argument("--vae-encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--gt-loss-weight", type=float, default=5.0)
    parser.add_argument("--gt-min-depth", type=float, default=0.1)
    parser.add_argument("--gt-max-depth", type=float, default=80.0)
    parser.add_argument("--depth-scale", type=float, default=256.0,
                        help="uint16 GT units per metre. MS2=256, RGBDT500=1000 (millimetres).")
    parser.add_argument("--min-gt-valid-fraction", type=float, default=0.0,
                        help=(
                            "Drop manifest rows whose gt_valid_fraction is below this "
                            "(RGBDT500 has a handful of near-empty depth frames that would "
                            "otherwise abort the masked SSI loss mid-epoch). MS2 manifests "
                            "carry no such field and are unaffected."
                        ))
    parser.add_argument(
        "--gt-sparsify",
        choices=("none", "ms2_lidar", "random"),
        default="none",
        help=(
            "Thin the TRAINING supervision; evaluation is untouched. Iris learns its "
            "text pathway under dense per-pixel GT and only *reports* on sparse test "
            "sets, whereas we always train on sparse LiDAR. This closes that gap from "
            "the other side: 'ms2_lidar' transfers a real MS2 LiDAR validity pattern "
            "onto each frame (structured blindness -- sky and far field go dark), "
            "'random' drops the same fraction uniformly, which separates 'fewer pixels' "
            "from 'blind exactly where text would help'."
        ),
    )
    parser.add_argument(
        "--gt-sparsify-source",
        default="/mnt/e/dataset/ms2/proj_depth/_2021-08-06-11-23-45/thr/depth_filtered/*.png",
        help="Glob of MS2 depth maps whose validity patterns get borrowed (ms2_lidar).",
    )
    parser.add_argument("--gt-sparsify-count", type=int, default=400,
                        help="How many MS2 validity patterns to cache and cycle through.")
    parser.add_argument("--gt-sparsify-density", type=float, default=0.0,
                        help="random mode: fraction of pixels kept. 0 = match the measured ms2_lidar density.")
    parser.add_argument("--gt-sparsify-seed", type=int, default=20260729)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--unet-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--unet-max-grad-norm", type=float, default=1.0)
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
            "empty reproduces the original Arm 6 behaviour exactly."
        ),
    )
    parser.add_argument(
        "--caption-dropout",
        type=float,
        default=0.1,
        help="With --caption-mode correct: per-visit probability of swapping in the empty prompt (Arm-1 uses 0.1).",
    )
    parser.add_argument(
        "--caption-contrast-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the paired caption-vs-empty hinge on the GT loss. The empty "
            "reference is recomputed under no_grad from the same condition and the same "
            "seeded noise, so the hinge can only be satisfied by improving the caption "
            "branch. 0.0 (default) leaves every existing recipe bit-identical."
        ),
    )
    parser.add_argument(
        "--caption-contrast-margin",
        type=float,
        default=0.02,
        help=(
            "Relative margin m: hinge = relu(gt_caption - gt_empty + m * gt_empty). "
            "Relative rather than absolute because gt_ssi_l1 is not AbsRel-scaled."
        ),
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
    parser.add_argument(
        "--dense-teacher-dir",
        type=Path,
        default=None,
        help=(
            "Task-4 arm C: directory of dense calibrated teacher disparities "
            "(teacher_disparity/<id>.npy from run_ms2_anythermal_midas.py "
            "--calibrate-to-gt). Adds a dense SSI L1 distillation term."
        ),
    )
    parser.add_argument(
        "--dense-teacher-weight",
        type=float,
        default=0.0,
        help="Weight of the dense-teacher term (0 = off; try 1.0).",
    )
    parser.add_argument(
        "--dense-teacher-align",
        choices=("ssi", "l1"),
        default="ssi",
        help=(
            "ssi: scale-shift-invariant L1 to the teacher (teaches geometry only, "
            "cannot anchor the output range). l1: direct L1 to the GT-calibrated "
            "metric disparity (teaches geometry AND pins the range -- the anchor "
            "role without the frozen-response suppression)."
        ),
    )
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


def read_train_manifest(path: Path, root: Path, min_gt_valid_fraction: float = 0.0):
    """Rows for training. `min_gt_valid_fraction` only filters manifests that
    carry a `gt_valid_fraction` field (RGBDT500); MS2 rows pass through."""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for manifest_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError(f"Non-Train row in Train manifest: {row.get('id')}")
            # RGBDT500 manifests name the GT field `depth_path`; MS2 uses
            # `thermal_depth_path` (it must be the thermal-view GT there).
            depth_field = row.get("thermal_depth_path") or row.get("depth_path")
            if not depth_field:
                raise ValueError(f"Row {row.get('id')} has no depth GT path")
            # Degenerate GT (e.g. RGBDT500 frames that are almost all holes)
            # would make the masked SSI loss raise mid-epoch; drop them up front.
            valid_fraction = row.get("gt_valid_fraction")
            if valid_fraction is not None and float(valid_fraction) < min_gt_valid_fraction:
                continue
            thermal_path = root / row["thermal_path"]
            depth_path = root / depth_field
            if not thermal_path.is_file():
                raise FileNotFoundError(f"Missing thermal input: {thermal_path}")
            if not depth_path.is_file():
                raise FileNotFoundError(f"Missing GT depth: {depth_path}")
            rows.append(
                {
                    "id": str(row["id"]),
                    "manifest_index": manifest_index,
                    "thermal_path": thermal_path,
                    "depth_path": depth_path,
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


def validate_protocol(args) -> None:
    if args.micro_batch_size != 1 or args.gradient_accumulation_steps != 4:
        raise ValueError("Arm 6 requires micro-batch 1 and accumulation 4.")
    if args.timestep != 999:
        raise ValueError("Arm 6 must match inference timestep 999.")
    if args.gt_loss_weight <= 0:
        raise ValueError("gt-loss-weight must be positive.")
    if args.vae_encoder_learning_rate <= 0:
        raise ValueError("vae-encoder-learning-rate must be positive.")
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


_SPARSIFY_MASKS: list | None = None
_SPARSIFY_DENSITY: float | None = None


def load_sparsify_masks(args) -> tuple[list, float]:
    """MS2 LiDAR validity patterns, cached once per process."""
    global _SPARSIFY_MASKS, _SPARSIFY_DENSITY
    if _SPARSIFY_MASKS is not None:
        return _SPARSIFY_MASKS, _SPARSIFY_DENSITY
    paths = sorted(glob.glob(args.gt_sparsify_source))[: args.gt_sparsify_count]
    if not paths:
        raise SystemExit(f"--gt-sparsify {args.gt_sparsify}: no depth maps matched {args.gt_sparsify_source}")
    masks = [np.asarray(Image.open(path)) > 0 for path in paths]
    _SPARSIFY_MASKS = masks
    _SPARSIFY_DENSITY = float(np.mean([mask.mean() for mask in masks]))
    print(
        f"[sparsify] {len(masks)} MS2 validity patterns from {Path(args.gt_sparsify_source).parent.parent.parent.name}, "
        f"mean density {_SPARSIFY_DENSITY * 100:.1f}%",
        flush=True,
    )
    return _SPARSIFY_MASKS, _SPARSIFY_DENSITY


def sparsify_gt(gt_disparity, valid_mask, row, args):
    """Zero the supervision outside a thinned mask. Deterministic per row, so the
    empty and caption arms of one comparison see byte-identical supervision."""
    height, width = valid_mask.shape[-2:]
    masks, measured = load_sparsify_masks(args)
    pattern = masks[int(row["manifest_index"]) % len(masks)]
    structured = np.asarray(
        Image.fromarray(pattern.astype(np.uint8) * 255).resize((width, height), Image.NEAREST)
    ) > 127

    if args.gt_sparsify == "ms2_lidar":
        keep = structured
    else:
        # Match the structured arm's pixel COUNT per frame, not the pattern's
        # density: MS2's LiDAR blindness overlaps RGBDT500's own holes (both are
        # the sky), so the intersection is smaller than the product. Sampling the
        # same number of pixels leaves spatial arrangement as the only difference.
        dense = valid_mask[0].numpy() > 0.5
        target = int((dense & structured).sum())
        if args.gt_sparsify_density:
            target = int(round(args.gt_sparsify_density * dense.sum()))
        offsets = np.flatnonzero(dense)
        rng = np.random.default_rng(args.gt_sparsify_seed + int(row["manifest_index"]))
        chosen = rng.choice(offsets, size=min(target, offsets.size), replace=False)
        keep = np.zeros(dense.size, bool)
        keep[chosen] = True
        keep = keep.reshape(dense.shape)
    keep_tensor = torch.from_numpy(keep.astype(np.float32))[None]
    thinned = valid_mask * keep_tensor
    remaining = int(thinned.sum())
    if remaining < MIN_VALID_PIXELS:
        raise RuntimeError(
            f"{row['id']}: --gt-sparsify {args.gt_sparsify} left {remaining} valid pixels "
            f"(floor {MIN_VALID_PIXELS}); raise --min-gt-valid-fraction to drop such frames."
        )
    return gt_disparity * thinned, thinned, remaining


def load_sample(row, args):
    thermal = thermal_to_lotus_input(row["thermal_path"], processing_res=0)
    if thermal.diagnostics["converted_uint8_std"] <= 0:
        raise RuntimeError(f"Constant thermal conversion: {row['id']}")
    gt_disparity, valid_mask = load_gt_disparity(
        row["depth_path"], args.gt_min_depth, args.gt_max_depth, args.depth_scale
    )
    dense_valid = int(valid_mask.sum())
    if args.gt_sparsify != "none":
        gt_disparity, valid_mask, _ = sparsify_gt(gt_disparity, valid_mask, row, args)
    diagnostics = {
        "gt_valid_pixels_dense": dense_valid,
        "id": row["id"],
        "manifest_index": row["manifest_index"],
        "thermal": thermal.diagnostics,
        "gt_valid_pixels": int(valid_mask.sum()),
    }
    return thermal.tensor, gt_disparity, valid_mask, diagnostics


def encode_condition(lotus, thermal_tensor, device, trainable):
    encoder_dtype = next(lotus.vae.encoder.parameters()).dtype
    with torch.set_grad_enabled(trainable):
        posterior = lotus.vae.encode(
            thermal_tensor.to(device=device, dtype=encoder_dtype)
        ).latent_dist
        condition = posterior.mode() * lotus.vae.config.scaling_factor
    return condition.to(dtype=torch.float32)


def forward_losses(
    row, thermal_tensor, gt_disparity, valid_mask,
    lotus, teacher_unet, student_unet, prompt, prompt_fp32,
    args, device, teacher_dtype, gt_vae=None, dense_teacher=None,
    empty_prompt_fp32=None, caption_active=False,
):
    condition = encode_condition(lotus, thermal_tensor, device, args.train_vae_encoder)
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
            torch.cat([condition.detach().to(teacher_dtype), latent_input], dim=1),
            timestep,
            encoder_hidden_states=prompt.to(dtype=teacher_dtype),
            class_labels=task_embedding(1, device, teacher_dtype),
            return_dict=False,
        )[0]
    student_response = student_unet(
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
    total = args.response_weight * response["total"] + args.gt_loss_weight * gt_loss
    metrics = {
        "total": total,
        "response_total": response["total"],
        "response_mse": response["mse"],
        "response_cosine": response["cosine"],
        "response_gradient_energy_ratio": response["gradient_energy_ratio"],
        "gt_ssi_l1": gt_loss,
        "gt_abs_rel": gt_abs_rel,
        "gt_valid_pixels": torch.tensor(float(gt_valid)),
    }
    if dense_teacher is not None and args.dense_teacher_weight > 0:
        if dense_teacher.shape[-2:] != predicted_disparity.shape[-2:]:
            raise RuntimeError(
                f"Dense teacher shape {tuple(dense_teacher.shape[-2:])} != "
                f"prediction {tuple(predicted_disparity.shape[-2:])}"
            )
        if args.dense_teacher_align == "l1":
            teacher_loss = torch.mean(torch.abs(predicted_disparity[None] - dense_teacher))
            metrics["dense_teacher_l1"] = teacher_loss
        else:
            dense_mask = torch.ones_like(dense_teacher)
            teacher_loss, teacher_abs_rel, _ = masked_ssi_l1(
                predicted_disparity[None], dense_teacher, dense_mask
            )
            metrics["dense_teacher_ssi_l1"] = teacher_loss
            metrics["dense_teacher_abs_rel"] = teacher_abs_rel
        metrics["total"] = total + args.dense_teacher_weight * teacher_loss
    if args.caption_contrast_weight > 0 and caption_active:
        # Paired empty-prompt reference: identical condition, identical seeded noise,
        # only the text differs.  Held under no_grad so the hinge cannot be satisfied
        # by degrading the empty branch -- the caption branch has to actually improve.
        with torch.no_grad():
            reference_response = student_unet(
                torch.cat([condition.detach(), latent_input.float()], dim=1),
                timestep,
                encoder_hidden_states=empty_prompt_fp32,
                class_labels=task_embedding(1, device, torch.float32),
                return_dict=False,
            )[0]
            reference_disparity = decode_to_disparity(
                lotus, reference_response, device, gt_vae=gt_vae
            )
            reference_gt_loss, _, _ = masked_ssi_l1(
                reference_disparity[None], gt_disparity, valid_mask
            )
        hinge = torch.relu(
            gt_loss - reference_gt_loss + args.caption_contrast_margin * reference_gt_loss
        )
        metrics["caption_contrast_hinge"] = hinge
        # Negative = caption beat empty on this sample.  This is the quantity the
        # whole experiment is about; log it every step.
        metrics["caption_gt_gap"] = (gt_loss - reference_gt_loss).detach()
        metrics["caption_reference_gt_ssi_l1"] = reference_gt_loss
        metrics["total"] = metrics["total"] + args.caption_contrast_weight * hinge
    return metrics


def serializable_settings(args):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in ("resume", "smoke_updates", "overfit_steps")
    }


def save_checkpoint(path, lotus, student_unet, optimizer, step, next_offset, permutation, args, manifest_hash):
    payload = {
        "format": CHECKPOINT_FORMAT,
        "train_mode": "thermal_vae_unet",
        "train_vae_encoder": bool(args.train_vae_encoder),
        "gt_loss_weight": args.gt_loss_weight,
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
    if args.train_vae_encoder:
        payload["vae_encoder_state_dict"] = {
            key: value.detach().cpu() for key, value in lotus.vae.encoder.state_dict().items()
        }
        payload["vae_quant_conv_state_dict"] = {
            key: value.detach().cpu() for key, value in lotus.vae.quant_conv.state_dict().items()
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
    rows = read_train_manifest(manifest, ms2_root, args.min_gt_valid_fraction)
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
        "route": (
            "Arm 6-E: Thermal-VAE condition (trainable encoder) + trained U-Net + masked GT"
            if args.train_vae_encoder
            else "Arm 6: genuine Thermal-VAE condition + trained Lotus U-Net + masked GT"
        ),
        "training_objective": (
            "frozen-teacher response consistency (regularizer; teacher follows current "
            "condition, pins U-Net mapping only) + masked scale-shift-invariant L1"
        ),
        "condition_source": (
            "Lotus VAE mode-encoding of thermal; encoder+quant_conv TRAINABLE (fp32), "
            "decoder frozen"
            if args.train_vae_encoder
            else "frozen Lotus VAE mode-encoding of thermal"
        ),
        "caption": (
            f"per-sample manifest caption, dropout {args.caption_dropout} (Arm-1 protocol)"
            if args.caption_mode == "correct"
            else "empty prompt throughout"
        ),
        "caption_mode": args.caption_mode,
        "dense_teacher_dir": str(args.dense_teacher_dir) if args.dense_teacher_dir else None,
        "dense_teacher_weight": args.dense_teacher_weight,
        "dense_teacher_align": args.dense_teacher_align,
        "gt_loss_weight": args.gt_loss_weight,
        "vae_encoder_learning_rate": (
            args.vae_encoder_learning_rate if args.train_vae_encoder else None
        ),
        "initialization": "pretrained Lotus (fp16 load; fp32 student U-Net; fp32 encoder if trainable)",
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
    gt_vae = None
    if args.gt_decode_fp32:
        gt_vae = copy.deepcopy(lotus.vae).to(device=device, dtype=torch.float32)
        gt_vae.requires_grad_(False).eval()
        gt_vae.encoder = None
    if args.train_vae_encoder:
        lotus.vae.encoder.to(dtype=torch.float32)
        lotus.vae.quant_conv.to(dtype=torch.float32)
        lotus.vae.encoder.requires_grad_(True).train()
        lotus.vae.quant_conv.requires_grad_(True).train()
    empty_prompt, _ = lotus.encode_prompt(
        prompt="", device=device, num_images_per_prompt=1, do_classifier_free_guidance=None
    )
    empty_prompt = empty_prompt.detach()
    empty_prompt_fp32 = empty_prompt.to(dtype=torch.float32)

    if args.caption_contrast_weight > 0 and args.caption_mode != "correct":
        raise SystemExit(
            "--caption-contrast-weight requires --caption-mode correct: the hinge "
            "compares a caption branch against an empty branch, and with caption-mode "
            "empty both branches are identical."
        )

    if args.caption_mode == "correct":
        missing_captions = [row["id"] for row in rows if not row["caption"].strip()]
        if missing_captions:
            raise SystemExit(
                f"--caption-mode correct but {len(missing_captions)} rows lack captions "
                f"(first: {missing_captions[:3]}); use the clip75 train manifest"
            )
        caption_rng = random.Random(args.seed + 424242)

        def prompt_for(row):
            # Returns (prompt, caption_active).  The RNG is drawn exactly once per
            # visit, in the same order as before, so the frozen 1015-sample dropout
            # roster is unchanged.
            if caption_rng.random() < args.caption_dropout:
                return empty_prompt, False
            encoded, _ = lotus.encode_prompt(
                prompt=row["caption"], device=device,
                num_images_per_prompt=1, do_classifier_free_guidance=None,
            )
            return encoded.detach(), True
    else:
        def prompt_for(row):
            del row
            return empty_prompt, False

    student_unet = copy.deepcopy(teacher_unet).to(device=device, dtype=torch.float32)
    student_unet.train().requires_grad_(True)
    encoder_parameters = (
        list(lotus.vae.encoder.parameters()) + list(lotus.vae.quant_conv.parameters())
        if args.train_vae_encoder
        else []
    )
    parameter_groups = [
        {"params": student_unet.parameters(), "lr": args.unet_learning_rate}
    ]
    if encoder_parameters:
        parameter_groups.append(
            {"params": encoder_parameters, "lr": args.vae_encoder_learning_rate}
        )
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)

    start_step = 0
    next_offset = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume.resolve(), map_location="cpu", weights_only=False)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise RuntimeError("Resume checkpoint is not a formal Arm 6 run.")
        if checkpoint.get("manifest_sha256") != manifest_hash:
            raise RuntimeError("Resume manifest hash differs from the frozen run.")
        if checkpoint.get("permutation") != permutation:
            raise RuntimeError("Resume sample permutation differs from the frozen run.")
        if checkpoint.get("settings") != serializable_settings(args):
            raise RuntimeError("Resume settings differ from the frozen Arm 6 run.")
        student_unet.load_state_dict(checkpoint["lotus_unet_state_dict"], strict=True)
        if args.train_vae_encoder:
            lotus.vae.encoder.load_state_dict(checkpoint["vae_encoder_state_dict"], strict=True)
            lotus.vae.quant_conv.load_state_dict(
                checkpoint["vae_quant_conv_state_dict"], strict=True
            )
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["global_step"])
        next_offset = int(checkpoint["next_sample_offset"])
        validate_resume_position(start_step, next_offset, len(rows), effective_batch)
    formal_run = args.smoke_updates is None and args.overfit_steps is None
    if start_step == 0 and next_offset == 0 and formal_run:
        save_checkpoint(
            output / "arm6_step_0000.pt",
            lotus, student_unet, optimizer, 0, 0, permutation, args, manifest_hash,
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
            metric_counts = {}
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
                sample_prompt, caption_active = prompt_for(row)
                dense_teacher = None
                if args.dense_teacher_dir is not None and args.dense_teacher_weight > 0:
                    teacher_path = args.dense_teacher_dir / f"{row['id']}.npy"
                    if not teacher_path.is_file():
                        raise FileNotFoundError(f"Missing dense teacher: {teacher_path}")
                    dense_teacher = torch.from_numpy(
                        np.load(teacher_path)
                    ).float()[None].to(device)
                losses = forward_losses(
                    row, thermal_tensor, gt_disparity, valid_mask,
                    lotus, teacher_unet, student_unet,
                    sample_prompt, sample_prompt.to(dtype=torch.float32),
                    args, device, teacher_dtype, gt_vae=gt_vae, dense_teacher=dense_teacher,
                    empty_prompt_fp32=empty_prompt_fp32, caption_active=caption_active,
                )
                if not bool(torch.isfinite(losses["total"])):
                    raise RuntimeError(f"Non-finite loss at step {step} sample {row['id']}.")
                (losses["total"] / len(batch_indices)).backward()
                for key, value in losses.items():
                    if value.ndim == 0:
                        metric_sums[key] = metric_sums.get(key, 0.0) + float(value.detach())
                        # Count per key, not per batch: the caption_* metrics only exist
                        # on non-dropout samples and would otherwise be under-reported.
                        metric_counts[key] = metric_counts.get(key, 0) + 1
            if any(parameter.grad is not None for parameter in teacher_unet.parameters()):
                raise RuntimeError("Frozen teacher U-Net unexpectedly owns gradients.")
            if any(
                parameter.grad is not None for parameter in lotus.vae.decoder.parameters()
            ):
                raise RuntimeError("Frozen VAE decoder unexpectedly owns gradients.")
            unet_grad_norm = torch.nn.utils.clip_grad_norm_(
                student_unet.parameters(), args.unet_max_grad_norm
            )
            if not bool(torch.isfinite(unet_grad_norm)):
                raise RuntimeError(f"Non-finite U-Net gradient at step {step}.")
            encoder_grad_norm = 0.0
            if encoder_parameters:
                encoder_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(encoder_parameters, float("inf"))
                )
                if not math.isfinite(encoder_grad_norm):
                    raise RuntimeError(f"Non-finite VAE-encoder gradient at step {step}.")
            optimizer.step()
            if args.overfit_steps is None:
                next_offset += len(batch_indices)
            last_record = {
                "step": step,
                "samples_seen": next_offset,
                "batch_size": len(batch_indices),
                "elapsed_seconds": time.time() - start_time,
                "unet_grad_norm": float(unet_grad_norm),
                "encoder_grad_norm": encoder_grad_norm,
                **{key: value / metric_counts[key] for key, value in metric_sums.items()},
            }
            log_handle.write(json.dumps(last_record) + "\n")
            log_handle.flush()
            if step % args.log_interval == 0 or step == 1 or step == total_updates:
                print(json.dumps(last_record), flush=True)
            if formal_run and (step in SAVE_STEPS or step == total_updates):
                name = "arm6_end.pt" if step == total_updates else f"arm6_step_{step:04d}.pt"
                save_checkpoint(
                    output / name,
                    lotus, student_unet, optimizer, step, next_offset,
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
        "end_checkpoint": str(output / "arm6_end.pt"),
        "end_checkpoint_sha256": sha256(output / "arm6_end.pt"),
        "elapsed_seconds": time.time() - start_time,
        "final_training_record": last_record,
        "frozen_modules": {
            "vae": frozen_audit(lotus.vae),
            "text_encoder": frozen_audit(lotus.text_encoder),
            "teacher_unet": frozen_audit(teacher_unet),
        },
        "unet_trainable_parameters": int(
            sum(p.numel() for p in student_unet.parameters() if p.requires_grad)
        ),
        "vae_encoder_trainable_parameters": int(sum(p.numel() for p in encoder_parameters)),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
