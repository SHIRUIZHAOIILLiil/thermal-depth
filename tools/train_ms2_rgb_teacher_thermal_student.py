"""RGB-Lotus teacher -> Thermal student distillation line (no caption).

A sibling of ``tools/train_ms2_joint_gt_v3.py`` that keeps the student branch
identical (Thermal -> frozen AnyThermal -> trainable Adapter V2.3 -> trainable
Lotus U-Net -> student condition / response / x0) but lets the *teacher*
condition come from a different modality:

    --teacher-mode thermal_vae   teacher condition = VAE.encode(thermal)   [reproduces gt_v3]
    --teacher-mode rgb_vae       teacher condition = VAE.encode(RGB)       [the new RGB-teacher line]
    --teacher-mode none          no teacher; only the masked GT loss       [Arm-7 analog]

All three losses are reused unchanged from ``train_ms2_joint_gt_v3``:
condition distillation, response consistency, and masked scale-shift-invariant
GT L1 (decoded through the fp32 VAE via ``--gt-decode-fp32`` by default here).

The original script is NOT modified; every shared helper is imported from it.

GEOMETRIC CAVEAT (frozen conclusion 15): in MS2 the RGB and thermal cameras
do NOT share a viewpoint (RGB 1224x384 AR 3.19 vs thermal 640x256 AR 2.50;
edge NCC 0.13-0.24; per-frame disparity drift +-10-26 px). Resizing RGB to the
thermal frame (done here so the two latents share a spatial size, requirement 8)
makes the tensors *shape-compatible* but NOT *content-aligned*. The condition /
response losses are spatial, so on real MS2 data rgb_vae distils a
view-misaligned target. That is a scientific limitation of the data, not a bug
in this script; the smoke self-test therefore uses content-aligned fake pairs.
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
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = ROOT / "lotus"
TOOLS = ROOT / "tools"
for path in (ROOT, LOTUS_ROOT, TOOLS):
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

# Reuse — never redefine — the losses and utilities from the base line so this
# script is guaranteed to train against the same objective.
from train_ms2_joint_gt_v3 import (  # noqa: E402
    decode_to_disparity,
    encode_caption,
    frozen_audit,
    load_gt_disparity,
    masked_log_depth_l1,
    masked_ssi_l1,
    multiscale_gradient_loss,
    seed_everything,
    task_embedding,
)

SAVE_STEPS = (0, 100, 500, 1000, 2000)
CHECKPOINT_FORMAT = "rgb_teacher_thermal_student_full_epoch"
TEACHER_MODES = ("thermal_vae", "rgb_vae", "none")
DEFAULT_MANIFEST = Path(
    "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/"
    "ms2_fixed_sequence_internvl3_8b_filtered_caption_train.jsonl"
)


# ----------------------------------------------------------------------------- args
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ms2", "vtd"), default="ms2")
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ms2-root", type=Path, default=Path("/mnt/e/dataset/ms2"))
    parser.add_argument("--vtd-root", type=Path, default=None,
                        help="VTD dataset root (required when --dataset vtd).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-mode", choices=TEACHER_MODES, required=True)
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gt-loss-form", choices=("ssi_disparity", "log_depth"), default="ssi_disparity")
    parser.add_argument("--gt-loss-weight", type=float, default=5.0)
    parser.add_argument("--gt-min-depth", type=float, default=0.1)
    parser.add_argument("--gt-max-depth", type=float, default=80.0)
    parser.add_argument("--depth-scale", type=float, default=256.0)
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
        default=True,
        help="fp32 VAE copy for the GT decode (default on; reuses the gt_v3 underflow fix).",
    )
    parser.add_argument("--no-gt-decode-fp32", dest="gt_decode_fp32", action="store_false")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=None)
    parser.add_argument("--overfit-steps", type=int, default=None)
    parser.add_argument(
        "--smoke-selftest",
        action="store_true",
        help="Run the 8-fake-sample forward/backward + gradient/RGB-sensitivity checks and exit.",
    )
    return parser.parse_args()


def validate_protocol(args) -> None:
    if args.micro_batch_size != 1 or args.gradient_accumulation_steps != 4:
        raise ValueError("This line requires micro-batch 1 and accumulation 4 (matches gt_v3).")
    if args.timestep != 999:
        raise ValueError("Must match the Lotus-G inference timestep 999.")
    if args.gt_loss_weight <= 0:
        raise ValueError("--gt-loss-weight must be positive.")
    if args.smoke_updates is not None and "smoke" not in args.output_dir.name:
        raise ValueError("--smoke-updates requires 'smoke' in the output dir name.")
    if args.overfit_steps is not None and "overfit" not in args.output_dir.name:
        raise ValueError("--overfit-steps requires 'overfit' in the output dir name.")


# --------------------------------------------------------------------------- data
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_train_manifest(path: Path, root: Path):
    """Each row must carry thermal_path, rgb_path and thermal_depth_path."""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for manifest_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError(f"Non-train row in train manifest: {row.get('id')}")
            thermal_path = root / row["thermal_path"]
            rgb_path = root / row["rgb_path"]
            depth_path = root / row["thermal_depth_path"]
            for tag, p in (("thermal", thermal_path), ("rgb", rgb_path), ("depth", depth_path)):
                if not p.is_file():
                    raise FileNotFoundError(f"Missing {tag} input: {p}")
            rows.append(
                {
                    "id": str(row["id"]),
                    "manifest_index": manifest_index,
                    "thermal_path": thermal_path,
                    "rgb_path": rgb_path,
                    "depth_path": depth_path,
                }
            )
    if not rows:
        raise ValueError("Train manifest is empty.")
    return rows


def rgb_to_lotus_input(rgb_path, target_hw) -> torch.Tensor:
    """Load an RGB image, normalise to [-1,1] and resize to the thermal frame.

    Resizing to the thermal H*W is what forces VAE.encode(RGB) to land on the
    same latent grid as VAE.encode(thermal) (requirement 8). It does NOT make
    the two views geometrically aligned (see module docstring / conclusion 15).
    """
    image = Image.open(rgb_path).convert("RGB")
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected [H,W,3] RGB, got {array.shape} from {rgb_path}.")
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    tensor = F.interpolate(
        tensor, size=tuple(int(v) for v in target_hw),
        mode="bilinear", align_corners=False, antialias=True,
    )
    return tensor


def load_raw_sample(row, anythermal, args, device):
    """Requirement 7 data interface: rgb, thermal, depth, valid_mask, metadata.

    ``thermal_features`` (the AnyThermal pyramid the adapter consumes) is carried
    alongside as the thermal-branch preprocessing product.
    """
    thermal = thermal_to_lotus_input(row["thermal_path"], processing_res=0)
    if thermal.diagnostics["converted_uint8_std"] <= 0:
        raise RuntimeError(f"Constant thermal conversion: {row['id']}")
    thermal_tensor = thermal.tensor.to(device=device, dtype=torch.float32)  # [1,3,Ht,Wt]
    target_hw = thermal_tensor.shape[-2:]
    rgb_tensor = rgb_to_lotus_input(row["rgb_path"], target_hw).to(device=device, dtype=torch.float32)

    features, _, anythermal_diag = extract_anythermal_feature_pyramid(
        anythermal, row["thermal_path"], enable_grad=False
    )
    features = [feature.detach().float().to(device) for feature in features]

    gt_disparity, valid_mask = load_gt_disparity(
        row["depth_path"], args.gt_min_depth, args.gt_max_depth, args.depth_scale
    )
    metadata = {
        "id": row["id"],
        "manifest_index": row["manifest_index"],
        "thermal_hw": list(map(int, thermal_tensor.shape[-2:])),
        "rgb_hw": list(map(int, rgb_tensor.shape[-2:])),
        "thermal_std": thermal.diagnostics["converted_uint8_std"],
        "anythermal_converted_std": anythermal_diag["converted_uint8_std"],
        "gt_valid_pixels": int(valid_mask.sum()),
    }
    return {
        "rgb": rgb_tensor,
        "thermal": thermal_tensor,
        "thermal_features": features,
        "depth": gt_disparity.to(device),
        "valid_mask": valid_mask.to(device),
        "metadata": metadata,
    }


# ------------------------------------------------------------------------- teacher
def build_teacher(sample, lotus, teacher_unet, empty_prompt, args, device, teacher_dtype):
    """Build the shared noisy latent and (optionally) the teacher targets.

    Requirement 1: teacher and student share timestep, sampled noise and the
    resulting noisy latent. Requirement 4: the teacher is frozen and every
    teacher forward runs inside ``torch.no_grad()``.
    """
    # Canonical latent grid comes from the thermal input (student is thermal-based).
    thermal_latent = encode_condition_latent(lotus.vae, sample["thermal"], posterior="mode").to(
        device=device, dtype=torch.float32
    )
    latent_shape = tuple(thermal_latent.shape)  # [1,4,ht,wt]

    if args.teacher_mode == "thermal_vae":
        teacher_condition = thermal_latent
    elif args.teacher_mode == "rgb_vae":
        teacher_condition = encode_condition_latent(lotus.vae, sample["rgb"], posterior="mode").to(
            device=device, dtype=torch.float32
        )
        if tuple(teacher_condition.shape) != latent_shape:
            raise RuntimeError(
                f"RGB latent {tuple(teacher_condition.shape)} != thermal latent {latent_shape}; "
                "RGB was not resized to the thermal frame."
            )
    else:  # none
        teacher_condition = None

    # Shared noisy depth latent: seeded per sample, identical for teacher & student.
    noise = seeded_noise(
        (1, *latent_shape[1:]),
        seed=args.seed + int(sample["metadata"]["manifest_index"]),
        device=device,
        dtype=teacher_dtype,
        scale=float(lotus.scheduler.init_noise_sigma),
    )
    timestep = torch.full((1,), args.timestep, device=device, dtype=torch.long)
    noisy_latent = lotus.scheduler.scale_model_input(noise, timestep)

    teacher_response = None
    if teacher_condition is not None:
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=teacher_dtype,
            enabled=device.type == "cuda" and teacher_dtype == torch.float16,
        ):
            teacher_response = teacher_unet(
                torch.cat([teacher_condition.to(teacher_dtype), noisy_latent], dim=1),
                timestep,
                encoder_hidden_states=empty_prompt.to(dtype=teacher_dtype),
                class_labels=task_embedding(1, device, teacher_dtype),
                return_dict=False,
            )[0].detach().float()

    return {
        "teacher_condition": None if teacher_condition is None else teacher_condition.detach().float(),
        "teacher_response": teacher_response,
        "noisy_latent": noisy_latent.detach().float(),
        "latent_shape": latent_shape,
    }


# ----------------------------------------------------------------- student + losses
def student_forward_losses(adapter, student_unet, lotus, sample, teacher_pack, empty_prompt, args, device, gt_vae):
    """Student forward + the three reused losses. Returns grad-bearing dict."""
    latent_hw = teacher_pack["latent_shape"][-2:]
    student_condition = adapter(
        sample["thermal_features"], sample["thermal"], target_size=tuple(latent_hw)
    )
    timestep = torch.full((student_condition.shape[0],), args.timestep, device=device, dtype=torch.long)
    student_response = student_unet(
        torch.cat([student_condition, teacher_pack["noisy_latent"]], dim=1),
        timestep,
        encoder_hidden_states=empty_prompt.repeat(student_condition.shape[0], 1, 1),
        class_labels=task_embedding(student_condition.shape[0], device, torch.float32),
        return_dict=False,
    )[0]

    teacher_condition = teacher_pack["teacher_condition"]
    teacher_response = teacher_pack["teacher_response"]
    zero = student_condition.new_zeros(())
    if teacher_condition is not None:
        condition = condition_distillation_losses(
            student_condition,
            teacher_condition,
            cosine_weight=args.cosine_loss_weight,
            channel_stats_weight=args.channel_stats_loss_weight,
            spatial_gradient_weight=args.spatial_gradient_loss_weight,
        )
        condition_multiscale = multiscale_gradient_loss(student_condition, teacher_condition)
        condition_total = condition["total"] + args.multiscale_gradient_loss_weight * condition_multiscale
        response = response_consistency_losses(
            student_response,
            teacher_response,
            cosine_weight=args.response_cosine_loss_weight,
            spatial_gradient_weight=args.response_spatial_gradient_weight,
            multiscale_gradient_weight=args.response_multiscale_gradient_weight,
            gradient_energy_weight=args.response_gradient_energy_weight,
        )
        response_total = response["total"]
        response_cosine = response["cosine"]
    else:
        condition_total = zero
        response_total = zero
        response_cosine = zero

    predicted_disparity = decode_to_disparity(lotus, student_response, device, gt_vae=gt_vae)
    if args.gt_loss_form == "log_depth":
        gt_loss, gt_abs_rel, gt_valid = masked_log_depth_l1(
            predicted_disparity[None], sample["depth"], sample["valid_mask"],
            args.gt_min_depth, args.gt_max_depth,
        )
    else:
        gt_loss, gt_abs_rel, gt_valid = masked_ssi_l1(
            predicted_disparity[None], sample["depth"], sample["valid_mask"]
        )

    total = (
        args.condition_weight * condition_total
        + args.response_weight * response_total
        + args.gt_loss_weight * gt_loss
    )
    return {
        "total_loss": total,
        "condition_loss": condition_total,
        "response_loss": response_total,
        "response_cosine": response_cosine,
        "gt_loss": gt_loss,
        "gt_abs_rel": gt_abs_rel,
        "gt_valid_pixels": torch.tensor(float(gt_valid)),
    }


# ------------------------------------------------------------------- model assembly
def build_models(args, device, teacher_dtype):
    from pipeline import LotusGPipeline

    lotus = LotusGPipeline.from_pretrained(
        args.lotus_model_path, torch_dtype=teacher_dtype, local_files_only=args.local_files_only,
    ).to(device)
    anythermal = AnyThermalEncoder(
        model_path=args.anythermal_model_path, device=str(device), local_files_only=args.local_files_only,
    )
    teacher_unet = lotus.unet
    for module in (lotus.vae, lotus.text_encoder, teacher_unet, anythermal.model):
        module.requires_grad_(False).eval()
    gt_vae = None
    if args.gt_decode_fp32:
        gt_vae = copy.deepcopy(lotus.vae).to(device=device, dtype=torch.float32)
        gt_vae.requires_grad_(False).eval()
        gt_vae.encoder = None  # decode-only copy
    empty_prompt = encode_caption(lotus, "", device)  # requirement 11: no caption, ever

    adapter = AnyThermalLotusAdapterV23().to(device=device, dtype=torch.float32).train()
    student_unet = copy.deepcopy(teacher_unet).to(device=device, dtype=torch.float32)
    student_unet.train().requires_grad_(True)
    # Requirement 5: optimizer holds Adapter + student U-Net only.
    optimizer = torch.optim.AdamW(
        [
            {"params": adapter.parameters(), "lr": args.adapter_learning_rate},
            {"params": student_unet.parameters(), "lr": args.unet_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    return lotus, anythermal, teacher_unet, student_unet, adapter, optimizer, gt_vae, empty_prompt


def save_checkpoint(path, adapter, student_unet, optimizer, step, args, manifest_hash):
    payload = {
        "format": CHECKPOINT_FORMAT,
        "train_mode": "adapter_unet_joint",
        "adapter_architecture": "v2_3_thermal_detail_skip",
        "teacher_mode": args.teacher_mode,
        "caption_training": "off",
        "gt_loss_weight": args.gt_loss_weight,
        "global_step": step,
        "manifest_sha256": manifest_hash,
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "adapter": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
        "lotus_unet_state_dict": {k: v.detach().cpu() for k, v in student_unet.state_dict().items()},
        "optimizer": optimizer.state_dict(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


# ------------------------------------------------------------------------- smoke test
def _write_fake_sample(directory: Path, index: int, seed: int):
    """Write one content-aligned fake (thermal, rgb, depth) triple to disk."""
    rng = np.random.default_rng(seed + index)
    ht, wt = 256, 640
    hr, wr = 384, 1224  # RGB native size differs; the loader resizes it down
    base = rng.integers(4000, 28000, size=(ht, wt), dtype=np.uint16)
    Image.fromarray(base, mode="I;16").save(directory / f"thermal_{index:03d}.png")
    # RGB "aligned": upsample the same base pattern + colour it, so shuffling RGB
    # across samples genuinely changes the teacher condition.
    base_rgb = np.stack([np.asarray(Image.fromarray(base, mode="I;16").resize((wr, hr)))] * 3, axis=-1)
    base_rgb = ((base_rgb.astype(np.float64) / 256.0) % 256).astype(np.uint8)
    base_rgb[..., 1] = (base_rgb[..., 1] + index * 7) % 256
    Image.fromarray(base_rgb, mode="RGB").save(directory / f"rgb_{index:03d}.png")
    depth = rng.integers(0, 80, size=(ht, wt)).astype(np.float64)
    depth[rng.random((ht, wt)) < 0.5] = 0.0  # ~50% invalid
    depth_u16 = (depth * 256.0).astype(np.uint16)
    Image.fromarray(depth_u16, mode="I;16").save(directory / f"depth_{index:03d}.png")
    return {
        "id": f"fake_{index:03d}",
        "manifest_index": index,
        "thermal_path": directory / f"thermal_{index:03d}.png",
        "rgb_path": directory / f"rgb_{index:03d}.png",
        "depth_path": directory / f"depth_{index:03d}.png",
    }


def run_smoke_selftest(args) -> None:
    device = torch.device(args.device)
    teacher_dtype = torch.float16 if args.teacher_dtype == "fp16" else torch.float32
    seed_everything(args.seed)
    lotus, anythermal, teacher_unet, student_unet, adapter, optimizer, gt_vae, empty_prompt = build_models(
        args, device, teacher_dtype
    )
    checks: list[tuple[str, bool, str]] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        rows = [_write_fake_sample(tmp_dir, i, args.seed) for i in range(8)]

        # (1) full forward/backward over 8 fake samples
        optimizer.zero_grad(set_to_none=True)
        total_ok = True
        packs = []
        for row in rows:
            sample = load_raw_sample(row, anythermal, args, device)
            pack = build_teacher(sample, lotus, teacher_unet, empty_prompt, args, device, teacher_dtype)
            packs.append((sample, pack))
            losses = student_forward_losses(
                adapter, student_unet, lotus, sample, pack, empty_prompt, args, device, gt_vae
            )
            if not bool(torch.isfinite(losses["total_loss"])):
                total_ok = False
            (losses["total_loss"] / len(rows)).backward()
        checks.append(("8 fake samples forward/backward finite", total_ok, ""))

        # (2) teacher params have no gradient
        teacher_grad = [p.grad is not None for p in teacher_unet.parameters()]
        vae_grad = [p.grad is not None for p in lotus.vae.parameters()]
        checks.append(
            ("teacher U-Net grads all None", not any(teacher_grad), f"{sum(teacher_grad)} nonzero"),
        )
        checks.append(
            ("frozen VAE grads all None", not any(vae_grad), f"{sum(vae_grad)} nonzero"),
        )

        # (3) student params carry nonzero gradient
        adapter_nonzero = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in adapter.parameters())
        unet_nonzero = any(
            p.grad is not None and float(p.grad.abs().sum()) > 0 for p in student_unet.parameters()
        )
        checks.append(("adapter has nonzero grad", adapter_nonzero, ""))
        checks.append(("student U-Net has nonzero grad", unet_nonzero, ""))

        # (4) shuffling RGB changes the teacher loss (rgb_vae mode only)
        if args.teacher_mode == "rgb_vae":
            sample0, _ = packs[0]
            sample1, _ = packs[1]
            pack_self = build_teacher(sample0, lotus, teacher_unet, empty_prompt, args, device, teacher_dtype)
            swapped = dict(sample0)
            swapped["rgb"] = sample1["rgb"]  # borrow a different image's RGB
            pack_swap = build_teacher(swapped, lotus, teacher_unet, empty_prompt, args, device, teacher_dtype)
            with torch.no_grad():
                delta = float((pack_self["teacher_condition"] - pack_swap["teacher_condition"]).abs().mean())
                resp_delta = float((pack_self["teacher_response"] - pack_swap["teacher_response"]).abs().mean())
            checks.append(("shuffling RGB changes teacher condition", delta > 1e-6, f"|d|={delta:.4e}"))
            checks.append(("shuffling RGB changes teacher response", resp_delta > 1e-6, f"|d|={resp_delta:.4e}"))
        else:
            checks.append((f"RGB-sensitivity (skipped for teacher_mode={args.teacher_mode})", True, "n/a"))

    print("\n=== RGB-teacher smoke self-test ===")
    all_pass = True
    for name, ok, note in checks:
        all_pass &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({note})" if note else ""))
    print(f"=== {'ALL PASS' if all_pass else 'FAILURES PRESENT'} ===")
    if not all_pass:
        raise SystemExit(1)


# ------------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    validate_protocol(args)
    if args.smoke_selftest:
        run_smoke_selftest(args)
        return

    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    teacher_dtype = torch.float16 if args.teacher_dtype == "fp16" else torch.float32
    effective_batch = args.micro_batch_size * args.gradient_accumulation_steps

    # Dataset dispatch: MS2 (default) or VTD (pixel-aligned RGB-thermal, the
    # legal venue for teacher_mode=rgb_vae). Both yield the same 5-key sample
    # interface, so only the loader differs downstream.
    vtd_cfg = None
    if args.dataset == "vtd":
        if args.vtd_root is None:
            raise SystemExit("--dataset vtd requires --vtd-root")
        import vtd_dataset
        vtd_cfg = vtd_dataset.VTDConfig.from_args(args)
        rows = vtd_dataset.scan_vtd(vtd_cfg)
        manifest_hash = "vtd:" + vtd_cfg.signature()
    else:
        manifest = args.train_manifest.resolve()
        ms2_root = args.ms2_root.resolve()
        manifest_hash = sha256(manifest)
        rows = read_train_manifest(manifest, ms2_root)
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1.")

    permutation = list(range(len(rows)))
    random.Random(args.seed).shuffle(permutation)
    sample_stream = permutation * args.epochs
    overfit_pool = permutation[:32]
    total_updates = math.ceil(len(sample_stream) / effective_batch)
    if args.smoke_updates is not None:
        total_updates = min(args.smoke_updates, total_updates)
    if args.overfit_steps is not None:
        total_updates = args.overfit_steps

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    lotus, anythermal, teacher_unet, student_unet, adapter, optimizer, gt_vae, empty_prompt = build_models(
        args, device, teacher_dtype
    )

    frozen_config = {
        "route": "RGB-Lotus teacher -> Thermal student (Adapter V2.3 + U-Net + masked GT)",
        "teacher_mode": args.teacher_mode,
        "objective": "condition distill + response consistency + masked SSI GT L1 (no caption)",
        "gt_decode_fp32": bool(args.gt_decode_fp32),
        "gt_loss_weight": args.gt_loss_weight,
        "gt_loss_form": args.gt_loss_form,
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "train_samples": len(rows),
        "epochs": args.epochs,
        "effective_batch_size": effective_batch,
        "optimizer_updates": total_updates,
        "adapter_audit": frozen_audit(adapter),
        "student_unet_audit": frozen_audit(student_unet),
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (output / "frozen_config.json").write_text(
        json.dumps(frozen_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log_path = output / "training_metrics.jsonl"
    diag_path = output / "sample_audit_first8.json"
    first_diag = []
    next_offset = 0
    start_time = time.time()
    with log_path.open("a", encoding="utf-8") as log_handle:
        for step in range(1, total_updates + 1):
            if args.overfit_steps is not None:
                base = ((step - 1) * effective_batch) % len(overfit_pool)
                batch_indices = [overfit_pool[(base + o) % len(overfit_pool)] for o in range(effective_batch)]
            else:
                batch_indices = sample_stream[next_offset : next_offset + effective_batch]
            if not batch_indices:
                raise RuntimeError(f"Empty batch at step {step}.")
            optimizer.zero_grad(set_to_none=True)
            sums: dict[str, float] = {}
            for row_index in batch_indices:
                row = rows[row_index]
                if vtd_cfg is not None:
                    import vtd_dataset
                    sample = vtd_dataset.load_vtd_raw_sample(row, anythermal, args, device, vtd_cfg)
                else:
                    sample = load_raw_sample(row, anythermal, args, device)
                pack = build_teacher(sample, lotus, teacher_unet, empty_prompt, args, device, teacher_dtype)
                if len(first_diag) < 8:
                    first_diag.append(sample["metadata"])
                    if len(first_diag) == 8:
                        diag_path.write_text(json.dumps(first_diag, indent=2, ensure_ascii=False), encoding="utf-8")
                losses = student_forward_losses(
                    adapter, student_unet, lotus, sample, pack, empty_prompt, args, device, gt_vae
                )
                if not bool(torch.isfinite(losses["total_loss"])):
                    raise RuntimeError(f"Non-finite loss at step {step} sample {row['id']}.")
                (losses["total_loss"] / len(batch_indices)).backward()
                for key, value in losses.items():
                    if torch.is_tensor(value) and value.ndim == 0:
                        sums[key] = sums.get(key, 0.0) + float(value.detach())

            # Requirement 4 guard: teacher must never accumulate gradients.
            if any(p.grad is not None for p in teacher_unet.parameters()):
                raise RuntimeError("Frozen teacher U-Net unexpectedly owns gradients.")
            adapter_grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), float("inf"))
            unet_grad_norm = torch.nn.utils.clip_grad_norm_(student_unet.parameters(), args.unet_max_grad_norm)
            if not bool(torch.isfinite(adapter_grad_norm)) or not bool(torch.isfinite(unet_grad_norm)):
                raise RuntimeError(f"Non-finite gradient at step {step}.")
            optimizer.step()
            if args.overfit_steps is None:
                next_offset += len(batch_indices)

            n = len(batch_indices)
            record = {
                "step": step,
                "samples_seen": next_offset,
                "condition_loss": sums.get("condition_loss", 0.0) / n,
                "response_loss": sums.get("response_loss", 0.0) / n,
                "gt_loss": sums.get("gt_loss", 0.0) / n,
                "total_loss": sums.get("total_loss", 0.0) / n,
                "gt_abs_rel": sums.get("gt_abs_rel", 0.0) / n,
                "adapter_grad_norm": float(adapter_grad_norm),
                "unet_grad_norm": float(unet_grad_norm),
                "elapsed_seconds": round(time.time() - start_time, 3),
            }
            if step % args.log_interval == 0 or step == total_updates:
                log_handle.write(json.dumps(record) + "\n")
                log_handle.flush()
                print(
                    f"step {step}/{total_updates} teacher={args.teacher_mode} "
                    f"cond {record['condition_loss']:.4f} resp {record['response_loss']:.4f} "
                    f"gt {record['gt_loss']:.4f} total {record['total_loss']:.4f} "
                    f"a_grad {record['adapter_grad_norm']:.2f} u_grad {record['unet_grad_norm']:.2f}"
                )
            formal = args.smoke_updates is None and args.overfit_steps is None
            if formal and (step in SAVE_STEPS):
                save_checkpoint(output / f"step_{step:04d}.pt", adapter, student_unet, optimizer, step, args, manifest_hash)

    if args.smoke_updates is None and args.overfit_steps is None:
        save_checkpoint(output / "rgb_teacher_end.pt", adapter, student_unet, optimizer, total_updates, args, manifest_hash)
    print("done.")


if __name__ == "__main__":
    main()
