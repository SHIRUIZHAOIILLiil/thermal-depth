
"""Overfit 32 fixed MS2 samples with AnyThermal -> Adapter -> Lotus-G."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = REPO_ROOT / "lotus"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(LOTUS_ROOT) not in sys.path:
    sys.path.insert(0, str(LOTUS_ROOT))

from diffusers import DDPMScheduler  # noqa: E402
from models.anythermal_encoder import AnyThermalEncoder  # noqa: E402
from models.anythermal_lotus_adapter import AnyThermalLotusAdapter  # noqa: E402
from models.anythermal_lotus_model import AnyThermalLotusModel  # noqa: E402
from pipeline import LotusGPipeline  # noqa: E402

DEFAULT_MANIFEST = "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/ms2_fixed_sequence_internvl3_8b_filtered_caption_train.jsonl"
DEFAULT_SAVE_STEPS = "0,10,50,100,250,500,1000,1500,2000"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ms2-root", default="/mnt/e/dataset/ms2")
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--sample-list", default=None)
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    p.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    p.add_argument("--anythermal-revision", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--adapter-lr", type=float, default=1e-4)
    p.add_argument("--lotus-unet-lr", type=float, default=1e-6)
    p.add_argument("--train-mode", choices=("adapter_only", "adapter_unet"), default="adapter_only")
    p.add_argument("--timestep", type=int, default=999)
    p.add_argument("--prediction-type", default="sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--half-precision", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--save-steps", default=DEFAULT_SAVE_STEPS)
    p.add_argument("--visual-samples", type=int, default=4)
    p.add_argument("--resume", default=None)
    p.add_argument("--output-dir", default="outputs/adapter_v0_overfit32")
    return p.parse_args()


def parse_save_steps(text, max_step):
    values = sorted({int(x.strip()) for x in text.split(",") if x.strip()})
    values = [x for x in values if 0 <= x <= max_step]
    if 0 not in values:
        values.insert(0, 0)
    if max_step not in values:
        values.append(max_step)
    return sorted(set(values))


def tensor_stats(t):
    v = t.detach().float().cpu()
    return {"shape": list(v.shape), "mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max()), "l2_norm": float(torch.linalg.vector_norm(v)), "finite": bool(torch.isfinite(v).all())}


def np_stats(a):
    v = np.asarray(a, dtype=np.float32)
    finite = np.isfinite(v)
    if not finite.any():
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "finite": False}
    good = v[finite]
    return {"mean": float(good.mean()), "std": float(good.std()), "min": float(good.min()), "max": float(good.max()), "finite": bool(finite.all())}


def grad_norm(params):
    total = torch.zeros((), device="cpu")
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().float().cpu().pow(2).sum()
    return float(total.sqrt())


def save_minmax(a, path, invalid_mask=None):
    v = np.asarray(a, dtype=np.float32)
    if v.ndim == 3:
        v = v[..., 0]
    finite = np.isfinite(v)
    if invalid_mask is not None:
        finite = finite & (~invalid_mask.astype(bool))
    if finite.any() and float(v[finite].max()) > float(v[finite].min()):
        s = (v - v[finite].min()) / (v[finite].max() - v[finite].min())
    else:
        s = np.zeros_like(v)
    img = (np.clip(s, 0, 1) * 255).round().astype(np.uint8)
    if invalid_mask is not None:
        img[invalid_mask.astype(bool)] = 32
    Image.fromarray(img, mode="L").save(path)


def load_depth(path):
    with Image.open(path) as im:
        depth = np.asarray(im).astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    valid = np.isfinite(depth) & (depth > 0)
    norm = np.zeros_like(depth, dtype=np.float32)
    if valid.any():
        vals = depth[valid]
        lo, hi = float(vals.min()), float(vals.max())
        if hi > lo:
            norm = np.clip((depth - lo) / (hi - lo), 0, 1)
            norm[~valid] = 0.0
    target = torch.from_numpy(norm).view(1, *norm.shape).repeat(3, 1, 1).float() * 2.0 - 1.0
    mask = torch.from_numpy(valid.astype(np.bool_)).view(1, *valid.shape)
    stats = {"raw_depth": np_stats(depth), "valid_fraction": float(valid.mean()), "normalization_uses_valid_pixels_only": True}
    return target, mask, depth, valid, stats


def resolve_path(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_manifest(path, root, limit):
    samples = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if len(samples) >= limit:
                break
            item = json.loads(line)
            thermal = resolve_path(root, item["thermal_path"])
            depth = resolve_path(root, item.get("thermal_depth_path") or item["depth_path"])
            if not thermal.is_file() or not depth.is_file():
                continue
            samples.append({
                "id": str(item.get("id", f"sample_{len(samples):06d}")),
                "image_id": str(item.get("image_id", thermal.stem)),
                "thermal": str(thermal),
                "depth": str(depth),
                "caption": str(item.get("caption", "")),
                "split": str(item.get("split", "train")),
                "sequence": str(item.get("sequence", "")),
            })
    return samples


def load_sample_list(path, root, limit):
    samples = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if len(samples) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                item = json.loads(line)
                thermal_value = item.get("thermal") or item.get("thermal_path")
                depth_value = item.get("depth") or item.get("depth_path") or item.get("thermal_depth_path")
                sample_id = item.get("id", f"sample_{len(samples):06d}")
                caption = item.get("caption", "")
                split = item.get("split", "train")
                sequence = item.get("sequence", "")
                image_id = item.get("image_id", Path(str(thermal_value)).stem)
            else:
                parts = line.replace(",", " ").split()
                thermal_value, depth_value = parts[:2]
                sample_id = f"sample_{len(samples):06d}"
                caption, split, sequence, image_id = "", "train", "", Path(thermal_value).stem
            thermal = resolve_path(root, str(thermal_value))
            depth = resolve_path(root, str(depth_value))
            if not thermal.is_file() or not depth.is_file():
                raise FileNotFoundError(f"Missing sample paths: thermal={thermal}, depth={depth}")
            samples.append({"id": str(sample_id), "image_id": str(image_id), "thermal": str(thermal), "depth": str(depth), "caption": str(caption), "split": str(split), "sequence": str(sequence)})
    return samples


def discover_samples(root, limit):
    samples = []
    for thermal in sorted((root / "sync_data").glob("*/thr/img_left/*.png")):
        seq_dir = thermal.parents[2].name
        seq = seq_dir.lstrip("_")
        stem = thermal.stem
        candidates = [
            root / "proj_depth" / seq_dir / "thr" / "depth_filtered" / f"{stem}.png",
            root / "proj_depth" / seq_dir / "thr" / "depth" / f"{stem}.png",
        ]
        depth = next((p for p in candidates if p.is_file()), None)
        if depth is None:
            continue
        samples.append({"id": f"{seq}_{stem}", "image_id": stem, "thermal": str(thermal), "depth": str(depth), "caption": "", "split": "train", "sequence": seq})
        if len(samples) >= limit:
            break
    return samples


def batch_indices_for_step(step, total, batch_size, seed):
    batches_per_epoch = math.ceil(total / batch_size)
    epoch = max(step - 1, 0) // batches_per_epoch
    batch_id = max(step - 1, 0) % batches_per_epoch
    indices = list(range(total))
    random.Random(seed + epoch).shuffle(indices)
    start = batch_id * batch_size
    batch = indices[start:start + batch_size]
    if len(batch) < batch_size:
        batch += indices[:batch_size - len(batch)]
    return batch


def stack_features(cache, batch_indices, device):
    return [torch.cat([cache[i][level] for i in batch_indices], dim=0).to(device) for level in range(len(cache[0]))]


def stack_items(items, batch_indices, device):
    return torch.stack([items[i] for i in batch_indices], dim=0).to(device)


def make_noise(batch_size, latent_size, step, batch_marker, seed, device, dtype):
    gen = torch.Generator(device=device).manual_seed(seed + step * 1009 + batch_marker * 9176)
    return torch.randn((batch_size, 4, latent_size[0], latent_size[1]), generator=gen, device=device, dtype=dtype)


def configure_trainability(model, mode):
    model.anythermal_encoder.model.eval().requires_grad_(False)
    model.lotus.vae.eval().requires_grad_(False)
    model.lotus.text_encoder.eval().requires_grad_(False)
    model.adapter.train().requires_grad_(True)
    if mode == "adapter_only":
        model.lotus.unet.eval().requires_grad_(False)
    else:
        model.lotus.unet.train().requires_grad_(True)


def train_params(model, mode):
    params = [p for p in model.adapter.parameters() if p.requires_grad]
    if mode == "adapter_unet":
        params.extend([p for p in model.lotus.unet.parameters() if p.requires_grad])
    return params


def make_optimizer(model, args):
    if args.train_mode == "adapter_only":
        groups = [{"params": model.adapter.parameters(), "lr": args.adapter_lr}]
    else:
        groups = [
            {"params": model.adapter.parameters(), "lr": args.adapter_lr},
            {"params": model.lotus.unet.parameters(), "lr": args.lotus_unet_lr},
        ]
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda _: 1.0)
    return opt, sched


def run_batch(model, feature_cache, depth_targets, valid_masks, batch_indices, step, seed, device, latent_size, timestep, return_decoded):
    features = stack_features(feature_cache, batch_indices, device)
    depth = stack_items(depth_targets, batch_indices, device)
    valid = stack_items(valid_masks, batch_indices, device)
    timesteps = torch.full((len(batch_indices),), timestep, device=device, dtype=torch.long)
    dtype = next(model.lotus.unet.parameters()).dtype
    noise = make_noise(len(batch_indices), latent_size, step, sum(batch_indices) + len(batch_indices), seed, device, dtype)
    return model(features=features, depth_values=depth, valid_mask=valid, timesteps=timesteps, noise=noise, return_decoded=return_decoded)


def evaluate_all(model, feature_cache, depth_targets, valid_masks, batch_size, step, seed, device, latent_size, timestep, decode):
    losses, adapter_stats, pred_batches, decoded_batches = [], [], [], []
    adapter_training = model.adapter.training
    unet_training = model.lotus.unet.training
    model.adapter.eval()
    model.lotus.unet.eval()
    with torch.no_grad():
        for start in range(0, len(depth_targets), batch_size):
            idx = list(range(start, min(start + batch_size, len(depth_targets))))
            out = run_batch(model, feature_cache, depth_targets, valid_masks, idx, step, seed, device, latent_size, timestep, decode)
            losses.append(float(out["loss"].detach().cpu()) * len(idx))
            adapter_stats.append(tensor_stats(out["condition_latent"]))
            pred_batches.append(out["model_pred"].detach().float().cpu())
            if decode and out["decoded"] is not None:
                decoded_batches.append(out["decoded"].detach().float().cpu())
    if adapter_training:
        model.adapter.train()
    if unet_training:
        model.lotus.unet.train()
    result = {
        "loss": sum(losses) / len(depth_targets),
        "adapter_output_mean": float(np.mean([s["mean"] for s in adapter_stats])),
        "adapter_output_std": float(np.mean([s["std"] for s in adapter_stats])),
        "model_pred": torch.cat(pred_batches, dim=0),
    }
    if decoded_batches:
        result["decoded"] = torch.cat(decoded_batches, dim=0)
    return result


def save_checkpoint(path, model, optimizer, scheduler, step, args, samples, history):
    ckpt = {
        "adapter": model.adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": step,
        "random_seed": args.seed,
        "config": vars(args),
        "samples": samples,
        "history": list(history),
    }
    if args.train_mode == "adapter_unet":
        ckpt["lotus_unet"] = model.lotus.unet.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.adapter.load_state_dict(ckpt["adapter"])
    if "lotus_unet" in ckpt:
        model.lotus.unet.load_state_dict(ckpt["lotus_unet"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt["global_step"]), list(ckpt.get("history", []))


def thermal_vis(path):
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    vals = arr.astype(np.float32)
    if vals.max() > vals.min():
        vals = (vals - vals.min()) / (vals.max() - vals.min())
    else:
        vals = np.zeros_like(vals)
    return (np.clip(vals, 0, 1) * 255).astype(np.uint8)


def make_panel(items, path):
    tiles, w, h = [], 320, 128
    for label, arr in items:
        img = Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB").resize((w, h), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, w, 18), fill=(0, 0, 0))
        draw.text((4, 3), label, fill=(255, 255, 255))
        tiles.append(img)
    canvas = Image.new("RGB", (w * len(tiles), h), (0, 0, 0))
    for i, img in enumerate(tiles):
        canvas.paste(img, (i * w, 0))
    canvas.save(path)


def save_step(output_dir, step, eval_result, samples, raw_depths, valid_np, visual_samples):
    step_dir = output_dir / f"step_{step:04d}"
    vis_dir = output_dir / "visual_comparison"
    step_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    np.save(step_dir / "predicted_latents.npy", eval_result["model_pred"].numpy())
    if "decoded" in eval_result:
        pred_depth = eval_result["decoded"].mean(dim=1).numpy()
        np.save(step_dir / "predicted_depth.npy", pred_depth)
        for i in range(min(visual_samples, len(samples))):
            invalid = ~valid_np[i]
            pred_png = step_dir / f"pred_depth_{i:02d}.png"
            gt_png = step_dir / f"gt_depth_{i:02d}.png"
            mask_png = step_dir / f"valid_mask_{i:02d}.png"
            save_minmax(pred_depth[i], pred_png)
            save_minmax(raw_depths[i], gt_png, invalid_mask=invalid)
            Image.fromarray((valid_np[i].astype(np.uint8) * 255), mode="L").save(mask_png)
            make_panel([
                ("thermal", thermal_vis(Path(samples[i]["thermal"]))),
                ("gt depth", np.asarray(Image.open(gt_png).convert("L"))),
                ("valid mask", np.asarray(Image.open(mask_png).convert("L"))),
                (f"pred step {step}", np.asarray(Image.open(pred_png).convert("L"))),
            ], vis_dir / f"sample_{i:02d}_step_{step:04d}.png")
    with (step_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump({"step": step, "loss": eval_result["loss"], "adapter_output_mean": eval_result["adapter_output_mean"], "adapter_output_std": eval_result["adapter_output_std"]}, f, indent=2)


def save_loss_curve(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    train = [h for h in history if h.get("train_loss") is not None]
    evals = [h for h in history if h.get("eval_loss") is not None]
    plt.figure(figsize=(8, 4.5))
    if train:
        plt.plot([h["step"] for h in train], [h["train_loss"] for h in train], label="train batch loss", linewidth=1)
    if evals:
        plt.plot([h["step"] for h in evals], [h["eval_loss"] for h in evals], marker="o", label="fixed-32 eval loss")
    plt.xlabel("optimizer step")
    plt.ylabel("latent MSE on valid mask")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def build_summary(history, args):
    records = [h for h in history if h.get("eval_loss") is not None] or history
    def loss_of(x):
        return float(x["eval_loss"] if x.get("eval_loss") is not None else x["train_loss"])
    initial, final = loss_of(records[0]), loss_of(records[-1])
    best = min(loss_of(r) for r in records)
    recent = [loss_of(r) for r in records[-3:]]
    plateau = len(recent) >= 3 and (max(recent) - min(recent)) / max(recent[0], 1e-8) < 0.03
    improved = (initial - final) / max(initial, 1e-8)
    has_structure = bool(improved > 0.35 and final < 0.25)
    return {
        "initial_loss": initial,
        "final_loss": final,
        "best_loss": best,
        "num_steps": int(args.steps),
        "adapter_output_std_start": float(records[0].get("adapter_output_std", 0.0)),
        "adapter_output_std_end": float(records[-1].get("adapter_output_std", 0.0)),
        "artifact_removed": bool(has_structure and not plateau),
        "prediction_has_scene_structure": has_structure,
        "loss_plateau_detected": bool(plateau),
        "loss_improved_fraction": improved,
        "train_mode": args.train_mode,
        "summary_heuristic_note": "artifact_removed and prediction_has_scene_structure are automatic heuristics; inspect visual_comparison for final judgment.",
    }


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"CUDA is not available for --device {args.device}")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_steps = parse_save_steps(args.save_steps, args.steps)
    root = Path(args.ms2_root)

    if args.sample_list:
        samples = load_sample_list(Path(args.sample_list), root, args.num_samples)
    elif args.manifest:
        samples = load_manifest(Path(args.manifest), root, args.num_samples)
    else:
        samples = discover_samples(root, args.num_samples)
    if len(samples) != args.num_samples:
        raise SystemExit(f"Expected {args.num_samples} MS2 samples, found {len(samples)}.")
    with (output_dir / "fixed_samples.json").open("w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    anythermal = AnyThermalEncoder(
        args.anythermal_model_path,
        device=args.device,
        revision=args.anythermal_revision,
        local_files_only=args.local_files_only,
    )
    lotus_dtype = torch.float16 if args.half_precision else torch.float32
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.lotus_model_path,
        subfolder="scheduler",
        local_files_only=args.local_files_only,
    )
    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
    lotus = LotusGPipeline.from_pretrained(
        args.lotus_model_path,
        scheduler=noise_scheduler,
        torch_dtype=lotus_dtype,
        safety_checker=None,
        local_files_only=args.local_files_only,
    ).to(device)
    if int(lotus.unet.config.in_channels) != 8:
        raise SystemExit(f"Expected Lotus-G U-Net in_channels=8, got {lotus.unet.config.in_channels}")

    model = AnyThermalLotusModel(
        anythermal_encoder=anythermal,
        lotus_pipeline=lotus,
        adapter=AnyThermalLotusAdapter().to(device),
        noise_scheduler=noise_scheduler,
    ).to(device)
    configure_trainability(model, args.train_mode)
    optimizer, lr_scheduler = make_optimizer(model, args)

    feature_cache, depth_targets, valid_masks = [], [], []
    raw_depths, valid_np, sample_stats, feature_infos = [], [], [], []
    for sample in samples:
        features, info, _ = model.extract_features(Path(sample["thermal"]))
        feature_cache.append([f.detach().cpu() for f in features])
        depth, mask, raw, valid, stats = load_depth(Path(sample["depth"]))
        depth_targets.append(depth)
        valid_masks.append(mask)
        raw_depths.append(raw)
        valid_np.append(valid)
        sample_stats.append(stats)
        feature_infos.append({
            "transformer_block_indices": list(info.transformer_block_indices),
            "grid_size": list(info.grid_size),
            "preprocessed_shape": list(info.preprocessed_shape),
            "original_shape": list(info.original_shape),
            "num_register_tokens": info.num_register_tokens,
            "has_cls_token": info.has_cls_token,
        })

    first_feature_shapes = [tuple(f.shape) for f in feature_cache[0]]
    first_depth_shape = tuple(depth_targets[0].shape)
    if any([tuple(f.shape) for f in item] != first_feature_shapes for item in feature_cache):
        raise SystemExit("All fixed samples must share AnyThermal feature shapes for batching.")
    if any(tuple(d.shape) != first_depth_shape for d in depth_targets):
        raise SystemExit("All fixed samples must share depth target shapes for batching.")
    latent_size = (first_depth_shape[-2] // 8, first_depth_shape[-1] // 8)

    history, start_step = [], 0
    if args.resume:
        start_step, history = load_checkpoint(Path(args.resume), model, optimizer, lr_scheduler)
        configure_trainability(model, args.train_mode)
        print(f"Resumed from {args.resume} at step {start_step}")

    run_info = {
        "status": "running",
        "models": {"anythermal": args.anythermal_model_path, "anythermal_revision": args.anythermal_revision, "lotus": args.lotus_model_path},
        "settings": {**vars(args), "save_steps": save_steps, "caption_used": False, "data_augmentation": "disabled", "fixed_training_order": "deterministic epoch shuffle from seed + epoch", "noise_sampling": "deterministic torch.Generator seed + step/batch marker"},
        "sample_ids": [s["id"] for s in samples],
        "feature_info_first_sample": feature_infos[0],
        "feature_shapes": [list(s) for s in first_feature_shapes],
        "depth_target_shape": list(first_depth_shape),
        "latent_size": list(latent_size),
        "sample_stats_first": sample_stats[0],
    }
    with (output_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    if start_step == 0 and not any(h.get("step") == 0 for h in history):
        eval_result = evaluate_all(model, feature_cache, depth_targets, valid_masks, args.batch_size, 0, args.seed, device, latent_size, args.timestep, True)
        history.append({"step": 0, "train_loss": None, "eval_loss": eval_result["loss"], "adapter_output_mean": eval_result["adapter_output_mean"], "adapter_output_std": eval_result["adapter_output_std"], "grad_norm": 0.0, "lr": [g["lr"] for g in optimizer.param_groups]})
        save_step(output_dir, 0, eval_result, samples, raw_depths, valid_np, args.visual_samples)
        save_checkpoint(output_dir / "checkpoint_step_0000.pt", model, optimizer, lr_scheduler, 0, args, samples, history)

    for step in range(start_step + 1, args.steps + 1):
        configure_trainability(model, args.train_mode)
        batch_idx = batch_indices_for_step(step, len(samples), args.batch_size, args.seed)
        out = run_batch(model, feature_cache, depth_targets, valid_masks, batch_idx, step, args.seed, device, latent_size, args.timestep, False)
        loss = out["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = grad_norm(train_params(model, args.train_mode))
        optimizer.step()
        lr_scheduler.step()
        cstats = tensor_stats(out["condition_latent"])
        record = {
            "step": step,
            "train_loss": float(loss.detach().cpu()),
            "eval_loss": None,
            "adapter_output_mean": cstats["mean"],
            "adapter_output_std": cstats["std"],
            "grad_norm": gnorm,
            "lr": [g["lr"] for g in optimizer.param_groups],
            "batch_indices": batch_idx,
            "batch_sample_ids": [samples[i]["id"] for i in batch_idx],
        }
        if step in save_steps:
            eval_result = evaluate_all(model, feature_cache, depth_targets, valid_masks, args.batch_size, step, args.seed, device, latent_size, args.timestep, True)
            record["eval_loss"] = eval_result["loss"]
            record["adapter_output_mean"] = eval_result["adapter_output_mean"]
            record["adapter_output_std"] = eval_result["adapter_output_std"]
            save_step(output_dir, step, eval_result, samples, raw_depths, valid_np, args.visual_samples)
            save_checkpoint(output_dir / f"checkpoint_step_{step:04d}.pt", model, optimizer, lr_scheduler, step, args, samples, [*history, record])
            save_checkpoint(output_dir / "checkpoint_latest.pt", model, optimizer, lr_scheduler, step, args, samples, [*history, record])
            print(f"step={step:04d} train_loss={record['train_loss']:.6f} eval_loss={record['eval_loss']:.6f} adapter_std={record['adapter_output_std']:.6f} grad_norm={gnorm:.6f}")
        elif step % 10 == 0:
            print(f"step={step:04d} train_loss={record['train_loss']:.6f} adapter_std={record['adapter_output_std']:.6f} grad_norm={gnorm:.6f}")
        history.append(record)
        with (output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    save_loss_curve(history, output_dir / "loss_curve.png")
    summary = build_summary(history, args)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    run_info["status"] = "complete"
    run_info["summary"] = summary
    with (output_dir / "run_info.json").open("w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)
    torch.save(model.adapter.state_dict(), output_dir / "adapter_final.pt")
    if args.train_mode == "adapter_unet":
        torch.save(model.lotus.unet.state_dict(), output_dir / "lotus_unet_final.pt")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
