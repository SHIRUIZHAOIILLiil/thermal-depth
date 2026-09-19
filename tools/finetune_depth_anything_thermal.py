"""Adapt Depth Anything V2 to MS2 thermal, under our supervision.

Zero-shot, DA2 scores AbsRel 0.153 on day against our 0.073 -- a two-fold lead
that says only that an RGB model has never seen thermal, which is the premise of
the field rather than a result. A table of unadapted comparators is a table
nobody has to believe. This trains one that has been adapted, on our data, so
the comparison has something at stake.

Their repository releases fine-tuning code only for the metric variant, and
metric is the wrong target for an affine-invariant table. So the recipe is
written here instead, which is the better outcome for fairness: the supervision
is then demonstrably the same one our own line trains under -- the official
8-sequence split, the completed pseudo depth with lidar written over it, the
same scale-shift-invariant loss in disparity space, the same val rule for
picking a checkpoint. Using their trainer would mean arguing that we had not
quietly given ourselves an easier target.

The encoder is not frozen. Their own metric recipe fine-tunes it at a tenth of
the decoder's rate, and freezing it would leave an RGB-pretrained encoder
looking at thermal -- an opponent with one hand tied, whose defeat would prove
nothing.

    python tools/finetune_depth_anything_thermal.py \
        --manifest $SCRATCH/manifests/.../ms2_train_official8_thermalcap_....jsonl \
        --ms2-root $SCRATCH/data/ms2 \
        --pseudo-dir $SCRATCH/runs/pseudo_gt/official_train/calibrated_pseudo_depth \
        --out-dir $SCRATCH/runs/da2_thermal/v1 --smoke
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

D_MIN, D_MAX = 1e-3, 80.0
MIN_VALID_PIXELS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ms2-root", required=True, type=Path)
    parser.add_argument("--pseudo-dir", type=Path,
                        help="Completed pseudo depth. Omit to supervise on the raw "
                             "sparse lidar alone, which is the other arm worth having.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--hf-id", default="depth-anything/Depth-Anything-V2-Large-hf")
    parser.add_argument("--stretch", default="percentile", choices=("percentile", "minmax"),
                        help="16-bit thermal to 8-bit. percentile is what our own line "
                             "trains under and what the zero-shot DA2 numbers used, so "
                             "changing it here would compare two different inputs.")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--encoder-lr", type=float, default=5e-6,
                        help="Their own metric recipe: encoder at a tenth of the decoder.")
    parser.add_argument("--decoder-lr", type=float, default=5e-5)
    parser.add_argument("--ckpt-step", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="20 steps, then stop.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def thermal_to_rgb(path: Path, stretch: str) -> np.ndarray:
    """16-bit thermal to the 3-channel 8-bit-valued float DA2 expects."""
    raw = np.asarray(Image.open(path), dtype=np.float32)
    if stretch == "percentile":
        low, high = (float(v) for v in np.percentile(raw, (1.0, 99.0)))
    else:
        low, high = float(raw.min()), float(raw.max())
    unit = (np.zeros(raw.shape, np.float32) if high <= low
            else np.clip((raw - low) / (high - low), 0.0, 1.0).astype(np.float32))
    return np.repeat(unit[None], 3, axis=0)


class ThermalFrames(Dataset):
    """Thermal in, target disparity out, plus the mask the loss is taken over."""

    def __init__(self, rows: list[dict], args: argparse.Namespace):
        self.rows, self.args = rows, args

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row, args = self.rows[index], self.args
        image = thermal_to_rgb(args.ms2_root / row["thermal_path"], args.stretch)
        gt = np.asarray(Image.open(args.ms2_root / row["thermal_depth_path"]),
                        dtype=np.float32) / 256.0
        real = np.isfinite(gt) & (gt > D_MIN) & (gt < D_MAX)
        if args.pseudo_dir is not None:
            pseudo = np.load(args.pseudo_dir / f"{row['id']}.npy").astype(np.float32)
            # The same completed target our own line trains on: pseudo depth
            # everywhere, with the real returns written over it.
            dense = np.clip(np.where(real, gt, pseudo), D_MIN, D_MAX)
            valid = np.ones_like(dense, bool)
        else:
            dense, valid = np.clip(gt, D_MIN, D_MAX), real
        return {
            "image": torch.from_numpy(image),
            # DA2 emits affine-invariant inverse depth, so the target is disparity
            # and the fit happens in that space. Supervising it against depth would
            # be the wrong function family -- worth a factor of thirty here before.
            "disparity": torch.from_numpy(1.0 / np.maximum(dense, D_MIN)),
            "valid": torch.from_numpy(valid),
        }


def masked_ssi_l1(prediction: torch.Tensor, gt_disparity: torch.Tensor,
                  valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Scale-shift-invariant masked L1 in disparity space.

    Lifted from tools/train_ms2_joint_gt_v3.py deliberately rather than written
    afresh: the comparison is only worth making if the opponent is trained under
    the loss our own line is trained under, down to the detached affine fit.
    """
    mask = valid > 0.5
    count = int(mask.sum())
    if count < MIN_VALID_PIXELS:
        raise RuntimeError(f"GT valid pixels {count} below minimum {MIN_VALID_PIXELS}.")
    pred, gt = prediction[mask], gt_disparity[mask]
    with torch.no_grad():
        design = torch.stack([pred, torch.ones_like(pred)], dim=1)
        solution = torch.linalg.lstsq(design.float(), gt.float()[:, None]).solution.squeeze(1)
        scale, shift = solution[0], solution[1]
        if not bool(torch.isfinite(scale)) or not bool(torch.isfinite(shift)):
            raise RuntimeError("Non-finite scale/shift in GT alignment.")
    aligned = scale * pred + shift
    loss = (aligned - gt).abs().mean()
    with torch.no_grad():
        # The loss above is untouched; only this diagnostic differs from the
        # original. Inverting an aligned disparity that has landed near zero
        # sends one pixel to thousands of metres -- a frame like that reported
        # AbsRel 617 in a bench test -- and the running mean watched during
        # training would then say nothing about whether the model is learning.
        # The official protocol clamps to [min_depth, max_depth] after aligning,
        # so the same clamp applies here.
        gt_depth = (1.0 / gt.clamp_min(1e-6)).clamp(D_MIN, D_MAX)
        pred_depth = (1.0 / aligned.detach().clamp_min(1e-6)).clamp(D_MIN, D_MAX)
        abs_rel = ((pred_depth - gt_depth).abs() / gt_depth).mean()
    return loss, abs_rel, count


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    rows = []
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if args.pseudo_dir is not None and not (
                    args.pseudo_dir / f"{row['id']}.npy").is_file():
                continue
            rows.append(row)
    if not rows:
        raise SystemExit("清单里没有一帧能配上伪深度")
    target = "补全伪 GT（激光覆写）" if args.pseudo_dir else "稀疏激光"
    print(f"[data] {len(rows)} 帧   目标 = {target}   thermal = {args.stretch}", flush=True)

    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained(args.hf_id).to(args.device)
    model.train()

    # Their split of the learning rate, by parameter name. A single rate would
    # either move the pretrained encoder too fast or leave the head too slow.
    encoder, decoder = [], []
    for name, parameter in model.named_parameters():
        (encoder if "backbone" in name or "encoder" in name else decoder).append(parameter)
    print(f"[model] 编码器 {sum(p.numel() for p in encoder)/1e6:.1f} M @ lr {args.encoder_lr}   "
          f"解码器 {sum(p.numel() for p in decoder)/1e6:.1f} M @ lr {args.decoder_lr}", flush=True)
    optimiser = torch.optim.AdamW([
        {"params": encoder, "lr": args.encoder_lr},
        {"params": decoder, "lr": args.decoder_lr},
    ])

    shuffle_rng = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(ThermalFrames(rows, args), batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True, generator=shuffle_rng)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recent: collections.deque[float] = collections.deque(maxlen=200)
    step = 0
    while step < args.steps:
        for batch in loader:
            image = batch["image"].to(args.device)
            disparity = batch["disparity"].to(args.device)
            valid = batch["valid"].to(args.device)

            predicted = model(pixel_values=image).predicted_depth
            if predicted.shape[-2:] != disparity.shape[-2:]:
                predicted = F.interpolate(predicted[:, None], disparity.shape[-2:],
                                          mode="bilinear", align_corners=False)[:, 0]
            # Per frame, because the affine is per frame: pooling the batch would
            # fit one scale to several frames and score a different quantity.
            losses, errors = [], []
            for i in range(predicted.shape[0]):
                loss_i, abs_rel_i, _ = masked_ssi_l1(predicted[i], disparity[i], valid[i])
                losses.append(loss_i)
                errors.append(abs_rel_i)
            loss = torch.stack(losses).mean()

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            recent.append(float(torch.stack(errors).mean()))
            if step % 50 == 0 or (args.smoke and step % 5 == 0):
                print(f"  step {step:6d}  SSI-L1 {loss.item():.5f}  "
                      f"AbsRel {recent[-1]:.4f}  近 {len(recent)} 步均值 "
                      f"{sum(recent)/len(recent):.4f}", flush=True)
            step += 1

            if args.smoke and step >= 20:
                print("[smoke] 20 步完成，无 NaN")
                return
            if step % args.ckpt_step == 0 or step >= args.steps:
                path = args.out_dir / f"step{step}"
                model.save_pretrained(path)
                print(f"[ckpt] step {step} -> {path}", flush=True)
            if step >= args.steps:
                return


if __name__ == "__main__":
    main()
