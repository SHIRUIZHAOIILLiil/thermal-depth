"""Adapt Depth Anything V2 to MS2 thermal, under its own training recipe.

Zero-shot, DA2 scores AbsRel 0.153 on day against our 0.073 -- a two-fold lead
that says only that an RGB model has never seen thermal, which is the premise of
the field rather than a result. A table of unadapted comparators is a table
nobody has to believe. This trains one that has been adapted, on our data, so
the comparison has something at stake.

What is shared and what is theirs
---------------------------------
The *data* is ours and shared with every other arm: the official 8-sequence
split, the completed pseudo depth with lidar written over it, the same val rule
for picking a checkpoint. The *objective and the procedure* are theirs. Their
repository releases fine-tuning code for the metric variant only, and metric is
the wrong target for an affine-invariant table, so the relative recipe is
assembled here from what they published -- each default below cites where it
comes from:

  objective   L_ssi + L_gm at 1:2                          DA2 paper
              L_ssi normalises by median and mean absolute
              deviation, not by a least-squares fit        DAv1 paper, the loss DA2 reuses
              L_gm on that normalised residual, 4 scales   MiDaS, the source DA2 cites
              top 10% largest-loss pixels ignored for a
              pseudo-labelled sample                       DA2 paper
  procedure   shorter side to 518 (bicubic image, nearest
              target, multiple of 14, lower bound), then a
              random 518x518 crop on one box               their metric_depth/dataset
              random horizontal flip                       their metric_depth/train.py
              AdamW, wd 0.01, encoder 5e-6, decoder x10    their metric_depth/train.py
              poly decay (1 - it/total)^0.9, no warmup     their metric_depth/train.py
              no gradient clipping, fp32                   their metric_depth/train.py

What cannot be matched on one GPU is disclosed rather than approximated: their
student ran at batch 192 for 480K iterations. What is deliberately not adopted:
DAv1's strong colour perturbation and CutMix on the unlabelled stream, which the
DA2 paper does not say it kept; and their rule setting sky to disparity 0, which
is part of how a target is built, and the target is the shared part.

History
-------
The first arm trained on our line's loss -- least-squares SSI-L1, no gradient
matching, whole frames, no flip, constant rate, clipped gradients. That is still
reachable, bit for bit, as

    --ssi-norm lstsq --gm-weight 0 --trim-top 0 --crop 0 --no-flip \\
    --lr-schedule constant --grad-clip 1.0

and is kept as an ablation. It is not the arm to report: an opponent trained
under another model's objective is not the opponent.

The encoder is not frozen. Their own recipe fine-tunes it at a tenth of the
decoder's rate, and freezing it would leave an RGB-pretrained encoder looking at
thermal -- an opponent with one hand tied, whose defeat would prove nothing.

    python tools/finetune_depth_anything_thermal.py \\
        --manifest $SCRATCH/manifests/.../ms2_train_official8_thermalcap_....jsonl \\
        --ms2-root $SCRATCH/data/ms2 \\
        --pseudo-dir $SCRATCH/runs/pseudo_gt/official_train/calibrated_pseudo_depth \\
        --out-dir $SCRATCH/runs/da2_thermal/v2 --smoke
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
import time
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
# Their NormalizeImage, verbatim.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


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
                        help="Their metric_depth/train.py: encoder at the base rate.")
    parser.add_argument("--decoder-lr", type=float, default=5e-5,
                        help="Their metric_depth/train.py: decoder at ten times it.")

    # --- their objective ---------------------------------------------------
    parser.add_argument("--ssi-norm", default="median_mad", choices=("median_mad", "lstsq"),
                        help="median_mad is DAv1's affine-invariant loss, which DA2 "
                             "reuses: t = median, s = mean |d - t|, both maps "
                             "normalised before the difference. lstsq is the "
                             "least-squares fit our own line uses -- the first arm.")
    parser.add_argument("--gm-weight", type=float, default=2.0,
                        help="L_gm relative to L_ssi. DA2: 'The weight ratio of L_ssi "
                             "and L_gm is set as 1:2'. 0 drops the term.")
    parser.add_argument("--trim-top", type=float, default=0.10,
                        help="Fraction of largest-loss pixels ignored per sample. DA2: "
                             "'For each pseudo-labeled sample, we ignore its top-n "
                             "largest-loss regions during training, where n is set as "
                             "10%%.' The completed target is a pseudo label. 0 = off.")

    # --- their procedure ---------------------------------------------------
    parser.add_argument("--crop", type=int, default=518,
                        help="Their training resolution: shorter side resized to this, "
                             "then a random square crop of it. 0 keeps the whole frame "
                             "through DPTImageProcessor, which is how the first arm ran.")
    parser.add_argument("--no-flip", action="store_true",
                        help="Their train.py flips at random; this turns it off.")
    parser.add_argument("--lr-schedule", default="poly", choices=("poly", "constant"),
                        help="poly: lr * (1 - it/total)^0.9, no warmup, as in their "
                             "train.py. constant is the first arm.")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Their train.py clips nothing. The first arm clipped at 1.0.")

    parser.add_argument("--ckpt-step", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="40 steps, then stop and report the steady pace.")
    parser.add_argument("--expect-initial-absrel", nargs=2, type=float,
                        default=(0.08, 0.30), metavar=("LOW", "HIGH"),
                        help="Band the first step's AbsRel must fall in. Before any "
                             "training this is zero-shot DA2 on thermal, which "
                             "measured 0.153 / 0.165 / 0.172 on the three test "
                             "conditions, so a first step outside this band means "
                             "the thermal conversion or the loss space does not "
                             "match the run those numbers came from -- and the two "
                             "would then not be comparable, which is the only "
                             "reason to train this at all. Pass 0 0 to disable.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def thermal_to_uint8_rgb(path: Path, stretch: str) -> np.ndarray:
    """16-bit thermal to the 3-channel uint8 image DA2's pipeline starts from.

    uint8 HWC, not a float tensor: in the whole-frame path the model's own
    DPTImageProcessor takes it from here (resize to 518 at a multiple of 14,
    rescale, ImageNet normalisation), and in the crop path `their_train_view`
    reproduces their training transforms instead. Feeding the network a raw
    [0,1] tensor skips all of it -- which is how a first step once landed at
    0.3069 instead of the 0.15 the zero-shot numbers say.
    """
    raw = np.asarray(Image.open(path), dtype=np.float32)
    if stretch == "percentile":
        low, high = (float(v) for v in np.percentile(raw, (1.0, 99.0)))
    else:
        low, high = float(raw.min()), float(raw.max())
    eight = (np.zeros(raw.shape, np.uint8) if high <= low
             else np.clip((raw - low) / (high - low) * 255.0, 0, 255).round().astype(np.uint8))
    return np.repeat(eight[:, :, None], 3, axis=2)


def constrain_to_multiple_of(x: float, multiple: int = 14, min_val: int = 0,
                             max_val: int | None = None) -> int:
    """Their Resize.constrain_to_multiple_of, verbatim in behaviour.

    np.round is round-half-to-even, and it matters here: MS2's long side scales
    to exactly 1295 = 14 x 92.5, which their code rounds down to 1288.
    """
    y = int(np.round(x / multiple) * multiple)
    if max_val is not None and y > max_val:
        y = int(np.floor(x / multiple) * multiple)
    if y < min_val:
        y = int(np.ceil(x / multiple) * multiple)
    return y


def lower_bound_size(height: int, width: int, target: int) -> tuple[int, int]:
    """Their Resize with keep_aspect_ratio and resize_method='lower_bound': one
    scale for both sides, the larger of the two, so neither falls below target."""
    scale = max(target / height, target / width)
    return (constrain_to_multiple_of(scale * height, min_val=target),
            constrain_to_multiple_of(scale * width, min_val=target))


class ThermalFrames(Dataset):
    """Thermal in, target disparity out, plus the mask the loss is taken over."""

    def __init__(self, rows: list[dict], args: argparse.Namespace, processor):
        self.rows, self.args, self.processor = rows, args, processor

    def __len__(self) -> int:
        return len(self.rows)

    def _targets(self, row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        args = self.args
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
        # DA2 emits affine-invariant inverse depth, so the target is disparity
        # and the loss lives in that space. Supervising it against depth would
        # be the wrong function family -- worth a factor of thirty here before.
        disparity = (1.0 / np.maximum(dense, D_MIN)).astype(np.float32)
        # The lidar, kept separately. The loss trains against the completed map,
        # but the number watched during training is scored on the real returns
        # -- because that is what the zero-shot figures were scored on. The
        # completed map is itself 6.25 AbsRel away from the lidar.
        lidar_disparity = (1.0 / np.maximum(np.clip(gt, D_MIN, D_MAX), D_MIN)).astype(np.float32)
        return disparity, valid, lidar_disparity, real

    def their_train_view(self, image_u8: np.ndarray, *maps: np.ndarray):
        """Their training transforms, in their order: Resize (lower bound, bicubic
        image, nearest target and mask), NormalizeImage, PrepareForNet, Crop at a
        random position on one box for every map; then the random flip their
        train.py applies.

        Random numbers come from torch, which DataLoader reseeds per worker from
        the loader's own generator, so the crops are reproducible across runs
        and not duplicated across workers.
        """
        import cv2

        crop = self.args.crop
        height, width = image_u8.shape[:2]
        new_h, new_w = lower_bound_size(height, width, crop)
        image = cv2.resize(image_u8.astype(np.float32) / 255.0, (new_w, new_h),
                           interpolation=cv2.INTER_CUBIC)
        image = ((image - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
        resized = [cv2.resize(m.astype(np.float32), (new_w, new_h),
                              interpolation=cv2.INTER_NEAREST) for m in maps]

        top = int(torch.randint(0, new_h - crop + 1, (1,)))
        left = int(torch.randint(0, new_w - crop + 1, (1,)))
        image = image[:, top:top + crop, left:left + crop]
        resized = [m[top:top + crop, left:left + crop] for m in resized]

        if not self.args.no_flip and float(torch.rand(1)) < 0.5:
            image = image[:, :, ::-1]
            resized = [m[:, ::-1] for m in resized]
        return (np.ascontiguousarray(image, np.float32),
                *[np.ascontiguousarray(m) for m in resized])

    def __getitem__(self, index: int):
        row, args = self.rows[index], self.args
        image = thermal_to_uint8_rgb(args.ms2_root / row["thermal_path"], args.stretch)
        disparity, valid, lidar_disparity, lidar_valid = self._targets(row)

        if args.crop > 0:
            pixels, disparity, valid, lidar_disparity, lidar_valid = self.their_train_view(
                image, disparity, valid, lidar_disparity, lidar_valid)
            pixel_values = torch.from_numpy(pixels)
            valid, lidar_valid = valid > 0.5, lidar_valid > 0.5
        else:
            # The first arm: the whole frame through their inference processor,
            # targets at native resolution, no augmentation.
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values[0]

        return {
            # Carried so that a failure can name the frame it happened on. Without
            # it the only record of a crash is a step number, which a shuffled
            # loader does not map back to anything.
            "id": row["id"],
            "pixel_values": pixel_values,
            "disparity": torch.from_numpy(np.ascontiguousarray(disparity, np.float32)),
            "valid": torch.from_numpy(np.ascontiguousarray(valid, bool)),
            "lidar_disparity": torch.from_numpy(np.ascontiguousarray(lidar_disparity, np.float32)),
            "lidar_valid": torch.from_numpy(np.ascontiguousarray(lidar_valid, bool)),
        }


def masked_ssi_l1(prediction: torch.Tensor, gt_disparity: torch.Tensor,
                  valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Least-squares SSI-L1 in disparity space, and the AbsRel it implies.

    This is the evaluation protocol's alignment (`ssi_disparity`), so it is what
    the training-time diagnostic reports: the number watched while training has
    to be the number the table will print. It was also the first arm's training
    loss, which is why `da2_objective(..., ssi_norm="lstsq")` reproduces it.
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
        # Inverting an aligned disparity that has landed near zero sends one
        # pixel to thousands of metres -- a frame like that reported AbsRel 617
        # in a bench test. The official protocol clamps to [min_depth,
        # max_depth] after aligning, so the same clamp applies here.
        gt_depth = (1.0 / gt.clamp_min(1e-6)).clamp(D_MIN, D_MAX)
        pred_depth = (1.0 / aligned.detach().clamp_min(1e-6)).clamp(D_MIN, D_MAX)
        abs_rel = ((pred_depth - gt_depth).abs() / gt_depth).mean()
    return loss, abs_rel, count


def median_mad_normalise(d: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """DAv1's normalisation: t(d) = median(d), s(d) = mean |d - t(d)|, over the
    valid pixels, applied to the whole map. The prediction's t and s are not
    detached -- they are part of the loss as written, not a fit to it."""
    values = d[mask]
    t = values.median()
    s = (values - t).abs().mean().clamp_min(eps)
    return (d - t) / s


def gradient_matching(residual: torch.Tensor, mask: torch.Tensor,
                      scales: int = 4) -> torch.Tensor:
    """MiDaS's multi-scale gradient matching term, on the normalised residual.

    Following the MiDaS reference: four scales by stride-2 subsampling, the
    absolute first differences summed over both axes and all scales, divided by
    the full-resolution count. A difference is counted only where both of its
    pixels are kept, so no gradient is taken across the edge of the mask --
    there the residual jumps for want of a target rather than for want of
    sharpness.
    """
    total = residual.new_zeros(())
    denominator = mask.sum().clamp_min(1).to(residual.dtype)
    for level in range(scales):
        stride = 2 ** level
        r, m = residual[::stride, ::stride], mask[::stride, ::stride]
        if r.shape[0] < 2 or r.shape[1] < 2:
            break
        dx = (r[:, 1:] - r[:, :-1]).abs() * (m[:, 1:] & m[:, :-1])
        dy = (r[1:, :] - r[:-1, :]).abs() * (m[1:, :] & m[:-1, :])
        total = total + dx.sum() + dy.sum()
    return total / denominator


def da2_objective(prediction: torch.Tensor, gt_disparity: torch.Tensor,
                  valid: torch.Tensor, *, ssi_norm: str, gm_weight: float,
                  trim_top: float, scales: int = 4
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`L_ssi + gm_weight * L_gm` for one frame, with the largest losses ignored.

    Normalisation is taken over every valid pixel; the trim then drops the
    largest per-pixel SSI residuals from both terms. That order is our reading
    of "ignore its top-n largest-loss regions during training" -- their
    relative-depth training code is not released, so the sentence is all there
    is -- and it keeps an outlier from choosing the scale that decides which
    pixels are outliers.

    With ssi_norm="lstsq", gm_weight=0 and trim_top=0 this is masked_ssi_l1's
    loss, term for term.
    """
    mask = valid > 0.5
    count = int(mask.sum())
    if count < MIN_VALID_PIXELS:
        raise RuntimeError(f"GT valid pixels {count} below minimum {MIN_VALID_PIXELS}.")

    if ssi_norm == "median_mad":
        with torch.no_grad():
            gt_hat = median_mad_normalise(gt_disparity, mask)
        residual = median_mad_normalise(prediction, mask) - gt_hat
    else:
        with torch.no_grad():
            pred, gt = prediction[mask], gt_disparity[mask]
            design = torch.stack([pred, torch.ones_like(pred)], dim=1)
            solution = torch.linalg.lstsq(design.float(), gt.float()[:, None]).solution.squeeze(1)
            scale, shift = solution[0], solution[1]
            if not bool(torch.isfinite(scale)) or not bool(torch.isfinite(shift)):
                raise RuntimeError("Non-finite scale/shift in GT alignment.")
        residual = (scale * prediction + shift) - gt_disparity

    per_pixel = residual.abs()
    keep = mask
    if trim_top > 0:
        with torch.no_grad():
            threshold = torch.quantile(per_pixel[mask].float(), 1.0 - trim_top)
        keep = mask & (per_pixel <= threshold)

    ssi = per_pixel[keep].mean()
    if gm_weight == 0.0:
        return ssi, ssi, residual.new_zeros(())
    gm = gradient_matching(residual, keep, scales)
    return ssi + gm_weight * gm, ssi, gm


def quiet_worker(_worker_id: int) -> None:
    """One thread per DataLoader worker, for OpenCV and for torch.

    Each worker otherwise opens its own OpenCV pool and its own intra-op torch
    pool, sized to the machine, so six workers on an eight-core allocation run
    dozens of threads against eight cores. On Aire that showed up as roughly
    five seconds a step for a batch the GPU clears in well under one -- the
    first v2 attempt reached step 100 after ten minutes, which at 20,000 steps
    is a day beyond its wall clock. Parallelism comes from the workers; inside
    each, one thread. This changes nothing about what a sample contains.
    """
    import cv2

    cv2.setNumThreads(1)
    torch.set_num_threads(1)


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
    print(f"[recipe] L_ssi({args.ssi_norm}) + {args.gm_weight:g} x L_gm   "
          f"trim top {args.trim_top:.0%}   crop {args.crop or 'whole frame'}   "
          f"flip {'off' if args.no_flip else 'on'}   lr {args.lr_schedule}   "
          f"clip {args.grad_clip or 'off'}", flush=True)

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained(args.hf_id)
    model = AutoModelForDepthEstimation.from_pretrained(args.hf_id).to(args.device)
    model.train()

    # Their split of the learning rate, by parameter name. A single rate would
    # either move the pretrained encoder too fast or leave the head too slow.
    #
    # DepthAnythingForDepthEstimation assigns exactly three submodules --
    # backbone, neck, head -- so the prefix is enough and a substring test is
    # not: the backbone is a DINOv2 whose own layers are named
    # backbone.encoder.layer.N, and matching "encoder" anywhere would keep
    # working by luck until something in the neck was named that way too.
    encoder, decoder = [], []
    for name, parameter in model.named_parameters():
        (encoder if name.startswith("backbone.") else decoder).append(parameter)
    encoder_size = sum(p.numel() for p in encoder)
    decoder_size = sum(p.numel() for p in decoder)
    print(f"[model] 编码器 {encoder_size/1e6:.1f} M @ lr {args.encoder_lr}   "
          f"解码器 {decoder_size/1e6:.1f} M @ lr {args.decoder_lr}", flush=True)
    # A silently empty group would train half the network at the wrong rate and
    # still converge to something reportable.
    if not encoder or not decoder:
        raise SystemExit(
            f"参数分组失败：编码器 {len(encoder)} 个张量，解码器 {len(decoder)} 个。"
            f"顶层模块是 {sorted({n.split('.')[0] for n, _ in model.named_parameters()})}")
    if encoder_size < decoder_size:
        raise SystemExit(
            f"编码器({encoder_size/1e6:.1f} M)不该小于解码器({decoder_size/1e6:.1f} M)；"
            "前缀多半对不上了")
    # Their train.py: AdamW, betas (0.9, 0.999), weight_decay 0.01 -- which are
    # also torch's defaults, so the first arm already matched here.
    optimiser = torch.optim.AdamW([
        {"params": encoder, "lr": args.encoder_lr},
        {"params": decoder, "lr": args.decoder_lr},
    ], betas=(0.9, 0.999), weight_decay=0.01)
    base_lrs = [group["lr"] for group in optimiser.param_groups]

    shuffle_rng = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(ThermalFrames(rows, args, processor), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers, drop_last=True,
                        generator=shuffle_rng, worker_init_fn=quiet_worker)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recent: collections.deque[float] = collections.deque(maxlen=200)
    step = 0
    # Seconds per step since the previous report, so a smoke run measures what
    # the wall clock has to cover instead of leaving it to a guess.
    clock, clock_step = time.time(), 0
    while step < args.steps:
        for batch in loader:
            if args.lr_schedule == "poly":
                # Their train.py sets the rate before each step from the step
                # about to be taken, so step 0 runs at the base rate.
                factor = (1.0 - step / args.steps) ** 0.9
                for group, base in zip(optimiser.param_groups, base_lrs):
                    group["lr"] = base * factor

            pixel_values = batch["pixel_values"].to(args.device)
            disparity = batch["disparity"].to(args.device)
            valid = batch["valid"].to(args.device)
            lidar_disparity = batch["lidar_disparity"].to(args.device)
            lidar_valid = batch["lidar_valid"].to(args.device)

            predicted = model(pixel_values=pixel_values).predicted_depth
            if predicted.shape[-2:] != disparity.shape[-2:]:
                predicted = F.interpolate(predicted[:, None], disparity.shape[-2:],
                                          mode="bilinear", align_corners=False)[:, 0]
            # Per frame, because the normalisation is per frame: pooling the
            # batch would fit one scale to several frames and score a different
            # quantity.
            ids = batch["id"]
            losses, ssi_terms, gm_terms, errors = [], [], [], []
            for i in range(predicted.shape[0]):
                loss_i, ssi_i, gm_i = da2_objective(
                    predicted[i], disparity[i], valid[i], ssi_norm=args.ssi_norm,
                    gm_weight=args.gm_weight, trim_top=args.trim_top)
                losses.append(loss_i)
                ssi_terms.append(ssi_i.detach())
                gm_terms.append(gm_i.detach())
                # Scored on the lidar, under the evaluation protocol's own
                # alignment, which is the reference the zero-shot numbers used.
                #
                # This is monitoring, not the objective, so it must not be able
                # to end the run. The v2 recipe's first attempt died at 22
                # minutes with no checkpoint and a log nobody could read, and
                # this call raises on two conditions a random crop can meet (too
                # few returns in the box, or a degenerate fit). A sample it cannot
                # score is named and skipped; a non-finite *loss* is caught
                # separately below and does stop the run.
                with torch.no_grad():
                    try:
                        _, abs_rel_i, _ = masked_ssi_l1(
                            predicted[i].detach(), lidar_disparity[i], lidar_valid[i])
                        errors.append(abs_rel_i)
                    except RuntimeError as err:
                        p = predicted[i].detach()
                        print(f"  [monitor] step {step} frame {ids[i]}: {err}   "
                              f"lidar px {int(lidar_valid[i].sum())}   pred finite "
                              f"{bool(torch.isfinite(p).all())} min {float(p.min()):.3g} "
                              f"max {float(p.max()):.3g} -- not scored this step", flush=True)
            loss = torch.stack(losses).mean()

            def dump_and_stop(what: str) -> None:
                print(f"\n⛔ step {step}: {what}", flush=True)
                for i in range(predicted.shape[0]):
                    p = predicted[i].detach()
                    print(f"   {ids[i]}  loss {float(losses[i]):.4g}  SSI {float(ssi_terms[i]):.4g}  "
                          f"GM {float(gm_terms[i]):.4g}  pred finite {bool(torch.isfinite(p).all())} "
                          f"min {float(p.min()):.3g} max {float(p.max()):.3g}  "
                          f"target finite {bool(torch.isfinite(disparity[i]).all())} "
                          f"max {float(disparity[i].max()):.3g}", flush=True)
                raise SystemExit(f"stopped at step {step}: {what}; weights left as of step {step - 1}")

            # Checked before the update rather than after, so that whatever went
            # wrong is reported against the frames that caused it and never
            # written into the weights.
            if not bool(torch.isfinite(loss)):
                dump_and_stop(f"non-finite loss {float(loss)}")
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            # max_norm=inf measures without clipping, so the recipe's "no
            # clipping" is unchanged; --grad-clip still applies its own bound.
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf")))
            if not np.isfinite(grad_norm):
                dump_and_stop(f"non-finite gradient norm {grad_norm}")
            optimiser.step()

            if errors:
                recent.append(float(torch.stack(errors).mean()))
            low, high = args.expect_initial_absrel
            if step == 0 and high > 0 and not recent:
                raise SystemExit("⛔ 第一步没有一个样本能在激光上打分，门禁无从判断。")
            if step == 0 and high > 0 and not low <= recent[0] <= high:
                raise SystemExit(
                    f"⛔ 第一步 AbsRel {recent[0]:.4f} 不在 [{low}, {high}] 内。\n"
                    "   未经训练时这就是 DA2 在热像上的零样本成绩（实测 0.153 / "
                    "0.165 / 0.172）。落在band外说明热像转换或损失空间与那次评估"
                    "不一致，训出来的数和零样本那一行不可比 —— 而可比正是做这条"
                    "线的唯一理由。先查，别训。")
            if step % 50 == 0 or (args.smoke and step % 5 == 0):
                absrel = (f"AbsRel {recent[-1]:.4f}  近 {len(recent)} 步均值 "
                          f"{sum(recent)/len(recent):.4f}" if recent else "AbsRel —")
                now = time.time()
                pace = (now - clock) / max(step - clock_step, 1)
                clock, clock_step = now, step
                print(f"  step {step:6d}  loss {loss.item():.5f}  "
                      f"(SSI {torch.stack(ssi_terms).mean().item():.5f} + "
                      f"{args.gm_weight:g}x GM {torch.stack(gm_terms).mean().item():.5f})  "
                      f"|grad| {grad_norm:.3g}  lr_enc {optimiser.param_groups[0]['lr']:.2e}  "
                      f"{absrel}  {pace:.2f} s/step", flush=True)
            step += 1
            if step == 10:
                # Worker start-up and the first CUDA kernels land in the first
                # few steps; the rate that decides the wall clock is after them.
                steady_from = (time.time(), step)

            if args.smoke and step >= 40:
                per_step = (time.time() - steady_from[0]) / (step - steady_from[1])
                hours = per_step * args.steps / 3600
                print(f"[smoke] 40 步完成，无 NaN。稳态 {per_step:.2f} s/step → "
                      f"{args.steps} 步约 {hours:.1f} 小时（不含存盘）")
                return
            if step % args.ckpt_step == 0 or step >= args.steps:
                path = args.out_dir / f"step{step}"
                model.save_pretrained(path)
                print(f"[ckpt] step {step} -> {path}", flush=True)
            if step >= args.steps:
                return


if __name__ == "__main__":
    main()
