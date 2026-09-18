"""Recover metres per pixel, from the thermal frame and its caption.

Our checkpoints emit y in [0,1], affine in log depth between that frame's own
2nd and 98th percentiles. Those two numbers are the scale, and they come from
the frame's GT, so at test time they are gone: every number this project reports
gets them back by fitting two parameters against test GT. That is why the model
cannot enter a table beside NeWCRF, which predicts metres outright -- the
obstacle is the output space, not the alignment (METRIC_ADAPTATION section 0).

This head predicts them instead, per pixel, from inputs available at test time:

    log D = A(thermal, caption) * y + B(thermal, caption)

A and B start as two scalars fitted on the training split, so at step 0 this is
exactly the global metric affine -- the two-frozen-constants design the ceiling
analysis judged too rigid for the depth range. The heads add zero-init per-pixel
residuals on top, so whatever the global constants already buy is kept and the
network only has to learn where they are wrong.

The shape of the recovery follows our target, not TR2M's. TR2M writes
D = 1/(A*Dr + B) because its relative depth is disparity-like; ours is affine in
log depth, so the matching form is the exponential above. Borrowing theirs would
fit the wrong function family, which in this project has cost a factor of thirty
(tests/test_log_truncnorm_roundtrip.py).

Supervision is the real lidar in metres, never the completed pseudo depth: that
map is itself scale-free, calibrated by a fit, so training metres against it
would be circular.

    python tools/train_metric_rescale_head.py \
        --manifest $SCRATCH/manifests/.../ms2_train_official8_thermalcap_....jsonl \
        --ms2-root $SCRATCH/data/ms2 \
        --relative-dir $SCRATCH/runs/eval/<train export>/raw_predictions \
        --out-dir $SCRATCH/runs/metric_head/v1 --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

D_MIN, D_MAX = 1e-3, 80.0
TRUNC_LO, TRUNC_HI = 0.02, 0.98


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ms2-root", required=True, type=Path)
    parser.add_argument("--relative-dir", required=True, type=Path,
                        help="raw_predictions/<id>.npy from --save-raw-pred on this manifest.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sd2-repo", default="Manojb/stable-diffusion-2-1-base",
                        help="Supplies the frozen CLIP text encoder -- the same one "
                             "the backbone was conditioned with, so a caption is "
                             "embedded here exactly as it was during training.")
    parser.add_argument("--no-captions", action="store_true",
                        help="Ablation arm: embed an empty string for every frame. "
                             "The only honest way to price the text branch.")
    parser.add_argument("--global-only", action="store_true",
                        help="Ablation arm: drop the per-pixel residuals and fit "
                             "the two scalars alone. That is the design the metric "
                             "adaptation line already had, so it is the number the "
                             "per-pixel maps have to beat to justify existing. "
                             "Text cannot reach two scalars, so this arm ignores "
                             "--no-captions: there is one global arm, not two.")
    parser.add_argument("--stretch", default="percentile", choices=("percentile", "minmax"),
                        help="Must match the arm that produced --relative-dir.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=96, help="Channels in the image trunk.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="20 steps, then stop.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def thermal_to_unit(path: Path, stretch: str) -> np.ndarray:
    raw = np.asarray(Image.open(path), dtype=np.float32)
    if stretch == "percentile":
        low, high = (float(v) for v in np.percentile(raw, (1.0, 99.0)))
    else:
        low, high = float(raw.min()), float(raw.max())
    if high <= low:
        return np.zeros(raw.shape, np.float32)
    return np.clip((raw - low) / (high - low), 0.0, 1.0).astype(np.float32)


class Frames(Dataset):
    def __init__(self, rows: list[dict], args: argparse.Namespace):
        self.rows, self.args = rows, args

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row, args = self.rows[index], self.args
        thermal = thermal_to_unit(args.ms2_root / row["thermal_path"], args.stretch)
        relative = np.load(args.relative_dir / f"{row['id']}.npy").astype(np.float32)
        gt = np.asarray(Image.open(args.ms2_root / row["thermal_depth_path"]),
                        dtype=np.float32) / 256.0
        if relative.shape != gt.shape:
            relative = F.interpolate(torch.from_numpy(relative)[None, None], gt.shape,
                                     mode="bilinear", align_corners=False)[0, 0].numpy()
        valid = np.isfinite(gt) & (gt > D_MIN) & (gt < D_MAX)
        return {
            "thermal": torch.from_numpy(thermal)[None],
            "relative": torch.from_numpy(relative)[None],
            "gt": torch.from_numpy(gt)[None],
            "valid": torch.from_numpy(valid)[None],
            "caption": "" if args.no_captions else str(row.get("caption") or ""),
        }


def collate(batch):
    out = {key: torch.stack([item[key] for item in batch])
           for key in ("thermal", "relative", "gt", "valid")}
    out["caption"] = [item["caption"] for item in batch]
    return out


class RescaleHead(nn.Module):
    """Per-pixel (A, B) for log D = A * y + B, from the frame and its caption.

    The text vector is not concatenated once at a bottleneck. A caption that only
    shifted a global bias could be replaced by a learned constant, and the
    no-caption arm would then differ from this one by a single number -- an
    ablation that cannot fail informatively. It enters as cross-attention over
    the image tokens instead, so it is able to say *where*, and removing it
    removes an expressible degree of freedom rather than one bias.
    """

    def __init__(self, width: int, text_dim: int, a_init: float, b_init: float,
                 global_only: bool = False):
        super().__init__()
        self.global_only = global_only
        self.a_global = nn.Parameter(torch.tensor(float(a_init)))
        self.b_global = nn.Parameter(torch.tensor(float(b_init)))
        if global_only:
            # Two scalars and nothing else: the metric adaptation design, fitted
            # rather than frozen. Built with no trunk at all rather than a trunk
            # whose output is discarded, so the arm cannot quietly carry the
            # optimiser state or the parameter count of the thing it controls for.
            return
        self.stem = nn.Sequential(
            nn.Conv2d(2, width, 5, stride=2, padding=2), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GroupNorm(8, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU(),
        )
        self.text_proj = nn.Linear(text_dim, width)
        self.attend = nn.MultiheadAttention(width, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.mix = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.GELU())
        # Zero-init, so the whole network starts as the two fitted scalars: the
        # global metric affine, which is the thing this is meant to beat. A run
        # that never improves is then a clean negative, not a bad initialisation.
        self.to_ab = nn.Conv2d(width, 2, 3, padding=1)
        nn.init.zeros_(self.to_ab.weight)
        nn.init.zeros_(self.to_ab.bias)

    def forward(self, thermal: torch.Tensor, relative: torch.Tensor,
                text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, _, h, w = thermal.shape
        if self.global_only:
            ones = torch.ones((n, 1, h, w), device=thermal.device, dtype=thermal.dtype)
            return self.a_global * ones, self.b_global * ones
        features = self.stem(torch.cat([thermal, relative], dim=1))
        _, channels, fh, fw = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        context = self.text_proj(text)
        attended, _ = self.attend(self.norm(tokens), context, context, need_weights=False)
        features = self.mix((tokens + attended).transpose(1, 2).reshape(n, channels, fh, fw))
        delta = F.interpolate(self.to_ab(features), (h, w), mode="bilinear", align_corners=False)
        return self.a_global + delta[:, :1], self.b_global + delta[:, 1:]


def fit_global_log_affine(rows: list[dict], args: argparse.Namespace,
                          sample: int = 400) -> tuple[float, float]:
    """Mean per-frame (hi - lo, lo) of log depth, from TRAIN frames and nothing else.

    These are the constants the head is initialised to, so step 0 reproduces the
    global affine rather than a random guess.
    """
    picks = random.Random(args.seed).sample(rows, min(sample, len(rows)))
    spans, lows = [], []
    for row in picks:
        gt = np.asarray(Image.open(args.ms2_root / row["thermal_depth_path"]),
                        dtype=np.float32) / 256.0
        valid = np.isfinite(gt) & (gt > D_MIN) & (gt < D_MAX)
        if valid.sum() < 100:
            continue
        log_depth = np.log(gt[valid])
        lo = float(np.quantile(log_depth, TRUNC_LO))
        hi = float(np.quantile(log_depth, TRUNC_HI))
        spans.append(hi - lo)
        lows.append(lo)
    if not spans:
        raise SystemExit("没有一帧有足够的激光点来拟合全局仿射")
    return float(np.mean(spans)), float(np.mean(lows))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    rows = []
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if (args.relative_dir / f"{row['id']}.npy").is_file():
                    rows.append(row)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"没有一帧在 {args.relative_dir} 里找到对应的相对深度")
    print(f"[data] {len(rows)} 帧同时有相对深度与激光", flush=True)

    span, low = fit_global_log_affine(rows, args)
    print(f"[init] 全局 log 仿射 A={span:.4f} B={low:.4f}  "
          f"(深度 {math.exp(low):.2f}-{math.exp(low + span):.2f} m)", flush=True)

    from transformers import CLIPTextModel, CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(args.sd2_repo, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.sd2_repo, subfolder="text_encoder").to(args.device).eval()
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)

    model = RescaleHead(args.width, text_encoder.config.hidden_size, span, low,
                        global_only=args.global_only).to(args.device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    arm = "global-only" if args.global_only else (
        "per-pixel, caption off" if args.no_captions else "per-pixel, caption on")
    print(f"[model] 臂 = {arm}   可训参数 {trainable / 1e6:.3f} M", flush=True)
    if args.global_only and not args.no_captions:
        print("[note] global-only 下文本到不了两个标量，这一臂与 --no-captions 等价",
              flush=True)

    loader = DataLoader(Frames(rows, args), batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, collate_fn=collate, drop_last=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            thermal = batch["thermal"].to(args.device)
            relative = batch["relative"].to(args.device)
            gt = batch["gt"].to(args.device)
            valid = batch["valid"].to(args.device)
            with torch.no_grad():
                tokens = tokenizer(batch["caption"], padding="max_length", truncation=True,
                                   max_length=tokenizer.model_max_length, return_tensors="pt")
                text = text_encoder(tokens.input_ids.to(args.device), return_dict=False)[0]

            a, b = model(thermal, relative, text)
            # Clamped before the exponential. An early step can put the exponent
            # anywhere, and inf inside a masked mean is nan, which kills the run
            # rather than costing it a step.
            predicted = torch.exp((a * relative + b).clamp(-9.0, 9.0))
            # Metres, on the real returns only: the completed pseudo map is
            # scale-free by construction, so supervising metres against it would
            # be circular. About 26% of pixels carry a genuine return, and those
            # are the only ones that know how far anything actually is.
            count = valid.sum().clamp(min=1)
            loss = ((predicted - gt).abs() * valid).sum() / count

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            if step % 50 == 0:
                with torch.no_grad():
                    relative_error = (((predicted - gt).abs()
                                       / gt.clamp(min=D_MIN)) * valid).sum() / count
                print(f"  epoch {epoch} step {step:6d}  L1 {loss.item():.4f} m  "
                      f"AbsRel {relative_error.item():.4f}", flush=True)
            step += 1
            if args.smoke and step >= 20:
                torch.save({"model": model.state_dict(), "a_init": span, "b_init": low},
                           args.out_dir / "head_smoke.pt")
                print("[smoke] 20 步完成，无 NaN")
                return

        torch.save({"model": model.state_dict(), "a_init": span, "b_init": low,
                    "epoch": epoch, "no_captions": args.no_captions},
                   args.out_dir / f"head_epoch{epoch}.pt")
        print(f"[ckpt] epoch {epoch} -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
