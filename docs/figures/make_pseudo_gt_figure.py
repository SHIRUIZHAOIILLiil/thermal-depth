"""What the completed target looks like next to the LiDAR it was built from.

The method section says the sparse thermal-view LiDAR cannot be used as a latent
target and that a calibrated, completed pseudo-depth map is used instead. This renders
the two side by side so the claim is visible rather than asserted.

Colouring is Iris's own `colorize_depth_map` -- the function `lotus/infer.py` calls --
on disparity with `reverse_color=True`, so these panels are directly comparable with
the ones `tools/export_qualitative.py` produces. `dilate` and `colour_shared` are lifted
from that file verbatim; if the convention changes there it must change here too.

Frames default to the three whose captions appear in the supplementary, so a reader can
match a sentence to the supervision that accompanied it.

    python docs/figures/make_pseudo_gt_figure.py
    python docs/figures/make_pseudo_gt_figure.py --output-dir "$PAPER/figures"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "lotus"))

from utils.image_utils import colorize_depth_map  # noqa: E402  (Iris official)

D_MIN, D_MAX, DEPTH_SCALE = 1e-3, 80.0, 256.0
SEQUENCE = "_2021-08-06-10-59-33"
FRAMES = ("000000", "000099", "000198")
LABEL_H = 34


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ms2-root", type=Path, default=Path("E:/dataset/ms2"))
    parser.add_argument("--pseudo-dir", type=Path, default=ROOT / "pseudo_gt_samples")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs" / "figures")
    parser.add_argument("--sequence", default=SEQUENCE)
    parser.add_argument("--frames", nargs="*", default=list(FRAMES))
    parser.add_argument("--gt-dilate", type=int, default=3,
                        help="Raw points are invisible at print size; 0 disables.")
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--panel-labels", choices=("on", "off"), default="off",
                        help="Labels inside panels are for slides. At print size they "
                             "are illegible and cost height, so the paper figure names "
                             "its columns in the caption instead.")
    parser.add_argument("--separate-panels", action="store_true",
                        help="Also write each panel as its own file. The pipeline "
                             "schematic places them individually as TikZ nodes, and a "
                             "composed strip cannot be taken apart again without "
                             "re-deriving the colour range.")
    return parser.parse_args()


def dilate(values: np.ndarray, valid: np.ndarray, window: int):
    """Verbatim from tools/export_qualitative.py."""
    if window <= 1:
        return values, valid
    filled = torch.from_numpy(np.where(valid, values, -np.inf))[None, None]
    pooled = F.max_pool2d(filled, window, stride=1, padding=window // 2)[0, 0].numpy()
    mask = np.isfinite(pooled)
    return np.where(mask, pooled, 0.0), mask


def colour_shared(disparity: np.ndarray, lo: float, hi: float, mask=None) -> Image.Image:
    """Verbatim from tools/export_qualitative.py.

    The official helper always min-max normalises what it is handed, which is what
    makes two panels incomparable. Clipping to a shared range and appending a
    one-pixel sentinel row carrying lo and hi makes it normalise against those
    bounds instead; the row is then cropped away, so every displayed pixel is real.
    """
    clipped = np.clip(np.asarray(disparity, np.float32), lo, hi)
    sentinel = np.full((1, clipped.shape[1]), lo, np.float32)
    sentinel[0, 0] = hi
    padded = np.vstack([clipped, sentinel])
    if mask is not None:
        mask = np.vstack([mask, np.zeros((1, mask.shape[1]), bool)])
        image = colorize_depth_map(padded, mask=torch.from_numpy(mask), reverse_color=True)
    else:
        image = colorize_depth_map(padded, reverse_color=True)
    return image.crop((0, 0, image.width, image.height - 1))


def strip(panels: list[tuple[str, Image.Image]], width: int, labels: bool) -> Image.Image:
    scaled = []
    for title, panel in panels:
        height = round(panel.height * width / panel.width)
        scaled.append((title, panel.resize((width, height), Image.LANCZOS)))
    label_h = LABEL_H if labels else 0
    row_h = max(p.height for _, p in scaled) + label_h
    out = Image.new("RGB", (width * len(scaled), row_h), "white")
    draw = ImageDraw.Draw(out)
    for column, (title, panel) in enumerate(scaled):
        x = column * width
        if labels:
            draw.text((x + 6, 10), title, fill="black")
        out.paste(panel, (x, label_h))
    return out


def main() -> int:
    args = parse_args()
    rows, fractions = [], []
    for frame in args.frames:
        # MS2's thermal frames are 16-bit; converting them straight to RGB clips
        # every value to white. Percentile stretch, same as export_qualitative.py.
        thermal = np.asarray(Image.open(
            args.ms2_root / "sync_data" / args.sequence / "thr" / "img_left" / f"{frame}.png"
        )).astype(np.float32)
        t_lo, t_hi = np.percentile(thermal, [1, 99])
        grey = Image.fromarray(
            (np.clip((thermal - t_lo) / max(t_hi - t_lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        ).convert("RGB")

        sparse = np.asarray(Image.open(
            args.ms2_root / "proj_depth" / args.sequence / "thr" / "depth_filtered" / f"{frame}.png"
        ), np.float32) / DEPTH_SCALE
        valid = np.isfinite(sparse) & (sparse > D_MIN) & (sparse < D_MAX)

        pseudo = np.load(args.pseudo_dir / f"{args.sequence.lstrip('_')}_{frame}.npy").astype(np.float32)
        if pseudo.shape != sparse.shape:
            raise SystemExit(f"{frame}: pseudo {pseudo.shape} != LiDAR {sparse.shape}")

        # Iris colours disparity, so the shared range lives in disparity. The 2%/98%
        # bounds are the same ones `trunc_disparity` uses to normalise the target, so
        # the panel shows the range training actually sees.
        pseudo_disparity = 1.0 / np.clip(pseudo, D_MIN, D_MAX)
        lo, hi = np.percentile(pseudo_disparity, [2, 98])

        sparse_disparity = np.zeros_like(sparse)
        sparse_disparity[valid] = 1.0 / sparse[valid]
        shown, mask = dilate(sparse_disparity, valid, args.gt_dilate)

        fractions.append((frame, float(valid.mean())))
        panels = [
            ("Thermal input", grey),
            (f"Projected LiDAR  ({valid.mean():.0%} valid, dilated)",
             colour_shared(shown, lo, hi, mask)),
            ("Completed target", colour_shared(pseudo_disparity, lo, hi)),
        ]
        if args.separate_panels:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            for slug, (_, panel) in zip(("thermal", "lidar", "completed"), panels):
                height = round(panel.height * args.panel_width / panel.width)
                path = args.output_dir / f"panel_{slug}_{frame}.png"
                panel.resize((args.panel_width, height), Image.LANCZOS).save(path, optimize=True)
                print(f"  panel -> {path}")
        rows.append(strip(panels, args.panel_width, args.panel_labels == "on"))

    figure = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
    y = 0
    for row in rows:
        figure.paste(row, (0, y))
        y += row.height

    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "pseudo_gt_strip.png"
    figure.save(destination, optimize=True)
    print(f"{figure.size}  {destination.stat().st_size / 1e6:.2f} MB  ->  {destination}")
    # With labels off these belong in the caption; print them so they are not retyped.
    print("valid fraction, top to bottom: "
          + ", ".join(f"{frame} {share:.0%}" for frame, share in fractions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
