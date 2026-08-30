"""How much of a latent target MS2's LiDAR can actually supervise.

The autoencoder reduces the image eightfold in each dimension, so one latent cell
covers an $8\\times8$ image region, and the Iris/Lotus trainer keeps a cell only when
every pixel under it carries a value (`train_iris_ms2_g.py`, latent mask construction:
invalidity is max-pooled over non-overlapping 8x8 cells and any cell containing one
invalid pixel is dropped). Pixel coverage therefore understates the problem badly --
a quarter of the pixels does not buy a quarter of the cells.

This measures both, so the method section can say why the raw supervision cannot be
used as a latent target instead of asserting it.

    python tools/latent_cell_coverage.py --ms2-root E:/dataset/ms2 --frames 40
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from PIL import Image

D_MIN, D_MAX, DEPTH_SCALE, CELL = 1e-3, 80.0, 256.0, 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ms2-root", required=True)
    parser.add_argument("--sequences", nargs="*", default=None,
                        help="Directory names under proj_depth/. Default: all present.")
    parser.add_argument("--frames", type=int, default=40,
                        help="Evenly spaced frames per sequence.")
    parser.add_argument("--gt-variant", default="depth_filtered",
                        help="Must match what training reads.")
    return parser.parse_args()


def measure(path: str) -> tuple[float, float]:
    depth = np.asarray(Image.open(path), dtype=np.float32) / DEPTH_SCALE
    valid = np.isfinite(depth) & (depth > D_MIN) & (depth < D_MAX)
    h, w = valid.shape
    H, W = h // CELL * CELL, w // CELL * CELL
    cells = valid[:H, :W].reshape(H // CELL, CELL, W // CELL, CELL)
    return float(valid.mean()), float(cells.all(axis=(1, 3)).mean())


def main() -> int:
    args = parse_args()
    root = os.path.join(args.ms2_root, "proj_depth")
    names = args.sequences or sorted(
        d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
    )
    print(f"{'sequence':26s} {'frames':>6s} {'pixel cov':>10s} {'full cells':>11s}")
    weighted_pixel = weighted_cell = counted = 0.0
    for name in names:
        files = sorted(glob.glob(os.path.join(root, name, "thr", args.gt_variant, "*.png")))
        if not files:
            print(f"{name:26s}  (no {args.gt_variant} frames)")
            continue
        step = max(1, len(files) // args.frames)
        picked = files[::step][:args.frames]
        pixel, cell = zip(*(measure(f) for f in picked))
        print(f"{name:26s} {len(picked):6d} {100*np.mean(pixel):9.2f}% {100*np.mean(cell):10.3f}%")
        weighted_pixel += np.mean(pixel) * len(picked)
        weighted_cell += np.mean(cell) * len(picked)
        counted += len(picked)
    if not counted:
        raise SystemExit("no frames measured")
    print(f"\n{'pooled':26s} {int(counted):6d} {100*weighted_pixel/counted:9.2f}% "
          f"{100*weighted_cell/counted:10.3f}%")
    print("\nPixel coverage is not the constraint; the fully covered cell fraction is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
