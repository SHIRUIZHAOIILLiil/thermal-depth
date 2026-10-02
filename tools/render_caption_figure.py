"""Draft caption-showcase figure: chosen frames across, rows down.

Rows: thermal | LiDAR | w/o captions | w/ captions, and with --errors two more rows
of per-pixel relative error on the LiDAR points, |d - d*| / d*, one per arm. Every
depth panel goes through the evaluator's per-image ssi_log fit and shares its
column's GT-derived scale (magma, near dark, far bright, no-data grey); the error
rows share one fixed scale across the figure.

The frames must come from rank_<cond>.txt (the top eight per condition by
caption-free minus captioned AbsRel), and the paper caption must say so and give
the mean gain -- this is a best-case illustration.

    python tools/render_caption_figure.py --bundle outputs/caption_showcase \
        --frames night:2021-08-13-22-03-03_005989 rainy:2021-08-06-16-19-00_004830 \
        --out outputs/caption_showcase/fig_caption_A
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from ms2_eval.official_protocol import official_valid_mask  # noqa: E402
from render_qualitative_figure import aligned_depth, colour, dilate_sparse, stretch  # noqa: E402

TITLES = {"day": "Day", "night": "Night", "rainy": "Rain"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", required=True, help="cond:frame_id, in column order")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--errors", action="store_true")
    parser.add_argument("--error-max", type=float, default=0.4)
    parser.add_argument("--width-in", type=float, default=7.16)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for spec in args.frames:
        cond, fid = spec.split(":")
        rank = (args.bundle / f"rank_{cond}.txt").read_text(encoding="utf-8")
        if fid not in rank:
            raise SystemExit(f"!! {fid} is not in rank_{cond}.txt -- frames must come from the ranked list")

    labels = ["Thermal", "LiDAR", "w/o captions", "w/ captions"]
    if args.errors:
        labels += ["Error, w/o", "Error, w/"]
    cols = len(args.frames)
    cell_w = (args.width_in - 0.45) / cols
    fig, axes = plt.subplots(len(labels), cols, squeeze=False, dpi=300,
                             figsize=(args.width_in, cell_w * 0.4 * len(labels) + 0.25),
                             gridspec_kw={"wspace": 0.02, "hspace": 0.04})
    fig.subplots_adjust(left=0.45 / args.width_in, right=0.998, bottom=0.003,
                        top=1 - 0.2 / fig.get_figheight())
    err_cmap = matplotlib.colormaps["viridis"]
    titles_seen: dict[str, int] = {}

    for c, spec in enumerate(args.frames):
        cond, fid = spec.split(":")
        th = stretch(np.asarray(Image.open(args.bundle / "inputs" / f"{cond}_{fid}_thermal.png"), np.float32))
        gt = np.asarray(Image.open(args.bundle / "inputs" / f"{cond}_{fid}_gt.png"), np.float32) / 256.0
        valid = official_valid_mask(gt, 1e-3, 80.0)
        lo, hi = (float(v) for v in np.percentile(gt[valid], (2, 98)))
        grown, mask = dilate_sparse(gt, valid, 1)
        depth = {arm: aligned_depth(np.load(args.bundle / arm / "raw_predictions" / f"{fid}.npy").astype(np.float32),
                                    gt, "ssi_log") for arm in ("nocap", "cap")}
        panels = [np.stack([th] * 3, -1), colour(grown, lo, hi, mask),
                  colour(depth["nocap"], lo, hi), colour(depth["cap"], lo, hi)]
        if args.errors:
            for arm in ("nocap", "cap"):
                err = np.zeros_like(gt)
                err[valid] = np.abs(depth[arm][valid] - gt[valid]) / gt[valid]
                e_grown, e_mask = dilate_sparse(err + 1e-6, valid, 1)  # nearer-wins is fine: all >= 0
                rgb = err_cmap(np.clip(e_grown / args.error_max, 0, 1))[..., :3]
                rgb[~e_mask] = 0.5
                panels.append(rgb)
        n = titles_seen.get(cond, 0) + 1
        titles_seen[cond] = n
        for r, img in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, interpolation="nearest", aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c == 0:
                ax.set_ylabel(labels[r], fontsize=6.5, labelpad=2)
        axes[0, c].set_title(TITLES[cond] if sum(1 for s in args.frames if s.startswith(cond + ":")) == 1
                             else f"{TITLES[cond]} {n}", fontsize=7.5, pad=2)
        print(f"[{cond}] {fid}  scale {lo:.0f}-{hi:.0f} m")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out.with_suffix(f".{ext}"), facecolor="white")
    print(f"[done] {args.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
