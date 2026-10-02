"""Contact sheets for choosing caption-showcase frames (working material, not a figure).

One row per frame: thermal | LiDAR | caption-free | captioned | where they differ.
Both arms go through the evaluator's own per-image ssi_log fit against the same GT
and share one colour scale per row (GT 2-98%), so a difference between the two
depth panels is a difference between the models, not between colour ranges. The
last panel is log(captioned depth / caption-free depth) on a diverging map: red
where the captioned model puts the surface farther, blue nearer.

Titles carry each frame's AbsRel for choosing; they are not for the paper.

    python tools/render_caption_showcase.py --bundle outputs/caption_showcase
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from ms2_eval.official_protocol import official_valid_mask  # noqa: E402
from render_qualitative_figure import aligned_depth, colour, dilate_sparse, stretch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for cond in ("day", "night", "rainy"):
        rank = (args.bundle / f"rank_{cond}.txt").read_text(encoding="utf-8").splitlines()
        rows = [re.split(r"\s+", l.strip()) for l in rank[2:] if l.strip()]
        fig, axes = plt.subplots(len(rows), 5, figsize=(22, 2.05 * len(rows)), dpi=90, squeeze=False)
        for r, (k, fid, nocap_abs, cap_abs, gain) in enumerate(rows):
            th = stretch(np.asarray(Image.open(args.bundle / "inputs" / f"{cond}_{fid}_thermal.png"), np.float32))
            gt = np.asarray(Image.open(args.bundle / "inputs" / f"{cond}_{fid}_gt.png"), np.float32) / 256.0
            valid = official_valid_mask(gt, 1e-3, 80.0)
            lo, hi = (float(v) for v in np.percentile(gt[valid], (2, 98)))
            grown, mask = dilate_sparse(gt, valid, 1)
            depth = {arm: aligned_depth(np.load(args.bundle / arm / "raw_predictions" / f"{fid}.npy").astype(np.float32),
                                        gt, "ssi_log") for arm in ("nocap", "cap")}
            ratio = np.log(depth["cap"] / depth["nocap"])
            lim = max(float(np.percentile(np.abs(ratio), 99)), 1e-3)
            panels = [
                (np.stack([th] * 3, -1), f"#{k} {fid}"),
                (colour(grown, lo, hi, mask), f"LiDAR  ({lo:.0f}-{hi:.0f} m)"),
                (colour(depth["nocap"], lo, hi), f"w/o captions  AbsRel {float(nocap_abs):.4f}"),
                (colour(depth["cap"], lo, hi), f"w/ captions  AbsRel {float(cap_abs):.4f}"),
                (matplotlib.colormaps["RdBu_r"](np.clip(ratio / lim * 0.5 + 0.5, 0, 1))[..., :3],
                 f"log(w/ / w/o), +-{lim:.2f}  (red: w/ farther)"),
            ]
            for c, (img, title) in enumerate(panels):
                ax = axes[r, c]
                ax.imshow(img, interpolation="nearest")
                ax.set_title(title, fontsize=9, loc="left")
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{cond}: {rank[0]}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / fig.get_figheight()))
        path = args.bundle / f"sheet_{cond}.png"
        fig.savefig(path, facecolor="white")
        plt.close(fig)
        print(f"[sheet] {path}")


if __name__ == "__main__":
    main()
