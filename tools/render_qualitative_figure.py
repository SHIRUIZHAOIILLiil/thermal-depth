"""Draw fig:qual from what slurm/qual_figure_preds.sbatch packs.

Columns are the three conditions; rows are thermal input, LiDAR ground truth,
then one row per model named in --rows. A model whose predictions are missing
from the bundle is left out rather than drawn blank; the printout says which
rows were drawn.

Two versions. The default, --rows lotus0 ours, is the one usable while the
adapted baselines' weights are locked under the retired account: our model
next to Lotus-G applied to thermal without adaptation (Table main's "Lotus-G,
no adaptation" row). ⛔ Lotus-G is NOT our starting point -- our line starts
from SD2.1-base, a text-to-image model with no depth head. The caption must
call Lotus-G "the same architecture trained on synthetic RGB depth, applied
to thermal unadapted", never "our model before adaptation", and must point
the reader to Table main for the adapted baselines. The full version is
--rows ours ppd da2.

Every prediction goes through the evaluator's own code before it is drawn --
`collapse_channels`, `resize_dense_prediction` to GT resolution, and the same
per-image two-parameter fit in the same space as Table main (ssi_log for ours
and PPD, ssi_disparity for DA2). So the picture is of exactly what was scored.
Unlike the scorer, nothing is clamped to 80 m: the colour scale does that job.

Colour: depth in metres on magma, near dark and far bright, one scale per
column taken from that column's ground truth (2nd-98th percentile), shared by
every row in the column so rows can be compared. Pixels with no LiDAR return
are grey -- black would read as "near". The sparse LiDAR is dilated by a
small max-filter for visibility, as the caption says. Thermal is the 1-99
percentile stretch the models were fed.

    python tools/render_qualitative_figure.py --bundle <unpacked qual_figure> --out figures/qualitative
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ms2_eval.official_protocol import (  # noqa: E402
    collapse_channels,
    fit_scale_shift,
    official_valid_mask,
)
from ms2_eval.resize import resize_dense_prediction  # noqa: E402

CONDITIONS = (("day", "Day"), ("night", "Night"), ("rainy", "Rain"))
# (bundle directory, row label, alignment space). Drawn in the order --rows gives.
MODELS = {
    "lotus0": ("lotus0", "Lotus-G (no adapt.)", "ssi_disparity"),
    "ours": ("ours_percentile", "Ours", "ssi_log"),
    "ppd": ("ppd", "PPD (adapted)", "ssi_log"),
    "da2": ("da2", "DA2 (adapted)", "ssi_disparity"),
}
D_MIN, D_MAX = 1e-3, 80.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True,
                        help="The unpacked qual_figure directory.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output path without extension; writes .png and .pdf.")
    parser.add_argument("--rows", nargs="+", default=["lotus0", "ours"], choices=sorted(MODELS),
                        help="Model rows under thermal and GT. Default is the version without "
                             "adapted baselines: the zero-shot starting point, then ours. "
                             "Full version: --rows ours ppd da2")
    parser.add_argument("--dilate", type=int, default=2,
                        help="Max-filter radius in pixels for the sparse LiDAR.")
    parser.add_argument("--width-in", type=float, default=7.16,
                        help="IEEE two-column text width.")
    return parser.parse_args()


def stretch(raw: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(raw, (1.0, 99.0))
    return np.clip((raw - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(raw)


def aligned_depth(pred: np.ndarray, gt: np.ndarray, space: str) -> np.ndarray:
    """evaluate_sample's fit, kept as a map instead of reduced to metrics."""
    valid = official_valid_mask(gt, D_MIN, D_MAX)
    if space == "ssi_log":
        target = np.zeros_like(gt, np.float32)
        target[valid] = np.log(np.maximum(gt[valid], 1e-6))
        s, t = fit_scale_shift(pred, target, valid)
        return np.exp(np.clip(pred.astype(np.float64) * s + t, -9.0, 9.0))
    if space == "ssi_disparity":
        target = np.zeros_like(gt, np.float32)
        target[valid] = 1.0 / gt[valid]
        s, t = fit_scale_shift(pred, target, valid)
        return 1.0 / np.clip(pred.astype(np.float64) * s + t, 1e-3, None)
    raise ValueError(space)


def dilate_sparse(depth: np.ndarray, valid: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    """Spread each return over its neighbourhood, nearer return winning."""
    if radius <= 0:
        return depth, valid
    filled = np.where(valid, depth, np.inf)
    out = filled.copy()
    h, w = depth.shape
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            shifted = np.full_like(filled, np.inf)
            ys, yd = slice(max(dy, 0), h + min(dy, 0)), slice(max(-dy, 0), h + min(-dy, 0))
            xs, xd = slice(max(dx, 0), w + min(dx, 0)), slice(max(-dx, 0), w + min(-dx, 0))
            shifted[yd, xd] = filled[ys, xs]
            out = np.minimum(out, shifted)
    grown = np.isfinite(out)
    return np.where(grown, out, 0.0), grown


def colour(depth: np.ndarray, lo: float, hi: float, valid: np.ndarray | None = None) -> np.ndarray:
    import matplotlib

    t = np.clip((depth - lo) / (hi - lo), 0, 1)
    rgb = matplotlib.colormaps["magma"](t)[..., :3]
    if valid is not None:
        rgb[~valid] = 0.5
    return rgb


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = [json.loads(l) for l in (args.bundle / "frames.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_cond = {}
    for cond, _ in CONDITIONS:
        hit = [f for f in frames if (args.bundle / "inputs" / f"{cond}_{f['id']}_gt.png").is_file()]
        if len(hit) != 1:
            raise SystemExit(f"!! {cond}: expected one frame in the bundle, found {len(hit)}")
        by_cond[cond] = hit[0]["id"]

    wanted = [MODELS[key] for key in args.rows]
    models = [m for m in wanted
              if all((args.bundle / m[0] / "raw_predictions" / f"{fid}.npy").is_file() for fid in by_cond.values())]
    skipped = [m[1] for m in wanted if m not in models]
    rows = ["Thermal", "LiDAR GT"] + [m[1] for m in models]
    print(f"[rows] {', '.join(rows)}" + (f"   (left out, predictions missing: {', '.join(skipped)})" if skipped else ""))

    aspect = 256 / 640
    cell_w = (args.width_in - 0.55) / 3
    fig, axes = plt.subplots(len(rows), 3, figsize=(args.width_in, cell_w * aspect * len(rows) + 0.25),
                             dpi=300, squeeze=False,
                             gridspec_kw={"wspace": 0.02, "hspace": 0.04})
    fig.subplots_adjust(left=0.55 / args.width_in, right=0.998, top=1 - 0.2 / fig.get_figheight(), bottom=0.003)

    for col, (cond, title) in enumerate(CONDITIONS):
        fid = by_cond[cond]
        thermal = np.asarray(Image.open(args.bundle / "inputs" / f"{cond}_{fid}_thermal.png"), np.float32)
        gt = np.asarray(Image.open(args.bundle / "inputs" / f"{cond}_{fid}_gt.png"), np.float32) / 256.0
        valid = official_valid_mask(gt, D_MIN, D_MAX)
        lo, hi = (float(v) for v in np.percentile(gt[valid], (2, 98)))
        gt_show, gt_mask = dilate_sparse(gt, valid, args.dilate)

        panels = [np.stack([stretch(thermal)] * 3, axis=-1), colour(gt_show, lo, hi, gt_mask)]
        for directory, label, space in models:
            pred = collapse_channels(np.load(args.bundle / directory / "raw_predictions" / f"{fid}.npy"))
            pred = resize_dense_prediction(pred, tuple(gt.shape))
            depth = aligned_depth(pred, gt, space)
            panels.append(colour(depth, lo, hi))
        print(f"[{cond}] {fid}   colour scale {lo:.1f}-{hi:.1f} m")

        axes[0, col].set_title(title, fontsize=8, pad=2)
        for row, panel in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(panel, interpolation="nearest", aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if col == 0:
                ax.set_ylabel(rows[row], fontsize=7, rotation=90, labelpad=3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out.with_suffix(f".{ext}"), facecolor="white")
    print(f"[done] {args.out.with_suffix('.png')}  and .pdf")


if __name__ == "__main__":
    main()
