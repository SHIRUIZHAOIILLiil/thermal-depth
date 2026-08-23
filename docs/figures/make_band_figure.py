"""Re-lay out the band comparison so the panels are large enough to read.

`tools/export_qualitative.py` writes one row per frame and one column per model. With
five columns that is the right shape for a slide and the wrong one for a two-column
paper: the full text width divided by five leaves each MS2 frame about $1.4$ inches
wide, and since the frames are $640\\times256$ letterbox, barely half an inch tall. The
upper band -- the entire subject -- becomes invisible.

Transposing fixes it without new pixels. Frames run across, conditions run down, so
three columns give each panel roughly $2.3$ inches, about $64\\%$ wider. The LiDAR
column is dropped: the caption has to spend three lines explaining that it is not a
target for the band, which is a sign it does not belong in a figure about the band.

Nothing is recomputed. The aligned depth maps written by `export_qualitative.py` are
read straight off disk, and the colouring, the dilation and the per-frame shared range
(1st and 99th percentiles of that frame's LiDAR disparity) are the same ones it uses.

    python docs/figures/make_band_figure.py
    python docs/figures/make_band_figure.py --output-dir "$PAPER/figures"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from make_pseudo_gt_figure import colour_shared  # noqa: E402

D_MIN, D_MAX, DEPTH_SCALE = 1e-3, 80.0, 256.0
SEQUENCE = "_2021-08-13-16-08-46"
FRAMES = ("000363", "001089", "001815")
# Row order is the argument's order: what goes wrong, then the two things that fix it.
MODELS = ("b_e5", "b_pseudo_e12", "b_skyloss20")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ms2-root", type=Path, default=Path("E:/dataset/ms2"))
    parser.add_argument("--aligned-dir", type=Path,
                        default=HERE.parents[1] / "pseudo_vs_sky" / "aligned")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--frames", nargs="*", default=list(FRAMES))
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--separate-panels", action="store_true",
                        help="Also write each panel as its own file, for the pipeline "
                             "schematic, which places the input and one prediction as "
                             "separate nodes and cannot take a composed strip apart.")
    parser.add_argument("--panel-width", type=int, default=640)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sequence_id = SEQUENCE.lstrip("_")

    columns = []
    for frame in args.frames:
        thermal = np.asarray(Image.open(
            args.ms2_root / "sync_data" / SEQUENCE / "thr" / "img_left" / f"{frame}.png"
        )).astype(np.float32)
        t_lo, t_hi = np.percentile(thermal, [1, 99])
        grey = Image.fromarray(
            (np.clip((thermal - t_lo) / max(t_hi - t_lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        ).convert("RGB")

        gt = np.asarray(Image.open(
            args.ms2_root / "proj_depth" / SEQUENCE / "thr" / "depth_filtered" / f"{frame}.png"
        ), np.float32) / DEPTH_SCALE
        valid = np.isfinite(gt) & (gt > D_MIN) & (gt < D_MAX)
        gt_disparity = np.zeros_like(gt)
        gt_disparity[valid] = 1.0 / gt[valid]
        lo = float(np.percentile(gt_disparity[valid], 1))
        hi = float(np.percentile(gt_disparity[valid], 99))

        panels = [grey]
        for model in args.models:
            path = args.aligned_dir / f"{model}__{sequence_id}_{frame}.npy"
            if not path.exists():
                raise SystemExit(f"missing {path}")
            aligned = np.load(path).astype(np.float32)
            panels.append(colour_shared(1.0 / np.maximum(aligned, 1e-6), lo, hi))
        if args.separate_panels:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            names = ["thermal"] + list(args.models)
            for slug, panel in zip(names, panels):
                height_i = round(panel.height * args.panel_width / panel.width)
                out = args.output_dir / f"panel_{slug}_{frame}.png"
                panel.resize((args.panel_width, height_i), Image.LANCZOS).save(out, optimize=True)
                print(f"  panel -> {out}")
        columns.append(panels)

    width = args.panel_width
    height = round(columns[0][0].height * width / columns[0][0].width)
    figure = Image.new("RGB", (width * len(columns), height * len(columns[0])), "white")
    for x, panels in enumerate(columns):
        for y, panel in enumerate(panels):
            figure.paste(panel.resize((width, height), Image.LANCZOS), (x * width, y * height))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "sky_band_strip.png"
    figure.save(destination, optimize=True)
    print(f"{figure.size}  {destination.stat().st_size / 1e6:.2f} MB  ->  {destination}")
    print("rows, top to bottom: thermal input, " + ", ".join(args.models))
    print("columns, left to right: " + ", ".join(args.frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
