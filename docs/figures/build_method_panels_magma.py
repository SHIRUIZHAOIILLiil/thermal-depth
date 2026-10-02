"""Depth thumbnails for the method pipeline figure, in the paper's one colour rule.

Every depth panel is log depth on magma, near dark and far bright, pixels with no
value grey -- black would read as "near". Thermal is the 1-99 percentile stretch the
models are fed. This replaces the Spectral/disparity panels of 2026-08, and fixes the
inference panel, which showed the old disparity line (`full8_thermalcap`) under a
"log-depth prediction" label.

Panel A, completed-target construction, is training frame 10-59-33_000099: its
thermal, its projected LiDAR and its completed target on one shared range (the target's
2nd-98th log-depth percentile), so the two depth panels can be compared.
Panel C, inference, is test frame 15-46-56_002640 and the raw output of the model in
Table main (log line, seed 43, step 18000, its own caption), taken from the bundle
`slurm/qual_figure_preds.sbatch` wrote and checked frame by frame against the test CSV.
That output is normalised log depth, so it is drawn as is, on its own 2-98 range.

    python docs/figures/build_method_panels_magma.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from render_qualitative_figure import colour, dilate_sparse, stretch  # noqa: E402

MS2 = Path("E:/dataset/ms2")
OUT = ROOT / "docs" / "figures" / "figma_method_assets"
TRAIN = ("2021-08-06-10-59-33", "000099")
TEST = ("2021-08-13-15-46-56", "002640")
BUNDLE = ROOT / "outputs" / "method_panel"


def save(rgb: np.ndarray, name: str) -> None:
    Image.fromarray((np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)).save(OUT / name)
    print(f"[panel] {name}  {rgb.shape[1]}x{rgb.shape[0]}")


def thermal(seq: str, stem: str) -> np.ndarray:
    raw = np.asarray(Image.open(MS2 / "sync_data" / f"_{seq}" / "thr" / "img_left" / f"{stem}.png"), np.float32)
    return np.stack([stretch(raw)] * 3, axis=-1)


def main() -> None:
    seq, stem = TRAIN
    save(thermal(seq, stem), f"panel_thermal_{stem}_pct.png")

    target = np.load(ROOT / "pseudo_gt_samples" / f"{seq}_{stem}.npy").astype(np.float64)
    log_target = np.log(np.clip(target, 1e-3, 80.0))
    lo, hi = (float(v) for v in np.percentile(log_target, (2, 98)))
    save(colour(log_target, lo, hi), f"panel_completed_{stem}_magma.png")

    lidar = np.asarray(Image.open(MS2 / "proj_depth" / f"_{seq}" / "thr" / "depth_filtered" / f"{stem}.png"),
                       np.float32) / 256.0
    valid = np.isfinite(lidar) & (lidar > 1e-3) & (lidar < 80.0)
    grown, mask = dilate_sparse(lidar, valid, 1)
    log_lidar = np.log(np.where(mask, grown, 1.0))
    save(colour(log_lidar, lo, hi, mask), f"panel_lidar_{stem}_magma.png")

    seq, stem = TEST
    save(thermal(seq, stem), f"panel_thermal_{stem}_pct.png")
    pred = np.load(BUNDLE / "ours_percentile" / "raw_predictions" / f"{seq}_{stem}.npy").astype(np.float64)
    plo, phi = (float(v) for v in np.percentile(pred, (2, 98)))
    save(colour(pred, plo, phi), f"panel_pred_logtn_s43_{stem}_magma.png")


if __name__ == "__main__":
    main()
