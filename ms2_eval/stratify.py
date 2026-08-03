"""Stratum definitions and the alignment mirror, shared by the region and figure tools.

These used to live in `tools/analyze_route_regions.py`, whose module body imports
torch and `train_route_suite` (and therefore diffusers).  That is the right
dependency for a tool that runs checkpoints, and the wrong one for a tool that
only reads saved `*.npy` predictions -- the qualitative figures need the exact
same strata but no model at all.  Duplicating the definitions would let the
figures and the tables drift apart silently, which is worse than either.

So the definitions live here, importable with numpy alone.  `boundary_mask` is
the one exception: its max/min pooling is torch's, so torch is imported inside
the function and only that stratum costs the dependency.

`analyze_route_regions` re-exports every name, so its callers are unaffected.
"""

from __future__ import annotations

import numpy as np

from ms2_eval.official_protocol import fit_scale_shift, median_scale_ratio

DEPTH_BANDS = ((0.0, 10.0, "near <10m"), (10.0, 30.0, "mid 10-30m"), (30.0, 80.0, "far >30m"))
ROW_BANDS = ((0.0, 1 / 3, "top"), (1 / 3, 2 / 3, "middle"), (2 / 3, 1.0, "bottom"))
BOUNDARY_WINDOW = 9          # neighbourhood side length, pixels
BOUNDARY_RATIO = 1.25        # local max/min depth ratio that counts as a discontinuity


def boundary_mask(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Valid pixels sitting on a depth discontinuity.

    Sparse LiDAR makes a gradient operator meaningless, so instead of
    differentiating we ask a rank question over a window: does the local
    neighbourhood of valid samples span more than BOUNDARY_RATIO in depth?
    Implemented with max/min pooling so it costs one pass on the GPU-free path.
    """
    import torch                      # only this stratum needs it
    import torch.nn.functional as F

    filled_high = np.where(valid, depth, -np.inf)
    filled_low = np.where(valid, depth, np.inf)
    tensor_high = torch.from_numpy(filled_high)[None, None]
    tensor_low = torch.from_numpy(filled_low)[None, None]
    pad = BOUNDARY_WINDOW // 2
    local_max = F.max_pool2d(tensor_high, BOUNDARY_WINDOW, stride=1, padding=pad)[0, 0].numpy()
    local_min = -F.max_pool2d(-tensor_low, BOUNDARY_WINDOW, stride=1, padding=pad)[0, 0].numpy()
    span = np.zeros_like(depth)
    usable = valid & np.isfinite(local_max) & np.isfinite(local_min) & (local_min > 0)
    span[usable] = local_max[usable] / np.maximum(local_min[usable], 1e-6)
    return valid & (span > BOUNDARY_RATIO)


def strata_for(gt: np.ndarray, valid: np.ndarray, *,
               include_boundary: bool = True) -> dict[str, np.ndarray]:
    """name -> boolean mask (already intersected with `valid`).

    `include_boundary=False` drops the two `structure/*` strata and with them the
    torch dependency; the depth bands and row thirds are unchanged either way.
    """
    height = gt.shape[0]
    rows = np.arange(height)[:, None] / height
    strata: dict[str, np.ndarray] = {"all": valid}
    for low, high, name in DEPTH_BANDS:
        strata[f"depth/{name}"] = valid & (gt >= low) & (gt < high)
    for low, high, name in ROW_BANDS:
        band = (rows >= low) & (rows < high)
        strata[f"row/{name}"] = valid & np.broadcast_to(band, gt.shape)
    if include_boundary:
        edge = boundary_mask(gt, valid)
        strata["structure/boundary"] = edge
        strata["structure/interior"] = valid & ~edge
    return strata


def align_prediction(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray,
                     align: str) -> np.ndarray:
    """Mirror of evaluate_sample's alignment branches, returning the aligned map.

    The protocol module is frozen and hands back metrics, not the aligned map, so
    anything that needs the map itself (strata, figures) mirrors it here.  Every
    caller re-scores at least one frame with the real `evaluate_sample` and
    aborts on disagreement, which is what keeps the mirror honest.
    """
    if align == "ssi":
        scale, shift = fit_scale_shift(pred, gt, valid)
        return pred.astype(np.float64) * scale + shift
    if align == "ssi_disparity":
        gt_disparity = np.zeros_like(gt, np.float64)
        gt_disparity[valid] = 1.0 / gt[valid].astype(np.float64)
        scale, shift = fit_scale_shift(pred, gt_disparity.astype(np.float32), valid)
        return 1.0 / np.clip(pred.astype(np.float64) * scale + shift, 1e-3, None)
    if align == "median":
        return pred.astype(np.float64) * median_scale_ratio(pred, gt, valid)
    return pred.astype(np.float64)
