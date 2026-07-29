"""MS2 bridge to the metric/alignment semantics of the Lotus evaluator."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from lotus.evaluation.util.alignment import align_depth_least_square, depth2disparity, disparity2depth
from lotus.evaluation.util import metric


OFFICIAL_METRICS = (
    "abs_relative_difference", "squared_relative_difference", "rmse_linear",
    "rmse_log", "log10", "delta1_acc", "delta2_acc", "delta3_acc",
    "i_rmse", "silog_rmse",
)


def align_lotus_disparity_to_ms2_depth(
    pred_disparity: np.ndarray,
    gt_depth_m: np.ndarray,
    valid_mask: np.ndarray,
    *,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, float, float]:
    pred = np.asarray(pred_disparity, np.float32)
    gt = np.asarray(gt_depth_m, np.float32)
    valid = np.asarray(valid_mask, bool)
    if pred.shape != gt.shape or gt.shape != valid.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}, mask={valid.shape}")
    if not valid.any(): raise ValueError("Empty MS2 valid mask")
    if not np.isfinite(pred).all(): raise ValueError("Prediction contains NaN/Inf")
    gt_disparity, gt_positive = depth2disparity(gt, return_mask=True)
    fit_mask = valid & gt_positive & (pred > 0)
    if int(fit_mask.sum()) < 2: raise ValueError("Fewer than two pixels available for disparity alignment")
    aligned_disparity, scale, shift = align_depth_least_square(
        gt_arr=gt_disparity, pred_arr=pred, valid_mask_arr=fit_mask,
        return_scale_shift=True, max_resolution=None,
    )
    aligned_disparity = np.clip(aligned_disparity, 1e-3, None)
    aligned_depth = disparity2depth(aligned_disparity)
    aligned_depth = np.clip(aligned_depth, min_depth_m, max_depth_m).astype(np.float32)
    return aligned_depth, float(np.asarray(scale).squeeze()), float(np.asarray(shift).squeeze())


def lotus_official_metrics(aligned_depth_m: np.ndarray, gt_depth_m: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    """Call the upstream Lotus metric functions on CPU for one image."""
    pred = torch.from_numpy(np.asarray(aligned_depth_m, np.float32))
    gt = torch.from_numpy(np.asarray(gt_depth_m, np.float32))
    valid = torch.from_numpy(np.asarray(valid_mask, bool))
    values: dict[str, float] = {}
    for name in OFFICIAL_METRICS:
        result = getattr(metric, name)(pred.clone(), gt.clone(), valid.clone())
        value = float(result.item())
        if not np.isfinite(value): raise ValueError(f"Lotus metric {name} is not finite")
        values[name] = value
    values["valid_pixels"] = int(valid.sum().item())
    return values


def aggregate_imagewise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows: raise ValueError("No metric rows")
    summary = {"image_count": len(rows), "aggregation": "image-wise macro mean", "metrics": {}}
    for name in OFFICIAL_METRICS:
        data = np.asarray([row[name] for row in rows], np.float64)
        summary["metrics"][name] = {
            "mean": float(data.mean()), "std": float(data.std(ddof=0)), "median": float(np.median(data)),
        }
    summary["total_valid_pixels"] = int(sum(int(row["valid_pixels"]) for row in rows))
    return summary
