"""Canonical MS2 depth metrics on explicit GT-valid pixels."""

from __future__ import annotations

from typing import Any

import numpy as np


class EmptyValidMaskError(ValueError):
    """No GT pixels are eligible for evaluation."""


class PredictionValueError(ValueError):
    """A prediction is invalid on at least one GT-valid pixel."""


RAW_METRICS = ("abs_rel", "sq_rel", "rmse_m", "rmse_log", "delta1", "delta2", "delta3")


def build_valid_mask(gt_m: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    gt = np.asarray(gt_m)
    if gt.ndim != 2:
        raise ValueError(f"GT must be HxW, got {gt.shape}")
    if not np.isfinite(min_depth) or not np.isfinite(max_depth) or min_depth >= max_depth:
        raise ValueError(f"Invalid depth range ({min_depth}, {max_depth})")
    return np.isfinite(gt) & (gt > float(min_depth)) & (gt < float(max_depth))


def validate_pair(pred_m, gt_m, valid):
    pred, gt, mask = np.asarray(pred_m, np.float32), np.asarray(gt_m, np.float32), np.asarray(valid, bool)
    if pred.ndim != 2 or gt.ndim != 2 or mask.ndim != 2:
        raise ValueError(f"Expected HxW arrays, got pred={pred.shape}, gt={gt.shape}, mask={mask.shape}")
    if pred.shape != gt.shape or gt.shape != mask.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}, mask={mask.shape}")
    if not mask.any():
        raise EmptyValidMaskError("No valid GT pixels remain after applying the MS2 depth mask")
    if not np.isfinite(gt[mask]).all() or (gt[mask] <= 0).any():
        raise ValueError("Valid GT pixels must be finite and strictly positive")
    bad = mask & (~np.isfinite(pred) | (pred <= 0))
    if bad.any():
        raise PredictionValueError(
            "Prediction has invalid values on GT-valid pixels: "
            f"nan={int((mask & np.isnan(pred)).sum())}, inf={int((mask & np.isinf(pred)).sum())}, "
            f"nonpositive={int((mask & np.isfinite(pred) & (pred <= 0)).sum())}"
        )
    return pred, gt, mask


def compute_depth_metrics(pred_m, gt_m, valid) -> dict[str, float | int]:
    pred, gt, mask = validate_pair(pred_m, gt_m, valid)
    p, g = pred[mask].astype(np.float64), gt[mask].astype(np.float64)
    diff, ratio = p - g, np.maximum(p / g, g / p)
    return {
        "abs_rel": float(np.mean(np.abs(diff) / g)),
        "sq_rel": float(np.mean(np.square(diff) / g)),
        "rmse_m": float(np.sqrt(np.mean(np.square(diff)))),
        "rmse_log": float(np.sqrt(np.mean(np.square(np.log(p) - np.log(g))))),
        "delta1": float(np.mean(ratio < 1.25)), "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)), "valid_pixels": int(mask.sum()),
    }


def align_affine(pred_m, gt_m, valid):
    pred, gt, mask = validate_pair(pred_m, gt_m, valid)
    x, y = pred[mask].astype(np.float64), gt[mask].astype(np.float64)
    scale, shift = np.linalg.lstsq(np.stack([x, np.ones_like(x)], 1), y, rcond=None)[0]
    aligned = pred.astype(np.float64) * float(scale) + float(shift)
    if not np.isfinite(aligned).all():
        raise PredictionValueError("Affine alignment produced NaN/Inf pixels")
    eps = float(np.finfo(np.float32).eps)
    clipped = int((aligned[mask] <= eps).sum())
    return np.maximum(aligned, eps).astype(np.float32), float(scale), float(shift), clipped


def rankdata(values):
    values = np.asarray(values, np.float64)
    order, ranks = np.argsort(values, kind="mergesort"), np.empty(values.size, np.float64)
    sorted_values, start = values[order], 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]: end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def correlation(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    return float("nan") if x.size < 2 or x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1])


def evaluate_depth_pair(pred_m, gt_m, valid, *, raw_is_metric=True, include_spearman=True):
    pred, gt, mask = validate_pair(pred_m, gt_m, valid)
    result: dict[str, Any] = {"valid_pixels": int(mask.sum()), "raw_metric_available": bool(raw_is_metric)}
    result.update(compute_depth_metrics(pred, gt, mask) if raw_is_metric else {k: None for k in RAW_METRICS})
    aligned, scale, shift, clipped = align_affine(pred, gt, mask)
    am = compute_depth_metrics(aligned, gt, mask)
    result.update({
        "alignment_scale": scale, "alignment_shift": shift, "alignment_nonpositive_clipped": clipped,
        "aligned_abs_rel": am["abs_rel"], "aligned_rmse_m": am["rmse_m"],
        "aligned_rmse_log": am["rmse_log"], "aligned_delta1": am["delta1"],
        "pearson_r": correlation(pred[mask], gt[mask]),
        "spearman_r": correlation(rankdata(pred[mask]), rankdata(gt[mask])) if include_spearman else None,
    })
    return result, aligned
