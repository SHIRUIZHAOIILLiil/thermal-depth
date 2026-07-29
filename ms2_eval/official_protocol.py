"""Official BridgeMultiSpectralDepth (MS2 / AnyThermal) depth evaluation protocol.

Faithful numpy port of BridgeMultiSpectralDepth @ f7e231de (cloned at
``E:/project/AnyThermal/baselines/depth/BridgeMultiSpectralDepth``):

* ``models/metrics/eval_metric.py::compute_depth_errors`` (metric path,
  per-image median scaling, clamp, official metric set);
* ``models/losses/midas_loss.py::compute_scale_and_shift`` +
  ``models/trainers/mono_depth/Midas.py::fit_scale_shfit_depth``
  (relative-model path: per-image affine fit in RAW network-output space).

Deliberate differences from ``ms2_eval.core`` (Iris unified protocol v1) --
this module must never be merged into it:

* relative models are aligned in the raw network-output space, never after a
  1/x disparity-to-depth conversion;
* metric models use per-image median scaling (scale only, no shift);
* the valid mask is ``gt > 1e-3 & gt < 80`` (v1 uses ``gt > 0.1``);
* aligned predictions are clamped to ``[min_depth, max_depth]`` before
  metrics are computed;
* extra official metrics: ``abs_diff`` and ``log10``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


OFFICIAL_METRICS = ("abs_diff", "abs_rel", "sq_rel", "log10", "rmse", "rmse_log", "a1", "a2", "a3")
DEFAULT_MIN_DEPTH_M = 1e-3
DEFAULT_MAX_DEPTH_M = 80.0  # official value for MS2 and ViViD
ALIGN_MODES = ("ssi", "ssi_disparity", "median", "none")

PROTOCOL_REFERENCE: dict[str, Any] = {
    "name": "BridgeMultiSpectralDepth-official-MS2-depth-eval",
    "paper": "Shin et al., ICRA 2025 (protocol shared with AnyThermal depth eval)",
    "source_repo": "https://github.com/UkcheolShin/BridgeMultiSpectralDepth",
    "source_commit": "f7e231de6acff7ee09e4ed1833bc9af1223bdbf7",
    "local_clone": "E:/project/AnyThermal/baselines/depth/BridgeMultiSpectralDepth",
    "alignment": {
        "ssi": "per-image least-squares scale+shift of RAW network output vs GT depth (MiDaS/DPT path)",
        "ssi_disparity": (
            "per-image least-squares scale+shift of RAW disparity output vs GT disparity (1/gt), "
            "inverted to depth; same 2-parameter budget as ssi but respects affine-disparity outputs. "
            "NOT in the upstream code -- extension for disparity-space models, everything after "
            "alignment (mask, clamp, metric formulas, aggregation) stays official"
        ),
        "median": "per-image median(gt)/median(pred) scaling, no shift (metric-model path)",
        "none": "no alignment (predictions already metric and calibrated)",
    },
    "post_alignment_clamp": "predictions clamped to [min_depth, max_depth] on valid pixels",
    "aggregation": "macro: metrics per image, then unweighted mean over images",
}


class OfficialProtocolError(ValueError):
    """A sample cannot be evaluated under the official protocol."""


def collapse_channels(raw: np.ndarray) -> np.ndarray:
    """Collapse a raw exported prediction to one dense HxW channel."""
    value = np.squeeze(np.asarray(raw))
    if value.ndim == 3 and value.shape[-1] in (1, 3):
        value = value.mean(axis=-1)
    if value.ndim == 3 and value.shape[0] in (1, 3):
        value = value.mean(axis=0)
    if value.ndim != 2:
        raise OfficialProtocolError(f"Expected one dense channel, got shape {np.asarray(raw).shape}")
    return value.astype(np.float32)


def official_valid_mask(gt_m: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    """Official mask: ``gt > min_depth & gt < max_depth`` (finite added for safety)."""
    gt = np.asarray(gt_m)
    if gt.ndim != 2:
        raise OfficialProtocolError(f"GT must be HxW, got {gt.shape}")
    if not np.isfinite(min_depth) or not np.isfinite(max_depth) or min_depth >= max_depth:
        raise OfficialProtocolError(f"Invalid depth range ({min_depth}, {max_depth})")
    return np.isfinite(gt) & (gt > float(min_depth)) & (gt < float(max_depth))


def fit_scale_shift(pred: np.ndarray, gt_m: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    """Closed-form per-image affine fit, port of ``compute_scale_and_shift``.

    Fits ``gt ~= scale * pred + shift`` over valid pixels, in the RAW
    prediction space (the official code never converts disparity to depth
    before this fit).
    """
    p = pred[valid].astype(np.float64)
    g = gt_m[valid].astype(np.float64)
    a_00, a_01, a_11 = float(np.sum(p * p)), float(np.sum(p)), float(p.size)
    b_0, b_1 = float(np.sum(p * g)), float(np.sum(g))
    det = a_00 * a_11 - a_01 * a_01
    if det <= 0 or not np.isfinite(det):
        raise OfficialProtocolError(
            f"Degenerate affine fit (det={det}); prediction is constant or invalid on valid pixels"
        )
    scale = (a_11 * b_0 - a_01 * b_1) / det
    shift = (-a_01 * b_0 + a_00 * b_1) / det
    if not np.isfinite(scale) or not np.isfinite(shift):
        raise OfficialProtocolError(f"Affine fit produced non-finite scale/shift ({scale}, {shift})")
    return float(scale), float(shift)


def _torch_style_median(values: np.ndarray) -> float:
    """``torch.median`` semantics: lower-middle element for even counts."""
    ordered = np.sort(values.ravel())
    return float(ordered[(ordered.size - 1) // 2])


def median_scale_ratio(pred: np.ndarray, gt_m: np.ndarray, valid: np.ndarray) -> float:
    """Per-image median-scaling ratio, port of the ``align=True`` metric path.

    Uses torch-style (lower-middle) medians so results match the official
    torch implementation bit-for-bit on even pixel counts.
    """
    pred_median = _torch_style_median(pred[valid].astype(np.float64))
    gt_median = _torch_style_median(gt_m[valid].astype(np.float64))
    if not np.isfinite(pred_median) or pred_median <= 0:
        raise OfficialProtocolError(f"Median scaling needs positive prediction median, got {pred_median}")
    return gt_median / pred_median


def official_depth_errors(
    pred: np.ndarray,
    gt_m: np.ndarray,
    valid: np.ndarray,
    *,
    min_depth: float = DEFAULT_MIN_DEPTH_M,
    max_depth: float = DEFAULT_MAX_DEPTH_M,
) -> dict[str, float | int]:
    """Official per-image metrics on (already aligned) predictions.

    Port of the ``compute_depth_errors`` body after the alignment decision:
    valid predictions are clamped to ``[min_depth, max_depth]`` first.
    """
    if not valid.any():
        raise OfficialProtocolError("No valid GT pixels under the official mask")
    g = gt_m[valid].astype(np.float64)
    p = np.clip(pred[valid].astype(np.float64), float(min_depth), float(max_depth))
    if not np.isfinite(p).all():
        raise OfficialProtocolError("Prediction has NaN/Inf on valid pixels")
    diff = g - p
    ratio = np.maximum(g / p, p / g)
    return {
        "abs_diff": float(np.mean(np.abs(diff))),
        "abs_rel": float(np.mean(np.abs(diff) / g)),
        "sq_rel": float(np.mean(np.square(diff) / g)),
        "log10": float(np.mean(np.abs(np.log10(g) - np.log10(p)))),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "rmse_log": float(np.sqrt(np.mean(np.square(np.log(g) - np.log(p))))),
        "a1": float(np.mean(ratio < 1.25)),
        "a2": float(np.mean(ratio < 1.25**2)),
        "a3": float(np.mean(ratio < 1.25**3)),
        "valid_pixels": int(valid.sum()),
    }


def evaluate_sample(
    pred: np.ndarray,
    gt_m: np.ndarray,
    *,
    align: str,
    min_depth: float = DEFAULT_MIN_DEPTH_M,
    max_depth: float = DEFAULT_MAX_DEPTH_M,
) -> dict[str, Any]:
    """Evaluate one prediction (already resized to GT resolution) officially.

    ``align='ssi'`` expects ``pred`` in RAW network-output space;
    ``align='median'`` and ``align='none'`` expect positive depth in metres.
    """
    if align not in ALIGN_MODES:
        raise OfficialProtocolError(f"Unknown align mode {align!r}; expected one of {ALIGN_MODES}")
    pred = np.asarray(pred, np.float32)
    gt = np.asarray(gt_m, np.float32)
    if pred.shape != gt.shape:
        raise OfficialProtocolError(f"Shape mismatch: pred={pred.shape}, gt={gt.shape}; resize before evaluating")
    valid = official_valid_mask(gt, min_depth, max_depth)
    if not valid.any():
        raise OfficialProtocolError("No valid GT pixels under the official mask")
    if not np.isfinite(pred[valid]).all():
        raise OfficialProtocolError("Raw prediction has NaN/Inf on valid pixels")

    row: dict[str, Any] = {"align_mode": align}
    if align == "ssi":
        scale, shift = fit_scale_shift(pred, gt, valid)
        aligned = pred.astype(np.float64) * scale + shift
        row.update({"alignment_scale": scale, "alignment_shift": shift})
    elif align == "ssi_disparity":
        gt_disparity = np.zeros_like(gt, np.float64)
        gt_disparity[valid] = 1.0 / gt[valid].astype(np.float64)
        scale, shift = fit_scale_shift(pred, gt_disparity.astype(np.float32), valid)
        aligned_disparity = np.clip(pred.astype(np.float64) * scale + shift, 1e-3, None)
        aligned = 1.0 / aligned_disparity
        row.update({"alignment_scale": scale, "alignment_shift": shift})
    elif align == "median":
        ratio = median_scale_ratio(pred, gt, valid)
        aligned = pred.astype(np.float64) * ratio
        row.update({"alignment_scale": ratio, "alignment_shift": 0.0})
    else:
        aligned = pred.astype(np.float64)
        row.update({"alignment_scale": 1.0, "alignment_shift": 0.0})

    row["clamped_below"] = int((aligned[valid] < min_depth).sum())
    row["clamped_above"] = int((aligned[valid] > max_depth).sum())
    row.update(official_depth_errors(aligned, gt, valid, min_depth=min_depth, max_depth=max_depth))
    return row
