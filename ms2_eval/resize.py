"""Prediction/mask resizing. Sparse GT is intentionally absent."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def resize_dense_prediction(prediction: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    pred = np.asarray(prediction, np.float32)
    if pred.ndim != 2: raise ValueError(f"Prediction must be HxW, got {pred.shape}")
    if tuple(pred.shape) == tuple(target_hw): return pred.copy()
    tensor = torch.from_numpy(pred)[None, None]
    return F.interpolate(tensor, target_hw, mode="bilinear", align_corners=False)[0, 0].numpy().astype(np.float32)


def resize_mask_nearest(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    value = np.asarray(mask, bool)
    if value.ndim != 2: raise ValueError(f"Mask must be HxW, got {value.shape}")
    if tuple(value.shape) == tuple(target_hw): return value.copy()
    tensor = torch.from_numpy(value.astype(np.float32))[None, None]
    return F.interpolate(tensor, target_hw, mode="nearest")[0, 0].numpy() > 0.5
