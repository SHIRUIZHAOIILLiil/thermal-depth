"""Dependency-light resizing for evaluator tests and production use."""

import numpy as np
from PIL import Image


def resize_dense_prediction(prediction, target_hw):
    pred = np.asarray(prediction, np.float32)
    if pred.ndim != 2: raise ValueError(f"Prediction must be HxW, got {pred.shape}")
    if tuple(pred.shape) == tuple(target_hw): return pred.copy()
    height, width = map(int, target_hw)
    return np.asarray(Image.fromarray(pred, mode="F").resize((width, height), Image.Resampling.BILINEAR), np.float32)


def resize_mask_nearest(mask, target_hw):
    value = np.asarray(mask, bool)
    if value.ndim != 2: raise ValueError(f"Mask must be HxW, got {value.shape}")
    if tuple(value.shape) == tuple(target_hw): return value.copy()
    height, width = map(int, target_hw)
    image = Image.fromarray(value.astype(np.uint8) * 255, mode="L")
    return np.asarray(image.resize((width, height), Image.Resampling.NEAREST)) > 0
