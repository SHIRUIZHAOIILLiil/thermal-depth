"""Canonical resize API (Pillow-backed; no sparse-GT interpolation)."""

from ms2_eval.resize_pil import resize_dense_prediction, resize_mask_nearest

__all__ = ["resize_dense_prediction", "resize_mask_nearest"]
