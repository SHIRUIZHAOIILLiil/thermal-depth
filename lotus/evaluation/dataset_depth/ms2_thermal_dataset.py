"""MS2 left-thermal dataset adapter for the upstream Lotus evaluator."""

from __future__ import annotations

import numpy as np

from .base_depth_dataset import BaseDepthDataset, DepthFileNameMode


class MS2ThermalDataset(BaseDepthDataset):
    """Decode official MS2 filtered thermal-view depth as metres."""

    def __init__(
        self,
        depth_scale: float = 256.0,
        min_depth: float = 0.1,
        max_depth: float = 80.0,
        **kwargs,
    ) -> None:
        super().__init__(
            min_depth=min_depth,
            max_depth=max_depth,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )
        self.depth_scale = float(depth_scale)
        if self.depth_scale <= 0:
            raise ValueError("depth_scale must be positive")

    def _read_rgb_file(self, rel_path) -> np.ndarray:
        thermal = self._read_image(rel_path)
        if thermal.ndim == 3:
            thermal = thermal[..., 0]
        if thermal.ndim != 2:
            raise ValueError(f"Expected a single-channel thermal image, got {thermal.shape}")
        return thermal[np.newaxis, ...].astype(int)

    def _read_depth_file(self, rel_path) -> np.ndarray:
        raw = self._read_image(rel_path)
        if raw.ndim == 3:
            raw = raw[..., 0]
        return raw.astype(np.float32) / self.depth_scale


class MS2RGBDataset(BaseDepthDataset):
    """Decode MS2 RGB-view filtered depth for native RGB-route evaluation."""

    def __init__(
        self,
        depth_scale: float = 256.0,
        min_depth: float = 0.1,
        max_depth: float = 80.0,
        **kwargs,
    ) -> None:
        super().__init__(
            min_depth=min_depth,
            max_depth=max_depth,
            has_filled_depth=False,
            name_mode=DepthFileNameMode.id,
            **kwargs,
        )
        self.depth_scale = float(depth_scale)
        if self.depth_scale <= 0:
            raise ValueError("depth_scale must be positive")

    def _read_depth_file(self, rel_path) -> np.ndarray:
        raw = self._read_image(rel_path)
        if raw.ndim == 3:
            raw = raw[..., 0]
        return raw.astype(np.float32) / self.depth_scale
