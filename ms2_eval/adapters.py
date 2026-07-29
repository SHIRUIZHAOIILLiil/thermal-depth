"""Model-route output adapters for the unified MS2 evaluator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AdapterDeclaration:
    route: str
    raw_representation_type: str
    orientation: str
    metric_scale_exists: bool
    conversion_to_positive_depth: str
    clipping_rules: str


@dataclass
class AdaptedPrediction:
    depth: np.ndarray
    declaration: AdapterDeclaration
    diagnostics: dict[str, Any]


class OutputAdapter:
    declaration: AdapterDeclaration

    def __init__(self, declaration: AdapterDeclaration, *, epsilon: float = 1e-6, clip_min=None, clip_max=None):
        self.declaration = declaration
        self.epsilon = float(epsilon)
        self.clip_min = clip_min
        self.clip_max = clip_max

    def _single_channel(self, raw: np.ndarray) -> np.ndarray:
        value = np.asarray(raw)
        value = np.squeeze(value)
        if value.ndim == 3 and value.shape[-1] in (1, 3): value = value.mean(axis=-1)
        if value.ndim == 3 and value.shape[0] in (1, 3): value = value.mean(axis=0)
        if value.ndim != 2:
            raise ValueError(f"Adapter {self.declaration.route} expected one dense channel, got {value.shape}")
        return value.astype(np.float32)

    def adapt(self, raw: np.ndarray) -> AdaptedPrediction:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        return asdict(self.declaration)


class MetricDepthAdapter(OutputAdapter):
    def adapt(self, raw: np.ndarray) -> AdaptedPrediction:
        depth = self._single_channel(raw)
        bad = ~np.isfinite(depth)
        if bad.any():
            raise ValueError(f"{self.declaration.route}: raw output has {int(bad.sum())} NaN/Inf values")
        before = depth.copy()
        depth = np.maximum(depth, self.epsilon)
        if self.clip_min is not None: depth = np.maximum(depth, float(self.clip_min))
        if self.clip_max is not None: depth = np.minimum(depth, float(self.clip_max))
        return AdaptedPrediction(depth.astype(np.float32), self.declaration, {
            "nonpositive_clipped": int((before <= 0).sum()), "raw_shape": list(np.asarray(raw).shape),
        })


class InverseDepthAdapter(OutputAdapter):
    def adapt(self, raw: np.ndarray) -> AdaptedPrediction:
        inverse = self._single_channel(raw)
        if not np.isfinite(inverse).all():
            raise ValueError(f"{self.declaration.route}: raw inverse depth contains NaN/Inf")
        clipped = np.maximum(inverse, self.epsilon)
        depth = 1.0 / clipped
        if self.clip_min is not None: depth = np.maximum(depth, float(self.clip_min))
        if self.clip_max is not None: depth = np.minimum(depth, float(self.clip_max))
        return AdaptedPrediction(depth.astype(np.float32), self.declaration, {
            "inverse_nonpositive_clipped": int((inverse <= 0).sum()), "raw_shape": list(np.asarray(raw).shape),
        })


def create_output_adapter(route: str, **kwargs) -> OutputAdapter:
    """Create one of the four registered route adapters.

    Iris/Lotus-family native outputs are treated as relative inverse depth, so
    raw metric errors are unavailable. SP-DiT restores metric depth and is the
    only default route with native metric scale.
    """
    key = route.strip().lower().replace("_", "-")
    declarations = {
        "iris-lotus": AdapterDeclaration("iris-lotus", "relative_disparity", "larger-is-nearer", False, "depth=1/max(disparity,epsilon)", "positive epsilon; optional configured depth clip"),
        "adapter-only": AdapterDeclaration("adapter-only", "lotus_relative_disparity", "larger-is-nearer", False, "depth=1/max(disparity,epsilon)", "positive epsilon; optional configured depth clip"),
        "adapter+u-net": AdapterDeclaration("adapter+u-net", "lotus_relative_disparity", "larger-is-nearer", False, "depth=1/max(disparity,epsilon)", "positive epsilon; optional configured depth clip"),
        "adapter-unet": AdapterDeclaration("adapter+u-net", "lotus_relative_disparity", "larger-is-nearer", False, "depth=1/max(disparity,epsilon)", "positive epsilon; optional configured depth clip"),
        "sp-dit": AdapterDeclaration("sp-dit", "metric_depth_m", "larger-is-farther", True, "identity", "positive epsilon; optional configured depth clip"),
        "spdit": AdapterDeclaration("sp-dit", "metric_depth_m", "larger-is-farther", True, "identity", "positive epsilon; optional configured depth clip"),
    }
    if key not in declarations:
        raise KeyError(f"Unknown route {route!r}; expected Iris/Lotus, Adapter-only, Adapter+U-Net, or SP-DiT")
    declaration = declarations[key]
    cls = MetricDepthAdapter if declaration.metric_scale_exists else InverseDepthAdapter
    return cls(declaration, **kwargs)
