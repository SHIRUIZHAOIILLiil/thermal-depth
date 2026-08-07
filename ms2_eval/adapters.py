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


class IdentityDepthAdapter(OutputAdapter):
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


class RelativeDepthAdapter(OutputAdapter):
    """Depth-oriented output on an arbitrary affine scale (MiDaS/DPT-style).

    Sign and scale are meaningless until the per-image scale+shift fit against GT
    depth, which is how the upstream protocol evaluates this family, so values pass
    through untouched -- clamping them positive first would discard the negative
    side of the network's range. ``nonpositive_raw`` records whether that side is
    actually populated; the affine alignment clamps its own output.
    """

    def adapt(self, raw: np.ndarray) -> AdaptedPrediction:
        value = self._single_channel(raw)
        if not np.isfinite(value).all():
            raise ValueError(f"{self.declaration.route}: raw output contains NaN/Inf")
        if self.clip_min is not None or self.clip_max is not None:
            raise ValueError(f"{self.declaration.route}: depth clips are meaningless before affine alignment")
        return AdaptedPrediction(value, self.declaration, {
            "nonpositive_raw": int((value <= 0).sum()), "raw_shape": list(np.asarray(raw).shape),
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


RELATIVE_DEPTH_CONVERSION = "identity; affine alignment supplies sign and scale"
_ADAPTER_BY_CONVERSION = {
    "identity": IdentityDepthAdapter,
    RELATIVE_DEPTH_CONVERSION: RelativeDepthAdapter,
    "depth=1/max(disparity,epsilon)": InverseDepthAdapter,
}


def create_output_adapter(route: str, **kwargs) -> OutputAdapter:
    """Create one of the registered route adapters.

    Iris/Lotus-family native outputs are treated as relative inverse depth, so
    raw metric errors are unavailable. SP-DiT restores metric depth and is the
    only default route with native metric scale.

    AnyThermal is relative like the Lotus family but depth-oriented: its released
    MiDaS/DPT path is evaluated upstream by fitting scale+shift of the raw output
    against GT depth (BridgeMultiSpectralDepth, our ``--align ssi``), never against
    GT disparity, and the fit runs on unclamped network output. Inverting or
    clamping it here would contradict that protocol, so it shares neither the Lotus
    inversion nor SP-DiT's positivity clamp despite matching each on one field --
    which is why the class is chosen by the declared conversion rather than by
    ``metric_scale_exists``.
    """
    key = route.strip().lower().replace("_", "-")
    declarations = {
        "iris-lotus": AdapterDeclaration("iris-lotus", "relative_disparity", "larger-is-nearer", False, "depth=1/max(disparity,epsilon)", "positive epsilon; optional configured depth clip"),
        "adapter-only": AdapterDeclaration("adapter-only", "lotus_relative_disparity", "larger-is-nearer", False, "depth=1/max(disparity,epsilon)", "positive epsilon; optional configured depth clip"),
        "adapter+u-net": AdapterDeclaration("adapter+u-net", "lotus_relative_disparity", "larger-is-nearer", False, "depth=1/max(disparity,epsilon)", "positive epsilon; optional configured depth clip"),
        "adapter-unet": AdapterDeclaration("adapter+u-net", "lotus_relative_disparity", "larger-is-nearer", False, "depth=1/max(disparity,epsilon)", "positive epsilon; optional configured depth clip"),
        "sp-dit": AdapterDeclaration("sp-dit", "metric_depth_m", "larger-is-farther", True, "identity", "positive epsilon; optional configured depth clip"),
        "spdit": AdapterDeclaration("sp-dit", "metric_depth_m", "larger-is-farther", True, "identity", "positive epsilon; optional configured depth clip"),
        "anythermal": AdapterDeclaration("anythermal", "relative_depth_affine", "larger-is-farther", False, RELATIVE_DEPTH_CONVERSION, "none before alignment; alignment clamps to positive epsilon"),
        "anythermal-midas": AdapterDeclaration("anythermal", "relative_depth_affine", "larger-is-farther", False, RELATIVE_DEPTH_CONVERSION, "none before alignment; alignment clamps to positive epsilon"),
    }
    if key not in declarations:
        raise KeyError(f"Unknown route {route!r}; expected Iris/Lotus, Adapter-only, Adapter+U-Net, SP-DiT, or AnyThermal")
    declaration = declarations[key]
    return _ADAPTER_BY_CONVERSION[declaration.conversion_to_positive_depth](declaration, **kwargs)
