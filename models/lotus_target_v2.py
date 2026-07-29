"""Audited Lotus-G depth-target utilities for Adapter V2.

This module is deliberately independent of the legacy MS2 loader.  Sparse
targets returned by :func:`trunc_disparity_target` contain NaN at invalid
pixels so they cannot accidentally be passed to the VAE before Phase C defines
an audited dense-target strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


LOTUS_TRUNC_QUANTILE = 0.02
LOTUS_NORMALIZATION_EPS = 1e-5


@dataclass(frozen=True)
class TruncDisparityTarget:
    """One per-image Lotus ``trunc_disparity`` target and its audit data."""

    values: torch.Tensor
    disparity: torch.Tensor
    valid_mask: torch.Tensor
    disparity_min: torch.Tensor
    disparity_max: torch.Tensor
    quantile_min: float
    quantile_max: float


@dataclass(frozen=True)
class SeededLatentNoise:
    """A reproducibly sampled VAE target latent and diffusion noise pair."""

    target_latent: torch.Tensor
    noise: torch.Tensor


def _floating_depth(depth: torch.Tensor) -> torch.Tensor:
    if not isinstance(depth, torch.Tensor):
        raise TypeError(f"depth must be a torch.Tensor, got {type(depth).__name__}.")
    if depth.ndim not in (2, 3):
        raise ValueError(
            "depth must describe one image with shape [H,W] or [1,H,W], "
            f"got {tuple(depth.shape)}."
        )
    if depth.ndim == 3 and depth.shape[0] != 1:
        raise ValueError(
            "A 3D depth target must have one channel [1,H,W], "
            f"got {tuple(depth.shape)}."
        )
    if not torch.isfinite(depth).all():
        raise ValueError("depth contains NaN or Inf; sanitize the source data explicitly.")
    if not depth.is_floating_point():
        depth = depth.to(torch.float32)
    return depth


def trunc_disparity_target(
    depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    *,
    trunc_quantile: float = LOTUS_TRUNC_QUANTILE,
    eps: float = LOTUS_NORMALIZATION_EPS,
) -> TruncDisparityTarget:
    """Convert one depth image to the upstream Lotus ``trunc_disparity`` convention.

    Only explicitly valid, positive depth values are inverted and used for
    quantiles.  The valid normalized values reproduce the upstream formula::

        d = 1 / depth
        lo, hi = quantile(d, q), quantile(d, 1-q)
        target = clip(2 * ((d-lo) / (hi-lo+1e-5) - 0.5), -1, 1)

    Invalid output positions are NaN by design.  They must not be filled and
    VAE-encoded without the separate dense-target decision required by Phase C.
    """

    depth = _floating_depth(depth)
    if not 0.0 <= trunc_quantile < 0.5:
        raise ValueError(
            f"trunc_quantile must satisfy 0 <= q < 0.5, got {trunc_quantile}."
        )
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")

    if valid_mask is None:
        valid = depth > 0
    else:
        if not isinstance(valid_mask, torch.Tensor):
            raise TypeError(
                f"valid_mask must be a torch.Tensor, got {type(valid_mask).__name__}."
            )
        if valid_mask.shape != depth.shape:
            raise ValueError(
                "valid_mask must have exactly the same shape as depth, "
                f"got {tuple(valid_mask.shape)} and {tuple(depth.shape)}."
            )
        valid = valid_mask.to(device=depth.device, dtype=torch.bool) & (depth > 0)

    if not bool(valid.any()):
        raise ValueError("No valid positive depth pixels are available for trunc_disparity.")

    valid_disparity = depth[valid].reciprocal()
    quantile_max = 1.0 - trunc_quantile
    disparity_min = torch.quantile(valid_disparity, trunc_quantile)
    disparity_max = torch.quantile(valid_disparity, quantile_max)
    valid_normalized = 2.0 * (
        (valid_disparity - disparity_min) / (disparity_max - disparity_min + eps) - 0.5
    )
    valid_normalized = valid_normalized.clamp(-1.0, 1.0)

    disparity = torch.full_like(depth, torch.nan)
    disparity[valid] = valid_disparity
    values = torch.full_like(depth, torch.nan)
    values[valid] = valid_normalized
    return TruncDisparityTarget(
        values=values,
        disparity=disparity,
        valid_mask=valid,
        disparity_min=disparity_min,
        disparity_max=disparity_max,
        quantile_min=trunc_quantile,
        quantile_max=quantile_max,
    )


def seeded_target_latent_and_noise(
    vae: Any,
    target_values: torch.Tensor,
    *,
    seed: int,
) -> SeededLatentNoise:
    """VAE-encode a finite dense target and reproducibly sample latent/noise.

    This helper intentionally rejects the sparse NaN-bearing result above.  A
    caller may use it only after supplying an audited dense target.
    """

    if not isinstance(target_values, torch.Tensor):
        raise TypeError(
            f"target_values must be a torch.Tensor, got {type(target_values).__name__}."
        )
    if target_values.ndim != 4 or target_values.shape[1] != 3:
        raise ValueError(
            "target_values must have shape [B,3,H,W], "
            f"got {tuple(target_values.shape)}."
        )
    if not target_values.is_floating_point() or not torch.isfinite(target_values).all():
        raise ValueError("target_values must be a finite floating-point dense target.")
    if not isinstance(seed, int) or seed < 0 or seed >= 2**63 - 1:
        raise ValueError("seed must be an integer in [0, 2**63 - 2].")

    latent_generator = torch.Generator(device=target_values.device).manual_seed(seed)
    with torch.no_grad():
        encoded = vae.encode(target_values)
        latent = encoded.latent_dist.sample(generator=latent_generator)
        latent = latent * vae.config.scaling_factor

    noise_generator = torch.Generator(device=latent.device).manual_seed(seed + 1)
    noise = torch.randn(
        latent.shape,
        generator=noise_generator,
        device=latent.device,
        dtype=latent.dtype,
    )
    return SeededLatentNoise(target_latent=latent, noise=noise)
