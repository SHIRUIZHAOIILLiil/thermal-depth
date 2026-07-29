"""Loss primitives for V2.4 frozen-U-Net response consistency."""

from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn.functional as F


def seeded_noise(
    shape: Sequence[int],
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    scale: float = 1.0,
) -> torch.Tensor:
    """Create reproducible diffusion noise without touching global RNG state."""

    if not isinstance(seed, int) or seed < 0 or seed >= 2**63:
        raise ValueError("seed must be in [0, 2**63 - 1].")
    if len(shape) != 4 or min(map(int, shape)) <= 0:
        raise ValueError(f"shape must be positive [B,C,H,W], got {tuple(shape)}.")
    if not math.isfinite(float(scale)):
        raise ValueError("scale must be finite.")
    generator = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.randn(
            tuple(map(int, shape)), generator=generator, device=device, dtype=dtype
        )
        * scale
    )


def gradient_energy(value: torch.Tensor) -> torch.Tensor:
    """Return mean absolute horizontal/vertical variation per sample."""

    dx = value[..., 1:] - value[..., :-1]
    dy = value[..., 1:, :] - value[..., :-1, :]
    return dx.abs().mean(dim=(1, 2, 3)) + dy.abs().mean(dim=(1, 2, 3))


def response_consistency_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    cosine_weight: float = 0.1,
    spatial_gradient_weight: float = 0.5,
    multiscale_gradient_weight: float = 0.5,
    gradient_energy_weight: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Match the frozen Lotus U-Net output including geometry-sensitive variation."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(
            f"prediction/target must share [B,C,H,W], got {prediction.shape}/{target.shape}."
        )
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("prediction and target must be finite.")
    if min(
        cosine_weight,
        spatial_gradient_weight,
        multiscale_gradient_weight,
        gradient_energy_weight,
    ) < 0:
        raise ValueError("Response loss weights must be non-negative.")

    pred = prediction.float()
    ref = target.detach().float()
    mse = F.mse_loss(pred, ref)
    cosine_loss = 1.0 - F.cosine_similarity(
        pred.flatten(1), ref.flatten(1), dim=1, eps=1e-8
    ).mean()
    pred_dx, ref_dx = pred[..., 1:] - pred[..., :-1], ref[..., 1:] - ref[..., :-1]
    pred_dy, ref_dy = pred[..., 1:, :] - pred[..., :-1, :], ref[..., 1:, :] - ref[..., :-1, :]
    spatial_gradient_loss = F.l1_loss(pred_dx, ref_dx) + F.l1_loss(pred_dy, ref_dy)
    multiscale_gradient_loss = pred.new_zeros(())
    for scale in (2, 4):
        pred_scaled = F.avg_pool2d(pred, scale, stride=scale)
        ref_scaled = F.avg_pool2d(ref, scale, stride=scale)
        pred_dx = pred_scaled[..., 1:] - pred_scaled[..., :-1]
        ref_dx = ref_scaled[..., 1:] - ref_scaled[..., :-1]
        pred_dy = pred_scaled[..., 1:, :] - pred_scaled[..., :-1, :]
        ref_dy = ref_scaled[..., 1:, :] - ref_scaled[..., :-1, :]
        multiscale_gradient_loss = multiscale_gradient_loss + F.l1_loss(
            pred_dx, ref_dx
        ) + F.l1_loss(pred_dy, ref_dy)
    prediction_energy_per_sample = gradient_energy(pred)
    target_energy_per_sample = gradient_energy(ref)
    prediction_energy = prediction_energy_per_sample.mean()
    target_energy = target_energy_per_sample.mean()
    energy_epsilon = torch.finfo(pred.dtype).eps
    gradient_energy_loss = torch.log(
        (prediction_energy_per_sample + energy_epsilon)
        / (target_energy_per_sample + energy_epsilon)
    ).abs().mean()
    total = (
        mse
        + cosine_weight * cosine_loss
        + spatial_gradient_weight * spatial_gradient_loss
        + multiscale_gradient_weight * multiscale_gradient_loss
        + gradient_energy_weight * gradient_energy_loss
    )
    return {
        "total": total,
        "mse": mse,
        "cosine_loss": cosine_loss,
        "cosine": 1.0 - cosine_loss,
        "spatial_gradient_loss": spatial_gradient_loss,
        "multiscale_gradient_loss": multiscale_gradient_loss,
        "gradient_energy_loss": gradient_energy_loss,
        "prediction_gradient_energy": prediction_energy,
        "target_gradient_energy": target_energy,
        "gradient_energy_ratio": prediction_energy / target_energy.clamp_min(1e-12),
    }
