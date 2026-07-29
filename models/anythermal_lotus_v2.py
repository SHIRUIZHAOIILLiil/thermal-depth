"""Adapter V2 condition-latent distillation primitives.

The dense teacher is produced only from the thermal input.  Sparse MS2 depth
is intentionally absent from every API in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Union

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from models.anythermal_encoder import AnyThermalEncoder


ThermalSource = Union[str, Path, Image.Image, np.ndarray]


@dataclass(frozen=True)
class ThermalLotusInput:
    """The shared audited thermal image used to create a Lotus VAE teacher."""

    tensor: torch.Tensor
    rgb_image: Image.Image
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class ConditionDistillationOutput:
    """Adapter prediction, frozen teacher, differentiable loss and diagnostics."""

    prediction: torch.Tensor
    target: torch.Tensor
    loss: torch.Tensor
    diagnostics: Dict[str, Any]


def condition_distillation_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    cosine_weight: float = 0.1,
    channel_stats_weight: float = 0.1,
    spatial_gradient_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """Return separately auditable V2.1 condition-distillation losses."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(
            f"prediction/target must share [B,C,H,W], got {prediction.shape}/{target.shape}."
        )
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("prediction and target must be finite.")
    if min(cosine_weight, channel_stats_weight, spatial_gradient_weight) < 0:
        raise ValueError("Composite loss weights must be non-negative.")

    pred = prediction.float()
    ref = target.detach().float()
    latent_mse = F.mse_loss(pred, ref)
    cosine_loss = 1.0 - F.cosine_similarity(
        pred.flatten(1), ref.flatten(1), dim=1, eps=1e-8
    ).mean()
    pred_mean = pred.mean(dim=(2, 3))
    ref_mean = ref.mean(dim=(2, 3))
    pred_std = pred.std(dim=(2, 3), unbiased=False)
    ref_std = ref.std(dim=(2, 3), unbiased=False)
    channel_stats_loss = F.mse_loss(pred_mean, ref_mean) + F.mse_loss(pred_std, ref_std)
    pred_dx, ref_dx = pred[..., 1:] - pred[..., :-1], ref[..., 1:] - ref[..., :-1]
    pred_dy, ref_dy = pred[..., 1:, :] - pred[..., :-1, :], ref[..., 1:, :] - ref[..., :-1, :]
    spatial_gradient_loss = F.l1_loss(pred_dx, ref_dx) + F.l1_loss(pred_dy, ref_dy)
    total = (
        latent_mse
        + cosine_weight * cosine_loss
        + channel_stats_weight * channel_stats_loss
        + spatial_gradient_weight * spatial_gradient_loss
    )
    return {
        "total": total,
        "latent_mse": latent_mse,
        "cosine_loss": cosine_loss,
        "channel_stats_loss": channel_stats_loss,
        "spatial_gradient_loss": spatial_gradient_loss,
    }


def _read_thermal_array(source: ThermalSource) -> Tuple[np.ndarray, str]:
    if isinstance(source, (str, Path)):
        with Image.open(source) as image:
            return np.asarray(image).copy(), image.mode
    if isinstance(source, Image.Image):
        return np.asarray(source).copy(), source.mode
    if isinstance(source, np.ndarray):
        return np.asarray(source).copy(), "not-applicable"
    raise TypeError(f"Unsupported thermal source type: {type(source).__name__}.")


def thermal_to_lotus_input(
    source: ThermalSource,
    *,
    processing_res: int = 768,
) -> ThermalLotusInput:
    """Decode high-bit thermal data and build the Lotus VAE input in ``[-1,1]``.

    The original array is decoded before RGB replication.  In particular, this
    function never calls ``convert('RGB')`` on a raw ``I;16`` image.
    """

    if processing_res < 0:
        raise ValueError(f"processing_res must be non-negative, got {processing_res}.")
    raw, pil_mode = _read_thermal_array(source)
    if raw.ndim == 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    if raw.ndim != 2:
        raise ValueError(f"Expected one-channel thermal array [H,W], got {raw.shape}.")
    if raw.size == 0 or not np.isfinite(raw).all():
        raise ValueError("Thermal input must be non-empty and finite.")

    converted = AnyThermalEncoder._array_to_uint8(raw)
    converted_min = int(converted.min())
    converted_max = int(converted.max())
    converted_std = float(converted.astype(np.float64).std())
    if converted_min == converted_max or converted_std == 0.0:
        raise ValueError("Converted thermal input is constant; refusing a saturated teacher.")
    if converted_min == 255 and converted_max == 255:
        raise ValueError("Converted thermal input is all white; refusing a saturated teacher.")

    rgb = np.repeat(converted[..., None], 3, axis=-1)
    rgb_image = Image.fromarray(rgb, mode="RGB")
    tensor = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor / 127.5 - 1.0

    original_height, original_width = raw.shape
    if processing_res > 0:
        scale = min(processing_res / original_width, processing_res / original_height)
        new_height = max(1, int(original_height * scale))
        new_width = max(1, int(original_width * scale))
        tensor = F.interpolate(
            tensor,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    diagnostics: Dict[str, Any] = {
        "pil_mode": pil_mode,
        "raw_dtype": str(raw.dtype),
        "raw_shape": list(raw.shape),
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "raw_mean": float(raw.astype(np.float64).mean()),
        "raw_std": float(raw.astype(np.float64).std()),
        "converted_uint8_min": converted_min,
        "converted_uint8_max": converted_max,
        "converted_uint8_mean": float(converted.astype(np.float64).mean()),
        "converted_uint8_std": converted_std,
        "converted_all_white": bool(np.all(converted == 255)),
        "rgb_channels_equal": bool(
            np.array_equal(rgb[..., 0], rgb[..., 1])
            and np.array_equal(rgb[..., 1], rgb[..., 2])
        ),
        "lotus_tensor_shape": list(tensor.shape),
        "lotus_tensor_min": float(tensor.min()),
        "lotus_tensor_max": float(tensor.max()),
        "lotus_tensor_std": float(tensor.std()),
        "processing_res": int(processing_res),
    }
    return ThermalLotusInput(tensor=tensor, rgb_image=rgb_image, diagnostics=diagnostics)


def encode_seeded_condition_latent(
    vae: Any,
    lotus_input: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Backward-compatible wrapper for a reproducible posterior sample."""

    return encode_condition_latent(
        vae,
        lotus_input,
        posterior="sample",
        seed=seed,
    )


def encode_condition_latent(
    vae: Any,
    lotus_input: torch.Tensor,
    *,
    posterior: str = "mode",
    seed: int | None = None,
) -> torch.Tensor:
    """Encode a dense thermal tensor with an explicit posterior policy.

    ``mode``/``mean`` are deterministic distillation teachers. ``sample``
    reproduces upstream Lotus conditioning and requires an explicit seed.
    """

    if lotus_input.ndim != 4 or lotus_input.shape[1] != 3:
        raise ValueError(
            f"lotus_input must have shape [B,3,H,W], got {tuple(lotus_input.shape)}."
        )
    if not lotus_input.is_floating_point() or not bool(torch.isfinite(lotus_input).all()):
        raise ValueError("lotus_input must be finite floating point.")
    posterior_policy = posterior
    if posterior_policy not in {"mode", "mean", "sample"}:
        raise ValueError(f"posterior must be mode, mean, or sample; got {posterior!r}.")
    if posterior_policy == "sample" and (
        not isinstance(seed, int) or seed < 0 or seed >= 2**63
    ):
        raise ValueError("sample posterior requires seed in [0, 2**63 - 1].")

    parameter = next(vae.parameters(), None) if hasattr(vae, "parameters") else None
    device = parameter.device if parameter is not None else lotus_input.device
    dtype = parameter.dtype if parameter is not None else lotus_input.dtype
    model_input = lotus_input.to(device=device, dtype=dtype)
    with torch.no_grad():
        posterior_distribution = vae.encode(model_input).latent_dist
        if posterior_policy == "sample":
            generator = torch.Generator(device=device).manual_seed(seed)
            latent = posterior_distribution.sample(generator=generator)
        elif posterior_policy == "mode":
            latent = posterior_distribution.mode()
        else:
            latent = posterior_distribution.mean
        latent = latent * vae.config.scaling_factor
    if latent.ndim != 4 or not bool(torch.isfinite(latent).all()):
        raise RuntimeError("Lotus VAE condition latent has invalid shape or NaN/Inf values.")
    return latent.detach()


def _channel_statistics(tensor: torch.Tensor) -> Tuple[list[float], list[float]]:
    values = tensor.detach().float()
    means = values.mean(dim=(0, 2, 3)).cpu().tolist()
    stds = values.std(dim=(0, 2, 3), unbiased=False).cpu().tolist()
    return means, stds


def distill_condition_latent(
    adapter: torch.nn.Module,
    features: Sequence[torch.Tensor],
    teacher_latent: torch.Tensor,
) -> ConditionDistillationOutput:
    """Match an Adapter output to the dense frozen thermal-VAE condition latent."""

    if teacher_latent.ndim != 4 or not bool(torch.isfinite(teacher_latent).all()):
        raise ValueError("teacher_latent must be a finite [B,C,H,W] tensor.")
    target = teacher_latent.detach()
    prediction = adapter(features, target_size=tuple(target.shape[-2:]))
    if prediction.shape != target.shape:
        raise RuntimeError(
            "Adapter V2 output must exactly match the VAE condition latent shape: "
            f"{tuple(prediction.shape)} != {tuple(target.shape)}."
        )
    if not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("Adapter V2 prediction contains NaN or Inf.")

    prediction_float = prediction.float()
    target_float = target.float()
    loss = F.mse_loss(prediction_float, target_float)
    cosine = F.cosine_similarity(
        prediction_float.flatten(1), target_float.flatten(1), dim=1, eps=1e-8
    ).mean()
    pred_centered = prediction_float.flatten(1) - prediction_float.flatten(1).mean(1, keepdim=True)
    target_centered = target_float.flatten(1) - target_float.flatten(1).mean(1, keepdim=True)
    pearson = F.cosine_similarity(pred_centered, target_centered, dim=1, eps=1e-8).mean()
    pred_mean, pred_std = _channel_statistics(prediction)
    target_mean, target_std = _channel_statistics(target)
    diagnostics = {
        "latent_mse": float(loss.detach().cpu()),
        "cosine_similarity": float(cosine.detach().cpu()),
        "pearson_correlation": float(pearson.detach().cpu()),
        "prediction_channel_mean": pred_mean,
        "prediction_channel_std": pred_std,
        "target_channel_mean": target_mean,
        "target_channel_std": target_std,
        "shape": list(target.shape),
    }
    return ConditionDistillationOutput(
        prediction=prediction,
        target=target,
        loss=loss,
        diagnostics=diagnostics,
    )
