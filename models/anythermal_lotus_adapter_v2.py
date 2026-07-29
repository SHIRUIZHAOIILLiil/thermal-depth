"""Adapter V2.1 learned spatial decoder for Lotus condition latents."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class ResidualSpatialBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.conv1(F.gelu(self.norm1(value)))
        value = self.conv2(F.gelu(self.norm2(value)))
        return value + residual


class AnyThermalLotusAdapterV2(nn.Module):
    """Fuse four DINO feature maps, then decode a spatial Lotus latent.

    Normalization is confined to residual branches.  The final output has no
    normalization, and a per-sample affine head preserves input-dependent VAE
    channel statistics.
    """

    architecture_name = "v2_1_spatial_decoder"

    def __init__(
        self,
        *,
        input_channels: int = 768,
        per_level_channels: int = 96,
        decoder_channels: int = 256,
        output_channels: int = 4,
        num_features: int = 4,
        native_blocks: int = 2,
        target_blocks: int = 2,
    ) -> None:
        super().__init__()
        if num_features != 4:
            raise ValueError("Adapter V2.1 expects exactly four feature maps.")
        if min(input_channels, per_level_channels, decoder_channels, output_channels) <= 0:
            raise ValueError("All channel counts must be positive.")
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.num_features = num_features
        self.projections = nn.ModuleList(
            [nn.Conv2d(input_channels, per_level_channels, 1) for _ in range(num_features)]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(num_features * per_level_channels, decoder_channels, 3, padding=1),
            nn.GELU(),
        )
        self.native_decoder = nn.Sequential(
            *[ResidualSpatialBlock(decoder_channels) for _ in range(native_blocks)]
        )
        self.target_decoder = nn.Sequential(
            *[ResidualSpatialBlock(decoder_channels) for _ in range(target_blocks)]
        )
        self.to_latent = nn.Conv2d(decoder_channels, output_channels, 3, padding=1)
        self.affine_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(decoder_channels, 128),
            nn.GELU(),
            nn.Linear(128, output_channels * 2),
        )
        nn.init.zeros_(self.affine_head[-1].weight)
        nn.init.zeros_(self.affine_head[-1].bias)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        if len(features) != self.num_features:
            raise ValueError(
                f"Adapter V2.1 expects {self.num_features} features, got {len(features)}."
            )
        if len(target_size) != 2 or min(map(int, target_size)) <= 0:
            raise ValueError(f"target_size must be positive (H,W), got {target_size!r}.")
        reference_shape = None
        projected = []
        for index, (feature, projection) in enumerate(zip(features, self.projections)):
            if feature.ndim != 4 or feature.shape[1] != self.input_channels:
                raise ValueError(
                    f"features[{index}] must be [B,{self.input_channels},H,W], "
                    f"got {tuple(feature.shape)}."
                )
            if not bool(torch.isfinite(feature).all()):
                raise ValueError(f"features[{index}] contains NaN/Inf.")
            shape = (feature.shape[0], feature.shape[-2], feature.shape[-1])
            if reference_shape is None:
                reference_shape = shape
            elif shape != reference_shape:
                raise ValueError("All V2.1 feature maps must share batch and spatial shape.")
            projected.append(projection(feature))

        value = self.native_decoder(self.fuse(torch.cat(projected, dim=1)))
        affine = self.affine_head(value)
        value = F.interpolate(
            value,
            size=tuple(map(int, target_size)),
            mode="bilinear",
            align_corners=False,
        )
        value = self.target_decoder(value)
        output = self.to_latent(value)
        scale_raw, bias_raw = affine.chunk(2, dim=1)
        scale = 1.0 + 0.25 * torch.tanh(scale_raw).view(
            output.shape[0], self.output_channels, 1, 1
        )
        bias = 0.25 * torch.tanh(bias_raw).view(
            output.shape[0], self.output_channels, 1, 1
        )
        output = output * scale + bias
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("Adapter V2.1 output contains NaN/Inf.")
        return output
