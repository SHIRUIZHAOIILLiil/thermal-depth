"""Adapter V2.2 progressive residual decoder for Lotus condition latents."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from models.anythermal_lotus_bridge import AnyThermalLotusBridge


class ResidualRefineBlock(nn.Module):
    """Pre-normalized residual refinement without output normalization."""

    def __init__(self, channels: int, groups: int = 8) -> None:
        super().__init__()
        if channels % groups:
            raise ValueError(f"channels ({channels}) must be divisible by groups ({groups}).")
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.conv1(F.gelu(self.norm1(value)))
        value = self.conv2(F.gelu(self.norm2(value)))
        return residual + value


class ProgressiveSkipStage(nn.Module):
    """Resize once, fuse a transformer-level skip, then refine spatially."""

    def __init__(self, channels: int, blocks: int) -> None:
        super().__init__()
        if blocks <= 0:
            raise ValueError("Each progressive stage requires at least one block.")
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            *[ResidualRefineBlock(channels) for _ in range(blocks)]
        )

    def forward(
        self,
        value: torch.Tensor,
        skip: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        value = F.interpolate(value, size=target_size, mode="bilinear", align_corners=False)
        skip = F.interpolate(skip, size=target_size, mode="bilinear", align_corners=False)
        return self.refine(self.fuse(torch.cat([value, skip], dim=1)))


class AnyThermalLotusAdapterV22(nn.Module):
    """Decode AnyThermal features progressively around the Direct condition.

    All four frozen AnyThermal transformer levels participate in a deep-to-shallow
    native-grid fusion.  Two learned stages then expand the representation to the
    exact Lotus latent size.  A zero-parameter Direct bridge is retained as an
    additive anchor, while the learned path predicts an unrestricted residual.
    """

    architecture_name = "v2_2_progressive_residual_decoder"

    def __init__(
        self,
        *,
        input_channels: int = 768,
        decoder_channels: int = 192,
        output_channels: int = 4,
        num_features: int = 4,
        native_blocks: int = 1,
        stage_blocks: int = 1,
    ) -> None:
        super().__init__()
        if num_features != 4:
            raise ValueError("Adapter V2.2 expects exactly four feature maps.")
        if min(input_channels, decoder_channels, output_channels) <= 0:
            raise ValueError("All channel counts must be positive.")
        if native_blocks <= 0:
            raise ValueError("native_blocks must be positive.")
        if input_channels % output_channels:
            raise ValueError("input_channels must be divisible by output_channels for Direct anchor.")

        self.input_channels = input_channels
        self.decoder_channels = decoder_channels
        self.output_channels = output_channels
        self.num_features = num_features
        self.direct_bridge = AnyThermalLotusBridge()
        self.lateral_projections = nn.ModuleList(
            [nn.Conv2d(input_channels, decoder_channels, 1) for _ in range(num_features)]
        )
        self.native_fusions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(decoder_channels * 2, decoder_channels, 3, padding=1),
                    nn.GELU(),
                    *[ResidualRefineBlock(decoder_channels) for _ in range(native_blocks)],
                )
                for _ in range(num_features - 1)
            ]
        )
        self.stage_one = ProgressiveSkipStage(decoder_channels, stage_blocks)
        self.stage_two = ProgressiveSkipStage(decoder_channels, stage_blocks)
        self.to_residual = nn.Conv2d(decoder_channels, output_channels, 3, padding=1)
        self.affine_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(decoder_channels, 128),
            nn.GELU(),
            nn.Linear(128, output_channels * 2),
        )

        # Start very close to the Direct route while preserving gradients through
        # the complete decoder on the first optimization step.
        nn.init.normal_(self.to_residual.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.to_residual.bias)
        nn.init.zeros_(self.affine_head[-1].weight)
        nn.init.zeros_(self.affine_head[-1].bias)

    def _validate_features(self, features: Sequence[torch.Tensor]) -> None:
        if len(features) != self.num_features:
            raise ValueError(
                f"Adapter V2.2 expects {self.num_features} features, got {len(features)}."
            )
        reference_shape = None
        for index, feature in enumerate(features):
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
                raise ValueError("All V2.2 feature maps must share batch and spatial shape.")

    def forward(
        self,
        features: Sequence[torch.Tensor],
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        self._validate_features(features)
        if len(target_size) != 2 or min(map(int, target_size)) <= 0:
            raise ValueError(f"target_size must be positive (H,W), got {target_size!r}.")
        target_size = tuple(map(int, target_size))

        laterals = [
            projection(feature)
            for projection, feature in zip(self.lateral_projections, features)
        ]
        value = laterals[-1]
        # Deep-to-shallow fusion uses every transformer level before expansion.
        for fusion, lateral in zip(self.native_fusions, reversed(laterals[:-1])):
            value = fusion(torch.cat([value, lateral], dim=1))

        native_height, native_width = value.shape[-2:]
        stage_one_size = (
            min(target_size[0], native_height * 2),
            min(target_size[1], native_width * 2),
        )
        value = self.stage_one(value, laterals[1], stage_one_size)
        value = self.stage_two(value, laterals[0], target_size)

        residual = self.to_residual(value)
        affine = self.affine_head(value)
        scale_raw, bias_raw = affine.chunk(2, dim=1)
        scale = 1.0 + 0.5 * torch.tanh(scale_raw).view(
            residual.shape[0], self.output_channels, 1, 1
        )
        bias = 0.5 * torch.tanh(bias_raw).view(
            residual.shape[0], self.output_channels, 1, 1
        )
        anchor = self.direct_bridge(
            features[-1], target_size=target_size, output_channels=self.output_channels
        )
        output = anchor * scale + bias + residual
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("Adapter V2.2 output contains NaN/Inf.")
        return output
