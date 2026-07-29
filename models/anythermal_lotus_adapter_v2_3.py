"""Adapter V2.3 with AnyThermal semantics and an audited thermal detail skip."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from models.anythermal_lotus_adapter_v2_2 import ProgressiveSkipStage, ResidualRefineBlock
from models.anythermal_lotus_bridge import AnyThermalLotusBridge


class AnyThermalLotusAdapterV23(nn.Module):
    """Fuse frozen AnyThermal features with local detail from the same thermal input."""

    architecture_name = "v2_3_thermal_detail_skip"
    requires_thermal_input = True

    def __init__(
        self,
        *,
        input_channels: int = 768,
        decoder_channels: int = 192,
        detail_channels: int = 64,
        output_channels: int = 4,
        num_features: int = 4,
        native_blocks: int = 1,
        stage_blocks: int = 1,
    ) -> None:
        super().__init__()
        if num_features != 4:
            raise ValueError("Adapter V2.3 expects exactly four feature maps.")
        if min(input_channels, decoder_channels, detail_channels, output_channels) <= 0:
            raise ValueError("All channel counts must be positive.")
        if input_channels % output_channels:
            raise ValueError("input_channels must be divisible by output_channels.")

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

        # The audited thermal tensor is resized to 8x the latent size, then reduced
        # by learned stride-2 convolutions. This preserves local contrast that is
        # absent from the lower-resolution transformer grid.
        self.detail_encoder = nn.Sequential(
            nn.Conv2d(1, detail_channels, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(detail_channels, detail_channels, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(detail_channels, decoder_channels, 3, stride=2, padding=1),
            nn.GELU(),
            ResidualRefineBlock(decoder_channels),
        )
        self.semantic_detail_fusion = nn.Sequential(
            nn.Conv2d(decoder_channels * 2, decoder_channels, 3, padding=1),
            nn.GELU(),
            ResidualRefineBlock(decoder_channels),
        )
        self.to_residual = nn.Conv2d(decoder_channels, output_channels, 3, padding=1)
        self.affine_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(decoder_channels, 128),
            nn.GELU(),
            nn.Linear(128, output_channels * 2),
        )
        nn.init.normal_(self.to_residual.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.to_residual.bias)
        nn.init.zeros_(self.affine_head[-1].weight)
        nn.init.zeros_(self.affine_head[-1].bias)

    def _validate_features(self, features: Sequence[torch.Tensor]) -> None:
        if len(features) != self.num_features:
            raise ValueError(
                f"Adapter V2.3 expects {self.num_features} features, got {len(features)}."
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
                raise ValueError("All V2.3 feature maps must share batch and spatial shape.")

    def _validate_thermal(self, thermal: torch.Tensor, batch_size: int) -> None:
        if thermal.ndim != 4 or thermal.shape[0] != batch_size or thermal.shape[1] not in (1, 3):
            raise ValueError(
                "thermal must be a batch-aligned [B,1,H,W] or [B,3,H,W] tensor, "
                f"got {tuple(thermal.shape)}."
            )
        if not thermal.is_floating_point() or not bool(torch.isfinite(thermal).all()):
            raise ValueError("thermal must be finite floating point.")
        variation = thermal.float().flatten(1).std(dim=1, unbiased=False)
        if bool((variation <= 0).any()):
            raise ValueError("thermal contains a constant/saturated sample.")

    def forward(
        self,
        features: Sequence[torch.Tensor],
        thermal: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        self._validate_features(features)
        self._validate_thermal(thermal, features[0].shape[0])
        if len(target_size) != 2 or min(map(int, target_size)) <= 0:
            raise ValueError(f"target_size must be positive (H,W), got {target_size!r}.")
        target_size = tuple(map(int, target_size))

        laterals = [
            projection(feature)
            for projection, feature in zip(self.lateral_projections, features)
        ]
        semantic = laterals[-1]
        for fusion, lateral in zip(self.native_fusions, reversed(laterals[:-1])):
            semantic = fusion(torch.cat([semantic, lateral], dim=1))
        native_height, native_width = semantic.shape[-2:]
        stage_one_size = (
            min(target_size[0], native_height * 2),
            min(target_size[1], native_width * 2),
        )
        semantic = self.stage_one(semantic, laterals[1], stage_one_size)
        semantic = self.stage_two(semantic, laterals[0], target_size)

        thermal_gray = thermal.mean(dim=1, keepdim=True).to(dtype=semantic.dtype)
        thermal_gray = F.interpolate(
            thermal_gray,
            size=(target_size[0] * 8, target_size[1] * 8),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        detail = self.detail_encoder(thermal_gray)
        if detail.shape[-2:] != target_size:
            raise RuntimeError(f"Detail branch shape {detail.shape[-2:]} != {target_size}.")
        fused = self.semantic_detail_fusion(torch.cat([semantic, detail], dim=1))

        residual = self.to_residual(fused)
        affine = self.affine_head(fused)
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
            raise RuntimeError("Adapter V2.3 output contains NaN/Inf.")
        return output
