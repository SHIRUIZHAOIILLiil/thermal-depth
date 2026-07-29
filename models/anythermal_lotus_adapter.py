"""Trainable AnyThermal-to-Lotus image-condition adapter."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class AnyThermalLotusAdapter(nn.Module):
    """Project four AnyThermal feature maps into a Lotus 4-channel latent."""

    def __init__(
        self,
        *,
        input_channels: int = 768,
        per_level_channels: int = 128,
        fusion_channels: int = 128,
        output_channels: int = 4,
        num_features: int = 4,
        use_output_group_norm: bool = True,
        interpolation_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}.")
        if per_level_channels <= 0:
            raise ValueError(
                f"per_level_channels must be positive, got {per_level_channels}."
            )
        if fusion_channels <= 0:
            raise ValueError(f"fusion_channels must be positive, got {fusion_channels}.")
        if output_channels <= 0:
            raise ValueError(f"output_channels must be positive, got {output_channels}.")
        if num_features != 4:
            raise ValueError("Adapter V0 expects exactly four AnyThermal feature maps.")

        self.input_channels = int(input_channels)
        self.per_level_channels = int(per_level_channels)
        self.fusion_channels = int(fusion_channels)
        self.output_channels = int(output_channels)
        self.num_features = int(num_features)
        self.interpolation_mode = interpolation_mode

        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(input_channels, per_level_channels, kernel_size=1),
                    nn.GELU(),
                )
                for _ in range(num_features)
            ]
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(
                num_features * per_level_channels,
                fusion_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(fusion_channels, output_channels, kernel_size=1),
        )
        self.output_norm = (
            nn.GroupNorm(num_groups=output_channels, num_channels=output_channels)
            if use_output_group_norm
            else nn.Identity()
        )
        self.latent_scale = nn.Parameter(torch.full((1, output_channels, 1, 1), 0.75))
        self.latent_bias = nn.Parameter(torch.zeros(1, output_channels, 1, 1))

    def forward(
        self,
        features: Sequence[torch.Tensor],
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Return a `[B,4,H,W]` Lotus image-condition latent."""
        if len(features) != self.num_features:
            raise ValueError(
                f"Adapter V0 expects {self.num_features} feature maps, "
                f"got {len(features)}."
            )
        if len(target_size) != 2:
            raise ValueError(f"target_size must be (height, width), got {target_size!r}.")
        target_height, target_width = int(target_size[0]), int(target_size[1])
        if target_height <= 0 or target_width <= 0:
            raise ValueError(f"target_size must be positive, got {target_size!r}.")

        resized = []
        reference_batch = None
        for index, (feature, projection) in enumerate(zip(features, self.projections)):
            if not torch.is_tensor(feature):
                raise TypeError(f"features[{index}] must be a tensor.")
            if feature.ndim != 4:
                raise ValueError(
                    f"features[{index}] must have shape [B,C,H,W], "
                    f"got {tuple(feature.shape)}."
                )
            if feature.shape[1] != self.input_channels:
                raise ValueError(
                    f"features[{index}] has {feature.shape[1]} channels; "
                    f"expected {self.input_channels}."
                )
            if reference_batch is None:
                reference_batch = feature.shape[0]
            elif feature.shape[0] != reference_batch:
                raise ValueError("All feature maps must have the same batch size.")
            if not bool(torch.isfinite(feature).all()):
                raise ValueError(f"features[{index}] contains NaN or Inf values.")

            projected = projection(feature)
            projected = F.interpolate(
                projected,
                size=(target_height, target_width),
                mode=self.interpolation_mode,
                align_corners=False if self.interpolation_mode in {"linear", "bilinear", "bicubic", "trilinear"} else None,
            )
            resized.append(projected)

        fused = torch.cat(resized, dim=1)
        output = self.fusion(fused)
        output = self.output_norm(output)
        output = output * self.latent_scale + self.latent_bias
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("Adapter output contains NaN or Inf values.")
        return output
