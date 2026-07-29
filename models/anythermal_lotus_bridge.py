"""Zero-parameter bridge between AnyThermal and Lotus-D interfaces."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


class AnyThermalLotusBridge(torch.nn.Module):
    """Apply zero-parameter spatial and grouped-channel transformations."""

    def __init__(self) -> None:
        super().__init__()

    def resize_spatial_features(
        self,
        features: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """Resize `[B,C,H,W]` features using bilinear interpolation."""
        if not torch.is_tensor(features):
            raise TypeError(f"features must be a torch.Tensor, got {type(features)!r}.")
        if features.ndim != 4:
            raise ValueError(
                "features must have shape [B,C,H,W], "
                f"got {tuple(features.shape)}."
            )
        if len(target_size) != 2:
            raise ValueError(
                "target_size must contain (target_height, target_width), "
                f"got {target_size!r}."
            )

        target_height, target_width = target_size
        if target_height <= 0 or target_width <= 0:
            raise ValueError(
                "target height and width must be greater than zero, "
                f"got {target_size}."
            )
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features contains NaN or infinite values.")

        resized = F.interpolate(
            features,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        if not bool(torch.isfinite(resized).all()):
            raise RuntimeError("Spatial resize produced NaN or infinite values.")
        return resized

    def project_channels(
        self,
        features: torch.Tensor,
        output_channels: int = 4,
    ) -> torch.Tensor:
        """Compress channels by averaging contiguous, equally sized groups."""
        if not torch.is_tensor(features):
            raise TypeError(f"features must be a torch.Tensor, got {type(features)!r}.")
        if features.ndim != 4:
            raise ValueError(
                "features must have shape [B,C,H,W], "
                f"got {tuple(features.shape)}."
            )
        if output_channels <= 0:
            raise ValueError(
                f"output_channels must be greater than zero, got {output_channels}."
            )

        batch_size, channels, height, width = features.shape
        if channels % output_channels != 0:
            raise ValueError(
                f"Input channels ({channels}) must be divisible by "
                f"output_channels ({output_channels})."
            )
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features contains NaN or infinite values.")

        channels_per_group = channels // output_channels
        projected = features.reshape(
            batch_size,
            output_channels,
            channels_per_group,
            height,
            width,
        ).mean(dim=2)
        if not bool(torch.isfinite(projected).all()):
            raise RuntimeError("Grouped channel mean produced NaN or infinite values.")
        return projected

    def forward(
        self,
        features: torch.Tensor,
        target_size: Tuple[int, int],
        output_channels: int = 4,
    ) -> torch.Tensor:
        resized = self.resize_spatial_features(features, target_size)
        return self.project_channels(
            resized,
            output_channels=output_channels,
        )