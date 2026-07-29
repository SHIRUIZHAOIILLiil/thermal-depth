"""Conditioning modules shared by the AnyThermal-to-Lotus training routes."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn

from models.anythermal_lotus_bridge import AnyThermalLotusBridge


class AnyThermalDirectConditioner(nn.Module):
    """Expose the zero-parameter Direct bridge through the Adapter interface.

    The learned Adapter consumes four hidden-state feature maps.  The Direct
    baseline consumes only the final AnyThermal feature map.  Keeping the same
    call signature lets the training wrapper swap the conditioner without
    changing the Lotus loss path.
    """

    route_name = "zero_parameter_direct_bridge"

    def __init__(self, *, output_channels: int = 4) -> None:
        super().__init__()
        if output_channels <= 0:
            raise ValueError("output_channels must be positive")
        self.output_channels = int(output_channels)
        self.bridge = AnyThermalLotusBridge()

    def forward(
        self,
        features: Sequence[torch.Tensor],
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        if not features:
            raise ValueError("Direct conditioner requires at least one feature map")
        return self.bridge(
            features[-1],
            target_size=target_size,
            output_channels=self.output_channels,
        )
