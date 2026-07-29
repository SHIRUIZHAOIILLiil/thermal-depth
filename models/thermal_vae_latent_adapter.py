"""Six-route line e: trainable adapter on the frozen Thermal-VAE latent.

``condition = latent + delta(latent)`` where ``delta`` is a small residual CNN
whose final convolution is zero-initialized, so an untrained adapter is an
exact identity and line e starts at the zero-training thermal baseline
(line c).  This is the single-variable counterpart of line f (AnyThermal
features -> Adapter): same trainable-module role, different feature source.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.conv1(self.activation(self.norm1(value)))
        value = self.conv2(self.activation(self.norm2(value)))
        return value + residual


class ThermalVAELatentAdapter(nn.Module):
    """4-channel latent in, 4-channel condition out, identity at initialization."""

    def __init__(self, latent_channels: int = 4, hidden_channels: int = 256, blocks: int = 6):
        super().__init__()
        if latent_channels <= 0 or hidden_channels % 8 or blocks <= 0:
            raise ValueError(
                f"Invalid adapter configuration: latent={latent_channels}, "
                f"hidden={hidden_channels} (must be divisible by 8), blocks={blocks}"
            )
        self.conv_in = nn.Conv2d(latent_channels, hidden_channels, 3, padding=1)
        self.blocks = nn.ModuleList(ResidualBlock(hidden_channels) for _ in range(blocks))
        self.norm_out = nn.GroupNorm(8, hidden_channels)
        self.activation = nn.SiLU()
        self.conv_out = nn.Conv2d(hidden_channels, latent_channels, 3, padding=1)
        # zero-init: untrained adapter == identity == line c
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4 or latent.shape[1] != self.conv_in.in_channels:
            raise ValueError(
                f"Expected [B,{self.conv_in.in_channels},H,W] latent, got {tuple(latent.shape)}"
            )
        value = self.conv_in(latent)
        for block in self.blocks:
            value = block(value)
        delta = self.conv_out(self.activation(self.norm_out(value)))
        return latent + delta
