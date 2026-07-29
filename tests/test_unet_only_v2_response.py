"""Tests for the V2 U-Net-only dense teacher route."""

from __future__ import annotations

import types
import unittest

import torch

from models.anythermal_lotus_conditioner import AnyThermalDirectConditioner
from tools.short_128_unet_only_v2_response_holdout import (
    gate,
    validate_protocol,
    validate_trainability,
)


class UnetOnlyV2Test(unittest.TestCase):
    def test_direct_conditioner_has_zero_parameters(self) -> None:
        direct = AnyThermalDirectConditioner()
        self.assertEqual(sum(parameter.numel() for parameter in direct.parameters()), 0)

    def test_only_unet_is_trainable(self) -> None:
        direct = AnyThermalDirectConditioner()
        unet = torch.nn.Conv2d(8, 4, 3, padding=1).requires_grad_(True)
        frozen = torch.nn.Conv2d(3, 4, 1).requires_grad_(False)
        validate_trainability(direct, unet, (frozen,))
        frozen.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "support module"):
            validate_trainability(direct, unet, (frozen,))

    def test_protocol_fixes_effective_batch_four(self) -> None:
        args = types.SimpleNamespace(
            micro_batch_size=1,
            gradient_accumulation_steps=4,
            timestep=999,
            optimizer_steps=300,
            log_interval=100,
            learning_rate=1e-6,
            max_grad_norm=1.0,
            response_cosine_loss_weight=0.1,
            response_spatial_gradient_weight=0.5,
            response_multiscale_gradient_weight=0.5,
            response_gradient_energy_weight=0.5,
        )
        validate_protocol(args)
        args.gradient_accumulation_steps = 2
        with self.assertRaisesRegex(ValueError, "accumulation 4"):
            validate_protocol(args)

    def test_holdout_gate_is_independent(self) -> None:
        initial = {"mse": 1.0}
        final = {"mse": 0.8, "cosine": 0.85, "gradient_energy_ratio": 0.9}
        self.assertTrue(gate(initial, final, holdout=True)["passed"])
        final["cosine"] = 0.7
        self.assertFalse(gate(initial, final, holdout=True)["passed"])


if __name__ == "__main__":
    unittest.main()
