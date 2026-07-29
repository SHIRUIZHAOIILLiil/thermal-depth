"""Unit tests for the V2.3 thermal-detail Adapter."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from models.anythermal_lotus_adapter_v2_3 import AnyThermalLotusAdapterV23
from models.anythermal_lotus_v2 import condition_distillation_losses, thermal_to_lotus_input


class AdapterV23Test(unittest.TestCase):
    def make_adapter(self) -> AnyThermalLotusAdapterV23:
        return AnyThermalLotusAdapterV23(
            input_channels=16,
            decoder_channels=32,
            detail_channels=16,
            output_channels=4,
        )

    def inputs(self):
        features = [torch.randn(2, 16, 5, 9) for _ in range(4)]
        thermal = torch.randn(2, 3, 96, 176)
        return features, thermal

    def test_shape_finite_and_both_branches_receive_gradients(self) -> None:
        torch.manual_seed(29)
        adapter = self.make_adapter()
        features, thermal = self.inputs()
        target = torch.randn(2, 4, 12, 22)
        output = adapter(features, thermal, target_size=(12, 22))
        condition_distillation_losses(output, target)["total"].backward()

        self.assertEqual(output.shape, target.shape)
        self.assertTrue(torch.isfinite(output).all())
        gradients = {name: parameter.grad for name, parameter in adapter.named_parameters()}
        self.assertTrue(all(gradient is not None for gradient in gradients.values()))
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients.values()))
        self.assertTrue(any(name.startswith("detail_encoder") for name in gradients))
        self.assertTrue(any(name.startswith("lateral_projections") for name in gradients))

    def test_thermal_detail_changes_output_with_fixed_semantics(self) -> None:
        torch.manual_seed(31)
        adapter = self.make_adapter().eval()
        features, thermal = self.inputs()
        first = adapter(features, thermal, target_size=(12, 22))
        changed = thermal.clone()
        changed[..., 20:60, 40:100] += 2.0
        second = adapter(features, changed, target_size=(12, 22))
        self.assertGreater(float((first - second).abs().mean().detach()), 0.0)

    def test_rejects_constant_or_nonfinite_thermal(self) -> None:
        adapter = self.make_adapter()
        features, thermal = self.inputs()
        with self.assertRaisesRegex(ValueError, "constant/saturated"):
            adapter(features, torch.ones_like(thermal), target_size=(12, 22))
        thermal[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite floating point"):
            adapter(features, thermal, target_size=(12, 22))

    def test_state_dict_roundtrip_is_exact(self) -> None:
        torch.manual_seed(37)
        adapter = self.make_adapter().eval()
        features, thermal = self.inputs()
        expected = adapter(features, thermal, target_size=(12, 22))
        clone = self.make_adapter().eval()
        clone.load_state_dict(adapter.state_dict(), strict=True)
        actual = clone(features, thermal, target_size=(12, 22))
        self.assertTrue(torch.equal(expected, actual))

    def test_audited_uint16_thermal_tensor_is_accepted(self) -> None:
        adapter = self.make_adapter().eval()
        features = [torch.randn(1, 16, 5, 9) for _ in range(4)]
        raw = np.linspace(3000, 5000, 96 * 176, dtype=np.uint16).reshape(96, 176)
        audited = thermal_to_lotus_input(raw)
        output = adapter(features, audited.tensor, target_size=(12, 22))
        self.assertEqual(tuple(output.shape), (1, 4, 12, 22))
        self.assertGreater(audited.diagnostics["converted_uint8_std"], 0.0)


if __name__ == "__main__":
    unittest.main()
