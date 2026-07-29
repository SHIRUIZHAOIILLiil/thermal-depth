"""Unit tests for the Adapter V2.2 progressive residual decoder."""

from __future__ import annotations

import unittest

import torch

from models.anythermal_lotus_adapter_v2_2 import AnyThermalLotusAdapterV22
from models.anythermal_lotus_bridge import AnyThermalLotusBridge
from models.anythermal_lotus_v2 import condition_distillation_losses


class AdapterV22Test(unittest.TestCase):
    def make_adapter(self) -> AnyThermalLotusAdapterV22:
        return AnyThermalLotusAdapterV22(
            input_channels=16,
            decoder_channels=32,
            output_channels=4,
            native_blocks=1,
            stage_blocks=1,
        )

    def test_shape_finite_and_full_decoder_gradients(self) -> None:
        torch.manual_seed(17)
        adapter = self.make_adapter()
        features = [torch.randn(2, 16, 5, 9) for _ in range(4)]
        target = torch.randn(2, 4, 12, 22)
        prediction = adapter(features, target_size=(12, 22))
        condition_distillation_losses(prediction, target)["total"].backward()

        self.assertEqual(prediction.shape, target.shape)
        self.assertTrue(torch.isfinite(prediction).all())
        gradients = {
            name: parameter.grad for name, parameter in adapter.named_parameters()
        }
        self.assertTrue(all(gradient is not None for gradient in gradients.values()))
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in gradients.values())
        )
        self.assertIsNone(target.grad)

    def test_initial_output_is_close_to_direct_anchor(self) -> None:
        torch.manual_seed(19)
        adapter = self.make_adapter().eval()
        features = [torch.randn(2, 16, 5, 9) for _ in range(4)]
        output = adapter(features, target_size=(12, 22))
        direct = AnyThermalLotusBridge()(features[-1], (12, 22), output_channels=4)
        initial_delta = float((output - direct).abs().mean().detach())
        self.assertLess(initial_delta, 0.03)
        self.assertGreater(initial_delta, 0.0)

        clone = self.make_adapter().eval()
        clone.load_state_dict(adapter.state_dict(), strict=True)
        self.assertTrue(torch.equal(output, clone(features, target_size=(12, 22))))

    def test_all_four_feature_levels_affect_output(self) -> None:
        torch.manual_seed(23)
        adapter = self.make_adapter().eval()
        features = [torch.randn(1, 16, 5, 9) for _ in range(4)]
        baseline = adapter(features, target_size=(12, 22))
        for index in range(4):
            changed = [feature.clone() for feature in features]
            changed[index] = changed[index] + 0.5
            candidate = adapter(changed, target_size=(12, 22))
            self.assertFalse(torch.equal(baseline, candidate), msg=f"level {index}")

    def test_rejects_mismatched_feature_grids_and_nonfinite_input(self) -> None:
        adapter = self.make_adapter()
        features = [torch.randn(1, 16, 5, 9) for _ in range(4)]
        features[1] = torch.randn(1, 16, 6, 9)
        with self.assertRaisesRegex(ValueError, "share batch and spatial shape"):
            adapter(features, target_size=(12, 22))

        features = [torch.randn(1, 16, 5, 9) for _ in range(4)]
        features[2][0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            adapter(features, target_size=(12, 22))


if __name__ == "__main__":
    unittest.main()
