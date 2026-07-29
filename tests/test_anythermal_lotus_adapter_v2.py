"""Unit tests for the Adapter V2.1 spatial decoder and composite loss."""

from __future__ import annotations

import unittest

import torch

from models.anythermal_lotus_adapter_v2 import AnyThermalLotusAdapterV2
from models.anythermal_lotus_v2 import condition_distillation_losses


class AdapterV21Test(unittest.TestCase):
    def make_adapter(self) -> AnyThermalLotusAdapterV2:
        return AnyThermalLotusAdapterV2(
            input_channels=16,
            per_level_channels=8,
            decoder_channels=32,
            native_blocks=1,
            target_blocks=1,
        )

    def test_shape_finite_and_only_adapter_gradients(self) -> None:
        torch.manual_seed(7)
        adapter = self.make_adapter()
        features = [torch.randn(2, 16, 5, 9) for _ in range(4)]
        target = torch.randn(2, 4, 8, 14)
        prediction = adapter(features, target_size=(8, 14))
        losses = condition_distillation_losses(prediction, target)
        losses["total"].backward()

        self.assertEqual(prediction.shape, target.shape)
        self.assertTrue(torch.isfinite(prediction).all())
        gradients = [parameter.grad for parameter in adapter.parameters()]
        self.assertTrue(any(gradient is not None for gradient in gradients))
        self.assertTrue(
            all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
        )
        self.assertIsNone(target.grad)

    def test_no_output_normalization_parameters_and_roundtrip_state(self) -> None:
        adapter = self.make_adapter()
        self.assertFalse(any(key.startswith("output_norm") for key in adapter.state_dict()))
        clone = self.make_adapter()
        clone.load_state_dict(adapter.state_dict(), strict=True)

    def test_composite_loss_components_are_zero_for_identical_tensors(self) -> None:
        target = torch.randn(2, 4, 6, 10)
        losses = condition_distillation_losses(target.clone(), target)
        for name, value in losses.items():
            self.assertAlmostEqual(float(value), 0.0, places=6, msg=name)

    def test_per_sample_outputs_are_not_forced_to_same_std(self) -> None:
        torch.manual_seed(11)
        adapter = self.make_adapter().eval()
        base = [torch.randn(1, 16, 5, 9) for _ in range(4)]
        features = [torch.cat([value, value * 3.0], dim=0) for value in base]
        output = adapter(features, target_size=(8, 14))
        stds = output.std(dim=(1, 2, 3), unbiased=False)
        self.assertNotAlmostEqual(
            float(stds[0].detach()), float(stds[1].detach()), places=4
        )

    def test_gradient_flows_through_frozen_response_model_only_to_adapter(self) -> None:
        torch.manual_seed(13)
        adapter = self.make_adapter()
        frozen_response = torch.nn.Conv2d(8, 4, 3, padding=1).requires_grad_(False)
        features = [torch.randn(1, 16, 5, 9) for _ in range(4)]
        teacher_condition = torch.randn(1, 4, 8, 14)
        noise = torch.randn_like(teacher_condition)
        student_condition = adapter(features, target_size=(8, 14))
        with torch.no_grad():
            teacher_response = frozen_response(torch.cat([teacher_condition, noise], dim=1))
        student_response = frozen_response(torch.cat([student_condition, noise], dim=1))
        torch.nn.functional.mse_loss(student_response, teacher_response).backward()

        self.assertTrue(any(parameter.grad is not None for parameter in adapter.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in frozen_response.parameters()))


if __name__ == "__main__":
    unittest.main()
