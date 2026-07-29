"""Phase C/D contract tests for Adapter V2 condition distillation."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from models.anythermal_lotus_v2 import (
    distill_condition_latent,
    encode_condition_latent,
    encode_seeded_condition_latent,
    thermal_to_lotus_input,
)


class _Posterior:
    def __init__(self, mean: torch.Tensor) -> None:
        self.mean = mean

    def sample(self, generator: torch.Generator) -> torch.Tensor:
        return self.mean + torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )

    def mode(self) -> torch.Tensor:
        return self.mean


class _VAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(scaling_factor=0.5)

    def encode(self, values: torch.Tensor) -> SimpleNamespace:
        mean = values.mean(1, keepdim=True).repeat(1, 4, 1, 1)
        return SimpleNamespace(latent_dist=_Posterior(mean))


class _Adapter(torch.nn.Module):
    def __init__(self, *, wrong_shape: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.wrong_shape = wrong_shape

    def forward(self, features, target_size):
        height, width = target_size
        if self.wrong_shape:
            width += 1
        return features[0][:, :1].mean((2, 3), keepdim=True).repeat(
            1, 4, height, width
        ) * self.weight


class ThermalLotusInputTest(unittest.TestCase):
    def test_uint16_is_decoded_before_rgb_replication_and_is_not_all_white(self) -> None:
        raw = np.array([[1000, 2000], [3000, 5000]], dtype=np.uint16)
        result = thermal_to_lotus_input(raw, processing_res=0)

        self.assertEqual(result.tensor.shape, (1, 3, 2, 2))
        self.assertEqual(result.diagnostics["raw_dtype"], "uint16")
        self.assertGreater(result.diagnostics["converted_uint8_std"], 0.0)
        self.assertFalse(result.diagnostics["converted_all_white"])
        self.assertTrue(result.diagnostics["rgb_channels_equal"])
        torch.testing.assert_close(result.tensor[:, 0], result.tensor[:, 1])
        torch.testing.assert_close(result.tensor[:, 1], result.tensor[:, 2])

    def test_constant_thermal_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "constant"):
            thermal_to_lotus_input(np.full((3, 4), 65535, dtype=np.uint16))

    def test_processing_resolution_preserves_aspect_ratio(self) -> None:
        raw = np.arange(8 * 4, dtype=np.uint16).reshape(4, 8)
        result = thermal_to_lotus_input(raw, processing_res=16)
        self.assertEqual(result.tensor.shape, (1, 3, 8, 16))


class ConditionTeacherTest(unittest.TestCase):
    def test_seeded_condition_latent_is_exactly_reproducible(self) -> None:
        source = torch.linspace(-1, 1, 48).reshape(1, 3, 4, 4)
        first = encode_seeded_condition_latent(_VAE(), source, seed=17)
        second = encode_seeded_condition_latent(_VAE(), source, seed=17)
        self.assertTrue(torch.equal(first, second))

    def test_mode_teacher_is_deterministic_and_has_no_sampling_noise(self) -> None:
        source = torch.linspace(-1, 1, 48).reshape(1, 3, 4, 4)
        first = encode_condition_latent(_VAE(), source, posterior="mode")
        second = encode_condition_latent(_VAE(), source, posterior="mode")
        expected = source.mean(1, keepdim=True).repeat(1, 4, 1, 1) * 0.5
        self.assertTrue(torch.equal(first, second))
        torch.testing.assert_close(first, expected)

    def test_adapter_output_matches_teacher_shape_and_only_adapter_gets_grad(self) -> None:
        adapter = _Adapter()
        features = [torch.ones((2, 4, 3, 5))]
        teacher = torch.randn((2, 4, 6, 10), requires_grad=True)
        output = distill_condition_latent(adapter, features, teacher)
        output.loss.backward()

        self.assertEqual(output.prediction.shape, teacher.shape)
        self.assertIsNotNone(adapter.weight.grad)
        self.assertIsNone(teacher.grad)
        self.assertIn("latent_mse", output.diagnostics)
        self.assertIn("cosine_similarity", output.diagnostics)
        self.assertIn("pearson_correlation", output.diagnostics)
        self.assertEqual(len(output.diagnostics["target_channel_mean"]), 4)

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exactly match"):
            distill_condition_latent(
                _Adapter(wrong_shape=True),
                [torch.ones((1, 4, 2, 2))],
                torch.ones((1, 4, 4, 4)),
            )


if __name__ == "__main__":
    unittest.main()
