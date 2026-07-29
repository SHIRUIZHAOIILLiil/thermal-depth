"""Tests for V2.4 response consistency losses."""

from __future__ import annotations

import unittest

import torch

from models.anythermal_lotus_v2_4 import response_consistency_losses, seeded_noise


class ResponseConsistencyV24Test(unittest.TestCase):
    def test_seeded_noise_is_exactly_reproducible_and_local(self) -> None:
        torch.manual_seed(43)
        state = torch.random.get_rng_state()
        first = seeded_noise(
            (2, 4, 8, 20), seed=1234, device=torch.device("cpu"), dtype=torch.float32
        )
        second = seeded_noise(
            (2, 4, 8, 20), seed=1234, device=torch.device("cpu"), dtype=torch.float32
        )
        different = seeded_noise(
            (2, 4, 8, 20), seed=1235, device=torch.device("cpu"), dtype=torch.float32
        )
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, different))
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_identical_responses_have_zero_losses_and_unit_energy_ratio(self) -> None:
        target = torch.randn(2, 4, 16, 40)
        losses = response_consistency_losses(target.clone(), target)
        for name in (
            "total",
            "mse",
            "cosine_loss",
            "spatial_gradient_loss",
            "multiscale_gradient_loss",
            "gradient_energy_loss",
        ):
            self.assertAlmostEqual(float(losses[name]), 0.0, places=6, msg=name)
        self.assertAlmostEqual(float(losses["cosine"]), 1.0, places=6)
        self.assertAlmostEqual(float(losses["gradient_energy_ratio"]), 1.0, places=6)

    def test_energy_loss_penalizes_smoothed_response_and_backpropagates(self) -> None:
        target = torch.tensor(
            [[[[float((row + column) % 2) for column in range(8)] for row in range(8)]]],
            dtype=torch.float32,
        )
        prediction = (target * 0.4).requires_grad_(True)
        losses = response_consistency_losses(
            prediction,
            target,
            cosine_weight=0.0,
            spatial_gradient_weight=0.0,
            multiscale_gradient_weight=0.0,
            gradient_energy_weight=1.0,
        )
        self.assertAlmostEqual(
            float(losses["gradient_energy_ratio"].detach()), 0.4, places=5
        )
        self.assertGreater(float(losses["gradient_energy_loss"].detach()), 0.9)
        losses["gradient_energy_loss"].backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)

    def test_response_loss_backpropagates_through_frozen_model_to_input_only(self) -> None:
        torch.manual_seed(41)
        frozen = torch.nn.Conv2d(8, 4, 3, padding=1).requires_grad_(False)
        student_condition = torch.randn(2, 4, 16, 40, requires_grad=True)
        teacher_condition = torch.randn_like(student_condition)
        noise = torch.randn_like(student_condition)
        with torch.no_grad():
            teacher_response = frozen(torch.cat([teacher_condition, noise], dim=1))
        student_response = frozen(torch.cat([student_condition, noise], dim=1))
        response_consistency_losses(student_response, teacher_response)["total"].backward()
        self.assertIsNotNone(student_condition.grad)
        self.assertTrue(torch.isfinite(student_condition.grad).all())
        self.assertTrue(all(parameter.grad is None for parameter in frozen.parameters()))

    def test_rejects_shape_nonfinite_and_negative_weight(self) -> None:
        value = torch.randn(1, 4, 8, 20)
        with self.assertRaisesRegex(ValueError, "share"):
            response_consistency_losses(value, value[..., :-1])
        bad = value.clone()
        bad[0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            response_consistency_losses(bad, value)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            response_consistency_losses(value, value, spatial_gradient_weight=-1.0)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            response_consistency_losses(value, value, gradient_energy_weight=-1.0)


if __name__ == "__main__":
    unittest.main()
