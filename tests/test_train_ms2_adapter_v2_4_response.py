"""Protocol tests for formal V2.4 one-epoch training."""

from __future__ import annotations

import unittest

import torch

from tools.train_ms2_adapter_v2_4_response import (
    expected_training_shape,
    multiscale_gradient_loss,
    validate_resume_position,
)


class FormalV24ProtocolTest(unittest.TestCase):
    def test_fixed_manifest_size_has_2611_updates_and_last_batch_one(self) -> None:
        self.assertEqual(
            expected_training_shape(10441, 4),
            {"optimizer_updates": 2611, "last_batch_size": 1},
        )

    def test_training_shape_rejects_invalid_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            expected_training_shape(0, 4)

    def test_resume_position_handles_final_short_batch(self) -> None:
        validate_resume_position(2611, 10441, 10441, 4)
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            validate_resume_position(2611, 10444, 10441, 4)

    def test_multiscale_gradient_is_zero_for_identical_latents(self) -> None:
        target = torch.randn(2, 4, 32, 80)
        self.assertAlmostEqual(
            float(multiscale_gradient_loss(target.clone(), target)), 0.0, places=6
        )


if __name__ == "__main__":
    unittest.main()
