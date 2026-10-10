"""Unit tests for the separated tail-shape and Train-alignment objectives.

Run:
    python -m pytest tests/test_image_depth_tail_loss.py -q
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lotus"))

from train_iris_ms2_g import (  # noqa: E402
    _ssi_log_aligned_depth,
    image_depth_tail_loss,
)


class StubVAE:
    class _Config:
        scaling_factor = 0.18215

    def __init__(self, normalised: torch.Tensor):
        self.config = self._Config()
        self._normalised = normalised

    def decode(self, latent, return_dict=False):
        del latent, return_dict
        return (self._normalised.repeat(1, 3, 1, 1),)


def normalise_log(depth: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    y = (torch.log(depth) - lo) / (hi - lo + 1e-5)
    return (y - 0.5) * 2.0


class TailShapeLossTest(unittest.TestCase):
    LO = math.log(2.0)
    HI = math.log(80.0)

    def _inputs(self, truth: torch.Tensor, predicted: torch.Tensor):
        bounds = torch.tensor([[self.LO, self.HI]] * truth.shape[0])
        valid = torch.ones_like(truth)
        vae = StubVAE(normalise_log(predicted, self.LO, self.HI))
        latent = torch.zeros(truth.shape[0], 4, 1, 1)
        return vae, latent, valid, bounds

    def test_perfect_unaligned_huber_is_zero(self):
        truth = torch.tensor([[[[3.0, 8.0], [20.0, 60.0]]]])
        vae, latent, valid, bounds = self._inputs(truth, truth)
        loss, count, stats = image_depth_tail_loss(
            vae, latent, truth, valid, bounds,
            norm_type="log_truncnorm", loss_type="huber", align="none",
        )
        self.assertEqual(count, 4)
        self.assertEqual(stats["frames_used"], 1)
        self.assertLess(float(loss), 1e-5)

    def test_huber_has_quadratic_core_and_linear_tail(self):
        truth = torch.tensor([[[[10.0, 10.0]]]])
        predicted = torch.tensor([[[[12.0, 20.0]]]])
        vae, latent, valid, bounds = self._inputs(truth, predicted)
        loss, _, _ = image_depth_tail_loss(
            vae, latent, truth, valid, bounds,
            norm_type="log_truncnorm", loss_type="huber", align="none",
            huber_delta_m=5.0,
        )
        # Huber(2) = 2; Huber(10) = 5 * (10 - 2.5) = 37.5.
        self.assertAlmostEqual(float(loss), (2.0 + 37.5) / 2.0, places=3)

    def test_sqrel_matches_official_per_pixel_expression(self):
        truth = torch.tensor([[[[5.0, 20.0]]]])
        predicted = torch.tensor([[[[7.0, 24.0]]]])
        vae, latent, valid, bounds = self._inputs(truth, predicted)
        loss, _, _ = image_depth_tail_loss(
            vae, latent, truth, valid, bounds,
            norm_type="log_truncnorm", loss_type="sqrel", align="none",
        )
        expected = ((2.0 ** 2) / 5.0 + (4.0 ** 2) / 20.0) / 2.0
        self.assertAlmostEqual(float(loss), expected, places=3)

    def test_empty_lidar_mask_is_zero_not_nan(self):
        truth = torch.tensor([[[[10.0, 20.0]]]])
        vae, latent, _, bounds = self._inputs(truth, truth)
        loss, count, stats = image_depth_tail_loss(
            vae, latent, truth, torch.zeros_like(truth), bounds,
            norm_type="log_truncnorm", loss_type="huber", align="none",
        )
        self.assertEqual(count, 0)
        self.assertEqual(stats["frames_used"], 0)
        self.assertEqual(float(loss), 0.0)


class SsiLogAlignmentTest(unittest.TestCase):
    def test_affine_raw_coordinate_is_absorbed(self):
        truth = torch.tensor([[[[2.0, 4.0], [8.0, 16.0]]]])
        scale, shift = 4.0, 0.2
        pred_y = (torch.log(truth) - shift) / scale
        normalised = pred_y * 2.0 - 1.0
        vae = StubVAE(normalised)
        bounds = torch.tensor([[math.log(2.0), math.log(80.0)]])
        loss, count, stats = image_depth_tail_loss(
            vae, torch.zeros(1, 4, 1, 1), truth, torch.ones_like(truth), bounds,
            norm_type="log_truncnorm", loss_type="huber", align="ssi_log",
        )
        self.assertEqual(count, 4)
        self.assertEqual(stats["frames_degenerate"], 0)
        self.assertLess(float(loss), 1e-8)
        self.assertAlmostEqual(stats["alignment_scale_min"], scale, places=5)
        self.assertAlmostEqual(stats["alignment_shift_min"], shift, places=5)

    def test_fit_remains_differentiable(self):
        pred_y = torch.tensor([[[[0.1, 0.3], [0.65, 0.9]]]], requires_grad=True)
        truth = torch.tensor([[[[2.0, 4.5], [9.0, 15.0]]]])
        mask = torch.ones_like(truth, dtype=torch.bool)
        aligned, frame_ok, _, _ = _ssi_log_aligned_depth(pred_y, truth, mask)
        self.assertEqual(frame_ok, [True])
        loss = (aligned - truth).square().mean()
        loss.backward()
        self.assertIsNotNone(pred_y.grad)
        self.assertTrue(torch.isfinite(pred_y.grad).all())
        self.assertGreater(float(pred_y.grad.abs().sum()), 0.0)

    def test_constant_prediction_is_counted_and_skipped(self):
        truth = torch.tensor([[[[2.0, 4.0], [8.0, 16.0]]]])
        vae = StubVAE(torch.zeros_like(truth))
        bounds = torch.tensor([[math.log(2.0), math.log(80.0)]])
        loss, count, stats = image_depth_tail_loss(
            vae, torch.zeros(1, 4, 1, 1), truth, torch.ones_like(truth), bounds,
            norm_type="log_truncnorm", loss_type="huber", align="ssi_log",
        )
        self.assertEqual(count, 0)
        self.assertEqual(stats["frames_used"], 0)
        self.assertEqual(stats["frames_degenerate"], 1)
        self.assertEqual(float(loss), 0.0)

    def test_ssi_log_rejects_non_log_target(self):
        truth = torch.tensor([[[[4.0, 8.0]]]])
        vae = StubVAE(torch.zeros_like(truth))
        with self.assertRaisesRegex(ValueError, "log_truncnorm"):
            image_depth_tail_loss(
                vae, torch.zeros(1, 4, 1, 1), truth, torch.ones_like(truth),
                torch.tensor([[2.0, 80.0]]), norm_type="truncnorm",
                loss_type="huber", align="ssi_log",
            )


if __name__ == "__main__":
    unittest.main()

