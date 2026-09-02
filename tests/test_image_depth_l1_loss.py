"""The metre-space image loss, exercised without a cluster, data, or a GPU.

The term is forty lines that had never run. Waiting for a queue slot to find out
whether it raises is the expensive way to learn it; the arithmetic only needs
tensors, and the decoder can be a stub that returns what a real one would.

What is checked here:

  * the per-frame normalisation inverts exactly, which is the part that fails
    silently -- a wrong inversion still produces a number, just the wrong one;
  * a perfect prediction scores zero, and a wrong one scores the metres it is
    wrong by, so the term means what its name says;
  * frames whose norm_type records no bounds (NaN) drop out of the mask instead
    of poisoning the mean;
  * a batch with no returns at all returns zero rather than dividing by it;
  * the far field is weighted more than the near, which is the whole reason the
    term exists.

Run: python -m pytest tests/test_image_depth_l1_loss.py -q
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lotus"))

from train_iris_ms2_g import image_depth_l1_loss  # noqa: E402


class StubVAE:
    """Returns a chosen normalised map, so the test controls the prediction.

    The real decoder maps a latent to [-1, 1] over three channels, and the loss
    takes the channel mean. Anything with that contract is enough here.
    """

    class _Config:
        scaling_factor = 0.18215

    def __init__(self, normalised: torch.Tensor):
        self.config = self._Config()
        self._normalised = normalised

    def decode(self, latent, return_dict=False):
        del latent, return_dict
        return (self._normalised.repeat(1, 3, 1, 1),)


def normalise(depth: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Exactly what the dataset does for trunc_disparity."""
    disparity = 1.0 / depth
    return ((disparity - lo) / (hi - lo + 1e-5) - 0.5) * 2.0


class ImageDepthL1LossTest(unittest.TestCase):
    LO, HI = 0.02, 0.20          # 1/m, i.e. 5 m to 50 m

    def _batch(self, depth: torch.Tensor, valid: torch.Tensor | None = None):
        bounds = torch.tensor([[self.LO, self.HI]] * depth.shape[0])
        if valid is None:
            valid = torch.ones_like(depth)
        return bounds, valid

    def test_perfect_prediction_scores_zero(self):
        depth = torch.tensor([[[[5.0, 10.0], [20.0, 50.0]]]])
        bounds, valid = self._batch(depth)
        vae = StubVAE(normalise(depth, self.LO, self.HI))
        loss, count, stats = image_depth_l1_loss(
            vae, torch.zeros(1, 4, 1, 1), depth, valid, bounds)
        self.assertEqual(count, 4)
        self.assertLess(float(loss), 1e-3, f"perfect prediction scored {float(loss)}")
        self.assertTrue(all(torch.isfinite(torch.tensor(v)).all()
                            for v in (stats["d_hat_min"], stats["d_hat_max"])))

    def test_error_is_measured_in_metres(self):
        truth = torch.tensor([[[[10.0, 10.0]]]])
        predicted = torch.tensor([[[[12.0, 12.0]]]])
        bounds, valid = self._batch(truth)
        vae = StubVAE(normalise(predicted, self.LO, self.HI))
        loss, _, _ = image_depth_l1_loss(
            vae, torch.zeros(1, 4, 1, 1), truth, valid, bounds)
        self.assertAlmostEqual(float(loss), 2.0, places=3)

    def test_far_field_costs_more_than_near(self):
        """The reason the term exists: an equal latent error, unequal metres."""
        offset = 0.02                      # the same step in normalised units
        losses = {}
        for depth_value in (5.0, 50.0):
            truth = torch.full((1, 1, 1, 1), depth_value)
            bounds, valid = self._batch(truth)
            shifted = normalise(truth, self.LO, self.HI) - offset
            vae = StubVAE(shifted)
            loss, _, _ = image_depth_l1_loss(
                vae, torch.zeros(1, 4, 1, 1), truth, valid, bounds)
            losses[depth_value] = float(loss)
        self.assertGreater(losses[50.0], 20 * losses[5.0],
                           f"far/near ratio was only {losses[50.0] / losses[5.0]:.1f}")

    def test_frames_without_bounds_drop_out(self):
        depth = torch.tensor([[[[10.0]]], [[[10.0]]]])
        bounds = torch.tensor([[self.LO, self.HI], [float("nan"), float("nan")]])
        valid = torch.ones_like(depth)
        vae = StubVAE(normalise(depth, self.LO, self.HI))
        loss, count, stats = image_depth_l1_loss(
            vae, torch.zeros(2, 4, 1, 1), depth, valid, bounds)
        self.assertEqual(count, 1, "the NaN-bounds frame should not be scored")
        self.assertTrue(torch.isfinite(loss), "a NaN frame must not poison the mean")
        self.assertEqual(stats["frames_without_bounds"], 1)

    def test_batch_without_returns_is_zero_not_nan(self):
        depth = torch.tensor([[[[10.0]]]])
        bounds, _ = self._batch(depth)
        valid = torch.zeros_like(depth)
        vae = StubVAE(normalise(depth, self.LO, self.HI))
        loss, count, _ = image_depth_l1_loss(
            vae, torch.zeros(1, 4, 1, 1), depth, valid, bounds)
        self.assertEqual(count, 0)
        self.assertEqual(float(loss), 0.0)

    def test_absurd_decoder_output_stays_finite(self):
        """A decoder can leave [-1, 1]; the term must not produce inf or NaN."""
        truth = torch.tensor([[[[10.0]]]])
        bounds, valid = self._batch(truth)
        vae = StubVAE(torch.full((1, 1, 1, 1), -50.0))
        loss, _, stats = image_depth_l1_loss(
            vae, torch.zeros(1, 4, 1, 1), truth, valid, bounds)
        self.assertTrue(torch.isfinite(loss), f"loss was {float(loss)}")
        self.assertLess(stats["d_hat_max"], 1e5)


if __name__ == "__main__":
    unittest.main()
