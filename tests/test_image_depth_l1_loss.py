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


class TruncnormInversionTest(unittest.TestCase):
    """The same term under depth normalisation, where there is no reciprocal."""

    LO, HI = 4.0, 60.0           # metres, the quantiles truncnorm records

    def _bounds(self, depth):
        return (torch.tensor([[self.LO, self.HI]] * depth.shape[0]),
                torch.ones_like(depth))

    @staticmethod
    def normalise(depth, lo, hi):
        """Exactly what the dataset does for truncnorm."""
        return ((depth - lo) / (hi - lo + 1e-5) - 0.5) * 2.0

    def test_perfect_prediction_scores_zero(self):
        depth = torch.tensor([[[[5.0, 12.0], [30.0, 55.0]]]])
        bounds, valid = self._bounds(depth)
        vae = StubVAE(self.normalise(depth, self.LO, self.HI))
        loss, count, _ = image_depth_l1_loss(
            vae, torch.zeros(1, 4, 1, 1), depth, valid, bounds,
            norm_type="truncnorm")
        self.assertEqual(count, 4)
        self.assertLess(float(loss), 1e-3, f"perfect prediction scored {float(loss)}")

    def test_error_is_measured_in_metres(self):
        truth = torch.tensor([[[[20.0, 20.0]]]])
        predicted = torch.tensor([[[[23.0, 23.0]]]])
        bounds, valid = self._bounds(truth)
        vae = StubVAE(self.normalise(predicted, self.LO, self.HI))
        loss, _, _ = image_depth_l1_loss(
            vae, torch.zeros(1, 4, 1, 1), truth, valid, bounds,
            norm_type="truncnorm")
        self.assertAlmostEqual(float(loss), 3.0, places=3)

    def test_near_and_far_cost_the_same(self):
        """The point of depth normalisation: one step in latent, one cost in metres.

        Under trunc_disparity the same step costs twenty times more at 50 m than
        at 5 m. Here it must not, and that difference is the whole reason the
        normalisation is worth swapping.
        """
        offset = 0.02
        losses = {}
        for depth_value in (6.0, 50.0):
            truth = torch.full((1, 1, 1, 1), depth_value)
            bounds, valid = self._bounds(truth)
            vae = StubVAE(self.normalise(truth, self.LO, self.HI) - offset)
            loss, _, _ = image_depth_l1_loss(
                vae, torch.zeros(1, 4, 1, 1), truth, valid, bounds,
                norm_type="truncnorm")
            losses[depth_value] = float(loss)
        self.assertAlmostEqual(losses[6.0], losses[50.0], places=4,
                               msg=f"near {losses[6.0]} vs far {losses[50.0]}")

    def test_reciprocal_would_be_caught(self):
        """A disparity inverse applied to depth bounds must not pass as correct.

        This is the actual bug the norm_type argument exists to prevent, so it
        gets a test of its own rather than trusting the branch to be reached.
        """
        depth = torch.tensor([[[[20.0]]]])
        bounds, valid = self._bounds(depth)
        vae = StubVAE(self.normalise(depth, self.LO, self.HI))
        wrong, _, _ = image_depth_l1_loss(
            vae, torch.zeros(1, 4, 1, 1), depth, valid, bounds,
            norm_type="trunc_disparity")
        self.assertGreater(float(wrong), 1.0,
                           "the disparity inverse on depth bounds scored as if correct")

    def test_absurd_decoder_output_stays_positive(self):
        truth = torch.tensor([[[[20.0]]]])
        bounds, valid = self._bounds(truth)
        vae = StubVAE(torch.full((1, 1, 1, 1), -50.0))
        loss, _, stats = image_depth_l1_loss(
            vae, torch.zeros(1, 4, 1, 1), truth, valid, bounds,
            norm_type="truncnorm")
        self.assertTrue(torch.isfinite(loss), f"loss was {float(loss)}")
        self.assertGreater(stats["d_hat_min"], 0.0, "depth must stay positive")

    def test_unknown_norm_type_raises(self):
        depth = torch.tensor([[[[20.0]]]])
        bounds, valid = self._bounds(depth)
        vae = StubVAE(self.normalise(depth, self.LO, self.HI))
        with self.assertRaises(ValueError):
            image_depth_l1_loss(vae, torch.zeros(1, 4, 1, 1), depth, valid, bounds,
                                norm_type="instnorm")


if __name__ == "__main__":
    unittest.main()
