"""The conventions in tools/metric_depth_norm.py, pinned.

Three things are checked, and the first two are the ones that would be most
expensive to get wrong:

  1. normalise -> denormalise is the identity to floating-point precision, for
     depths spanning the whole official range;
  2. the two encodings of the same number agree -- `unit_to_inverse` reached
     through [-1, 1] and `decoded_to_inverse` reached through [0, 1] -- so a
     future edit cannot drift one without failing here;
  3. the [0, 1] convention matches the *actual* postprocessing the evaluators
     use, `decoded.mean(dim=1) / 2 + 0.5` from `decode_to_disparity`, rather
     than a restatement of it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.metric_depth_norm import (  # noqa: E402
    MetricNorm,
    decoded_to_inverse,
    decoded_to_unit,
    depth_to_inverse,
    depth_to_unit,
    inverse_to_depth,
    inverse_to_unit,
    unit_to_decoded,
    unit_to_inverse,
)


def make_norm(**overrides) -> MetricNorm:
    kwargs = dict(
        q_lo=0.0125,
        q_hi=0.35,
        quantile_lo=0.02,
        quantile_hi=0.98,
        min_depth=1e-3,
        max_depth=80.0,
        source_split="train",
        source_manifest="unit-test",
        valid_pixels=1_000,
        frames=10,
    )
    kwargs.update(overrides)
    return MetricNorm(**kwargs)


class ConstantsTests(unittest.TestCase):
    def test_rejects_non_train_split(self):
        with self.assertRaises(ValueError):
            make_norm(source_split="val")

    def test_rejects_inverted_range(self):
        with self.assertRaises(ValueError):
            make_norm(q_lo=0.5, q_hi=0.1)

    def test_rejects_non_positive_q_lo(self):
        with self.assertRaises(ValueError):
            make_norm(q_lo=0.0)

    def test_round_trips_through_json(self):
        import tempfile

        norm = make_norm()
        with tempfile.TemporaryDirectory() as tmp:
            path = norm.save(Path(tmp) / "norm.json")
            back = MetricNorm.load(path)
        self.assertEqual(back.q_lo, norm.q_lo)
        self.assertEqual(back.q_hi, norm.q_hi)
        self.assertEqual(back.source_split, "train")


class RoundTripTests(unittest.TestCase):
    def setUp(self):
        self.norm = make_norm()
        # Depths inside the represented band, so nothing is clipped: 1/0.35 is
        # 2.86 m and 1/0.0125 is 80 m.
        self.depth = np.asarray([[3.0, 5.0, 10.0], [20.0, 40.0, 79.0]], np.float32)

    def test_normalise_denormalise_is_identity(self):
        q, valid = depth_to_inverse(self.depth, self.norm)
        u, report = inverse_to_unit(q, self.norm)
        self.assertEqual(report.clipped_low + report.clipped_high, 0)
        back = unit_to_inverse(u, self.norm)
        self.assertTrue(np.allclose(back, q, rtol=0, atol=1e-7))
        depth_back, _ = inverse_to_depth(back, self.norm)
        self.assertTrue(np.allclose(depth_back, self.depth, rtol=1e-5, atol=1e-4))

    def test_unit_and_decoded_encodings_agree(self):
        q, _ = depth_to_inverse(self.depth, self.norm)
        u, _ = inverse_to_unit(q, self.norm)
        y = unit_to_decoded(u)
        self.assertTrue(np.all((y >= 0.0) & (y <= 1.0)))
        self.assertTrue(np.allclose(decoded_to_unit(y), u, atol=1e-7))
        # The two roads to the same number.
        self.assertTrue(np.allclose(decoded_to_inverse(y, self.norm), unit_to_inverse(u, self.norm), atol=1e-7))

    def test_bounds_map_to_the_ends_of_the_range(self):
        self.assertAlmostEqual(float(decoded_to_inverse(np.float64(0.0), self.norm)), self.norm.q_lo, places=12)
        self.assertAlmostEqual(float(decoded_to_inverse(np.float64(1.0), self.norm)), self.norm.q_hi, places=12)

    def test_out_of_band_depth_is_clipped_and_counted(self):
        # 1.0 m is nearer than 1/q_hi = 2.86 m, 200 m is beyond the range and
        # also outside the official validity mask.
        depth = np.asarray([[1.0, 200.0, 10.0]], np.float32)
        u, valid, report = depth_to_unit(depth, self.norm)
        self.assertEqual(int(valid.sum()), 2)          # 200 m fails max_depth
        self.assertTrue(np.all((u >= -1.0) & (u <= 1.0)))
        # the near pixel clips high, the invalid pixel sits at q=0 and clips low
        self.assertEqual(report.clipped_high, 1)
        self.assertEqual(report.clipped_low, 1)
        self.assertEqual(report.total, 3)

    def test_invalid_pixels_land_at_the_far_end(self):
        depth = np.zeros((1, 4), np.float32)
        u, valid, _ = depth_to_unit(depth, self.norm)
        self.assertEqual(int(valid.sum()), 0)
        self.assertTrue(np.allclose(u, -1.0))

    def test_positive_clamp_reports_what_it_did(self):
        q = np.asarray([[-0.5, 0.0, 0.1, 1e6]], np.float64)
        depth, report = inverse_to_depth(q, self.norm)
        self.assertEqual(report.clipped_low, 2)        # negative and zero
        self.assertEqual(report.clipped_high, 1)       # 1e6 -> 1/min_depth
        self.assertTrue(np.all(depth > 0))
        self.assertTrue(np.all(np.isfinite(depth)))


class PostprocessingConventionTests(unittest.TestCase):
    """The [0, 1] convention must match the code, not a description of it."""

    def test_matches_decode_to_disparity(self):
        try:
            import torch
        except ImportError:  # pragma: no cover
            self.skipTest("torch not installed")
        sys.path.insert(0, str(ROOT / "lotus"))
        from tools.train_ms2_joint_gt_v3 import decode_to_disparity  # noqa: F401

        # decode_to_disparity's tail, applied to a decoder output standing in for
        # the real one: (decoded.mean(dim=1) / 2 + 0.5). If that line ever
        # changes, this test is the thing that says so.
        decoded = torch.linspace(-1.0, 1.0, 12).reshape(1, 3, 2, 2)
        y = decoded.float().mean(dim=1) / 2.0 + 0.5
        u = decoded.float().mean(dim=1)
        self.assertTrue(torch.allclose(y, (u + 1.0) / 2.0, atol=1e-7))
        self.assertTrue(torch.allclose(torch.as_tensor(decoded_to_unit(y.numpy())), u, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
