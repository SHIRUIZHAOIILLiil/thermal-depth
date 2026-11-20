"""The zero-shot evaluator's log-space alignment matches the MS2 ssi_log protocol."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'lotus'))
from evaluation.util.alignment import align_depth_least_square, align_depth_least_square_log  # noqa: E402
from ms2_eval.official_protocol import evaluate_sample  # noqa: E402


class LogAlignment(unittest.TestCase):
    def test_exact_log_affine_prediction_is_recovered(self):
        rng = np.random.default_rng(0)
        gt = rng.uniform(.5, 80., (40, 60))
        valid = rng.random((40, 60)) < .7
        pred = (np.log(gt) - .3)/2.5  # a unit map that is affine in log depth
        out, scale, shift = align_depth_least_square_log(gt, pred, valid)
        np.testing.assert_allclose(out[valid], gt[valid], rtol=1e-9)
        self.assertAlmostEqual(float(np.squeeze(scale)), 2.5, places=9)
        self.assertAlmostEqual(float(np.squeeze(shift)), .3, places=9)

    def test_same_metrics_as_ms2_ssi_log(self):
        rng = np.random.default_rng(1)
        gt = rng.uniform(1., 79., (48, 64)).astype(np.float32)
        gt[rng.random(gt.shape) < .5] = 0
        valid = (gt > .001) & (gt < 80.)
        pred = ((np.log(np.maximum(gt, 1e-3)) - 1.)/3. + rng.normal(0, .05, gt.shape)).astype(np.float32)
        out, _, _ = align_depth_least_square_log(gt, pred, valid)
        p = np.clip(out[valid], .001, 80.)
        g = gt[valid].astype(np.float64)
        ref = evaluate_sample(pred, gt, align='ssi_log', min_depth=.001, max_depth=80.)
        self.assertAlmostEqual(float(np.mean(np.abs(g-p)/g)), ref['abs_rel'], places=5)
        self.assertAlmostEqual(float(np.mean(np.maximum(g/p, p/g) < 1.25)), ref['a1'], places=5)

    def test_invalid_and_nonpositive_gt_are_ignored(self):
        gt = np.array([[2., 4.], [0., 8.]])
        valid = np.array([[True, True], [True, True]])  # the 0 must still be excluded
        pred = np.log(np.where(gt > 0, gt, 1.))
        out, _, _ = align_depth_least_square_log(gt, pred, valid)
        np.testing.assert_allclose(out[gt > 0], gt[gt > 0], rtol=1e-9)

    def test_existing_depth_alignment_unchanged(self):
        gt = np.array([[1., 2.], [3., 4.]])
        out, s, b = align_depth_least_square(gt, gt*2+1, np.ones_like(gt, bool))
        np.testing.assert_allclose(out, gt, rtol=1e-9)


if __name__ == '__main__':
    unittest.main()
