from __future__ import annotations

import unittest

import numpy as np

from ms2_eval.lotus_official import align_lotus_disparity_to_ms2_depth, lotus_official_metrics


class LotusOfficialBridgeTests(unittest.TestCase):
    def test_affine_disparity_alignment_recovers_depth(self):
        gt = np.asarray([[1.0, 2.0], [4.0, 8.0]], np.float32)
        gt_disparity = 1.0 / gt
        pred_disparity = 2.5 * gt_disparity + 0.2
        valid = np.ones_like(gt, bool)
        aligned, scale, shift = align_lotus_disparity_to_ms2_depth(
            pred_disparity, gt, valid, min_depth_m=0.1, max_depth_m=80.0)
        np.testing.assert_allclose(aligned, gt, atol=1e-5)
        self.assertAlmostEqual(scale, 0.4, places=5)
        self.assertAlmostEqual(shift, -0.08, places=5)
        metrics = lotus_official_metrics(aligned, gt, valid)
        self.assertLess(metrics["rmse_linear"], 1e-5)
        self.assertAlmostEqual(metrics["delta1_acc"], 1.0)

    def test_invalid_gt_does_not_enter_fit(self):
        gt = np.asarray([[0.0, 1.0], [2.0, 4.0]], np.float32)
        pred = np.asarray([[9999.0, 3.0], [1.5, 0.75]], np.float32)
        valid = gt > 0.1
        aligned_a, _, _ = align_lotus_disparity_to_ms2_depth(pred, gt, valid, min_depth_m=0.1, max_depth_m=80.0)
        pred[0, 0] = 0.001
        aligned_b, _, _ = align_lotus_disparity_to_ms2_depth(pred, gt, valid, min_depth_m=0.1, max_depth_m=80.0)
        np.testing.assert_allclose(aligned_a[valid], aligned_b[valid], atol=1e-6)


if __name__ == "__main__": unittest.main()
