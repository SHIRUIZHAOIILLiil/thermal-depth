from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import diagnose_pred_binned_far as tool


def table(pred, gt):
    totals = {}
    tool.accumulate(totals, 'm', 'val', pred, gt)
    return {(r['binned_by'], r['band']): r for r in tool.rows(totals)}


class PredBinned(unittest.TestCase):
    def test_noise_alone_biases_gt_bins_but_not_prediction_bins(self):
        rng = np.random.default_rng(0)
        gt = rng.uniform(1, 80, 400000)
        pred = np.clip(gt+rng.normal(0, 10, gt.shape), .001, 80.)
        # A flat prior leaves the prediction top bin slightly optimistic, not compressed.
        out = table(pred, gt)
        self.assertLess(out['gt', '60-80']['bias_m'], -1.)
        self.assertGreater(out['prediction', '60-80']['bias_m'], 0.)

    def test_real_compression_stays_negative_in_prediction_bins(self):
        gt = np.linspace(1, 79.9, 40000)
        pred = gt*.8
        out = table(pred, gt)
        self.assertLess(out['gt', '60-80']['bias_m'], -10.)
        self.assertLess(out['prediction', '40-60']['bias_m'], -10.)

    def test_prediction_clipped_to_eighty_stays_in_the_last_bin(self):
        out = table(np.array([80.]), np.array([70.]))
        self.assertEqual(out['prediction', '60-80']['points'], 1)

    def test_pooling_weights_points_not_calls(self):
        totals = {}
        tool.accumulate(totals, 'm', 'val', np.array([70.]), np.array([60.]))
        tool.accumulate(totals, 'm', 'val', np.array([70., 70., 70.]), np.array([70., 70., 70.]))
        out = {(r['binned_by'], r['band']): r for r in tool.rows(totals)}
        self.assertEqual(out['prediction', '60-80']['points'], 4)
        self.assertEqual(out['prediction', '60-80']['bias_m'], 2.5)

    def test_control_needs_both_values(self):
        self.assertTrue(tool.control(tool.NEWCRF_VAL_FAR_BIAS_M, 2.7, 2.7)['passed'])
        self.assertFalse(tool.control(-9.5, 2.7, 2.7)['passed'])
        self.assertFalse(tool.control(tool.NEWCRF_VAL_FAR_BIAS_M, 2.7, 2.71)['passed'])


if __name__ == '__main__':
    unittest.main()
