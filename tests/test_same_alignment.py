from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import diagnose_same_alignment as tool


def frame():
    rng = np.random.default_rng(0)
    gt = np.zeros((256, 640), np.float32)
    gt[::2, ::2] = rng.uniform(5, 79, gt[::2, ::2].shape)
    return gt


class SameAlignment(unittest.TestCase):
    def test_exact_output_scores_perfectly_with_any_offset(self):
        gt = frame()
        log_range = 3.2
        raw = ((np.log(np.maximum(gt, 5.))-1.7)/log_range).astype(np.float32)
        result = tool.scores(tool.depth_from_output(raw, log_range), gt, 'ssi')
        self.assertLess(result['abs_rel'], 1e-3)
        self.assertGreater(result['a1'], .999)

    def test_wrong_range_is_worse_than_right_range(self):
        gt = frame()
        log_range = 3.2
        raw = ((np.log(np.maximum(gt, 5.))-1.7)/log_range).astype(np.float32)
        wrong = tool.scores(tool.depth_from_output(raw, log_range*.5), gt, 'ssi')
        self.assertGreater(wrong['rmse'], .1)

    def test_gaps_are_against_newcrf_under_ssi(self):
        out = tool.compare({'newcrf ssi': dict(abs_rel=.08, rmse=2.4, a1=.95),
                            'decoder ssi x1.0': dict(abs_rel=.09, rmse=2.5, a1=.94)})
        row = out['decoder ssi x1.0']
        self.assertAlmostEqual(row['rmse_gap_m'], .1)
        self.assertAlmostEqual(row['absrel_gap'], .01)
        self.assertAlmostEqual(row['a1_gap'], -.01)

    def test_control_needs_both_reference_values(self):
        self.assertTrue(tool.control(tool.NEWCRF_CONTROL_M, 2.7, 2.7001)['passed'])
        self.assertFalse(tool.control(tool.NEWCRF_CONTROL_M+.01, 2.7, 2.7)['passed'])
        self.assertFalse(tool.control(tool.NEWCRF_CONTROL_M, 2.7, 2.71)['passed'])

    def test_sensitivity_contains_the_frozen_range(self):
        self.assertIn(1., tool.SENSITIVITY)


if __name__ == '__main__':
    unittest.main()
