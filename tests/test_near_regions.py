from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import diagnose_near_regions as tool

IDENT = '2021-08-06-10-59-33_000001'


class NearRegions(unittest.TestCase):
    def test_shares_add_to_one_and_point_at_the_worse_region(self):
        totals = {}
        tool.merge(totals, ('q98_decoder', 'val'), {('edge', '0-10'): (10, 0., 90.), ('interior', '0-10'): (10, 0., 15.)})
        tool.merge(totals, ('newcrf', 'val'), {('edge', '0-10'): (10, 0., 15.), ('interior', '0-10'): (10, 0., 10.)})
        rows = tool.table(totals)
        self.assertAlmostEqual(sum(r['difference_share'] for r in rows), 1.)
        by = {r['region']: r for r in rows}
        self.assertAlmostEqual(by['edge']['difference_share'], .9375)

    def test_different_points_are_refused(self):
        totals = {}
        tool.merge(totals, ('q98_decoder', 'val'), {('edge', '0-10'): (10, 0., 90.)})
        tool.merge(totals, ('newcrf', 'val'), {('edge', '0-10'): (11, 0., 15.)})
        with self.assertRaises(ValueError):
            tool.table(totals)

    def test_offset_split_is_exact(self):
        residual = np.array([1., 1., 3., 3.])
        offset = (residual.size, float(residual.size*residual.mean()**2), float((residual**2).sum()))
        out = tool.offset_share([offset])
        self.assertAlmostEqual(out['frame_offset_rmse_m']**2+out['within_rmse_m']**2, out['rmse_m']**2)
        self.assertAlmostEqual(out['within_rmse_m'], 1.)

    def test_frame_stats_counts_only_heldout_points_below_forty(self):
        rng = np.random.default_rng(0)
        gt = np.zeros((256, 640), np.float32)
        gt[::2, ::2] = rng.uniform(5, 79, gt[::2, ::2].shape)
        gt[1::2, 1::2] = rng.uniform(5, 79, gt[1::2, 1::2].shape)
        rows, offset = tool.frame_stats(gt.copy(), gt, IDENT)
        valid = tool.official_valid_mask(gt, .001, 80.)
        held = tool.source_tool.holdout_mask(valid, IDENT)
        self.assertEqual(offset[0], int((held & (gt < 40.)).sum()))
        self.assertEqual(offset[2], 0.)
        self.assertLessEqual(sum(v[0] for v in rows.values()), offset[0])


if __name__ == '__main__':
    unittest.main()
