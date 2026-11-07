from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import diagnose_error_scales as tool


class Scales(unittest.TestCase):
    def test_parts_add_to_the_whole(self):
        rng = np.random.default_rng(0)
        rows, cols = rng.integers(0, 256, 5000), rng.integers(0, 640, 5000)
        residual = rng.normal(.2, 2, 5000)
        parts = tool.decompose(residual, rows, cols)
        self.assertAlmostEqual(parts.sum(), (residual**2).sum(), places=6)
        self.assertTrue((parts >= -1e-9).all())

    def test_each_scale_lands_in_its_own_part(self):
        rows, cols = np.meshgrid(np.arange(64), np.arange(64), indexing='ij')
        rows, cols = rows.ravel(), cols.ravel()
        cases = dict(frame=np.ones(rows.shape), large=np.where(cols < 32, 1., -1.),
                     mid=np.where(cols % 16 < 8, 1., -1.), fine=np.where(cols % 2 == 0, 1., -1.))
        for index, residual in enumerate(cases.values()):
            parts = tool.decompose(residual, rows, cols)
            self.assertEqual(int(np.argmax(parts)), index)
            self.assertAlmostEqual(parts[index], (residual**2).sum())

    def test_table_shares_sum_to_one(self):
        totals = {}
        for band, _, _ in tool.BANDS:
            tool.merge(totals, 'q98_decoder', {band: (10, np.array([1., 4., 2., 9.]))})
            tool.merge(totals, 'newcrf', {band: (10, np.array([1., 2., 2., 8.]))})
        rows = tool.table(totals)
        near = [r for r in rows if r['band'] == '0-40']
        self.assertAlmostEqual(sum(r['difference_share'] for r in near), 1.)
        self.assertAlmostEqual(near[1]['difference_share'], 2/3)

    def test_different_points_are_refused(self):
        totals = {}
        for band, _, _ in tool.BANDS:
            tool.merge(totals, 'q98_decoder', {band: (10, np.ones(4))})
            tool.merge(totals, 'newcrf', {band: (11, np.ones(4))})
        with self.assertRaises(ValueError):
            tool.table(totals)


if __name__ == '__main__':
    unittest.main()
