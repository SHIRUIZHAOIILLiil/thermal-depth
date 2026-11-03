import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.train_iris_cross_region_pairs import cross_region_pairs


class CrossRegionTests(unittest.TestCase):
    def test_range_and_spatial_boundaries(self):
        gt = np.full((1, 70), np.nan)
        gt[0, [0, 1, 63, 64, 65]] = [10, 30, 50, 40, 80]
        np.testing.assert_array_equal(cross_region_pairs(gt, 43), [[0, 64]])
        self.assertEqual(cross_region_pairs(gt[:, :64], 43).shape, (0, 2))
        self.assertEqual(cross_region_pairs(gt[:, 63:], 43).shape, (0, 2))

    def test_repeatable_unique_bounded_pairs(self):
        gt = np.full((128, 32), 20.)
        gt[64:] = 60.
        pairs = cross_region_pairs(gt, 43)
        self.assertEqual(len(pairs), 256)
        self.assertEqual(len(np.unique(pairs, axis=0)), 256)
        a, b = pairs.T
        self.assertTrue(np.all(gt.flat[a] == 20))
        self.assertTrue(np.all(gt.flat[b] == 60))
        self.assertTrue(np.all(b // 32 - a // 32 >= 64))
        np.testing.assert_array_equal(pairs, cross_region_pairs(gt, 43))


if __name__ == '__main__':
    unittest.main()
