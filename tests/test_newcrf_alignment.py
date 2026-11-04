import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.diagnose_newcrf_alignment import compare_frame


class AlignmentTests(unittest.TestCase):
    def test_log_power_law_preserves_mask_and_recovers_depth(self):
        gt = np.array([[0, 4, 10], [60, 75, 80]], np.float32)
        raw = np.sqrt(gt)
        rows, near = compare_frame(raw, gt)
        a,b = rows
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(a[0][-1], 2)
        self.assertLess(b[3], 1e-4)
        self.assertGreater(a[3], 1.)
        self.assertIn(near[0], [1,2])

    def test_nonpositive_valid_depth_fails_without_dropping_pixel(self):
        gt = np.array([[0, 4], [60, 70]], np.float32)
        raw = gt.copy()
        raw[1,0] = 0
        with self.assertRaisesRegex(ValueError, 'positive'):
            compare_frame(raw, gt)


if __name__ == '__main__':
    unittest.main()
