import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools import diagnose_pair_rmse as d


class AttributionTests(unittest.TestCase):
    def test_bias_terms_with_unequal_frame_sizes_and_empty_bins(self):
        # Frame errors [1,3], [-1], [0] against a perfect B.
        n = np.zeros((3, len(d.BANDS)), int)
        n[:, 0] = [2, 1, 1]
        signed = np.zeros_like(n, dtype=float)
        squared = signed.copy()
        signed[:2, 0] = [4, -1]
        squared[:2, 0] = [10, 1]
        r = np.array([np.sqrt(5), 1., 0.])
        a = (n, signed, squared, r)
        b = (n, signed*0, squared*0, r*0)
        rows = d.breakdown(a, b, np.array(['one','two','two']))
        first = rows[0]
        mean = (2/np.sqrt(5) - 1) / (1/np.sqrt(5) + 1)
        self.assertAlmostEqual(first['mean_error_a_m'], mean)
        self.assertAlmostEqual(first['common_bias_gap_m'], mean**2*(1/np.sqrt(5)+1)/3)
        self.assertAlmostEqual(first['within_frame_gap_m'], 1/np.sqrt(5)/3)
        self.assertGreater(first['between_frames_gap_m'], 0)
        self.assertAlmostEqual(first['bin_gap_m'], (np.sqrt(5)+1)/3)
        self.assertIsNone(rows[1]['mean_error_a_m'])

    def test_native_alignment_clamp_and_signed_band_cancellation(self):
        gt = np.array([[0, 4, 5, 12], [40, 60, 77, 80]], dtype=np.float32)
        raw = np.array([[0, .1, .2, .3], [.4, .6, .95, 1]], dtype=np.float32)
        stats = [tuple(np.asarray([x]) for x in d.moments(raw, gt, align)) for align in ('ssi_log', 'ssi')]
        a,b = stats
        np.testing.assert_array_equal(a[0][0], [1,1,1,0,0,1,2])
        rows = d.breakdown(a, b, np.array(['one']))[:len(d.BANDS)]
        contributions = [r['bin_gap_m'] for r in rows]
        self.assertLess(min(contributions), 0)
        self.assertGreater(max(contributions), 0)
        self.assertAlmostEqual(sum(contributions), float(a[3][0]-b[3][0]))


if __name__ == '__main__':
    unittest.main()
