import unittest
import numpy as np
from tools.far_structure import structure, paired_summary


class FarStructureTest(unittest.TestCase):
    def test_compression_keeps_order(self):
        g = np.linspace(40, 79, 1000)
        result = structure(g, 40 + .3 * (g - 40), g)
        self.assertAlmostEqual(result['spearman'], 1.)
        self.assertAlmostEqual(result['span_ratio'], .3)
        self.assertEqual(result['order_accuracy'], 1.)

    def test_reversal_and_constant(self):
        g = np.linspace(40, 79, 1000)
        reversed_result = structure(g, 120 - g, -g)
        self.assertAlmostEqual(reversed_result['spearman'], -1.)
        self.assertEqual(reversed_result['order_accuracy'], 0.)
        constant = structure(g, np.full_like(g, 50), np.ones_like(g))
        self.assertIsNone(constant['spearman'])
        self.assertEqual(constant['span_ratio'], 0.)
        self.assertEqual(constant['order_tie_share'], 1.)

    def test_ties_sampling_and_narrow_gt(self):
        g = np.repeat(np.linspace(40, 79, 1000), 10)
        a = structure(g, g, g, seed=7)
        b = structure(g, g * 2, g * 2, seed=7)
        self.assertEqual(a['sample_pixels'], 4096)
        self.assertEqual(a['pairs'], b['pairs'])
        self.assertAlmostEqual(a['spearman'], 1.)
        narrow = structure(np.ones(100) * 50, np.ones(100), np.ones(100))
        self.assertIsNone(narrow['span_ratio'])
        self.assertIsNone(narrow['order_accuracy'])

    def fixture(self):
        return [dict(model=model, id=str(i), sequence='s1' if i < 2 else 's2',
                     band='>=40m', pixels=100, mse=value,
                     full_frame_mse_contribution=value / 10)
                for model, values in [('base', [2, 2, 2]), ('ours', [6, 1, 3])]
                for i, value in enumerate(values)]

    def test_signed_gap_and_excess(self):
        result = paired_summary(self.fixture(), 'base')[0]
        self.assertAlmostEqual(result['overall']['mean_mse_gap'], 4 / 3)
        self.assertAlmostEqual(result['by_sequence']['s1']['signed_gap_share'], 3 / 4)
        self.assertAlmostEqual(result['by_sequence']['s1']['positive_excess_share'], 4 / 5)
        self.assertEqual(result['top10pct']['ids'], ['0'])
        weighted = paired_summary(self.fixture(), 'base', 'full_frame_mse_contribution')[0]
        self.assertAlmostEqual(weighted['overall']['mean_mse_gap'], 4 / 30)

    def test_reject_unpaired_and_zero_gap(self):
        rows = self.fixture()
        with self.assertRaisesRegex(ValueError, 'frame IDs differ'):
            paired_summary(rows[:-1], 'base')
        for row in rows:
            row['mse'] = 1.
        result = paired_summary(rows, 'base')[0]
        self.assertIsNone(result['overall']['signed_gap_share'])
        self.assertIsNone(result['overall']['positive_excess_share'])


if __name__ == '__main__':
    unittest.main()
