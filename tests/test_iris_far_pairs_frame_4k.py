import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import extend_iris_far_pairs as ext


class FrameContinuationTests(unittest.TestCase):
    def test_stream_resumes_after_last_consumed_image(self):
        stride = ext.BATCH * ext.ACCUM
        whole = list(ext.pair.ImageStream(75688, 43, 0, 4000 * stride))
        self.assertEqual(list(ext.stream(75688)), whole[2000 * stride:])

    def test_original_guards_and_incremental_rmse_gain(self):
        original = dict(rmse=3., abs_rel=.07936, a1=.94227)
        base = dict(original, rmse=2.865, abs_rel=.07913, a1=.94321)
        rows = {2000: base, 3000: dict(base, rmse=2.846), 4000: dict(base, rmse=2.8, a1=.940)}
        self.assertEqual(ext.select(rows, original), 2000)
        rows[3000]['rmse'] = base['rmse'] - .02
        self.assertEqual(ext.select(rows, original), 3000)
        rows[4000] = dict(base, rmse=2.8, abs_rel=.080)
        self.assertEqual(ext.select(rows, original), 3000)
        rows[4000]['abs_rel'] = base['abs_rel']
        self.assertEqual(ext.select(rows, original), 4000)
        rows[4000]['rmse'] = rows[3000]['rmse']
        self.assertEqual(ext.select(rows, original), 3000)


if __name__ == '__main__':
    unittest.main()
