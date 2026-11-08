from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import diagnose_train_val_gap as tool


def rows(train, val):
    return dict(train=dict(abs_rel=.05, rmse=train, a1=.98), val=dict(abs_rel=.08, rmse=val, a1=.94))


class TrainValGap(unittest.TestCase):
    def test_gap_reduction_boundary(self):
        base = rows(1.48, 2.79)
        self.assertEqual(tool.decision(dict(tmax1=base, candidate=rows(1.70, 2.81)))['next'], 'tune_augmentation')
        self.assertEqual(tool.decision(dict(tmax1=base, candidate=rows(1.60, 2.80)))['next'], 'close_augmentation')

    def test_reduction_sign(self):
        out = tool.decision(dict(tmax1=rows(1.5, 2.8), candidate=rows(1.4, 2.8)))
        self.assertAlmostEqual(out['reduction_m'], -.1)

    def test_control_needs_both_splits(self):
        self.assertTrue(tool.control(rows(tool.ANCHOR['train'], tool.ANCHOR['val']))['passed'])
        self.assertFalse(tool.control(rows(tool.ANCHOR['train']+.01, tool.ANCHOR['val']))['passed'])

    def test_far_rows_keep_heldout_fit_only_60_80(self):
        base = dict(source='m', q=None, stage='model', frames=1, points=1)
        records = [dict(base, split='val', id='a', mode=tool.gap_tool.HELD, mask='heldout', band='60-80', bias_m=-9., rmse_m=10.),
                   dict(base, split='val', id='a', mode=tool.gap_tool.HELD, mask='fit', band='60-80', bias_m=-1., rmse_m=2.)]
        self.assertEqual(tool.far_rows(records), dict(val=dict(bias_m=-9., rmse_m=10.)))


if __name__ == '__main__':
    unittest.main()
