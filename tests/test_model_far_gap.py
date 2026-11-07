from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import diagnose_model_far_gap as tool

MODE = tool.source_tool.HELDOUT_ALIGN


def rows(source, biases, stage, q, frames=9, points=99):
    out = []
    for split, bias in biases.items():
        for band, _, _ in tool.source_tool.BANDS:
            out.append(dict(split=split, source=source, q=q, stage=stage, mode=MODE, mask='heldout',
                            band=band, frames=frames, points=points, bias_m=bias, rmse_m=abs(bias)+1))
    return out


class FarGap(unittest.TestCase):
    def test_decomposition_adds_up(self):
        table = rows(tool.TMAX1, dict(train=-3., val=-9.), 'model', None)
        reference = rows('anythermal', dict(train=-1.7, val=-6.85), 'vae', 1.)
        row = tool.decomposition(table, reference)[-1]
        self.assertEqual(row['band'], '60-80')
        self.assertAlmostEqual(row['train_shortfall_m'], -1.3)
        self.assertAlmostEqual(row['train_to_val_m'], -6.)
        self.assertAlmostEqual(row['train_shortfall_m']+row['train_to_val_m'], row['total_m'])

    def test_rows_must_cover_the_same_frames_and_points(self):
        table = rows(tool.TMAX1, dict(train=-3., val=-9.), 'model', None, points=98)
        reference = rows('anythermal', dict(train=-1.7, val=-6.85), 'vae', 1.)
        with self.assertRaises(ValueError):
            tool.decomposition(table, reference)

    def test_control_reads_the_whole_frame_row(self):
        base = dict(split='val', source=tool.CONTROL_MODEL, q=None, stage='model', mode='ssi_log',
                    mask='control_all', band='60-80', frames=9, points=99, rmse_m=9.)
        self.assertTrue(tool.control([dict(base, bias_m=-12.)])['passed'])
        self.assertFalse(tool.control([dict(base, bias_m=-14.)])['passed'])

    def test_heldout_gt_does_not_move_the_fit_alignment(self):
        rng = np.random.default_rng(0)
        gt = np.zeros((256, 640), np.float32)
        gt[::2, ::2] = rng.uniform(5, 79, gt[::2, ::2].shape)
        raw = np.log(np.maximum(gt, 1)).astype(np.float32)+rng.normal(0, .02, gt.shape).astype(np.float32)
        ident = '2021-08-06-10-59-33_000001'
        valid = tool.official_valid_mask(gt, .001, 80.)
        held = tool.source_tool.holdout_mask(valid, ident)
        changed = gt.copy()
        changed[held] = np.minimum(changed[held]+5, 79.)
        pick = lambda records: [r for r in records if r['mode'] == MODE and r['mask'] == 'fit']
        self.assertEqual(pick(tool.frame_records(raw, gt, ident)), pick(tool.frame_records(raw, changed, ident)))


if __name__ == '__main__':
    unittest.main()
