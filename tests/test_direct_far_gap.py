from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import diagnose_direct_far_gap as tool

IDENT = '2021-08-06-10-59-33_000001'


def frame():
    rng = np.random.default_rng(0)
    gt = np.zeros((256, 640), np.float32)
    gt[::2, ::2] = rng.uniform(5, 79, gt[::2, ::2].shape)
    return gt


def row(split, ident, bias, band='60-80', source='newcrf', mode='ssi_log', points=5):
    return dict(split=split, id=ident, source=source, mode=mode, band=band, points=points, bias_m=bias, rmse_m=abs(bias)+1)


class DirectGap(unittest.TestCase):
    def test_perfect_metric_prediction_has_no_bias_in_either_alignment(self):
        gt = frame()
        raw = np.where(gt > 0, gt, 10.).astype(np.float32)
        records = tool.frame_records(raw, gt, IDENT, ('ssi_log', 'ssi'))
        self.assertEqual({r['mode'] for r in records}, {'ssi_log', 'ssi'})
        self.assertTrue(all(abs(r['bias_m']) < .01 for r in records))

    def test_heldout_gt_does_not_change_the_fit(self):
        gt = frame()
        rng = np.random.default_rng(1)
        raw = (np.where(gt > 0, gt, 10.)*rng.uniform(.9, 1.1, gt.shape)).astype(np.float32)
        valid = tool.official_valid_mask(gt, .001, 80.)
        held = tool.source_tool.holdout_mask(valid, IDENT)
        changed = gt.copy()
        changed[held] = np.minimum(changed[held]+5, 79.)
        fit = valid & ~held
        for mode in ('ssi_log', 'ssi'):
            value = np.log(raw) if mode == 'ssi_log' else raw
            np.testing.assert_array_equal(tool.chain.aligned_depth(value, gt, fit, mode),
                                          tool.chain.aligned_depth(value, changed, fit, mode))

    def test_gap_is_val_minus_train_and_groups_by_sequence(self):
        records = [row('train', IDENT, -2.), row('train', '2021-08-06-11-37-46_000002', -4.),
                   row('val', '2021-08-13-16-14-48_000003', -12.)]
        table = tool.summarise(records)
        self.assertEqual(len({r['sequence'] for r in table}), 4)
        gap, = tool.gaps(table)
        self.assertEqual((gap['train_bias_m'], gap['val_bias_m'], gap['gap_m']), (-3., -12., -9.))

    def test_mask_mismatch_is_refused(self):
        model = [row('val', IDENT, -1., source=tool.MODEL)]
        tool.check_masks([row('val', IDENT, -2.)], model)
        with self.assertRaises(ValueError):
            tool.check_masks([row('val', IDENT, -2., points=6)], model)

    def test_control_uses_the_cached_frames(self):
        self.assertTrue(tool.control(dict(a=2., b=3.), dict(a=2.001, b=3.001))['passed'])
        self.assertFalse(tool.control(dict(a=2.), dict(a=2.5))['passed'])
        with self.assertRaises(ValueError):
            tool.control(dict(a=2.), dict(b=2.))

    def test_model_rows_keep_only_heldout_fit_only_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'p.csv'
            head = 'split,id,source,q,stage,mode,mask,band,points,bias_m,rmse_m\n'
            good = f'val,{IDENT},{tool.MODEL},,model,{tool.source_tool.HELDOUT_ALIGN},heldout,60-80,3,-9,10\n'
            other = f'val,{IDENT},{tool.MODEL},,model,{tool.source_tool.HELDOUT_ALIGN},fit,60-80,3,-9,10\n'
            path.write_text(head+good+other)
            rows = tool.model_records(path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['mode'], 'ssi_log')


if __name__ == '__main__':
    unittest.main()
