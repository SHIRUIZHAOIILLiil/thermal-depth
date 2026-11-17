"""The per-band AbsRel / delta1 split in tools/val_pixel_gap.py adds up to the evaluator exactly."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ms2_eval.official_protocol import official_depth_errors, official_valid_mask
from tools import val_pixel_gap as tool


def frame(seed, h=48, w=64):
    rng = np.random.default_rng(seed)
    gt = rng.uniform(1., 85., (h, w)).astype(np.float32)
    gt[rng.random((h, w)) < .6] = 0  # sparse LiDAR; also some > 80 m that the mask drops
    valid = official_valid_mask(gt, .001, 80.)
    aligned = {m: np.clip(gt.astype(np.float64)*rng.uniform(.7, 1.4, (h, w)), .001, 80.) for m in tool.MODELS}
    return gt, valid, aligned


class Split(unittest.TestCase):
    def test_cells_of_a_frame_equal_the_evaluator(self):
        cells = tool.Cells()
        for seed in range(5):
            gt, valid, aligned = frame(seed)
            got = cells.add(gt, valid, aligned)
            for m in tool.MODELS:
                ref = official_depth_errors(aligned[m], gt, valid, min_depth=.001, max_depth=80.)
                self.assertAlmostEqual(got[m]['abs_rel'], ref['abs_rel'], delta=tool.EXACT_TOL)
                self.assertAlmostEqual(got[m]['a1'], ref['a1'], delta=tool.EXACT_TOL)

    def test_bands_add_up_to_the_frame_mean_gap(self):
        cells, refs = tool.Cells(), {m: [] for m in tool.MODELS}
        for seed in range(6):
            gt, valid, aligned = frame(seed)
            cells.add(gt, valid, aligned)
            for m in tool.MODELS:
                refs[m].append(official_depth_errors(aligned[m], gt, valid, min_depth=.001, max_depth=80.))
        t = cells.table()
        mean = {m: {k: np.mean([r[k] for r in refs[m]]) for k in ('abs_rel', 'a1')} for m in tool.MODELS}
        for m in tool.MODELS:
            self.assertAlmostEqual(t['frame_mean'][m]['abs_rel'], mean[m]['abs_rel'], places=12)
            self.assertAlmostEqual(t['frame_mean'][m]['a1'], mean[m]['a1'], places=12)
        gap = mean['synth']['abs_rel']-mean['newcrf']['abs_rel']
        self.assertAlmostEqual(sum(b['absrel_gap'] for b in t['bands'].values()), gap, places=12)
        self.assertAlmostEqual(sum(c['absrel_gap'] for c in t['cells']), gap, places=12)
        d1 = mean['newcrf']['a1']-mean['synth']['a1']
        self.assertAlmostEqual(sum(b['a1_gap'] for b in t['bands'].values()), d1, places=12)
        if gap > 0:
            self.assertAlmostEqual(sum(b['absrel_gap_share'] for b in t['bands'].values()), 1., places=9)

    def test_a_gap_placed_in_one_band_is_found_there(self):
        cells = tool.Cells()
        for seed in range(4):
            gt, valid, aligned = frame(seed)
            aligned['newcrf'] = aligned['synth'].copy()
            mid = valid & (gt > 20) & (gt <= 40)
            aligned['synth'][mid] = np.clip(aligned['synth'][mid]*1.3, .001, 80.)  # synth worse at 20-40 m only
            cells.add(gt, valid, aligned)
        t = cells.table()
        self.assertGreater(t['bands']['20-40m']['absrel_gap_share'], .5)
        self.assertEqual(t['absrel_reading']['reading'], 'mid_10_40')

    def test_reading_rule(self):
        r = tool.band_reading
        self.assertEqual(r({'0-10m': .6, '10-20m': .1, '20-40m': .1, '40-80m': .2})['reading'], 'near_0_10')
        self.assertEqual(r({'0-10m': .1, '10-20m': .2, '20-40m': .3, '40-80m': .4})['reading'], 'mid_10_40')
        self.assertEqual(r({'0-10m': .2, '10-20m': .1, '20-40m': .1, '40-80m': .6})['reading'], 'far_40_80')
        self.assertEqual(r({'0-10m': .3, '10-20m': .2, '20-40m': .1, '40-80m': .4})['reading'], 'spread')
        self.assertEqual(r({'0-10m': .6, '10-20m': .4, '20-40m': .3, '40-80m': -.3})['reading'], 'ambiguous')


if __name__ == '__main__':
    unittest.main()
