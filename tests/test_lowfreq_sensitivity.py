from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ms2_eval.io import ManifestSample
from tools import diagnose_lowfreq_sensitivity as tool


def summary(ours, ref):
    rise = lambda v: {f'{tool.DECISION_AMPLITUDE:.2f}': dict(rise=dict(rmse_near=v))}
    return dict(tmax1=rise(ours), newcrf=rise(ref))


class LowFrequencyField(unittest.TestCase):
    def test_field_is_smooth_normalised_and_seeded(self):
        field = tool.low_field('2021-08-06-10-59-33_000100', 0, (256, 640))
        self.assertAlmostEqual(field.mean(), 0., places=6)
        self.assertAlmostEqual(field.std(), 1., places=6)
        self.assertLess(np.abs(np.diff(field, axis=1)).max(), .2)
        self.assertTrue(np.array_equal(field, tool.low_field('2021-08-06-10-59-33_000100', 0, (256, 640))))
        self.assertFalse(np.allclose(field, tool.low_field('2021-08-06-10-59-33_000100', 1, (256, 640))))

    def test_perturbation_size_follows_the_frame_range(self):
        rng = np.random.default_rng(0)
        raw = rng.integers(20000, 30000, (256, 640)).astype(np.uint16)
        lo, hi = np.percentile(raw, (1, 99))
        moved = tool.perturb(raw, tool.low_field('x', 0, raw.shape), .1)
        self.assertEqual(moved.dtype, np.uint16)
        delta = moved.astype(np.float64)-raw
        self.assertAlmostEqual(delta.std()/(hi-lo), .1, delta=.002)
        self.assertLess(abs(delta.mean()), 1.)

    def test_inputs_round_trip_and_keep_everything_but_the_path(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            raw = np.random.default_rng(1).integers(20000, 30000, (256, 640)).astype(np.uint16)
            Image.fromarray(raw).save(d/'a.png')
            s = ManifestSample('seq_a', str(d/'a.png'), str(d/'gt.png'), 'seq', 'day', 'val', 7, 'a road')
            sets = tool.write_inputs([s], d/'out')
            self.assertEqual(len(sets), 1+len(tool.AMPLITUDES)*tool.FIELDS)
            self.assertIs(sets['clean'][0], s)
            moved = sets['a0.10_k0'][0]
            self.assertEqual((moved.sample_id, moved.manifest_line, moved.caption, moved.thermal_gt_path),
                             (s.sample_id, s.manifest_line, s.caption, s.thermal_gt_path))
            self.assertNotEqual(moved.thermal_path, s.thermal_path)
            self.assertFalse(np.array_equal(np.asarray(Image.open(moved.thermal_path)), raw))

    def test_unchanged_prediction_has_no_change(self):
        gt = np.where(np.random.default_rng(2).random((32, 64)) < .3, np.linspace(5, 70, 64)[None], 0.).astype(np.float32)
        raw = np.log(np.linspace(5, 70, 64)[None].repeat(32, 0)).astype(np.float32)
        row, aligned = tool.frame_row(raw, gt, 'ssi_log', None)
        again, _ = tool.frame_row(raw, gt, 'ssi_log', aligned)
        self.assertEqual(again['change_near'], 0.)
        self.assertEqual(again['rmse'], row['rmse'])

    def test_decision_boundaries(self):
        self.assertEqual(tool.decision(summary(.05, .025))['verdict'], 'sensitive')
        self.assertEqual(tool.decision(summary(.049, .0))['verdict'], 'not_sensitive')
        self.assertEqual(tool.decision(summary(.08, .05))['verdict'], 'not_sensitive')
        self.assertEqual(tool.decision(summary(.06, -.02))['verdict'], 'sensitive')

    def test_control_tolerance(self):
        clean = lambda t, n: dict(tmax1=dict(clean=dict(rmse=t)), newcrf=dict(clean=dict(rmse=n)))
        self.assertTrue(tool.control(clean(2.799, 2.46))['passed'])
        self.assertFalse(tool.control(clean(2.801, 2.46))['passed'])
        self.assertFalse(tool.control(clean(2.799, 2.47))['passed'])


if __name__ == '__main__':
    unittest.main()
