"""CPU checks for the fixed correction, evaluator, fit split and decision."""
import sys
import unittest
import tempfile
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from ms2_eval.official_protocol import evaluate_sample
from tools import iris_far_stretch as s


class StretchTests(unittest.TestCase):
    def test_wrong_checkpoint_step_or_text_fails_before_loading(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'weights.pt'
            base = dict(route='b_thermal_unet', global_step=16000, caption_mode='correct', state_dicts={'unet': {}})
            for payload, arm in ((dict(base, global_step=18000), 'frame'), (base, 'empty')):
                torch.save(payload, path)
                with self.assertRaisesRegex(ValueError, 'Wrong converted'):
                    s.load_arm(None, path, arm)

    def test_raw_tail_and_zero_match_official_with_sparse_gt(self):
        raw = np.arange(20, dtype=np.float32).reshape(4,5)/10
        excess = s.tail(raw)
        moved = s.stretch(raw, excess, .5)
        self.assertIs(s.stretch(raw, excess, 0), raw)
        np.testing.assert_array_equal(raw.ravel()[:16], moved.ravel()[:16])
        self.assertTrue(np.all(np.diff(moved.ravel()) > 0))
        gt = np.zeros_like(raw)
        gt.flat[[0, 2, 5, 9, 19]] = [1, 2, 5, 10, 40]
        for pred, metrics in zip((raw, moved), s.score_pixels(raw, gt, [0,.5])):
            expected = evaluate_sample(pred, gt, align='ssi_log')
            for k in s.OFFICIAL_METRICS:
                self.assertAlmostEqual(metrics[k], expected[k], places=10)

    def test_fit_recovers_known_tail_without_validation(self):
        raw = np.linspace(-1,1,120,dtype=np.float32).reshape(10,12)
        gt = np.exp(2+s.stretch(raw, s.tail(raw), .35))
        sample = SimpleNamespace(sample_id='a')
        scores = s.measure([sample], [('a', raw, gt)], s.GRID)
        self.assertEqual(s.choose_beta(scores), .35)
        flat = dict.fromkeys(s.GRID, scores[0.])
        self.assertEqual(s.choose_beta(flat), 0.)
        with self.assertRaises(ValueError):
            s.measure([sample], [('wrong', raw, gt)], [0.])

    def test_training_sequence_selection_and_gates(self):
        samples = [SimpleNamespace(sample_id=f'{seq}_{i}', manifest_line=n*4+i+1)
                   for n,seq in enumerate(sorted(s.TRAIN)) for i in range(4)]
        chosen = s.sample_fit(samples, 2)
        self.assertEqual(chosen, s.sample_fit(samples, 2))
        self.assertEqual(len({x.sample_id for x in chosen}), 16)
        with self.assertRaises(ValueError):
            s.sample_fit(samples+[SimpleNamespace(sample_id='validation_0')], 2)
        zero = dict(rmse=3., abs_rel=.08, a1=.94)
        improved = dict(rmse=2.94, abs_rel=.0804, a1=.9391)
        bad = dict(improved, abs_rel=.0806)
        self.assertTrue(all(s.gates(zero, improved).values()))
        self.assertFalse(all(s.gates(zero, bad).values()))
        self.assertFalse(s.gates(zero, zero)['rmse'])
        self.assertEqual(s.choose_beta({0.:zero, .25:bad, .5:improved}), .5)


if __name__ == '__main__':
    unittest.main()
