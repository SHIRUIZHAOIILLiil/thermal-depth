from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import diagnose_checkpoint_average as tool


class CheckpointAverage(unittest.TestCase):
    def test_float_tensors_average_and_keep_dtype(self):
        states = [dict(a=torch.tensor([1., 3.], dtype=torch.float16)), dict(a=torch.tensor([3., 5.], dtype=torch.float16))]
        out = tool.average_states(states)['a']
        self.assertEqual(out.dtype, torch.float16)
        self.assertTrue(torch.equal(out, torch.tensor([2., 4.], dtype=torch.float16)))

    def test_non_float_buffers_must_agree(self):
        same = [dict(a=torch.tensor([1])), dict(a=torch.tensor([1]))]
        self.assertTrue(torch.equal(tool.average_states(same)['a'], torch.tensor([1])))
        with self.assertRaises(ValueError):
            tool.average_states([dict(a=torch.tensor([1])), dict(a=torch.tensor([2]))])

    def test_key_mismatch_is_refused(self):
        with self.assertRaises(ValueError):
            tool.average_states([dict(a=torch.ones(1)), dict(b=torch.ones(1))])

    def test_gate_boundaries(self):
        base = dict(abs_rel=.08, rmse=2.8, a1=.93)
        self.assertTrue(tool.gate(dict(base, rmse=2.75, abs_rel=.082), base)['passed'])
        self.assertFalse(tool.gate(dict(base, rmse=2.76), base)['passed'])
        self.assertFalse(tool.gate(dict(base, rmse=2.7, abs_rel=.0821), base)['passed'])

    def test_window_is_frozen_and_contains_the_baseline(self):
        self.assertEqual((tool.WINDOW[0], tool.WINDOW[-1], len(tool.WINDOW)), (12000, 20000, 9))
        self.assertIn(tool.BASELINE, tool.WINDOW)


if __name__ == '__main__':
    unittest.main()
