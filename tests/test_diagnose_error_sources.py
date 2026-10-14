"""CPU regression tests for validation-only diagnostics."""
import tempfile
import unittest
import json
import sys
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.diagnose_error_sources import boundary_regions, require_predictions, tail_sequences, aligned_depth, main
from ms2_eval.official_protocol import evaluate_sample, official_valid_mask


class DiagnosticsTest(unittest.TestCase):
    def regions(self, gt, valid=None):
        return boundary_regions(gt, np.ones_like(gt, bool) if valid is None else valid,
                                7, 1.25, 1.05, 10)

    def test_smooth_slope_not_edge(self):
        gt = np.tile(np.exp(np.arange(15) * 0.045), (15, 1))
        regions = self.regions(gt)
        self.assertFalse(regions['edge'].any())
        self.assertTrue(regions['interior'][7, 7])

    def test_step_boundary_and_partition(self):
        gt = np.ones((15, 15))
        gt[:, 8:] = 3
        regions = self.regions(gt)
        self.assertTrue(regions['edge'][7, 7])
        self.assertTrue(regions['interior'][7, 0])
        np.testing.assert_array_equal(sum(regions.values()), np.ones_like(gt))

    def test_hole_does_not_create_depth_edge(self):
        gt = np.ones((15, 15))
        gt[:, 8:] = 3
        valid = np.ones_like(gt, bool)
        valid[:, 7:9] = False
        regions = self.regions(gt, valid)
        self.assertFalse(regions['edge'].any())
        np.testing.assert_array_equal(sum(regions.values()), valid.astype(int))

    def test_missing_prediction_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            np.save(directory / 'a.npy', np.ones((2, 2)))
            samples = [SimpleNamespace(sample_id=s) for s in ('a', 'b')]
            with self.assertRaisesRegex(ValueError, 'missing 1/2'):
                require_predictions(samples, [('ours', directory, 'ssi')])

    def test_tails_distinguish_own_error_and_excess(self):
        rows = []
        for i, (ours, base) in enumerate(((10, 11), (8, 1), (2, 2), (1, 1))):
            row = dict(id=str(i), sequence='A' if i == 0 else 'B', condition='day')
            for metric in ('abs_rel', 'rmse', 'a1'):
                row[f'ours_{metric}'], row[f'base_{metric}'] = ours, base
            rows.append(row)
        result = tail_sequences(rows, ['ours', 'base'])
        own = next(r for r in result if r['metric'] == 'rmse' and r['tail'] == 'ours_own'
                   and r['top_pct'] == 25 and r['sequence'] == 'A')
        paired = next(r for r in result if r['metric'] == 'rmse' and r['tail'] == 'paired_excess'
                      and r['top_pct'] == 25 and r['sequence'] == 'B')
        self.assertEqual(own['ids'], ['0'])
        self.assertEqual(paired['ids'], ['1'])
        self.assertEqual(paired['sequence_rate'], 1 / 3)

    def test_alignment_matches_official(self):
        gt = np.array([[1, 3, 4], [0, 8, 10]], np.float32)
        pred = np.array([[.2, .8, .7], [.1, 1.5, 2]], np.float32)
        valid = official_valid_mask(gt, .001, 80)
        for mode in ('ssi', 'ssi_log', 'ssi_disparity', 'median', 'none'):
            aligned = np.clip(aligned_depth(pred, gt, valid, mode)[valid], .001, 80)
            expected = evaluate_sample(pred, gt, align=mode)
            self.assertAlmostEqual(float(np.mean(abs(aligned - gt[valid]) / gt[valid])),
                                   expected['abs_rel'], places=10)

    def test_complete_analysis_writes_audit_and_tail_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt = np.arange(1, 65, dtype=np.float32).reshape(8, 8)
            samples = [SimpleNamespace(sample_id='a', thermal_gt_path='unused',
                                       sequence='seq', condition='day')]
            for name in ('ours', 'base'):
                (root / name).mkdir()
                np.save(root / name / 'a.npy', gt)
            args = ['diagnose', '--manifest', 'unused', '--data-root', tmp,
                    '--model', f'ours={root / "ours"}:ssi',
                    '--model', f'base={root / "base"}:ssi', '--out', str(root / 'out')]
            with patch.object(sys, 'argv', args), \
                    patch('tools.diagnose_error_sources.load_manifest', return_value=(samples, {})), \
                    patch('tools.diagnose_error_sources.load_ms2_gt', return_value=(None, gt)):
                self.assertEqual(main(), 0)
            summary = json.loads((root / 'out/summary.json').read_text())
            self.assertEqual(summary['frames'], 1)
            self.assertEqual(summary['headline']['ours']['rmse'], 0)
            self.assertTrue((root / 'out/tail_by_sequence.csv').is_file())
            self.assertIn('ours_alignment_scale', (root / 'out/per_frame.csv').read_text())


if __name__ == '__main__':
    unittest.main()
