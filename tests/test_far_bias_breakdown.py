"""CPU-only regression checks, including the complete diagnostic entry point."""
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tools import far_bias_breakdown as diag


class FarBiasTest(unittest.TestCase):
    def test_macro_and_pixel_aggregation(self):
        cell = diag.new_cell()
        diag.add(cell, np.log(np.full(100, .5)), np.full(100, .5), 200)
        diag.add(cell, np.zeros(1000), np.zeros(1000), 200)
        result = diag.finish(cell)
        self.assertAlmostEqual(result['pixel_abs_rel'], 50 / 1100)
        self.assertAlmostEqual(result['frame_mean_abs_rel'], .25)
        self.assertEqual(result['frames'], 2)

    def test_tail_ranking_and_zero_error(self):
        records = [('seq', str(i), 10, 0., 5, 10. * i, float(i)) for i in range(20)]
        tail = diag.error_tail(records)
        self.assertEqual(tail['ids'], ['19', '18'])
        self.assertAlmostEqual(tail['pixel_sse_share'], 37 / 190)
        self.assertAlmostEqual(tail['frame_mse_share'], 37 / 190)
        self.assertIsNone(diag.error_tail([('seq', 'a', 10, 0., 0, 0., 0.)])['pixel_sse_share'])

    def run_fixture(self, root, constant_all=False):
        samples = [SimpleNamespace(sample_id=s, sequence='seq', thermal_gt_path=Path(s))
                   for s in ('good', 'degenerate', 'sparse')]
        gt = np.tile(np.array([[10, 20, 50, 60]], dtype=np.float32), (2, 1))
        gts = {s: gt.copy() for s in ('good', 'degenerate')}
        gts['sparse'] = np.tile(np.array([[10, 20, 30, 60]], dtype=np.float32), (2, 1))
        argv = ['far_bias', '--manifest', str(root / 'manifest'), '--data-root', str(root),
                '--out', str(root / 'out'), '--min-fit-pixels', '3', '--frame-min-far', '1']
        for model, align in (('a', 'ssi_log'), ('b', 'ssi')):
            directory = root / model
            directory.mkdir()
            for s in samples:
                raw = np.log(gts[s.sample_id]) if model == 'a' else gts[s.sample_id].copy()
                if model == 'a' and s.sample_id == 'degenerate':
                    raw[:, 2:] = 3.
                if constant_all and model == 'a' and s.sample_id == 'good':
                    raw[:] = 1.
                np.save(directory / f'{s.sample_id}.npy', raw)
            argv += ['--model', f'{model}={directory}:{align}']
        with patch('sys.argv', argv), patch.object(diag, 'load_manifest', return_value=(samples, {})), \
                patch.object(diag, 'load_ms2_gt', side_effect=lambda p, scale: (None, gts[p.name])), \
                contextlib.redirect_stdout(io.StringIO()):
            diag.main()

    def test_shared_cohort_and_exclusion_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.run_fixture(root)
            summary = json.loads((root / 'out/far_bias_summary.json').read_text())
            self.assertEqual(summary['paired_ids'], ['good'])
            self.assertLessEqual(summary['largest_absrel_mismatch_vs_official'], 1e-9)
            with (root / 'out/far_bias_by_bin.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(all(int(r['frames']) == 1 for r in rows if r['cohort'] == 'paired'))
            available = [r for r in rows if r['cohort'] == 'available' and r['bin'] == '>=40m'
                         and r['sequence'] == 'ALL' and r['fit'] == 'all']
            self.assertEqual([int(r['frames']) for r in available], [3, 3])
            with (root / 'out/excluded_fits.csv').open() as f:
                excluded = list(csv.DictReader(f))
            self.assertEqual(len(excluded), 3)
            self.assertTrue(any('Degenerate' in r['reason'] for r in excluded))
            self.assertTrue(any(r['reason'] == 'insufficient_fit_pixels' for r in excluded))

    def test_all_fit_failure_aborts(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(SystemExit, 'all fit failed'):
                self.run_fixture(Path(tmp), constant_all=True)


if __name__ == '__main__':
    unittest.main()
