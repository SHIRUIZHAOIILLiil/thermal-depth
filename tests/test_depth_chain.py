"""CPU gates for the read-only three-bound depth-chain diagnostic."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools.diagnose_depth_chain import (ARMS, select_rows, frame_seed, inverse_log_unit,
    score_stage, models, score)
from ms2_eval.io import sha256_file


class DepthChainTest(unittest.TestCase):
    def test_actual_dataset_three_upper_bounds(self):
        # Actual target loader, not a copied quantile formula. No model or GPU.
        from tests.test_truncnorm_max import TruncnormMaxTest
        TruncnormMaxTest.setUpClass()
        try:
            fixture = TruncnormMaxTest()
            lows, highs = [], []
            for q in ARMS.values():
                dataset, example = fixture.item(truncnorm_min=.02, truncnorm_max=q)
                dense, _, _ = dataset._completed_depth(dataset.rows[0])
                bounds = example['norm_bounds'].numpy()
                unit = example['depth_values'][0].numpy()/2+.5
                back = inverse_log_unit(unit, bounds)
                expected = np.exp(np.clip(np.log(dense[0].numpy()), bounds[0], bounds[1]+1e-5))
                np.testing.assert_allclose(back, expected, rtol=2e-6)
                lows.append(bounds[0])
                highs.append(bounds[1])
            self.assertEqual(len(set(lows)), 1)
            self.assertTrue(highs[0] < highs[1] <= highs[2])
        finally:
            TruncnormMaxTest.tearDownClass()

    def test_selection_fixed_and_rejects_foreign_sequences(self):
        seq = '2021-08-06-10-59-33'
        rows = [dict(id=f'{seq}_{i:06}', caption='road',
                     thermal_path=f'sync_data/_{seq}/thr/img_left/{i:06}.png',
                     thermal_depth_path=f'proj_depth/_{seq}/thr/depth_filtered/{i:06}.png') for i in range(20)]
        self.assertEqual(select_rows(rows, 8, 43, {seq}), select_rows(rows, 8, 43, {seq}))
        with self.assertRaisesRegex(ValueError, 'outside'):
            select_rows(rows, 8, 43, {'2021-08-13-16-08-46'})
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            select_rows(rows + [rows[0]], 8, 43, {seq})
        self.assertEqual(frame_seed(rows[0]['id'], 43), frame_seed(rows[0]['id'], 43))

    def test_inverse_and_clipping_distinguishable(self):
        gt = np.array([[2, 5, 20, 65, 79]], np.float32)
        lo, hi = np.log(2), np.log(60)
        unit = (np.log(gt)-lo)/(hi-lo+1e-5)
        np.testing.assert_allclose(inverse_log_unit(unit, (lo, hi)), gt, rtol=1e-6)
        clipped = inverse_log_unit(np.clip(unit, 0, 1), (lo, hi))
        self.assertLess(clipped[0, -1], 61)
        self.assertGreater(abs(clipped[0, -1]-gt[0, -1]), 18)

    def test_same_alignment_once_for_all_bins(self):
        gt = np.array([[2, 5, 20, 65, 79]], np.float32)
        raw = np.log(gt)*.3+.2
        rows = score_stage(raw, gt, 'val', 'q098', 'model', 'x', log_input=True, aligned_only=True)
        self.assertTrue(all(r['alignment'] == 'ssi_log' for r in rows))
        self.assertEqual(len({r['alignment_scale'] for r in rows}), 1)
        self.assertLess(rows[0]['rmse'], 1e-4)
        raw[0, 0] = np.nan
        with self.assertRaises(ValueError):
            score_stage(raw, gt, 'val', 'q098', 'model', 'x', log_input=True)

    def test_model_phase_only_calls_eval_with_same_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            checkpoint = out/'step16000_weights.pt'
            checkpoint.write_bytes(b'fake-test-weight')
            config = dict(checkpoints={arm: dict(path=str(checkpoint), sha256=sha256_file(checkpoint)) for arm in ARMS},
                          data_root=tmp, model_path='local-model', seed=20261017)
            with patch('tools.diagnose_depth_chain.subprocess.run') as run:
                models(SimpleNamespace(out=out), config)
            self.assertEqual(run.call_count, 6)
            for call in run.call_args_list:
                command = call.args[0]
                self.assertIn('--eval-checkpoint', command)
                self.assertEqual(command[command.index('--seed')+1], '20261017')
                self.assertNotIn('--resume-from-checkpoint', command)
                self.assertEqual(call.kwargs['env']['IRIS_DECODE_CLAMP'], '0')

    def test_end_to_end_score_saved_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            gt = np.array([[2, 5, 20, 65, 79], [3, 6, 21, 66, 78]], np.float32)
            Image.fromarray((gt*256).astype(np.uint16)).save(out/'gt.png')
            row = dict(id='frame', thermal_depth_path=str(out/'gt.png'))
            for split in ('train', 'val'):
                (out/f'{split}.jsonl').write_text(json.dumps(row)+'\n')
            (out/'targets').mkdir()
            bounds = np.array([np.log(2), np.log(79)])
            unit = ((np.log(gt)-bounds[0])/(bounds[1]-bounds[0]+1e-5)).astype(np.float32)
            for arm in ARMS:
                np.savez(out/'targets'/f'{arm}_frame.npz', gt=gt, dense=gt, pseudo=gt,
                         unit=unit, vae_mode=unit, vae_sample=unit, bounds=bounds)
                for split in ('train', 'val'):
                    directory = out/'models'/arm/split/'raw_predictions'
                    directory.mkdir(parents=True)
                    np.save(directory/'frame.npy', unit)
            score(SimpleNamespace(out=out), {})
            payload = json.loads((out/'analysis/summary.json').read_text())
            self.assertTrue(payload['diagnostic_only'])
            self.assertTrue((out/'analysis/stage_deltas.csv').is_file())
            val = [r for r in payload['rows'] if r['split'] == 'val']
            self.assertTrue(all(r['alignment'] == 'ssi_log' for r in val))
            self.assertTrue(all(r['rmse'] < 1e-4 for r in val))


if __name__ == '__main__':
    unittest.main()
