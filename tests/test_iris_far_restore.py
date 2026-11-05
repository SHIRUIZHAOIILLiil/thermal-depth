import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.diagnose_iris_far_restore import (
    STAGES, aggregate, check_reproduction, restore_far, score_frame,
)
from tools.diagnose_depth_chain import inverse_log_unit
from ms2_eval.official_protocol import official_valid_mask


class FarRestore(unittest.TestCase):
    def fixture(self):
        gt = np.array([[0, 5, 20, 40, 59, 60, 65, 70, 79, 80]], np.float32)
        bounds = np.log(np.array([5, 65], np.float32))
        raw = 2*((np.log(np.maximum(gt, .001))-bounds[0])/(bounds[1]-bounds[0]+1e-5)-.5)
        target = np.repeat(np.clip(raw, -1, 1)[None], 3, axis=0)
        return gt, bounds, target

    def test_restore_only_censored_real_far_and_preserve_input(self):
        gt, bounds, target = self.fixture()
        original = target.copy()
        valid = official_valid_mask(gt, .001, 80.)
        valid[0, 8] = False  # high-valued pseudo-only position must not be restored
        restored, mask = restore_far(target, gt, valid, bounds)
        self.assertEqual(np.flatnonzero(mask).tolist(), [7])
        np.testing.assert_array_equal(target, original)
        np.testing.assert_array_equal(restored[:,~mask], original[:,~mask])
        self.assertGreater(float(restored[:,mask].min()), 1.)
        depth = inverse_log_unit(restored[0]/2+.5, bounds)
        np.testing.assert_allclose(depth[mask], gt[mask], atol=3e-5, rtol=0)

    def test_no_far_identity_and_bad_inputs_rejected(self):
        gt, bounds, target = self.fixture()
        restored, mask = restore_far(target, gt, np.zeros_like(gt, bool), bounds)
        self.assertFalse(mask.any())
        np.testing.assert_array_equal(target, restored)
        with self.assertRaises(ValueError):
            restore_far(target, gt, gt>0, [1, 1])
        broken = target.copy()
        broken[:,0,7] = .5
        with self.assertRaises(ValueError):
            restore_far(broken, gt, gt>0, bounds)

    def test_identity_roundtrip_retention_and_near_preservation(self):
        gt, bounds, target = self.fixture()
        restored, mask = restore_far(target, gt, gt>0, bounds)
        before, after = target[0]/2+.5, restored[0]/2+.5
        maps = dict(zip(STAGES, (before, after, before, after)))
        rows = score_frame(gt, maps, bounds, mask)
        rows = [dict(id='seq_1', sequence='seq', **r) for r in rows]
        table, changes = aggregate(rows)
        retention = [r for r in changes if r['sequence']=='ALL' and r['transition']=='retention'][0]
        self.assertAlmostEqual(retention['signed_retention_ratio'], 1.)
        near = [r for r in changes if r['mode']=='target_scale' and r['depth']=='0-20m']
        self.assertTrue(all(r['delta_abs_rel']==0 for r in near))
        changed = [r for r in rows if r['stage']=='restored_unit' and r['mode']=='target_scale'
                   and r['depth']=='restored_points'][0]
        self.assertLess(changed['rmse_m'], 3e-5)
        self.assertEqual(changed['points'], 2)
        # SSI-log fitting includes all depths: restoring the far tail can change near alignment.
        aligned_near = [r for r in changes if r['mode']=='ssi_log' and r['depth']=='0-20m']
        self.assertTrue(any(abs(r['delta_abs_rel'])>1e-5 for r in aligned_near))
        with self.assertRaises(ValueError):
            aggregate(rows[:-1])

    def test_reference_reproduction_fails_on_changed_results_or_coverage(self):
        gt, bounds, target = self.fixture()
        unit = target[0]/2+.5
        rows = score_frame(gt, dict.fromkeys(STAGES, unit), bounds, gt>=60)
        reference = {'table': [dict(sequence='ALL', mode=r['mode'],
            stage={'original_unit':'target_unit','original_vae':'vae_mode'}[r['stage']],
            frames=1, points=r['points'], bias_m=r['bias_m'], rmse_m=r['rmse_m'])
            for r in rows if r['depth']=='60-80m' and r['stage'] in ('original_unit','original_vae')]}
        check_reproduction(rows, reference)
        reference['table'][0]['bias_m'] += .01
        with self.assertRaises(ValueError):
            check_reproduction(rows, reference)
        reference['table'][0]['bias_m'] -= .01
        reference['table'][0]['points'] += 1
        with self.assertRaises(ValueError):
            check_reproduction(rows, reference)


if __name__ == '__main__':
    unittest.main()
