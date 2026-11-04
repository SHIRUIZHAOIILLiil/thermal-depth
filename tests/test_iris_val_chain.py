import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.diagnose_iris_val_chain import STAGES, aggregate, score_frame


class ValChain(unittest.TestCase):
    def test_whole_frame_alignment_before_far_bin_and_separate_target_clipping(self):
        gt = np.array([[0, 5, 20, 59, 60, 70, 79, 80]], np.float32)
        log_gt = np.log(np.maximum(gt,.001))
        bounds = (np.log(5), np.log(65))
        unit = np.clip((log_gt-bounds[0])/(bounds[1]-bounds[0]+1e-5), 0, 1)
        maps = dict(pseudo=gt.copy(), pseudo_only_unit=unit, pseudo_only_vae=unit, lidar_written=gt.copy(), target_unit=unit,
                    vae_mode=unit.copy(), model4000=2*log_gt+3)
        bounds = dict.fromkeys(STAGES, bounds)
        rows = score_frame(gt, maps, bounds)
        self.assertTrue(all(r['points'] == 3 for r in rows))
        by = {(r['mode'],r['stage']):r for r in rows}
        self.assertAlmostEqual(by['ssi_log','model4000']['bias_m'], 0, places=4)
        self.assertLess(by['target_scale','target_unit']['bias_m'], -5)
        self.assertLess(by['ssi_log','target_unit']['bias_m'], -1)
        self.assertEqual(by['target_scale','lidar_written']['rmse_m'], 0)
        self.assertNotIn(('target_scale','model4000'), by)
        self.assertEqual(score_frame(gt[:,:4], {k:v[:,:4] for k,v in maps.items()}, bounds), [])

    def test_paired_signed_deltas_are_frame_weighted(self):
        records = [dict(stage=stage, mode='ssi_log', sequence='seq', points=n, bias_m=error*i,
                        rmse_m=abs(error*i), too_near_pct=100.)
                   for i,stage in enumerate(STAGES) for error,n in [(-4,1),(-2,100)]]
        table = [r for r in aggregate(records) if r['sequence']=='ALL']
        self.assertEqual([r['delta_bias_m'] for r in table], [None,-3,-3,-9,-3,-3,-3])
        self.assertTrue(all(r['points']==101 and r['frames']==2 for r in table))


if __name__ == '__main__':
    unittest.main()
