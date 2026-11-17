"""CPU checks for tools/check_near_truncation.py on a tiny synthetic MS2 frame."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tools import check_near_truncation as tool


def write_frame(root, name, dense, gt):
    thermal = (np.arange(dense.size, dtype=np.uint16).reshape(dense.shape)+20000)
    Image.fromarray(thermal).save(root/f'{name}_t.png')
    Image.fromarray((gt*256).astype(np.uint16)).save(root/f'{name}_d.png')
    np.save(root/'pseudo'/f'{name}.npy', dense.astype(np.float32))
    return json.dumps(dict(id=name, thermal_path=f'{name}_t.png', thermal_depth_path=f'{name}_d.png', caption='a road'))


class FrameCosts(unittest.TestCase):
    def test_hand_computed_clip_cost(self):
        dense = np.full((10, 10), 20., np.float32)
        dense[0, :5] = [1., 2., 3., 4., 5.]
        log_dense = torch.log(torch.from_numpy(dense))
        valid = torch.ones_like(log_dense, dtype=torch.bool)
        lidar = np.zeros((10, 10), np.float32)
        lidar[0, 0], lidar[5, 5] = 1., 20.
        lidar_valid = lidar > 0
        costs, bounds, n = tool.frame_costs(log_dense, valid, lidar, lidar_valid, (.02, 0.))
        lo = float(np.exp(bounds[.02]))
        self.assertGreater(lo, 1.)  # the 1 m pixel lies below the 2% quantile
        self.assertEqual(n, 2)
        self.assertAlmostEqual(costs[.02]['0-10m'], abs(lo-1.)/1./2, places=6)
        self.assertAlmostEqual(costs[.02]['10-20m'], 0., places=9)
        self.assertAlmostEqual(costs[0.]['0-10m'], 0., places=9)  # q=0: the bound is the frame minimum
        self.assertEqual(costs[0.]['clipped_points'], 0)
        self.assertEqual(costs[.02]['clipped_share_0-10m'], 1.)

    def test_reading_and_suggestion(self):
        self.assertEqual(tool.reading(.0012), 'material')
        self.assertEqual(tool.reading(.0003), 'excluded')
        self.assertEqual(tool.reading(.0006), 'partial')
        means = {.02: {'0-10m': .002}, .01: {'0-10m': .0005}, .005: {'0-10m': .0003}, 0.: {'0-10m': 0.}}
        self.assertEqual(tool.suggested_q(means), .005)
        means[.005]['0-10m'] = .001
        self.assertIsNone(tool.suggested_q(means))  # only q=0 would do; never suggested


class EndToEnd(unittest.TestCase):
    def test_control_passes_on_the_real_dataset_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'pseudo').mkdir()
            rows = []
            rng = np.random.default_rng(0)
            for k in range(3):
                dense = rng.uniform(15., 60., (64, 160)).astype(np.float32)
                dense[-6:, :] = np.linspace(2., 9., 160)  # near road at the bottom
                gt = np.zeros_like(dense)
                gt[-6:, ::4] = dense[-6:, ::4]
                gt[20:40, ::8] = dense[20:40, ::8]
                rows.append(write_frame(root, f'f{k}', dense, gt))
            (root/'train.jsonl').write_text('\n'.join(rows)+'\n')
            out = root/'out'
            tool.run(SimpleNamespace(data_root=root, train_manifest=root/'train.jsonl', pseudo_dir=root/'pseudo',
                                     out=out, frames=3, seed=1))
            r = json.loads((out/'summary.json').read_text())
            self.assertTrue(r['control']['passed'])
            self.assertEqual(r['frames'], 3)
            q02, q0 = r['quantiles']['0.02'], r['quantiles']['0.0']
            self.assertGreater(q02['cost_by_band']['0-10m'], 0.)
            self.assertAlmostEqual(q0['cost_by_band']['0-10m'], 0., places=9)
            self.assertEqual(q0['lidar_0_10_below_bound_share'], 0.)
            self.assertGreater(q02['lidar_0_10_below_bound_share'], 0.)


if __name__ == '__main__':
    unittest.main()
