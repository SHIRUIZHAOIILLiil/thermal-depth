import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'lotus'))
from tools import build_self_pseudo_gt as tool


def scene(seed=0):
    rng = np.random.default_rng(seed)
    truth = np.linspace(4, 70, 64)[None].repeat(32, 0)
    gt = np.where(rng.random((32, 64)) < .4, truth, 0.).astype(np.float32)
    return truth, gt


class SelfPseudo(unittest.TestCase):
    def test_log_affine_prediction_is_brought_to_metres(self):
        truth, gt = scene()
        raw = (.3*np.log(truth)-1.2).astype(np.float32)
        depth = tool.calibrated(raw, gt)
        self.assertEqual(depth.dtype, np.float32)
        self.assertLess(np.abs(depth-truth).max(), 1e-3*truth.max())

    def test_too_little_lidar_is_refused(self):
        truth, _ = scene()
        gt = np.zeros_like(truth, dtype=np.float32)
        gt[0, :50] = 10.
        with self.assertRaises(ValueError):
            tool.calibrated(np.log(truth).astype(np.float32), gt)

    def test_shards_cover_every_row_once(self):
        rows = list(range(103))
        parts = [tool.shard(rows, i, 4) for i in range(4)]
        self.assertEqual(sorted(sum(parts, [])), rows)
        with self.assertRaises(ValueError):
            tool.shard(rows, 4, 4)

    def test_dataset_reads_the_map_and_writes_lidar_over_it(self):
        from utils.ms2_thermal_dataset import MS2ThermalDataset
        truth, gt = scene(1)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d/'thr').mkdir()
            Image.fromarray(np.round(gt*256).astype(np.uint16)).save(d/'thr'/'a.png')
            maps = d/'self'/'calibrated_pseudo_depth'
            maps.mkdir(parents=True)
            stored = tool.calibrated((.5*np.log(truth)+.1).astype(np.float32), np.round(gt*256)/256)
            np.save(maps/'f_a.npy', stored*1.1)
            (d/'m.jsonl').write_text(json.dumps(dict(id='f_a', thermal_path='img/a.png', thermal_depth_path='thr/a.png'))+'\n')
            dense, sparse, mask = MS2ThermalDataset(d/'m.jsonl', d, maps, norm_type='log_truncnorm')._completed_depth(
                dict(id='f_a', depth_path=str(d/'thr'/'a.png')))
            real = mask[0].numpy() > 0
            self.assertTrue(np.allclose(dense[0].numpy()[real], np.round(gt*256)[real]/256))
            self.assertTrue(np.allclose(dense[0].numpy()[~real], np.clip(stored*1.1, .001, 80.)[~real]))


if __name__ == '__main__':
    unittest.main()
