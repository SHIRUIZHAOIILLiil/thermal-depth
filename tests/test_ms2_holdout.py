import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'lotus'))
from utils import ms2_holdout as holdout
from tools import diagnose_far_target_source as source_tool
from tools import diagnose_holdout_fit as tool
from tools.build_anythermal_pseudo_gt import calibrate_pseudo_depth


def write_frame(folder, ident, depth, raw):
    path = folder/'thr'/f'{ident}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(depth*256).astype(np.uint16)).save(path)
    np.save(folder/'raw'/f'{ident}.npy', raw.astype(np.float32))
    return path


class Fixture:
    def __init__(self, folder, frames=3):
        self.folder = folder
        (folder/'raw').mkdir()
        pseudo = folder/'pseudo'/'calibrated_pseudo_depth'
        pseudo.mkdir(parents=True)
        rng = np.random.default_rng(0)
        rows = []
        for i in range(frames):
            ident = f'seq_{i:06d}'
            truth = np.linspace(4, 70, 64)[None].repeat(32, 0) + rng.normal(0, 1, (32, 64))
            depth = np.where(rng.random((32, 64)) < .4, truth, 0.)
            raw = 1/truth[::2, ::2] + rng.normal(0, .002, (16, 32))
            write_frame(folder, ident, depth, raw)
            gt = np.round(depth*256)/256
            real = (gt > .001) & (gt < 80)
            np.save(pseudo/f'{ident}.npy', calibrate_pseudo_depth(raw.astype(np.float32), gt, real, min_fit_pixels=10)[0].astype(np.float32))
            rows.append(dict(id=ident, thermal_path=f'img/{ident}.png', thermal_depth_path=f'thr/{ident}.png', caption=''))
        (folder/'pseudo'/'build_config.json').write_text(json.dumps(dict(
            depth_scale=256., gt_min_depth=.001, gt_max_depth=80., dtype='float32', min_fit_pixels=10)))
        self.manifest = folder/'train.jsonl'
        self.manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.pseudo = pseudo

    def dataset(self):
        with mock.patch.dict(os.environ, IRIS_HOLDOUT_RAW_DIR=str(self.folder/'raw')):
            return holdout.HoldoutDataset(self.manifest, self.folder, self.pseudo, norm_type='log_truncnorm', truncnorm_max=1.)


class HoldoutTarget(unittest.TestCase):
    def test_cells_match_the_diagnostic(self):
        rng = np.random.default_rng(1)
        for i in range(10):
            valid = rng.random((256, 640)) < .1
            ident = f'2021-08-06-10-59-33_{i:06d}'
            self.assertTrue(np.array_equal(holdout.holdout_mask(valid, ident), source_tool.holdout_mask(valid, ident)))

    def test_target_hides_held_lidar(self):
        with tempfile.TemporaryDirectory() as d:
            fixture = Fixture(Path(d))
            dataset = fixture.dataset()
            row = dataset.rows[0]
            gt, real = dataset._lidar(row)
            held = holdout.holdout_mask(real, row['id'])
            fit = real & ~held
            self.assertTrue(.1 < held.sum()/real.sum() < .3)
            dense, sparse, mask = (x[0].numpy() for x in dataset._completed_depth(row))
            self.assertTrue(np.array_equal(mask > 0, fit))
            self.assertTrue(np.all(sparse[held] == 0) and np.allclose(sparse[fit], gt[fit]))
            self.assertTrue(np.allclose(dense[fit], gt[fit]))
            raw = np.load(fixture.folder/'raw'/f"{row['id']}.npy")
            expected = calibrate_pseudo_depth(raw, gt, fit, min_fit_pixels=10)[0].astype(np.float32)
            self.assertTrue(np.allclose(dense[held], np.clip(expected[held], .001, 80.)))
            self.assertFalse(np.allclose(dense[held], gt[held], atol=1e-3))

    def test_held_values_never_reach_the_target(self):
        with tempfile.TemporaryDirectory() as d:
            fixture = Fixture(Path(d))
            dataset = fixture.dataset()
            row = dataset.rows[0]
            before = [x.numpy() for x in dataset._completed_depth(row)]
            gt, real = dataset._lidar(row)
            changed = np.where(holdout.holdout_mask(real, row['id']), gt+5, gt)
            Image.fromarray(np.round(changed*256).astype(np.uint16)).save(row['depth_path'])
            after = [x.numpy() for x in dataset._completed_depth(row)]
            for x, y in zip(before, after):
                self.assertTrue(np.array_equal(x, y))

    def test_wrong_raw_predictions_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            fixture = Fixture(Path(d))
            path = fixture.folder/'raw'/'seq_000001.npy'
            np.save(path, np.load(path)*1.01+.001)
            with self.assertRaises(ValueError):
                fixture.dataset()

    def test_entry_point_replaces_only_the_dataset(self):
        source = (ROOT/'lotus/train_iris_ms2_holdout.py').read_text()
        self.assertIn('trainer.MS2ThermalDataset = HoldoutDataset', source)
        self.assertIn('trainer.main()', source)

    def test_decision_boundaries(self):
        self.assertEqual(tool.decision(1.5, 1.2, 1.4)['verdict'], 'copies_target')
        self.assertEqual(tool.decision(1.5, 1.2, 1.3)['verdict'], 'learns_beyond_target')
        self.assertEqual(tool.decision(1.5, 1.2, 1.35)['verdict'], 'partial')
        with self.assertRaises(ValueError):
            tool.decision(1.1, 1.2, 1.3)

    def test_control_tolerance(self):
        self.assertTrue(tool.control(1.1290, 1.1305)['passed'])
        self.assertFalse(tool.control(1.1290, 1.1320)['passed'])


if __name__ == '__main__':
    unittest.main()
