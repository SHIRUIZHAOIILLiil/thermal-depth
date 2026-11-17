"""log_truncnorm for the synthetic pretraining loaders (docs/IRIS_SYNTH_PRETRAIN_V1.md).

Runs in the iris env (torch, cv2): iris_synth_pretrain.sbatch calls it before training.
"""
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'lotus'))
import cv2  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402

from utils.hypersim_dataset import HypersimImageDepthNormalTransform  # noqa: E402
from utils.log_target import log_truncnorm  # noqa: E402
from utils.vkitti_dataset import VKITTIDataset, VKITTITransform  # noqa: E402


def ms2_formula(depth, valid, q_min, q_max):
    """The log_truncnorm branch of lotus/utils/ms2_thermal_dataset.py, line for line."""
    log_depth = torch.log(depth.clamp(min=1e-6))
    dmin = torch.quantile(log_depth[valid], q_min)
    dmax = torch.quantile(log_depth[valid], q_max)
    return ((log_depth - dmin) / (dmax - dmin + 1e-5) - 0.5) * 2.0


class Helper(unittest.TestCase):
    def test_ms2_source_still_has_this_formula(self):
        src = (ROOT/'lotus'/'utils'/'ms2_thermal_dataset.py').read_text(encoding='utf-8')
        for line in ('log_depth = torch.log(depth.clamp(min=1e-6))',
                     'dmin = torch.quantile(log_depth[valid_mask], self.truncnorm_min)',
                     'dmax = torch.quantile(log_depth[valid_mask], self.truncnorm_max)',
                     'depth_norm = ((log_depth - dmin) / (dmax - dmin + 1e-5) - 0.5) * 2.0'):
            self.assertIn(line, src)

    def test_equals_ms2_near_low_far_high(self):
        g = torch.Generator().manual_seed(0)
        depth = torch.rand(40, 60, generator=g, dtype=torch.float64)*79 + 1
        valid = depth < 70
        out = log_truncnorm(depth, valid, .02, 1.0)
        torch.testing.assert_close(out, ms2_formula(depth, valid, .02, 1.0))
        flat, d = out.flatten(), depth.flatten()
        far = torch.argmax(torch.where(valid.flatten(), d, torch.zeros_like(d)))
        self.assertLess(flat[torch.argmin(d)].item(), flat[far].item())
        self.assertAlmostEqual(flat[far].item(), 1.0, places=3)  # q_max 1.0 puts the farthest valid pixel at +1

    def test_rejects_bad_bounds_and_empty_mask(self):
        depth = torch.ones(4, 4)
        with self.assertRaises(ValueError):
            log_truncnorm(depth, depth > 0, .02, .4)
        with self.assertRaises(ValueError):
            log_truncnorm(depth, depth > 5, .02, 1.0)


class Hypersim(unittest.TestCase):
    def run_transform(self, norm, tmax):
        t = HypersimImageDepthNormalTransform((24, 32), False, norm, .02, truncnorm_max=tmax)
        image = Image.fromarray(np.zeros((48, 64, 3), np.uint8))
        depth = np.tile(np.linspace(1, 50, 64, dtype=np.float32), (48, 1))  # near on the left
        normal = np.zeros((48, 64, 3), np.float32)
        normal[..., 2] = 1
        _, d, _ = t(image, depth, normal)
        return d

    def test_log_is_near_low_and_disparity_is_reversed(self):
        log = self.run_transform('log_truncnorm', 1.0)
        self.assertEqual(tuple(log.shape), (3, 24, 32))
        self.assertTrue(bool((log >= -1).all() and (log <= 1).all()))
        self.assertLess(log[0, :, 0].mean().item(), log[0, :, -1].mean().item())
        self.assertAlmostEqual(log.max().item(), 1.0, places=3)
        disp = self.run_transform('trunc_disparity', None)
        self.assertGreater(disp[0, :, 0].mean().item(), disp[0, :, -1].mean().item())

    def test_default_upper_bound_is_unchanged(self):
        self.assertAlmostEqual(HypersimImageDepthNormalTransform((8, 8), False, 'trunc_disparity', .02).truncnorm_max, .98)


class VKITTI(unittest.TestCase):
    def test_log_target_and_sky(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'vk'
            ds_kw = dict(transform=VKITTITransform(random_flip=False), norm_type='log_truncnorm', truncnorm_min=.02)
            for scene in ('02', '06', '18', '20'):
                for cond in ('15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right', 'clone',
                             'fog', 'morning', 'overcast', 'rain', 'sunset'):
                    for cam in ('0', '1'):
                        (root/f'Scene{scene}'/cond/'frames'/'rgb'/f'Camera_{cam}').mkdir(parents=True)
            cam0 = root/'Scene02'/'clone'/'frames'
            Image.fromarray(np.zeros((375, 1242, 3), np.uint8)).save(cam0/'rgb'/'Camera_0'/'rgb_00000.jpg')
            for kind in ('depth', 'normal'):
                (cam0/kind/'Camera_0').mkdir(parents=True)
            depth_cm = np.tile(np.linspace(100, 6000, 1242), (375, 1)).astype(np.uint16)  # 1-60 m, near on the left
            depth_cm[:60] = 65535  # sky, 655 m
            cv2.imwrite(str(cam0/'depth'/'Camera_0'/'depth_00000.png'), depth_cm)
            Image.fromarray(np.full((375, 1242, 3), 128, np.uint8)).save(cam0/'normal'/'Camera_0'/'normal_00000.png')
            ds = VKITTIDataset(str(root), truncnorm_max=1.0, **ds_kw)
            self.assertEqual(len(ds), 1)
            d = ds[0]['depth_values']
            self.assertEqual(tuple(d.shape), (3, 352, 1216))
            self.assertTrue(bool((d[:, :30] == 1).all()))  # sky rows (60 minus the 23-row crop) clip to far
            ground = d[0, 100:]
            self.assertLess(ground[:, 0].mean().item(), ground[:, -1].mean().item())
            self.assertAlmostEqual(ground.min().item(), -1.0, delta=.05)
            self.assertEqual(VKITTIDataset(str(root), **ds_kw).truncnorm_max, .98)


if __name__ == '__main__':
    unittest.main()
