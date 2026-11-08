import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'lotus'))
from utils import ms2_train_augment as aug

SETTINGS = {'IRIS_AUG_MAX_SCALE': '1.5', 'IRIS_AUG_GAMMA': '0.7,1.4'}


def maps():
    columns = torch.arange(640, dtype=torch.float32)[None, None].repeat(1, 256, 1)
    image = (columns/639*2-1).repeat(3, 1, 1)
    mask = torch.zeros(1, 256, 640)
    mask[:, ::4, ::4] = 1.
    return image, columns.clone()+1, columns*mask, mask


class StrongAugmentation(unittest.TestCase):
    def test_crop_keeps_shape_and_alignment(self):
        torch.manual_seed(0)
        for _ in range(20):
            image, depth, gt, mask = aug.random_scale_crop(*maps(), max_scale=1.5)
            self.assertEqual((image.shape, depth.shape), ((3, 256, 640), (1, 256, 640)))
            recovered = (image[0]+1)/2*639
            self.assertLess(float((recovered-(depth[0]-1)).abs().max()), 1.5)
            self.assertTrue(set(mask.unique().tolist()) <= {0., 1.})
            self.assertTrue(((gt > 0) <= (mask > 0)).all())
            self.assertTrue(torch.equal(depth, depth.round()))

    def test_zoom_reaches_past_the_old_limit_and_stays_bounded(self):
        torch.manual_seed(1)
        spans = []
        for _ in range(200):
            _, depth, *_ = aug.random_scale_crop(*maps(), max_scale=1.5)
            spans.append(float(depth.max()-depth.min()))
        self.assertGreaterEqual(min(spans), 639/1.5-2)
        self.assertLess(min(spans), 639/1.15)

    def test_gamma_is_nonlinear_stays_in_range_and_keeps_end_points(self):
        torch.manual_seed(2)
        image = torch.linspace(-1, 1, 101).reshape(1, 1, 101).repeat(3, 1, 1)
        out = aug.random_gamma(image, .7, 1.4)
        self.assertEqual(float(out.min()), -1.)
        self.assertEqual(float(out.max()), 1.)
        self.assertGreater(float((out-image).abs().max()), .01)
        self.assertTrue(torch.equal(out[0], out[1]))

    def test_transform_moves_all_maps_and_only_image_gets_gamma(self):
        with mock.patch.dict(os.environ, SETTINGS):
            transform = aug.StrongTransform(random_flip=True)
        torch.manual_seed(3)
        image, depth, gt, mask = transform(*maps())
        self.assertEqual(image.shape, (3, 256, 640))
        self.assertTrue(torch.equal(depth, depth.round()))
        self.assertTrue(set(mask.unique().tolist()) <= {0., 1.})

    def test_scale_one_turns_the_crop_off(self):
        with mock.patch.dict(os.environ, dict(SETTINGS, IRIS_AUG_MAX_SCALE='1')):
            transform = aug.StrongTransform(random_flip=False)
        torch.manual_seed(4)
        image, depth, gt, mask = transform(*maps())
        original = maps()
        self.assertTrue(torch.equal(depth, original[1]) and torch.equal(mask, original[3]))
        self.assertFalse(torch.equal(image, original[0]))

    def test_invalid_settings_are_refused(self):
        with mock.patch.dict(os.environ, dict(SETTINGS, IRIS_AUG_GAMMA='1.2,1.4')):
            with self.assertRaises(ValueError):
                aug.StrongTransform(random_flip=True)

    def test_entry_point_replaces_only_the_transform(self):
        source = (ROOT/'lotus/train_iris_ms2_aug.py').read_text()
        self.assertIn('trainer.MS2ThermalTransform = StrongTransform', source)
        self.assertIn('trainer.main()', source)
        script = (ROOT/'lotus/train_scripts/train_iris_ms2_g_depth.sh').read_text()
        self.assertIn('"${TRAIN_ENTRY:-train_iris_ms2_g.py}"', script)


if __name__ == '__main__':
    unittest.main()
