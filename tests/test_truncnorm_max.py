"""The far bound of the log_truncnorm target can be moved on its own.

Run:
    python -m pytest tests/test_truncnorm_max.py -q
"""
from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ms2_thermal_dataset_t", ROOT / "lotus/utils/ms2_thermal_dataset.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MS2ThermalDataset = module.MS2ThermalDataset


class TruncnormMaxTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        rng = np.random.default_rng(0)
        (root / "pseudo").mkdir()
        h, w = 64, 160
        thermal = rng.integers(20000, 30000, size=(h, w)).astype(np.uint16)
        Image.fromarray(thermal).save(root / "thr.png")
        # depth ramps 3 m -> 75 m left to right; lidar on 30% of pixels
        depth = np.tile(np.exp(np.linspace(np.log(3.0), np.log(75.0), w)), (h, 1)).astype(np.float32)
        sparse = np.where(rng.random((h, w)) < 0.3, depth, 0.0)
        Image.fromarray((sparse * 256).astype(np.uint16)).save(root / "gt.png")
        np.save(root / "pseudo" / "f0.npy", depth)
        (root / "m.jsonl").write_text(json.dumps(
            {"id": "f0", "thermal_path": "thr.png", "thermal_depth_path": "gt.png", "caption": ""}) + "\n")
        cls.root = root
        cls.depth = depth

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def item(self, **kwargs):
        ds = MS2ThermalDataset(self.root / "m.jsonl", self.root, self.root / "pseudo",
                               norm_type="log_truncnorm", **kwargs)
        return ds, ds[0]

    def test_default_is_the_symmetric_pair(self):
        ds, ex = self.item(truncnorm_min=0.02)
        self.assertAlmostEqual(ds.truncnorm_max, 0.98)
        log_depth = torch.log(torch.from_numpy(self.depth))
        hi = float(torch.quantile(log_depth.reshape(-1), 0.98))
        self.assertAlmostEqual(float(ex["norm_bounds"][1]), hi, places=4)
        # about 2% of pixels sit at +1 (the far clip)
        top = float((ex["depth_values"][0] >= 1.0).float().mean())
        self.assertGreater(top, 0.01)

    def test_max_one_keeps_the_far_field(self):
        ds, ex = self.item(truncnorm_min=0.02, truncnorm_max=1.0)
        lo_default = float(self.item(truncnorm_min=0.02)[1]["norm_bounds"][0])
        self.assertAlmostEqual(float(ex["norm_bounds"][0]), lo_default, places=6)  # near bound unchanged
        self.assertAlmostEqual(float(ex["norm_bounds"][1]), math.log(float(self.depth.max())), places=4)
        top = float((ex["depth_values"][0] >= 1.0 - 1e-6).float().mean())
        self.assertLess(top, 0.01)  # only the farthest column touches the bound

    def test_bounds_are_checked(self):
        with self.assertRaises(ValueError):
            self.item(truncnorm_min=0.02, truncnorm_max=0.4)


if __name__ == "__main__":
    unittest.main()
