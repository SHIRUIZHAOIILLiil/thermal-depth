from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

import numpy as np

from ms2_eval.official_protocol import (
    OfficialProtocolError,
    collapse_channels,
    evaluate_sample,
    fit_scale_shift,
    median_scale_ratio,
    official_depth_errors,
    official_valid_mask,
)


def find_bmsd_root() -> Path | None:
    candidates = [os.environ.get("BMSD_ROOT"),
                  "/mnt/e/project/AnyThermal/baselines/depth/BridgeMultiSpectralDepth",
                  "E:/project/AnyThermal/baselines/depth/BridgeMultiSpectralDepth"]
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    return None


def load_module_from_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfficialProtocolTests(unittest.TestCase):
    def setUp(self):
        self.gt = np.asarray([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]], np.float32)
        self.valid = np.ones_like(self.gt, dtype=bool)

    def test_valid_mask_bounds_are_exclusive(self):
        gt = np.asarray([[0.0, 1e-3, 0.0011, 79.9, 80.0, np.nan]], np.float32)
        mask = official_valid_mask(gt, 1e-3, 80.0)
        np.testing.assert_array_equal(mask, [[False, False, True, True, False, False]])

    def test_ssi_recovers_affine_transform(self):
        pred = 0.5 * self.gt - 3.0
        scale, shift = fit_scale_shift(pred, self.gt, self.valid)
        self.assertAlmostEqual(scale, 2.0, places=6)
        self.assertAlmostEqual(shift, 6.0, places=6)
        row = evaluate_sample(pred, self.gt, align="ssi")
        self.assertLess(row["abs_rel"], 1e-6)
        self.assertAlmostEqual(row["a1"], 1.0)
        self.assertAlmostEqual(row["rmse"], 0.0, places=5)

    def test_ssi_rejects_constant_prediction(self):
        with self.assertRaises(OfficialProtocolError):
            fit_scale_shift(np.full_like(self.gt, 5.0), self.gt, self.valid)

    def test_median_scaling_fixes_pure_scale_but_not_shift(self):
        row = evaluate_sample(2.0 * self.gt, self.gt, align="median")
        self.assertLess(row["abs_rel"], 1e-6)
        self.assertAlmostEqual(row["alignment_scale"], 0.5, places=6)
        row_shifted = evaluate_sample(2.0 * self.gt + 5.0, self.gt, align="median")
        self.assertGreater(row_shifted["abs_rel"], 0.05)

    def test_median_scaling_rejects_nonpositive_median(self):
        with self.assertRaises(OfficialProtocolError):
            median_scale_ratio(np.zeros_like(self.gt), self.gt, self.valid)

    def test_clamp_is_applied_before_metrics(self):
        gt = np.full((2, 2), 79.0, np.float32)
        pred = np.full((2, 2), 200.0, np.float32)
        row = evaluate_sample(pred, gt, align="none")
        self.assertEqual(row["clamped_above"], 4)
        self.assertAlmostEqual(row["abs_diff"], 1.0, places=5)  # clamped to 80 vs gt 79

    def test_disparity_style_input_gets_negative_scale(self):
        pred = (1.0 / self.gt).astype(np.float32)  # larger-is-nearer, like Lotus raw output
        row = evaluate_sample(pred, self.gt, align="ssi")
        self.assertLess(row["alignment_scale"], 0.0)
        for key in ("abs_rel", "rmse", "a1"):
            self.assertTrue(np.isfinite(row[key]))
        # affine in disparity space cannot fully linearise 1/x: strictly worse than perfect
        self.assertGreater(row["abs_rel"], 1e-3)

    def test_ssi_disparity_recovers_affine_disparity(self):
        # model outputs an affine transform of GT *disparity* (the Lotus case)
        pred = (3.0 / self.gt + 0.25).astype(np.float32)
        row = evaluate_sample(pred, self.gt, align="ssi_disparity")
        self.assertAlmostEqual(row["alignment_scale"], 1.0 / 3.0, places=5)
        self.assertAlmostEqual(row["alignment_shift"], -0.25 / 3.0, places=5)
        self.assertLess(row["abs_rel"], 1e-5)
        self.assertAlmostEqual(row["a1"], 1.0)

    def test_ssi_disparity_beats_plain_ssi_on_disparity_output(self):
        pred = (1.0 / self.gt).astype(np.float32)
        plain = evaluate_sample(pred, self.gt, align="ssi")
        disparity = evaluate_sample(pred, self.gt, align="ssi_disparity")
        self.assertLess(disparity["abs_rel"], plain["abs_rel"])

    def test_collapse_channels_shapes(self):
        base = np.arange(20, dtype=np.float32).reshape(4, 5)
        for shaped in (base, base[None], base[..., None], np.stack([base] * 3, 0), np.stack([base] * 3, -1)):
            np.testing.assert_allclose(collapse_channels(shaped), base)
        with self.assertRaises(OfficialProtocolError):
            collapse_channels(np.zeros((2, 3, 4)))

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(OfficialProtocolError):
            evaluate_sample(np.ones((3, 3), np.float32), self.gt, align="none")

    def test_empty_mask_is_rejected(self):
        gt = np.zeros((2, 2), np.float32)
        with self.assertRaises(OfficialProtocolError):
            evaluate_sample(np.ones((2, 2), np.float32), gt, align="none")

    def test_hand_computed_abs_rel(self):
        gt = np.asarray([[2.0, 4.0]], np.float32)
        pred = np.asarray([[1.0, 5.0]], np.float32)
        metrics = official_depth_errors(pred, gt, np.ones_like(gt, bool))
        self.assertAlmostEqual(metrics["abs_rel"], (1.0 / 2.0 + 1.0 / 4.0) / 2.0, places=6)
        self.assertAlmostEqual(metrics["abs_diff"], 1.0, places=6)


@unittest.skipIf(find_bmsd_root() is None, "BridgeMultiSpectralDepth clone not found (set BMSD_ROOT)")
class CrossCheckAgainstOfficialTorch(unittest.TestCase):
    """Numerically validate the numpy port against the cloned official code."""

    @classmethod
    def setUpClass(cls):
        import torch  # noqa: F401  (skip the whole class if torch is unavailable)

        root = find_bmsd_root()
        cls.eval_metric = load_module_from_file("bmsd_eval_metric", root / "models" / "metrics" / "eval_metric.py")
        cls.midas_loss = load_module_from_file("bmsd_midas_loss", root / "models" / "losses" / "midas_loss.py")
        rng = np.random.default_rng(20260716)
        cls.gt = (rng.uniform(0.5, 60.0, size=(24, 48))).astype(np.float32)
        cls.gt[rng.uniform(size=cls.gt.shape) < 0.3] = 0.0  # sparse invalid pixels

    def to_torch(self, array, shape):
        import torch
        return torch.from_numpy(np.ascontiguousarray(array, np.float32)).reshape(shape)

    def test_median_path_matches_official(self):
        import torch
        rng = np.random.default_rng(7)
        pred = (self.gt * 1.7 + rng.normal(0, 1.0, self.gt.shape)).clip(0.05).astype(np.float32)
        official = self.eval_metric.compute_depth_errors(
            self.to_torch(self.gt, (1, *self.gt.shape)), self.to_torch(pred, (1, 1, *pred.shape)),
            align=True, dataset="MS2")
        mine = evaluate_sample(pred, self.gt, align="median")
        for value, name in zip(official, ("abs_diff", "abs_rel", "sq_rel", "log10", "rmse", "rmse_log", "a1", "a2", "a3")):
            self.assertAlmostEqual(mine[name], float(value), places=4, msg=name)

    def test_ssi_path_matches_official(self):
        import torch
        rng = np.random.default_rng(11)
        pred = (0.02 * self.gt - 0.5 + rng.normal(0, 0.05, self.gt.shape)).astype(np.float32)
        gt_t = self.to_torch(self.gt, (1, *self.gt.shape))
        pred_t = self.to_torch(pred, (1, *pred.shape))
        mask = (gt_t > 1e-3) & (gt_t < 80.0)
        scale, shift = self.midas_loss.compute_scale_and_shift(pred_t, gt_t, mask.float())
        fitted = scale.view(-1, 1, 1) * pred_t + shift.view(-1, 1, 1)
        official = self.eval_metric.compute_depth_errors(gt_t, fitted.unsqueeze(1), align=False, dataset="MS2")
        mine = evaluate_sample(pred, self.gt, align="ssi")
        self.assertAlmostEqual(mine["alignment_scale"], float(scale[0]), places=4)
        self.assertAlmostEqual(mine["alignment_shift"], float(shift[0]), places=4)
        for value, name in zip(official, ("abs_diff", "abs_rel", "sq_rel", "log10", "rmse", "rmse_log", "a1", "a2", "a3")):
            self.assertAlmostEqual(mine[name], float(value), places=4, msg=name)


if __name__ == "__main__":
    unittest.main()
