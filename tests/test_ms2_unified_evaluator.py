from __future__ import annotations

import unittest

import numpy as np

from ms2_eval.adapters import create_output_adapter
from ms2_eval.aggregate import bootstrap_mean_ci, paired_comparison
from ms2_eval.core import EmptyValidMaskError, PredictionValueError, build_valid_mask, compute_depth_metrics, evaluate_depth_pair
from ms2_eval.resize import resize_dense_prediction, resize_mask_nearest


class UnifiedEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.gt = np.asarray([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]], np.float32)
        self.valid = np.ones_like(self.gt, dtype=bool)

    def test_identity_is_near_perfect(self):
        metrics = compute_depth_metrics(self.gt, self.gt, self.valid)
        self.assertLess(metrics["rmse_m"], 1e-8)
        self.assertLess(metrics["abs_rel"], 1e-8)
        self.assertAlmostEqual(metrics["delta1"], 1.0)
        self.assertAlmostEqual(metrics["delta2"], 1.0)
        self.assertAlmostEqual(metrics["delta3"], 1.0)

    def test_double_gt_is_bad_raw_and_perfect_aligned(self):
        metrics, aligned = evaluate_depth_pair(2.0 * self.gt, self.gt, self.valid)
        self.assertGreater(metrics["abs_rel"], 0.9)
        self.assertGreater(metrics["rmse_m"], 1.0)
        self.assertLess(metrics["aligned_rmse_m"], 1e-5)
        np.testing.assert_allclose(aligned, self.gt, atol=1e-5)

    def test_invalid_gt_values_cannot_change_metrics(self):
        mask = self.valid.copy(); mask[0, 0] = False
        gt_a, gt_b = self.gt.copy(), self.gt.copy(); gt_a[0, 0] = 0.0; gt_b[0, 0] = 999999.0
        pred = self.gt.copy(); pred[0, 0] = 7.0
        self.assertEqual(compute_depth_metrics(pred, gt_a, mask), compute_depth_metrics(pred, gt_b, mask))

    def test_strict_valid_mask_boundaries(self):
        gt = np.asarray([[0.1, 0.1001, 79.9, 80.0, np.nan]], np.float32)
        np.testing.assert_array_equal(build_valid_mask(gt, 0.1, 80.0), [[False, True, True, False, False]])

    def test_resize_prediction_bilinear_and_mask_nearest(self):
        pred = np.asarray([[1.0, 2.0], [3.0, 4.0]], np.float32)
        mask = np.asarray([[True, False], [False, True]])
        resized_pred = resize_dense_prediction(pred, (4, 4)); resized_mask = resize_mask_nearest(mask, (4, 4))
        self.assertEqual(resized_pred.shape, (4, 4)); self.assertEqual(resized_mask.dtype, np.bool_)
        self.assertEqual(set(np.unique(resized_mask)), {False, True})
        np.testing.assert_array_equal(resized_mask[:2, :2], np.ones((2, 2), bool))
        np.testing.assert_array_equal(resized_mask[:2, 2:], np.zeros((2, 2), bool))

    def test_empty_valid_mask_is_clear_error(self):
        with self.assertRaisesRegex(EmptyValidMaskError, "No valid GT pixels"):
            compute_depth_metrics(self.gt, self.gt, np.zeros_like(self.valid))

    def test_nan_inf_predictions_are_reported(self):
        for value in (np.nan, np.inf, -np.inf):
            pred = self.gt.copy(); pred[0, 0] = value
            with self.assertRaisesRegex(PredictionValueError, "invalid values"):
                compute_depth_metrics(pred, self.gt, self.valid)

    def test_nonmetric_adapter_suppresses_raw_metric_labels(self):
        adapter = create_output_adapter("iris-lotus")
        adapted = adapter.adapt(np.asarray([[1.0, 2.0], [4.0, 8.0]], np.float32))
        self.assertFalse(adapted.declaration.metric_scale_exists)
        metrics, _ = evaluate_depth_pair(adapted.depth, np.asarray([[8.0, 4.0], [2.0, 1.0]], np.float32), np.ones((2, 2), bool), raw_is_metric=False)
        self.assertIsNone(metrics["rmse_m"]); self.assertIsNotNone(metrics["aligned_rmse_m"])

    def test_registered_routes_have_explicit_declarations(self):
        for route in ("iris-lotus", "adapter-only", "adapter+u-net", "sp-dit", "anythermal", "anythermal_midas"):
            declaration = create_output_adapter(route).metadata()
            for key in ("raw_representation_type", "orientation", "metric_scale_exists", "conversion_to_positive_depth", "clipping_rules"):
                self.assertIn(key, declaration)

    def test_anythermal_is_relative_but_neither_inverted_nor_clamped(self):
        """Upstream fits scale+shift on unclamped output, so both would break the protocol."""
        adapter = create_output_adapter("anythermal")
        raw = np.asarray([[-1.0, 2.0], [4.0, 8.0]], np.float32)
        adapted = adapter.adapt(raw)
        np.testing.assert_allclose(adapted.depth, raw)
        self.assertEqual(adapted.diagnostics["nonpositive_raw"], 1)
        self.assertFalse(adapted.declaration.metric_scale_exists)
        self.assertEqual(adapted.declaration.orientation, "larger-is-farther")
        # SP-DiT shares the identity conversion yet must clamp; Lotus shares
        # metric_scale_exists=False yet must invert. AnyThermal does neither.
        self.assertGreater(create_output_adapter("sp-dit").adapt(raw).depth.min(), 0.0)
        np.testing.assert_allclose(create_output_adapter("iris-lotus").adapt(raw).depth[0, 1], 0.5)

    def test_depth_clips_are_rejected_before_alignment(self):
        with self.assertRaisesRegex(ValueError, "meaningless before affine alignment"):
            create_output_adapter("anythermal", clip_max=80.0).adapt(np.ones((2, 2), np.float32))

    def test_bootstrap_and_paired_direction(self):
        ci = bootstrap_mean_ci([1.0, 2.0, 3.0], iterations=200, seed=1)
        self.assertLessEqual(ci["ci_low"], ci["mean"]); self.assertGreaterEqual(ci["ci_high"], ci["mean"])
        a = [{"sample_id": "x", "rmse_m": 1.0}, {"sample_id": "y", "rmse_m": 2.0}]
        b = [{"sample_id": "x", "rmse_m": 2.0}, {"sample_id": "y", "rmse_m": 3.0}]
        result = paired_comparison(a, b, label_a="A", label_b="B", metrics=["rmse_m"], iterations=200)
        self.assertEqual(result["metrics"]["rmse_m"]["win_rate"], 1.0)


if __name__ == "__main__": unittest.main()
