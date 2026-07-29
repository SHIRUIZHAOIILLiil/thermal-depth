"""Phase B tests for the audited Adapter V2 Lotus target convention."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from models.lotus_target_v2 import (
    LOTUS_NORMALIZATION_EPS,
    LOTUS_TRUNC_QUANTILE,
    seeded_target_latent_and_noise,
    trunc_disparity_target,
)


class _FakeLatentDistribution:
    def __init__(self, mean: torch.Tensor) -> None:
        self.mean = mean

    def sample(self, generator: torch.Generator) -> torch.Tensor:
        return self.mean + torch.randn(
            self.mean.shape,
            generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )


class _FakeVAE:
    config = SimpleNamespace(scaling_factor=0.25)

    def encode(self, values: torch.Tensor) -> SimpleNamespace:
        mean = values.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
        return SimpleNamespace(latent_dist=_FakeLatentDistribution(mean))


class TruncDisparityTargetTest(unittest.TestCase):
    def test_matches_upstream_quantile_clip_and_mapping_formula(self) -> None:
        depth = torch.tensor([[2.0, 4.0, 10.0, 50.0]], dtype=torch.float32)
        result = trunc_disparity_target(depth)

        disparity = depth.reciprocal()
        lo = torch.quantile(disparity, LOTUS_TRUNC_QUANTILE)
        hi = torch.quantile(disparity, 1.0 - LOTUS_TRUNC_QUANTILE)
        expected = (
            2.0
            * ((disparity - lo) / (hi - lo + LOTUS_NORMALIZATION_EPS) - 0.5)
        ).clamp(-1.0, 1.0)

        torch.testing.assert_close(result.values, expected, rtol=0, atol=0)
        torch.testing.assert_close(result.disparity_min, lo, rtol=0, atol=0)
        torch.testing.assert_close(result.disparity_max, hi, rtol=0, atol=0)

    def test_near_depth_has_larger_disparity_and_normalized_target(self) -> None:
        depth = torch.tensor([[2.0, 10.0, 50.0]])
        result = trunc_disparity_target(depth)

        self.assertGreater(result.disparity[0, 0], result.disparity[0, 1])
        self.assertGreater(result.disparity[0, 1], result.disparity[0, 2])
        self.assertGreater(result.values[0, 0], result.values[0, 1])
        self.assertGreater(result.values[0, 1], result.values[0, 2])

    def test_invalid_content_does_not_change_valid_statistics_or_values(self) -> None:
        valid = torch.tensor([[True, True, False, True, False]])
        depth_a = torch.tensor([[2.0, 10.0, 0.0, 50.0, -1.0]])
        depth_b = torch.tensor([[2.0, 10.0, 1.0e9, 50.0, 0.25]])

        result_a = trunc_disparity_target(depth_a, valid)
        result_b = trunc_disparity_target(depth_b, valid)

        torch.testing.assert_close(result_a.values[valid], result_b.values[valid])
        torch.testing.assert_close(result_a.disparity_min, result_b.disparity_min)
        torch.testing.assert_close(result_a.disparity_max, result_b.disparity_max)
        self.assertTrue(torch.isnan(result_a.values[~valid]).all())
        self.assertTrue(torch.isnan(result_b.values[~valid]).all())

    def test_zero_depth_is_excluded_before_reciprocal(self) -> None:
        depth = torch.tensor([[2.0, 0.0, 10.0, 50.0]])
        result = trunc_disparity_target(depth)

        self.assertFalse(result.valid_mask[0, 1])
        self.assertTrue(torch.isnan(result.disparity[0, 1]))
        self.assertTrue(torch.isfinite(result.disparity[result.valid_mask]).all())

    def test_empty_valid_mask_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "No valid positive depth pixels"):
            trunc_disparity_target(
                torch.tensor([[2.0, 10.0]]),
                torch.zeros((1, 2), dtype=torch.bool),
            )

    def test_nan_and_inf_depth_raise_clear_error(self) -> None:
        for bad_value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad_value=bad_value):
                with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                    trunc_disparity_target(torch.tensor([[2.0, bad_value]]))


class SeededTargetLatentTest(unittest.TestCase):
    def test_same_seed_reproduces_target_latent_and_noise_exactly(self) -> None:
        vae = _FakeVAE()
        target = torch.linspace(-1.0, 1.0, 48).reshape(1, 3, 4, 4)

        first = seeded_target_latent_and_noise(vae, target, seed=20260703)
        second = seeded_target_latent_and_noise(vae, target, seed=20260703)

        self.assertTrue(torch.equal(first.target_latent, second.target_latent))
        self.assertTrue(torch.equal(first.noise, second.noise))

    def test_sparse_target_is_rejected_before_vae_encode(self) -> None:
        sparse = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
        sparse[:, :, 0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "finite floating-point dense target"):
            seeded_target_latent_and_noise(_FakeVAE(), sparse, seed=1)


if __name__ == "__main__":
    unittest.main()
