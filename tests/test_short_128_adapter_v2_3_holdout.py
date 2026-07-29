"""Tests for the frozen V2.3 128/16 Train partition and gate."""

from __future__ import annotations

import unittest

from tools.short_128_adapter_v2_3_holdout import gate_metrics, partition_uniform_cache


class Short128HoldoutTest(unittest.TestCase):
    def test_partition_is_disjoint_deterministic_and_has_fixed_sizes(self) -> None:
        cache = [{"id": f"sample_{index:03d}"} for index in range(144)]
        first_train, first_holdout = partition_uniform_cache(cache, 128, 16)
        second_train, second_holdout = partition_uniform_cache(cache, 128, 16)
        self.assertEqual(len(first_train), 128)
        self.assertEqual(len(first_holdout), 16)
        self.assertFalse(
            {item["id"] for item in first_train}
            & {item["id"] for item in first_holdout}
        )
        self.assertEqual(
            [item["id"] for item in first_train],
            [item["id"] for item in second_train],
        )
        self.assertEqual(
            [item["id"] for item in first_holdout],
            [item["id"] for item in second_holdout],
        )
        self.assertTrue(all(item["short_split"] == "train" for item in first_train))
        self.assertTrue(
            all(item["short_split"] == "holdout" for item in first_holdout)
        )

    def test_gate_requires_correlation_and_spatial_energy(self) -> None:
        initial = {"mse": 1.0}
        final = {
            "mse": 0.2,
            "cosine": 0.95,
            "pearson": 0.94,
            "gradient_energy_ratio": 0.8,
            "fixed_eight": [{"finite": True}],
        }
        self.assertTrue(gate_metrics(initial, final, mse_ratio_limit=0.25)["passed"])
        final["gradient_energy_ratio"] = 0.2
        self.assertFalse(gate_metrics(initial, final, mse_ratio_limit=0.25)["passed"])

    def test_partition_rejects_wrong_cache_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected"):
            partition_uniform_cache([{"id": "only"}], 128, 16)


if __name__ == "__main__":
    unittest.main()
