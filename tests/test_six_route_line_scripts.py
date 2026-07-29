from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.thermal_vae_latent_adapter import ThermalVAELatentAdapter


class LatentAdapterTests(unittest.TestCase):
    def test_identity_at_initialization(self):
        adapter = ThermalVAELatentAdapter(hidden_channels=32, blocks=2)
        latent = torch.randn(2, 4, 32, 80)
        with torch.no_grad():
            output = adapter(latent)
        torch.testing.assert_close(output, latent)  # zero-init conv_out => exact identity

    def test_gradient_reaches_all_parameters(self):
        adapter = ThermalVAELatentAdapter(hidden_channels=32, blocks=2)
        latent = torch.randn(1, 4, 16, 40)
        loss = adapter(latent).square().mean()
        loss.backward()
        missing = [name for name, p in adapter.named_parameters() if p.grad is None]
        self.assertEqual(missing, [])

    def test_shape_and_channel_validation(self):
        adapter = ThermalVAELatentAdapter(hidden_channels=32, blocks=2)
        with self.assertRaises(ValueError):
            adapter(torch.randn(1, 3, 16, 40))
        with self.assertRaises(ValueError):
            adapter(torch.randn(4, 16, 40))
        with self.assertRaises(ValueError):
            ThermalVAELatentAdapter(hidden_channels=30, blocks=2)  # not divisible by 8

    def test_default_capacity_is_comparable_to_line_f(self):
        adapter = ThermalVAELatentAdapter()
        parameters = sum(p.numel() for p in adapter.parameters())
        self.assertGreater(parameters, 5_000_000)  # line f adapter is ~9.4M
        self.assertLess(parameters, 12_000_000)


class DownscaledSizeTests(unittest.TestCase):
    def test_native_kept_when_zero_or_small(self):
        from tools.train_ms2_rgb_unet_gt import downscaled_size

        self.assertEqual(downscaled_size(384, 1224, 0), (384, 1224))
        self.assertEqual(downscaled_size(384, 1224, 2000), (384, 1224))

    def test_downscale_is_divisible_by_8(self):
        from tools.train_ms2_rgb_unet_gt import downscaled_size

        height, width = downscaled_size(384, 1224, 768)
        self.assertEqual(height % 8, 0)
        self.assertEqual(width % 8, 0)
        self.assertLessEqual(max(height, width), 768)
        self.assertGreater(min(height, width), 0)
        # aspect ratio approximately preserved
        self.assertAlmostEqual(width / height, 1224 / 384, delta=0.15)


class ScriptSyntaxTests(unittest.TestCase):
    SCRIPTS = (
        "tools/run_ms2_lotus_rgb_official.py",
        "tools/run_ms2_lotus_thermal_vae_official.py",
        "tools/run_ms2_lotus_direct_official.py",
        "tools/run_ms2_lotus_trained_official.py",
        "tools/train_ms2_rgb_unet_gt.py",
        "tools/train_ms2_vae_latent_adapter_gt.py",
        "tools/run_official_ms2_evaluation.py",
    )

    def test_all_six_route_scripts_parse(self):
        for relative in self.SCRIPTS:
            source = (ROOT / relative).read_text(encoding="utf-8")
            ast.parse(source, filename=relative)


if __name__ == "__main__":
    unittest.main()
