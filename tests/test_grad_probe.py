"""Unit tests for the per-term gradient probe in train_iris_ms2_g.

Run:
    python -m pytest tests/test_grad_probe.py -q
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lotus"))

from train_iris_ms2_g import _grad_probe_group, probe_gradient_terms  # noqa: E402


class Net(nn.Module):
    def __init__(self, reentrant: bool | None):
        super().__init__()
        self.down_blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.conv_out = nn.Linear(4, 2)
        self.reentrant = reentrant

    def forward(self, x):
        for block in self.down_blocks:
            if self.reentrant is None:
                x = torch.tanh(block(x))
            else:
                x = torch.utils.checkpoint.checkpoint(
                    lambda t, b=block: torch.tanh(b(t)), x, use_reentrant=self.reentrant)
        return self.conv_out(x)


def reference(net, x, fa, fb):
    ga = torch.autograd.grad(fa(net(x)), list(net.parameters()))
    gb = torch.autograd.grad(fb(net(x)), list(net.parameters()))
    va = torch.cat([g.reshape(-1) for g in ga])
    vb = torch.cat([g.reshape(-1) for g in gb])
    return float(va @ vb / (va.norm() * vb.norm())), float(va.norm()), float(vb.norm())


class ProbeTest(unittest.TestCase):
    def test_matches_separate_autograd_with_and_without_checkpointing(self):
        torch.manual_seed(0)
        x = torch.randn(8, 4, requires_grad=True)
        fa = lambda y: y[:, 0].pow(2).mean()  # noqa: E731
        fb = lambda y: (y[:, 1] - y[:, 0]).abs().mean()  # noqa: E731
        base = Net(None)
        expected = reference(base, x, fa, fb)
        for reentrant in (None, True, False):
            net = Net(reentrant)
            net.load_state_dict(base.state_dict())
            out = net(x)
            result = probe_gradient_terms(net, {"a": fa(out), "b": fb(out),
                                                "off": out.new_zeros(())}, scale=1024.0)
            self.assertTrue(result["finite"])
            self.assertEqual(set(result["terms"]), {"a", "b"})  # zero-weight term skipped
            self.assertAlmostEqual(result["global"]["cos"]["a|b"], expected[0], places=5)
            self.assertAlmostEqual(result["global"]["norm"]["a"], expected[1], places=5)
            self.assertAlmostEqual(result["global"]["norm"]["b"], expected[2], places=5)
            self.assertEqual(set(result["groups"]), {"down_blocks.0", "down_blocks.1", "conv_out"})
            self.assertTrue(all(p.grad is None for p in net.parameters()))

    def test_opposed_terms_have_negative_cosine(self):
        net = Net(None)
        out = net(torch.randn(8, 4))
        loss = out.pow(2).mean()
        result = probe_gradient_terms(net, {"up": loss, "down": -0.5 * loss}, scale=1.0)
        self.assertAlmostEqual(result["global"]["cos"]["up|down"], -1.0, places=6)

    def test_group_names(self):
        self.assertEqual(_grad_probe_group("module.up_blocks.3.resnets.0.conv1.weight"), "up_blocks.3")
        self.assertEqual(_grad_probe_group("mid_block.attentions.0.proj_in.weight"), "mid_block")
        self.assertEqual(_grad_probe_group("conv_in.weight"), "conv_in")


if __name__ == "__main__":
    unittest.main()
