import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import train_iris_local_gate as g


class GateTests(unittest.TestCase):
    def test_direction_visual_reference_and_gate_bound(self):
        base = torch.linspace(1., 20., 60).reshape(6, 10)
        raw = base.sin()
        delta = g.shape_delta(raw, base)
        self.assertTrue(torch.allclose(delta, g.shape_delta(3*raw+7, 2*base), atol=2e-6))
        visual = torch.randn(1, 128, 4, 5)
        gate = g.LocalGate()
        self.assertIs(g.estimate(None, base, visual, delta), base)
        torch.testing.assert_close(gate(visual, base.shape), torch.full_like(base, g.W_INIT))
        with torch.no_grad():
            nn.init.normal_(gate.net[-1].weight, std=20)
        w = gate(visual, base.shape)
        self.assertTrue(((w >= 0) & (w <= g.W_MAX)).all())
        for direction in (delta, -delta):
            pred = g.estimate(gate, base, visual, direction)
            torch.testing.assert_close((pred/base).log(), w*direction, atol=1e-6, rtol=1e-5)

    def test_training_metric_official_and_gradient(self):
        base = torch.linspace(3., 19., 120).reshape(10, 12)
        gt = base + base.sin()
        gt[0,:4] = torch.tensor([0., .001, 80., float("nan")])
        gate = g.LocalGate()
        pred = g.estimate(gate, base, torch.randn(1,128,4,4), base.sin())
        loss = g.ssi_absrel(pred, gt)
        expected = g.old.evaluate_sample(pred.detach().numpy(), gt.numpy(), align="ssi")
        self.assertAlmostEqual(loss.item(), expected["abs_rel"], places=12)
        loss.backward()
        self.assertGreater(gate.net[-1].weight.grad.norm().item(), 0)
        self.assertIsNone(g.ssi_absrel(pred, torch.zeros_like(gt)))

    def test_matched_prompts_and_zero_selection(self):
        self.assertEqual(g.prompt_for("empty", object()), "")
        self.assertEqual(g.prompt_for("frame", SimpleNamespace(caption="left car")), "left car")
        curve = {s: {"ssi":{"abs_rel": .07}} for s in g.STEPS}
        self.assertEqual(g.select(curve), 0)
        curve[2000]["ssi"]["abs_rel"] -= .0001
        self.assertEqual(g.select(curve), 2000)

    def test_matched_input_path_shares_visual_features_and_learns_in_either_arm(self):
        image = torch.linspace(1., 4., 48).reshape(6, 8)
        feat = torch.randn(1,128,3,4)
        base = SimpleNamespace(fusions=[None, nn.Identity()])
        def frozen_forward(model, batch):
            model.fusions[1](feat)
            return image[None]
        class Data:
            samples = [SimpleNamespace(caption="left car", manifest_line=17)]
            def __getitem__(self, i):
                return dict(iris_image=image[None], gt=image)
        calls = []
        class Iris:
            def encode_prompt(self, text): return text
            def predict_disparity(self, row, thermal, prompt):
                calls.append((row["manifest_index"], prompt))
                return image.sin() if prompt else image.cos()
        results = []
        for arm in ("frame", "empty"):
            inputs = g.FrozenInputs(base, Iris(), arm)
            with patch.object(g.old, "forward_batch", side_effect=frozen_forward):
                depth, visual, delta, _ = inputs.get(Data(), 0)
            inputs.handle.remove()
            torch.testing.assert_close(visual, feat)
            torch.testing.assert_close(depth, image)
            self.assertGreater(delta.abs().mean().item(), 0.)
            results.append(delta)
        self.assertEqual(calls, [(16, "left car"), (16, "")])
        self.assertFalse(torch.equal(*results))

    def test_submission_links_both_jobs_to_the_same_smoke(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            runs = tmp/"runs"
            for rel in ("outputs/lotus_line_v2/da2_iris_fusion_matched_pair_v2/empty_s43/step006000.pt",
                        "outputs/lotus_line_v2/da2_iris_fusion_matched_pair_v2/empty_s43/summary.json",
                        "outputs/lotus_line_v2/da2_iris_fusion_matched_pair_v2/empty_s43/selected_full/val_step006000_frames.csv",
                        "da2_caption_assets/assets.json"):
                p = runs/rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
            fake = tmp/"sbatch"
            fake.write_text('#!/bin/bash\necho "$*" >> "$CALLS"\necho 12345\n')
            fake.chmod(0o755)
            calls = tmp/"calls"
            env = dict(os.environ, PATH=d+":"+os.environ["PATH"], IRIS_REPO=str(root),
                       IRIS_RUNS=str(runs), CALLS=str(calls), DA2_CAPTION_ASSETS=str(runs/"da2_caption_assets/assets.json"))
            subprocess.run(["bash", str(root/"slurm/submit_iris_local_gate.sh")], env=env, check=True, capture_output=True)
            smoke, frame, empty = calls.read_text().splitlines()
            self.assertIn("ACTION=smoke", smoke)
            for cmd, arm in ((frame, "frame"), (empty, "empty")):
                self.assertIn("--dependency=afterok:12345", cmd)
                self.assertIn("ACTION=train,ARM="+arm, cmd)


if __name__ == "__main__":
    unittest.main()
