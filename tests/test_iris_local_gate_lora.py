import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import train_iris_local_gate_lora as l


class LoRATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_zero_init_and_actual_native_path_backprop(self):
        # Actual diffusers cross attention, native RouteModel forward, frozen
        # VAE decoder, normalized correction and official differentiable loss.
        from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
        from tools.train_route_suite import RouteModel
        torch.manual_seed(43)
        unet = UNet2DConditionModel(sample_size=8, in_channels=8, out_channels=4,
            down_block_types=('CrossAttnDownBlock2D',)*3+('DownBlock2D',),
            up_block_types=('UpBlock2D',)+('CrossAttnUpBlock2D',)*3,
            block_out_channels=(32,64,64,64), layers_per_block=2, norm_num_groups=8,
            cross_attention_dim=32, attention_head_dim=(1,2,2,2), use_linear_projection=True,
            class_embed_type='projection', projection_class_embeddings_input_dim=4)
        vae = AutoencoderKL(in_channels=3, out_channels=3,
            down_block_types=('DownEncoderBlock2D',)*4, up_block_types=('UpDecoderBlock2D',)*4,
            block_out_channels=(32,32,32,32), layers_per_block=1, norm_num_groups=8,
            latent_channels=4, sample_size=64).requires_grad_(False).eval()
        iris = RouteModel.__new__(RouteModel)
        iris.device = torch.device('cpu')
        iris.args = SimpleNamespace(seed=20260703, num_inference_steps=1, backbone='g', timestep=999,
                                    condition_latent='mode', shuffle_condition='none')
        iris.condition_mode = 'vae'
        # RouteModel.condition selects self.condition_kind in the real loader;
        # using its actual VAE encoding avoids a fabricated trainable input.
        iris.condition = lambda row, image, *unused: iris.vae_latent(image)
        iris.unet = unet.requires_grad_(False).eval()
        iris.lotus = SimpleNamespace(vae=vae, scheduler=DDIMScheduler(prediction_type='sample'),
                                     text_encoder=nn.Linear(1,1).requires_grad_(False))
        iris.gt_vae = copy.deepcopy(vae).requires_grad_(False)
        iris.encode_prompt = lambda text: torch.randn(1, 5 if text else 2, 32, generator=torch.Generator().manual_seed(7))
        image = torch.randn(3,64,64)
        base_depth = torch.linspace(2,30,64*64).reshape(64,64)
        features = torch.randn(1,128,5,5)
        base = nn.Module()
        base.fusions = nn.ModuleList([nn.Identity(), nn.Identity()])
        base.requires_grad_(False)
        def forward(model, batch):
            model.fusions[1](features)
            return base_depth[None]
        class Data:
            samples = [SimpleNamespace(sample_id='sample', manifest_line=17, caption='car')]
            def __getitem__(self, i):
                return dict(iris_image=image, gt=base_depth + .8*base_depth.sin())
        os.environ['IRIS_DECODE_CLAMP'] = '0'
        with patch.object(l.g.old, 'forward_batch', side_effect=forward):
            inputs = l.g.FrozenInputs(base, iris, 'frame')
            original = inputs.get(Data(), 0)
            torch.manual_seed(43)
            layers = l.inject(unet, strict=False)
            gate = l.g.LocalGate()
            model = l.Adaptation(inputs, gate, layers)
            anchor, prediction = l.check_initial(model, Data(), 0)
            torch.testing.assert_close(prediction.detach(), l.g.estimate(gate, *original[:3]).detach(), rtol=1e-6, atol=1e-6)
            before = model.frozen_hash()
            loss = l.g.ssi_absrel(prediction, anchor[3])
            loss.backward()
            for kv in ('k','v'):
                grad = sum(float(m.b.grad.abs().sum()) for n,m in layers.items() if n.endswith('to_'+kv))
                self.assertGreater(grad, 0)
            self.assertGreater(gate.net[-1].weight.grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None for p in iris.gt_vae.parameters()))
            opt = torch.optim.AdamW([p for m in layers.values() for p in (m.a,m.b)], lr=1e-4)
            opt.step()
            self.assertEqual(before, model.frozen_hash())
            self.assertTrue(any(m.b.detach().abs().sum() > 0 for m in layers.values()))
            # Disabled LoRA still reproduces the frozen source after an update.
            updated_anchor = model.anchor(Data(), 0)
            torch.testing.assert_close(updated_anchor[2], anchor[2], rtol=0, atol=0)
            state = model.state()
            with torch.no_grad():
                p1 = model.predict(Data(), 0, updated_anchor)
                model.restore(state)
                torch.testing.assert_close(p1, model.predict(Data(), 0, updated_anchor), rtol=0, atol=0)
            inputs.handle.remove()

    def test_bound_zero_derivative_and_selection(self):
        self.assertNotEqual(l.digest_tensors([('scalar', torch.tensor(0.))]),
                            l.digest_tensors([('scalar', torch.tensor(1.))]))
        b = torch.linspace(1,30,100)
        r0 = torch.randn(100)*.05
        candidate = r0.clone().requires_grad_()
        pred = l.bounded_prediction(b, r0, candidate)
        torch.testing.assert_close(pred, b*r0.exp(), rtol=0, atol=0)
        pred.sum().backward()
        self.assertTrue((candidate.grad > 0).all())
        for offset in (-100., 100.):
            ratio = l.bounded_prediction(b, r0, r0+offset)/(b*r0.exp())
            self.assertGreaterEqual(float(ratio.min()), 1/1.05-1e-6)
            self.assertLessEqual(float(ratio.max()), 1.05+1e-6)
        curve = {s:{'ssi':dict(abs_rel=.07, rmse=2., a1=.96)} for s in l.STEPS}
        self.assertEqual(l.select(curve), 0)
        curve[500]['ssi'].update(abs_rel=.06, rmse=2.01)
        curve[1000]['ssi'].update(abs_rel=.065, a1=.95)
        self.assertEqual(l.select(curve), 0)
        curve[2000]['ssi']['abs_rel'] = .069
        self.assertEqual(l.select(curve), 2000)
        self.assertAlmostEqual(l.lr_factor(1), .01)
        self.assertEqual(l.lr_factor(100), 1)
        self.assertAlmostEqual(l.lr_factor(4000), .1)

    def test_pair_and_checkpoint_reject_mismatches(self):
        common = dict(experiment=l.NAME, parents={'frame':2000, 'empty':6000})
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for arm in l.PARENTS:
                (root/arm).mkdir()
                l.g.old.json_write(root/arm/'smoke.json', dict(arm=arm, common=common, trace='same',
                                    initial_reproduction=True, frozen_unchanged=True))
            l.pair_smoke(root)
            self.assertEqual(l.g.old.read(root/'passed.json'), common)
            entry = l.g.old.read(root/'empty/smoke.json')
            entry['trace'] = 'different image order'
            l.g.old.json_write(root/'empty/smoke.json', entry)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                l.pair_smoke(root)
        ck = dict(arm='empty', common=common, step=500)
        with self.assertRaises(ValueError): l.check_checkpoint(ck, 'frame', common, 500)
        with self.assertRaises(ValueError): l.check_checkpoint(ck, 'empty', common, 1000)

    def test_submit_dependency_and_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            runs = tmp/'runs'
            root = runs/'outputs/lotus_line_v2'
            files = [root/'da2_iris_fusion_matched_pair_v2/empty_s43/step006000.pt',
                     runs/'da2_caption_assets/assets.json']
            for arm, step in l.PARENTS.items():
                parent = root/'iris_visual_gate_matched_v1'/arm
                files.extend(parent/name for name in ('summary.json', 'contract.json', 'full.csv', f'step{step:06d}.pt'))
            for p in files:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
            fake = tmp/'sbatch'
            fake.write_text('#!/bin/bash\necho "$*" >> "$CALLS"\necho 12345\n')
            fake.chmod(0o755)
            calls = tmp/'calls'
            env = dict(os.environ, PATH=d+':'+os.environ['PATH'], IRIS_REPO=str(ROOT), IRIS_RUNS=str(runs),
                       CALLS=str(calls), DA2_CAPTION_ASSETS=str(runs/'da2_caption_assets/assets.json'))
            command = ['bash', str(ROOT/'slurm/submit_iris_local_gate_lora.sh')]
            subprocess.run(command, env=env, check=True, capture_output=True)
            smoke, frame, empty = calls.read_text().splitlines()
            self.assertIn('ACTION=smoke', smoke)
            for cmd, arm in ((frame,'frame'), (empty,'empty')):
                self.assertIn('--dependency=afterok:12345', cmd)
                self.assertIn('--kill-on-invalid-dep=yes', cmd)
                self.assertIn('ACTION=train,ARM='+arm, cmd)
            result = subprocess.run(command, env=env, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(len(calls.read_text().splitlines()), 3)


if __name__ == '__main__':
    unittest.main()
