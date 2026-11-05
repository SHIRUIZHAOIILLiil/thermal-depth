"""Small CPU contract tests; the real-checkpoint reproduction is the GPU smoke."""
import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from tools import train_iris_decoder_only as runner
from tools.iris_decoder_only import InferenceInput, fingerprint, predict_frozen, make_decoder, decode_unit, select
from tools.train_ms2_joint_gt_v3 import _denormalise


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(1, 1, 1)
        self.post_quant_conv = nn.Conv2d(1, 1, 1)
        self.decoder = nn.Conv2d(1, 3, 1)
        self.config = SimpleNamespace(scaling_factor=.18215)

    def decode(self, z, return_dict=False):
        return (self.decoder(self.post_quant_conv(z)),)


def model_fixture():
    return SimpleNamespace(unet=nn.Conv2d(1, 1, 1), gt_vae=TinyVAE(),
        lotus=SimpleNamespace(vae=TinyVAE(), text_encoder=nn.Linear(2, 2)), modality='thermal')


class DecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)

    def test_initial_output_and_only_decoder_update(self):
        model = model_fixture()
        decoder, params = make_decoder(model)
        latent = torch.randn(1, 1, 16, 32)
        with torch.no_grad():
            expected = _denormalise(model.gt_vae.decode(latent/.18215)[0])
        actual = decode_unit(decoder, latent)[0, 0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        before = runner.fingerprint(runner.frozen_parameters(model, decoder))
        initial = runner.fingerprint(params.items())
        self.assertEqual(set(params), {'decoder.weight', 'decoder.bias'})
        optimizer = torch.optim.AdamW(params.values(), lr=1e-5)
        (actual-.3).square().mean().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in params.values()))
        self.assertTrue(all(p.grad is None for _, p in runner.frozen_parameters(model, decoder)))
        optimizer.step()
        self.assertEqual(before, runner.fingerprint(runner.frozen_parameters(model, decoder)))
        self.assertNotEqual(initial, runner.fingerprint(params.items()))

    def test_gt_free_inference_boundary_and_backward(self):
        class Sample:
            sample_id, thermal_path, caption, manifest_line = 'frame', '/image', 'road', 7
            @property
            def thermal_gt_path(self):
                raise AssertionError('GT path read during prediction')
        model = model_fixture()
        decoder, _ = make_decoder(model)
        def predict(row, image, prompt, **kwargs):
            self.assertEqual(row, dict(id='frame', image_path='/image', manifest_index=6))
            self.assertEqual(kwargs, dict(target_latent=None, return_latent=True, noise_offset=0))
            self.assertFalse(torch.is_grad_enabled())
            return image[0, 0], image*2
        model.encode_prompt = lambda text: torch.tensor([len(text)])
        model.predict_disparity = predict
        with patch('tools.train_route_suite.load_input_tensor', return_value=(torch.ones(1,1,8,8), {})):
            raw, latent = predict_frozen(model, InferenceInput.from_sample(Sample()), SimpleNamespace())
        self.assertFalse(latent.requires_grad)
        self.assertFalse(torch.is_inference(latent))
        decode_unit(decoder, latent).square().mean().backward()
        self.assertGreater(float(decoder.decoder.weight.grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            predict_frozen(model, Sample(), SimpleNamespace())

    def test_real_existing_losses_reach_decoder_but_not_latent(self):
        # Execute the actual two loss primitives without importing the heavyweight
        # training CLI or downloading models. No duplicate loss implementation.
        path = ROOT/'lotus/train_iris_ms2_g.py'
        names = {'decode_image_depth', '_ssi_log_aligned_depth'}
        tree = ast.parse(path.read_text(encoding='utf8'))
        module = ModuleType('train_iris_ms2_g')
        module.torch = torch
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)
                                      and n.name in names], type_ignores=[]),str(path),'exec'), module.__dict__)
        model = model_fixture()
        decoder, params = make_decoder(model)
        latent = torch.randn(2,1,16,32, requires_grad=True)
        gt = torch.linspace(5,79,512).reshape(1,16,32)
        examples = [dict(gt_depth_values=gt.clone(), gt_valid_values=torch.ones_like(gt,dtype=torch.bool),
                         norm_bounds=torch.tensor([1.,4.])) for _ in range(2)]
        with patch.dict(sys.modules, {'train_iris_ms2_g':module}):
            loss, stats = runner.loss_for_batch(decoder,latent,examples,[1,2])
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats['far_frames'], 2)
        self.assertGreater(stats['pairs'], 0)
        loss.backward()
        self.assertIsNone(latent.grad)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in params.values()),0)
        self.assertTrue(all(p.grad is None for _,p in runner.frozen_parameters(model,decoder)))

    def test_fingerprint_reads_channels_last_one_by_one_convolutions(self):
        values = torch.arange(16, dtype=torch.float32).reshape(4, 4, 1, 1)
        stored = values.clone(memory_format=torch.channels_last)
        self.assertNotEqual(stored.stride(-1), 1)
        with self.assertRaises(RuntimeError):
            stored.contiguous().view(torch.uint8)
        self.assertEqual(fingerprint([('conv', stored)]), fingerprint([('conv', values)]))

    def test_selection_keeps_start_or_accepts_bounded_tradeoff(self):
        ref = dict(abs_rel=.08, rmse=2.47, a1=.95)
        base = dict(abs_rel=.079, rmse=2.74, a1=.945)
        tiny = dict(abs_rel=.079, rmse=2.73, a1=.945)
        chosen, report = select({0:base,500:tiny,1000:tiny},ref)
        self.assertEqual(chosen,0)
        self.assertEqual(set(report['candidates']),{0,500,1000})
        better = dict(abs_rel=.080, rmse=2.60, a1=.945)
        bad = dict(abs_rel=.082, rmse=2.50, a1=.94)
        chosen, report = select({0:base,500:better,1000:bad},ref)
        self.assertEqual((chosen, report['status']),(500,'narrowed_gap_on_val'))

    def test_submission_only_smoke_then_train(self):
        bash = 'D:/Git/bin/bash.exe' if os.name=='nt' else shutil.which('bash')
        if not bash or not Path(bash).exists():
            self.skipTest('Bash unavailable')
        # Git Bash environment paths on Windows may corrupt non-ASCII TEMP.
        with tempfile.TemporaryDirectory(prefix='decoder_submit_', dir=ROOT) as tmp:
            root = Path(tmp)
            base = root/'runs/outputs/lotus_line_v2'
            parent = base/'iris_far_bias_square_l050_frame_v1/run_parent'
            parent.mkdir(parents=True)
            (parent/'jobs.txt').write_text('frame train+val -> 9171487 (afterok:9171486)\n')
            for file in ('summary.json','contract.json','conv_4000.pt'):
                (parent/file).write_text('{}')
            archive = base/'iris_pair_rmse_val_v1/job_9167854'
            archive.mkdir(parents=True)
            (archive/'summary.json').write_text('{}')
            binary = root/'bin'
            binary.mkdir()
            stub = binary/'sbatch'
            stub.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\necho 12345\n',newline='\n')
            stub.chmod(0o755)
            calls = root/'calls'
            env = dict(os.environ,IRIS_REPO=ROOT.as_posix(), IRIS_RUNS=(root/'runs').as_posix(),
                       CALLS=calls.as_posix(), MOCK_BIN=binary.as_posix())
            result = subprocess.run([bash,'-c','export PATH="$(cd "$MOCK_BIN" && pwd):$PATH"; bash slurm/submit_iris_decoder_only.sh'],
                                    cwd=ROOT,env=env,text=True,encoding='utf8',errors='replace',capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            lines = calls.read_text().splitlines()
            self.assertEqual(len(lines),2)
            self.assertIn(' smoke ',lines[0])
            self.assertIn('--dependency=afterok:12345',lines[1])
            self.assertIn(' train ',lines[1])
            self.assertNotIn('test', '\n'.join(lines))


if __name__=='__main__':
    unittest.main()
