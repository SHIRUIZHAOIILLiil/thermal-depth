"""CPU gradient/scope and dependency checks; actual checkpoints are gated on Aire."""
import ast
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace, MethodType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from tools import train_iris_decoder_joint as run
from tools.train_route_suite import RouteModel


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(1,1,1)
        self.post_quant_conv = nn.Conv2d(1,1,1)
        self.decoder = nn.Conv2d(1,3,1)
        self.config = SimpleNamespace(scaling_factor=.18215)
    def decode(self,z,return_dict=False):
        return (self.decoder(self.post_quant_conv(z)),)
    def enable_gradient_checkpointing(self):
        pass
    def disable_gradient_checkpointing(self):
        pass


class Up(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnets = nn.ModuleList([nn.Sequential(nn.Conv2d(2,2,1),nn.GroupNorm(1,2),nn.SiLU())])
        self.upsamplers = nn.ModuleList([nn.Conv2d(2,2,1)])
        self.attentions = nn.ModuleList([nn.Conv2d(2,2,1)])
    def forward(self,x):
        return x+.1*self.upsamplers[0](self.resnets[0](x))+.01*self.attentions[0](x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Conv2d(2,2,1)
        self.up_blocks = nn.ModuleList([Up(),Up(),Up()])
        self.conv_out = nn.Conv2d(2,1,1)
    def forward(self,x,t,encoder_hidden_states=None,class_labels=None,return_dict=False):
        x = self.down(x)+encoder_hidden_states.mean()*.001
        for block in self.up_blocks:
            x = block(x)
        return (self.conv_out(x),)


def fixture():
    recipe = SimpleNamespace(seed=43,timestep=999,num_inference_steps=1,backbone='g')
    scheduler = SimpleNamespace(init_noise_sigma=1.,config=SimpleNamespace(prediction_type='sample'),
        set_timesteps=lambda *a,**k:None,scale_model_input=lambda x,t:x)
    model = SimpleNamespace(unet=UNet(),gt_vae=VAE(),modality='thermal',device=torch.device('cpu'),args=recipe,
        lotus=SimpleNamespace(vae=VAE(),text_encoder=nn.Linear(2,2),scheduler=scheduler))
    model.condition = lambda row,image,*a:model.lotus.vae.encoder(image).detach()
    model.encode_prompt = lambda text:model.lotus.text_encoder(torch.ones(1,2)).detach()
    model.predict_disparity = MethodType(RouteModel.predict_disparity,model)
    decoder,_ = run.old.make_decoder(model)
    return model,decoder,recipe


def loss_module():
    # Exercise the actual old loss primitives without importing its training CLI.
    path = ROOT/'lotus/train_iris_ms2_g.py'
    tree = ast.parse(path.read_text(encoding='utf8'))
    module = ModuleType('train_iris_ms2_g')
    module.torch = torch
    names = {'decode_image_depth','_ssi_log_aligned_depth'}
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)
        and n.name in names],type_ignores=[]),str(path),'exec'),module.__dict__)
    return module


class JointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)
        torch.set_num_threads(1)

    def test_actual_deployed_forward_gradients_scope_and_initial_equality(self):
        image = torch.linspace(-1,1,512).reshape(1,1,16,32)
        item = run.old.InferenceInput('f','/image','a road',0)
        gt = torch.linspace(5,79,512).reshape(1,16,32)
        examples = [dict(gt_depth_values=gt,gt_valid_values=torch.ones_like(gt,dtype=torch.bool),
                         norm_bounds=torch.tensor([1.,4.]))]
        hashes = []
        for arm in run.ARMS:
            torch.manual_seed(43)
            model,decoder,recipe = fixture()
            dp,up = run.configure(model,decoder,arm)
            eligible = {n for n in dict(model.unet.named_parameters())
                if n.startswith(('up_blocks.1.','up_blocks.2.')) and ('.resnets.0.0.' in n or '.upsamplers.0.' in n)}
            self.assertEqual(set(up),eligible if arm=='joint' else set())
            frozen = run.old.fingerprint(run.old.frozen_parameters(model,decoder))
            initial = run.initial_state(model,dp)
            trace = hashlib.sha256()
            with patch('tools.train_route_suite.load_input_tensor',return_value=(image,{})):
                raw,z0 = run.old.predict_frozen(model,item,recipe)
                z = run.predict_latent(model,item,recipe,arm=='joint',trace)
            hashes.append(trace.hexdigest())
            torch.testing.assert_close(z.detach(),z0,rtol=0,atol=0)
            torch.testing.assert_close(run.old.decode_unit(decoder,z)[0,0].detach(),raw,rtol=0,atol=0)
            self.assertEqual(z.requires_grad,arm=='joint')
            if arm=='joint':
                z.retain_grad()
            with patch.dict(sys.modules,{'train_iris_ms2_g':loss_module()}):
                loss,stats = run.loss_for_batch(decoder,z,examples,[0])
                old_loss,_ = run.old.loss_for_batch(decoder,z,examples,[0])
            torch.testing.assert_close(loss.detach(),old_loss.detach(),rtol=0,atol=0)
            self.assertGreater(stats['pairs'],0)
            self.assertEqual(stats['far_frames'],1)
            optimizer = run.Optimizer(dp,up)
            (loss*run.LOSS_SCALE).backward()
            if arm=='joint':
                self.assertGreater(float(z.grad.abs().sum()),0)
            self.assertTrue(all(p.grad is None for _,p in run.old.frozen_parameters(model,decoder)))
            norms = optimizer.step()
            self.assertGreater(norms['decoder'],0)
            self.assertEqual(norms['unet']>0,arm=='joint')
            self.assertEqual(frozen,run.old.fingerprint(run.old.frozen_parameters(model,decoder)))
            final = run.initial_state(model,dp)
            self.assertNotEqual(initial['decoder'],final['decoder'])
            self.assertEqual(initial['unet']!=final['unet'],arm=='joint')
            # Saved actual parameters reproduce the post-update output after restoration.
            d,u = run.cpu_state(dp),run.cpu_state(up)
            with torch.no_grad():
                for p in dp.values(): p.add_(1)
            run.restore(dp,d)
            run.restore(up,u)
            self.assertEqual(final,run.initial_state(model,dp))
        self.assertEqual(hashes[0],hashes[1])

    def test_fp32_masters_accumulate_sub_half_precision_updates(self):
        p = nn.Parameter(torch.tensor([.5],dtype=torch.float16))
        d = nn.Parameter(torch.tensor([.1]))
        optimizer = run.Optimizer({'decoder':d},{'conv':p})
        for i in range(200):
            p.grad = torch.ones_like(p)*run.LOSS_SCALE
            d.grad = torch.zeros_like(d)
            optimizer.step()
            if i==0:
                self.assertLess(float(optimizer.master['conv'].detach()),.5)
                self.assertEqual(float(p.detach()),.5)  # rounded forward, retained master update
        self.assertLess(float(p.detach()),.5)
        self.assertEqual(optimizer.master['conv'].dtype,torch.float32)
        self.assertEqual(p.dtype,torch.float16)

    def test_gt_is_excluded_from_prediction_and_cursor_advances(self):
        class Sample:
            sample_id,thermal_path,caption,manifest_line = 'f','/image','road',4
            @property
            def thermal_gt_path(self):
                raise AssertionError('GT used as input')
        item = run.old.InferenceInput.from_sample(Sample())
        self.assertEqual(item.manifest_index,3)
        model,decoder,recipe = fixture()
        run.configure(model,decoder,'joint')
        with patch('tools.train_route_suite.load_input_tensor',return_value=(torch.ones(1,1,16,32),{})):
            self.assertTrue(run.predict_latent(model,item,recipe,True).requires_grad)
        self.assertEqual(run.CURSOR,48000+1000*run.old.BATCH*run.old.ACCUM)
        whole = list(run.pair.ImageStream(75688,43,48000,12012))
        self.assertEqual(list(run.pair.ImageStream(75688,43,run.CURSOR,12)),whole[-12:])

    def test_source_must_be_verified_q98_decoder1000(self):
        with patch.object(run.prior,'selection',return_value=({'budget':1000,'image_cursor':48000},{},500,Path('x'))):
            with self.assertRaises(ValueError):
                run.source_info(Path('unused'))

    def test_short_paired_loop_preserves_frozen_weights_and_input_order(self):
        samples = [SimpleNamespace(sample_id=f'f{i}',thermal_path=f'/image{i}',caption='a road',
                                   manifest_line=i+1) for i in range(4)]
        image = torch.linspace(-1,1,512).reshape(1,1,16,32)
        gt = torch.linspace(5,79,512).reshape(1,16,32)
        dataset = [dict(gt_depth_values=gt,gt_valid_values=torch.ones_like(gt,dtype=torch.bool),
                        norm_bounds=torch.tensor([1.,4.])) for _ in samples]
        reports = []
        with tempfile.TemporaryDirectory() as tmp:
            for arm in run.ARMS:
                torch.manual_seed(43)
                model,decoder,recipe = fixture()
                dp,up = run.configure(model,decoder,arm)
                with patch('tools.train_route_suite.load_input_tensor',return_value=(image,{})), \
                     patch.dict(sys.modules,{'train_iris_ms2_g':loss_module()}), patch('builtins.print'):
                    result = run.train(model,decoder,dp,up,dataset,samples,recipe,Path(tmp),arm,True)
                self.assertEqual(result['updates'],4)
                self.assertTrue(result['frozen_unchanged'])
                reports.append(result)
            self.assertEqual(reports[0]['initial'],reports[1]['initial'])
            self.assertEqual(reports[0]['input_trace'],reports[1]['input_trace'])

    def test_submission_smoke_two_training_arms_and_cpu_report_no_test(self):
        bash = 'D:/Git/bin/bash.exe' if os.name=='nt' else shutil.which('bash')
        if not bash or not Path(bash).exists(): self.skipTest('Bash unavailable')
        with tempfile.TemporaryDirectory(prefix='decoder_joint_submit_',dir=ROOT) as tmp:
            root = Path(tmp)
            source = root/'runs/outputs/lotus_line_v2/iris_vae_decoder_only_v1/run_parent'
            for file in ('contract.json','train/summary.json','train/decoder_1000.pt','train/test_selected_v1/reproduced.json'):
                path = source/file
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text('{}')
            (source/'jobs.txt').write_text('val reproduction -> 9213840\n')
            binary = root/'bin'
            binary.mkdir()
            stub = binary/'sbatch'
            stub.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\necho "$(wc -l < "$CALLS")"\n',newline='\n')
            stub.chmod(0o755)
            calls = root/'calls'
            env = dict(os.environ,IRIS_REPO=ROOT.as_posix(),IRIS_RUNS=(root/'runs').as_posix(),
                       CALLS=calls.as_posix(),MOCK_BIN=binary.as_posix())
            result = subprocess.run([bash,'-c','export PATH="$(cd "$MOCK_BIN" && pwd):$PATH"; bash slurm/submit_iris_decoder_joint.sh'],
                cwd=ROOT,env=env,text=True,encoding='utf8',errors='replace',capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            lines = calls.read_text().splitlines()
            self.assertEqual(len(lines),4)
            self.assertIn(' smoke ',lines[0])
            for i,arm in ((1,'decoder'),(2,'joint')):
                self.assertIn('--dependency=afterok:1',lines[i])
                self.assertTrue(lines[i].endswith(' '+arm))
                self.assertIn('--gres=gpu:1',lines[i])
            self.assertIn('--dependency=afterok:2:3',lines[3])
            self.assertNotIn('--gres',lines[3])
            self.assertNotIn('test', '\n'.join(lines))


if __name__=='__main__':
    unittest.main()
