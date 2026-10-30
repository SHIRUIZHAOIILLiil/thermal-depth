"""Necessary CPU gates; real tiny SD modules, no downloads or tests-package imports."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.nn import functional as F
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTextConfig
from models.da2_iris_fusion import FusionDepth, IrisFeatures, padded_coordinates, id_noise
from models.da2_caption import masked_log_l1
from tools.train_da2_iris_fusion import (optimizer_for, save, restore, tensor_hash, pair_smoke,
                                       make_tokens, SMOKE_STEPS, FusionData, evaluate)


class Core(nn.Module):
    """Small frozen spatial decoder; hooks are at the production module names."""
    def __init__(self):
        super().__init__()
        self.pretrained = nn.Conv2d(3, 8, 3, padding=1)
        self.depth_head = nn.Module()
        self.depth_head.scratch = nn.Module()
        self.depth_head.scratch.layer2_rn = nn.Conv2d(8, 8, 3, padding=1)
        self.depth_head.scratch.layer3_rn = nn.Conv2d(8, 8, 3, padding=1)
        self.depth_head.output = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        f = self.pretrained(x)
        a = self.depth_head.scratch.layer2_rn(F.avg_pool2d(f, 4))
        b = self.depth_head.scratch.layer3_rn(F.avg_pool2d(f, 8))
        f = a + F.interpolate(b, a.shape[-2:], mode='bilinear', align_corners=False)
        return F.interpolate(self.depth_head.output(f), x.shape[-2:], mode='bilinear', align_corners=False).sigmoid().squeeze(1)*80


def model():
    unet = UNet2DConditionModel(sample_size=8, in_channels=8, out_channels=4,
        down_block_types=('CrossAttnDownBlock2D',)*3+('DownBlock2D',),
        up_block_types=('UpBlock2D',)+('CrossAttnUpBlock2D',)*3,
        block_out_channels=(32,64,64,64), layers_per_block=2, norm_num_groups=8,
        cross_attention_dim=32, attention_head_dim=(1,2,2,2), use_linear_projection=True,
        class_embed_type='projection', projection_class_embeddings_input_dim=4)
    vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
        down_block_types=('DownEncoderBlock2D',)*4, up_block_types=('UpDecoderBlock2D',)*4,
        block_out_channels=(32,)*4, layers_per_block=1, norm_num_groups=8, sample_size=64)
    text = CLIPTextModel(CLIPTextConfig(vocab_size=50, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=77))
    return FusionDepth(Core(), IrisFeatures(unet, vae, text, (64,32)), source_channels=(64,32), width=8)


def batch():
    image = torch.rand(1,3,64,128)
    ids = torch.tensor([[1,4,5,6,2]+[2]*72]); mask = torch.zeros_like(ids); mask[:,:5] = 1
    return image, image*2-1, ids, mask, ['frame-a']


class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(43)

    def test_exact_initial_baseline_and_paired_initialization(self):
        m = model().train(); args = batch()
        other = model()
        self.assertEqual(tensor_hash(m.fusions), tensor_hash(other.fusions))
        torch.testing.assert_close(m(*args), m.baseline(args[0]), rtol=0, atol=0)
        empty = args[2].clone(); empty[:,1:] = 2
        torch.testing.assert_close(m(args[0],args[1],empty,args[3],args[4]),m.baseline(args[0]),rtol=0,atol=0)
        self.assertFalse(m.core.training); self.assertFalse(m.iris.training)
        self.assertIsNone(m._context); self.assertIsNone(m.iris._capture)
        opt,_ = optimizer_for(m)
        self.assertEqual({id(p) for g in opt.param_groups for p in g['params']}, {id(p) for p in m.fusions.parameters()})

    def test_gradient_crosses_frozen_decoder_and_text_path_survives(self):
        m = model(); opt,sched = optimizer_for(m); args = batch()
        before = (tensor_hash(m.core), tensor_hash(m.iris))
        for _ in range(3):
            opt.zero_grad(); p=m(*args); gt=torch.zeros_like(p); gt[:,::3,::4]=12
            loss,_=masked_log_l1(p,gt); loss.backward(); opt.step(); sched.step()
        for fusion in m.fusions:
            self.assertGreater(float(fusion.project[0].weight.grad.abs().sum()),0.)
        self.assertEqual(before,(tensor_hash(m.core),tensor_hash(m.iris)))
        self.assertTrue(all(p.grad is None for part in (m.core,m.iris) for p in part.parameters()))
        # Nonzero fusion weights expose caption dependence through real frozen CLIP/U-Net.
        with torch.no_grad():
            for f in m.fusions: f.merge[-1].weight.normal_(0,.1)
        changed=args[2].clone(); changed[:,1:4]=torch.tensor([12,13,14])
        a=m(*args); b=m(args[0],args[1],changed,args[3],args[4])
        self.assertGreater(float((a-b).detach().abs().max()),0.)
        torch.testing.assert_close(a,m(args[0],args[1],args[2],torch.zeros_like(args[3]),args[4]),rtol=0,atol=0)
        with self.assertRaises(ValueError): m(args[0],args[1],args[2][:,:7],args[3][:,:7],args[4])

    def test_padding_registration_and_noise_pairing(self):
        ramp=torch.arange(16,dtype=torch.float32).reshape(1,1,4,4)
        torch.testing.assert_close(padded_coordinates(ramp,(4,4),(6,6),(6,6)),F.pad(ramp,(0,2,0,2),mode='reflect'),rtol=0,atol=1e-6)
        torch.testing.assert_close(padded_coordinates(ramp,(4,4),(4,4),(8,8)),F.interpolate(ramp,(8,8),mode='bilinear',align_corners=False))
        z=torch.zeros(2,4,8,16); a=id_noise(z,['a','b']); torch.rand(20)
        b=torch.cat((id_noise(z[:1],['a']),id_noise(z[1:],['b'])))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertFalse(torch.equal(a[0],a[1]))

    def test_both_normalizations_share_pixels_and_native_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path=Path(tmp)/'thermal.png'; gt_path=Path(tmp)/'gt.png'
            raw=np.arange(256*640,dtype=np.uint32).reshape(256,640)%4000+2000
            Image.fromarray(raw.astype(np.uint16)).save(image_path)
            gt=np.zeros((256,640),np.uint16); gt[::4,::3]=12*256
            Image.fromarray(gt).save(gt_path)
            sample=SimpleNamespace(thermal_path=image_path,thermal_gt_path=gt_path,sample_id='2021-08-06-10-59-33_000001')
            row=FusionData([sample])[0]
            mean=torch.tensor([.485,.456,.406])[:,None,None]; std=torch.tensor([.229,.224,.225])[:,None,None]
            da=((row['image']*std+mean)*255).round()
            ir=((row['iris_image']+1)*127.5).round()
            torch.testing.assert_close(da,ir,rtol=0,atol=0)
            torch.testing.assert_close(row['gt'],torch.from_numpy(gt.astype(np.float32)/256),rtol=0,atol=0)
            self.assertGreater(float(row['thermal_stats'][2]),0)

    def test_resume_exactly_and_small_checkpoint(self):
        m=model(); opt,sched=optimizer_for(m); args=batch(); contract={'arm':'frame'}
        def update():
            opt.zero_grad(); m(*args).square().mean().backward(); opt.step(); sched.step()
        update()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'last.pt'; save(path,m,opt,sched,1,contract)
            state=torch.load(path,weights_only=False)
            self.assertEqual(set(state),{'fusions','step','image_cursor','contract','optimizer','scheduler','rng'})
            update(); expected=tensor_hash(m.fusions)
            self.assertEqual(restore(path,m,contract,opt,sched),1); update()
            self.assertEqual(expected,tensor_hash(m.fusions))
            with self.assertRaises(ValueError): restore(path,m,{'arm':'empty'})

    def test_pair_gate_rejects_noise_or_recipe_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            for arm in ('frame','empty'):
                d=base/arm; d.mkdir()
                (d/'contract.json').write_text(json.dumps(dict(arm=arm,initial_fusion_sha256='same')))
                (d/'completed.json').write_text(json.dumps(dict(mode='smoke',step=SMOKE_STEPS,smoke_checks_passed=True,
                    initial_prediction_sha256='same',initial_baseline_sha256='same')))
                rows=[dict(step=s,data_ids_sha256=str(s),input_sha256='input',noise_sha256='noise') for s in range(1,SMOKE_STEPS+1)]
                (d/'train.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            pair_smoke(base)
            rows[0]['noise_sha256']='different'; (base/'empty/train.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaises(ValueError): pair_smoke(base)
        class Tokenizer:
            def __call__(self,texts,**kwargs):
                self.asserted=texts==[''] and kwargs['max_length']==77 and kwargs['padding']=='max_length'
                if not self.asserted: raise AssertionError('Wrong empty encoding')
                return dict(input_ids=torch.ones(1,77,dtype=torch.long),attention_mask=torch.ones(1,77,dtype=torch.bool))
        result=make_tokens([None]*3,Tokenizer(),{}, {},'empty',Path('.'),False)
        self.assertEqual(result['input_ids'].shape,(3,77))

    @unittest.skipUnless(shutil.which('bash'), 'Submission dry-run requires bash')
    def test_submission_chain_without_real_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'slurm').mkdir(); (root/'bin').mkdir()
            for name in ('submit_da2_iris_fusion.sh','da2_iris_fusion.sbatch'):
                shutil.copyfile(ROOT/'slurm'/name,root/'slurm'/name)
            runs=root/'runs'
            for name in ('da2_caption_v1/no-caption_s43/step018000.pt','da2_caption_assets/assets.json',
                         'iris_ms2/sd2_cap_pct_logtn_s43/converted/step18000_weights.pt'):
                p=runs/name; p.parent.mkdir(parents=True,exist_ok=True); p.touch()
            calls=root/'calls'
            fake=root/'bin/sbatch'
            fake.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALLS"\nprintf "12345\\n"\n')
            fake.chmod(0o755)
            env=dict(os.environ,IRIS_REPO=str(root),IRIS_RUNS=str(runs),CALLS=str(calls),PATH=str(root/'bin')+os.pathsep+os.environ['PATH'])
            for key in ('IRIS_FEATURE_CKPT','DA2_CAPTION_ASSETS'): env.pop(key,None)
            subprocess.run(['bash',str(root/'slurm/submit_da2_iris_fusion.sh')],env=env,check=True,capture_output=True)
            lines=calls.read_text().splitlines()
            self.assertEqual(len(lines),3)
            self.assertIn('ACTION=smoke',lines[0])
            for arm,line in zip(('frame','empty'),lines[1:]):
                self.assertIn('--dependency=afterok:12345',line)
                self.assertIn('--kill-on-invalid-dep=yes',line)
                self.assertIn(f'ACTION=train,ARM={arm},RESUME=0',line)


if __name__ == '__main__': unittest.main()
