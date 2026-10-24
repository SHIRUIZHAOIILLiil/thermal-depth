"""Necessary CPU tests, including actual Diffusers SD2-style block topology.

Run as a file; never import the ambiguous top-level site-packages `tests`.
No model downloads, GPU, real manifests or old experiment writes.
"""
import copy
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from diffusers import UNet2DConditionModel
from transformers import CLIPTextConfig, CLIPTextModel
from models.sd21_tadp import SD21Depth, DepthHead
from models.da2_silog import masked_silog
from tools.da2_caption_data import ImageStream
from tools.train_sd21_tadp_pair import (ThermalDirect, make_tokens, tensor_hash, optimizer_for,
    save, restore, repair_log, pair_smoke, selected_step, lr_factor)


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.config = SimpleNamespace(scaling_factor=.18215)
        self.calls = []

    def encode(self, image):
        self.calls.append(tuple(image.shape))
        z = self.conv(F.avg_pool2d(image, 8))
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: z))


def tiny_model():
    unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4,
        down_block_types=('CrossAttnDownBlock2D','CrossAttnDownBlock2D','CrossAttnDownBlock2D','DownBlock2D'),
        up_block_types=('UpBlock2D','CrossAttnUpBlock2D','CrossAttnUpBlock2D','CrossAttnUpBlock2D'),
        block_out_channels=(32,64,64,64), layers_per_block=2, norm_num_groups=8,
        cross_attention_dim=32, attention_head_dim=(1,2,2,2), use_linear_projection=True)
    unet.enable_gradient_checkpointing()
    text = CLIPTextModel(CLIPTextConfig(vocab_size=50, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=77))
    return SD21Depth(unet, TinyVAE(), text, DepthHead((32,64,64,64), width=32, filters=8, out=16))


class DirectDepthTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(43)

    def test_real_diffusers_taps_backward_frozen_and_text_effect(self):
        model = tiny_model()
        frozen = (tensor_hash(model.vae), tensor_hash(model.text_encoder))
        initial = (tensor_hash(model.unet), tensor_hash(model.head))
        x = torch.rand(2,3,64,128)*2-1
        ids = torch.randint(0,50,(2,77)); empty = torch.zeros_like(ids)
        model.eval()
        captured=[]
        hook=model.head.register_forward_pre_hook(
            lambda module,args: captured.append([f.detach().clone() for f in args[0]]))
        with torch.no_grad():
            a = model(x,ids,torch.ones_like(ids))
            b = model(x,empty,torch.ones_like(ids))
            c = model(x,ids,torch.zeros_like(ids))
        hook.remove()
        self.assertEqual(a.shape,(2,64,128))
        self.assertTrue(bool(torch.isfinite(a).all() and (a>0).all() and (a<80).all()))
        # Tiny random depth head can round to a constant initially. Test the
        # actual text-conditioned features, not a fabricated depth-quality gate.
        self.assertTrue(any(float((u-v).abs().max())>0 for u,v in zip(captured[0],captured[1])))
        torch.testing.assert_close(a,c,rtol=0,atol=0)
        opt,sched = optimizer_for(model); model.train()
        p = model(x,empty)
        gt = torch.zeros_like(p); gt[:,::3,::4] = 12.
        loss,_ = masked_silog(p,gt); loss.backward()
        self.assertIsNone(model._capture)
        for module in (model.vae,model.text_encoder):
            self.assertFalse(module.training)
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        cross = [p.grad for n,p in model.unet.named_parameters() if 'attn2.to_k.weight' in n]
        self.assertTrue(any(g is not None and bool(g.abs().sum()>0) for g in cross))
        opt.step(); sched.step()
        self.assertEqual(frozen,(tensor_hash(model.vae),tensor_hash(model.text_encoder)))
        self.assertNotEqual(initial[0],tensor_hash(model.unet))
        self.assertNotEqual(initial[1],tensor_hash(model.head))
        self.assertTrue(all(s==(2,3,64,128) for s in model.vae.calls))
        with self.assertRaises(ValueError): model(x[...,:63,:],ids)

    def test_identical_parameter_initialization_not_zero_caption_branch(self):
        a=tiny_model(); torch.manual_seed(43); b=tiny_model()
        self.assertEqual(tensor_hash(a),tensor_hash(b))
        self.assertEqual([(n,p.shape,p.requires_grad) for n,p in a.named_parameters()],
                         [(n,p.shape,p.requires_grad) for n,p in b.named_parameters()])
        opt,_=optimizer_for(a)
        actual=[p for g in opt.param_groups for p in g['params']]
        expected=[p for p in a.parameters() if p.requires_grad]
        self.assertEqual({id(p) for p in actual},{id(p) for p in expected})
        self.assertEqual(len(actual),len({id(p) for p in actual}))

    def test_sparse_silog_formula_mask_and_gradient(self):
        p=torch.tensor([[[2.,4.,30.,40.,3.]]],requires_grad=True)
        gt=torch.tensor([[[1.,2.,0.,float('nan'),80.]]])
        loss,n=masked_silog(p,gt)
        expected=torch.tensor(2.).log()*(.5**.5)
        torch.testing.assert_close(loss,expected)
        loss.backward(); self.assertEqual(n,0)
        torch.testing.assert_close(p.grad[...,2:],torch.zeros_like(p.grad[...,2:]))
        with self.assertRaises(ValueError): masked_silog(p,torch.zeros_like(gt))
        q=torch.ones(1,2,2,requires_grad=True)
        zero,_=masked_silog(q,torch.ones_like(q)); zero.backward()
        self.assertTrue(torch.isfinite(q.grad).all())

    def test_uint16_native_preprocess_real_gt_only_and_empty_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            image=Path(tmp)/'thermal.png'; gt=Path(tmp)/'lidar.png'
            raw=np.linspace(15000,40000,256*640).reshape(256,640).astype(np.uint16)
            lidar=np.zeros((256,640),np.uint16); lidar[10,20]=2560
            Image.fromarray(raw).save(image); Image.fromarray(lidar).save(gt)
            sample=SimpleNamespace(thermal_path=str(image),thermal_gt_path=str(gt),sample_id='seq_000001')
            class Tokenizer:
                def __call__(self,texts,**kwargs):
                    if texts!=[''] or kwargs['max_length']!=77: raise AssertionError('Wrong empty encoding')
                    ids=torch.full((1,77),2,dtype=torch.long); ids[:,0]=1
                    mask=torch.zeros_like(ids); mask[:,:2]=1
                    return dict(input_ids=ids,attention_mask=mask)
            tok=make_tokens([sample],Tokenizer(),{},'', 'empty',Path(tmp),False)
            item=ThermalDirect([sample],tok)[0]
            self.assertEqual(item['image'].shape,(3,256,640))
            self.assertGreater(float(item['image'].std()),.1)
            self.assertGreaterEqual(float(item['image'].min()),-1.)
            self.assertLessEqual(float(item['image'].max()),1.)
            self.assertEqual(float(item['gt'][10,20]),10.)
            self.assertEqual(int(torch.count_nonzero(item['gt'])),1)
            self.assertEqual(tok['input_ids'].shape,(1,77))
            self.assertEqual(tok['input_ids'][0,:2].tolist(),[1,2])
            with self.assertRaises(ValueError): make_tokens([],Tokenizer(),{},'', 'off',Path(tmp),False)

    def test_resume_optimizer_rng_cursor_and_log_repair(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__(); self.unet=nn.Linear(3,3); self.head=nn.Linear(3,1)
        a=Model(); opt,sched=optimizer_for(a); contract={'arm':'frame'}
        def update(m,o,s):
            o.zero_grad(); loss=m.head(m.unet(torch.rand(2,3))).square().mean()
            loss.backward(); o.step(); s.step()
        update(a,opt,sched)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'last.pt'
            save(path,a,opt,sched,1,contract,True)
            update(a,opt,sched); expected=tensor_hash(a)
            b=Model(); ob,sb=optimizer_for(b)
            self.assertEqual(restore(path,b,contract,ob,sb),1)
            update(b,ob,sb); self.assertEqual(expected,tensor_hash(b))
            self.assertEqual(sched.state_dict(),sb.state_dict())
            with self.assertRaises(ValueError): restore(path,b,{'arm':'empty'},ob,sb)
            log=Path(tmp)/'train.jsonl'
            log.write_text('{"step":1}\n{"step":2}\npartial crash tail')
            repair_log(log,1); self.assertEqual(len(log.read_text().splitlines()),1)
        full=list(ImageStream(17,43,0,48))
        self.assertEqual(full[12:],list(ImageStream(17,43,12,36)))
        self.assertGreater(lr_factor(0),0); self.assertEqual(lr_factor(499),1.)
        self.assertEqual(lr_factor(20000),0.)

    def test_paired_gate_and_selection_no_test_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            for arm in ('frame','empty'):
                d=base/arm; d.mkdir()
                (d/'contract.json').write_text(json.dumps(dict(arm=arm,initial='same')))
                (d/'completed.json').write_text(json.dumps(dict(step=32,mode='smoke',smoke_checks_passed=True)))
                (d/'train.jsonl').write_text('\n'.join(json.dumps(dict(step=s,data_ids_sha256=str(s))) for s in range(1,33)))
            pair_smoke(base)
            p=base/'empty/contract.json'; c=json.loads(p.read_text()); c['initial']='wrong'; p.write_text(json.dumps(c))
            with self.assertRaises(ValueError): pair_smoke(base)
        self.assertEqual(selected_step({4000:dict(metrics={'ssi':{'abs_rel':.1}}),
                                        2000:dict(metrics={'ssi':{'abs_rel':.1}})}),2000)
        submit=(ROOT/'slurm/submit_sd21_tadp_pair.sh').read_text()
        job=(ROOT/'slurm/sd21_tadp_pair.sbatch').read_text()
        self.assertIn('afterok:$smoke',submit)
        self.assertIn('--kill-on-invalid-dep=yes',submit)
        self.assertIn('RESUME=0',submit)
        self.assertIn('for arm in frame empty',submit)
        self.assertNotIn('test-manifest',job)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Bash mock runs on Linux/WSL')
    def test_submission_chain_with_mock_sbatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); binary=root/'sbatch'; log=root/'submitted.log'
            binary.write_text('#!/bin/bash\n'
                'printf "%s\\n" "$*" >> "$SUBMIT_LOG"\n'
                '[[ "${FAIL_SMOKE:-0}" == 1 ]] && exit 1\n'
                'if [[ "$*" == *ACTION=smoke* ]]; then echo 100; '
                'elif [[ "$*" == *ARM=frame* ]]; then echo 101; else echo 102; fi\n')
            binary.chmod(0o755)
            env=dict(os.environ, PATH=str(root)+os.pathsep+os.environ['PATH'],
                     IRIS_REPO=str(ROOT), IRIS_RUNS=str(root/'runs'), SCRATCH=str(root), SUBMIT_LOG=str(log))
            command=['bash',str(ROOT/'slurm/submit_sd21_tadp_pair.sh')]
            done=subprocess.run(command,env=env,capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            jobs=log.read_text().splitlines(); self.assertEqual(len(jobs),3)
            for line,arm in zip(jobs[1:],('frame','empty')):
                self.assertIn('--dependency=afterok:100',line)
                self.assertIn('--kill-on-invalid-dep=yes',line)
                self.assertIn('ARM='+arm,line)
            before=len(jobs)
            failed=subprocess.run(command,env=dict(env,FAIL_SMOKE='1'),capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0)
            self.assertEqual(len(log.read_text().splitlines()),before+1)


if __name__=='__main__': unittest.main()
