"""Necessary dual-task tests. Run as a file (Aire has a conflicting tests package)."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from diffusers import UNet2DConditionModel,AutoencoderKL,DDPMScheduler
from transformers import CLIPTextConfig,CLIPTextModel
from models.sd21_tadp import DepthHead
from models.sd21_dual import DualDepth,expand_input,task_labels,decode_log_depth,four_losses,LOSS_WEIGHTS
from tools.sd21_dual_data import log_target
from tools.train_sd21_dual_pair import (optimizer_for,save,restore,pair_smoke,CODE,STEPS,BATCH,ACCUM,tensor_hash)


def tiny():
    unet=UNet2DConditionModel(sample_size=8,in_channels=4,out_channels=4,
        down_block_types=('CrossAttnDownBlock2D','CrossAttnDownBlock2D','CrossAttnDownBlock2D','DownBlock2D'),
        up_block_types=('UpBlock2D','CrossAttnUpBlock2D','CrossAttnUpBlock2D','CrossAttnUpBlock2D'),
        block_out_channels=(32,64,64,64),layers_per_block=2,norm_num_groups=8,
        cross_attention_dim=32,attention_head_dim=(1,2,2,2),use_linear_projection=True,
        class_embed_type='projection',projection_class_embeddings_input_dim=4)
    expand_input(unet); unet.enable_gradient_checkpointing()
    vae=AutoencoderKL(in_channels=3,out_channels=3,latent_channels=4,
        down_block_types=('DownEncoderBlock2D',)*4,up_block_types=('UpDecoderBlock2D',)*4,
        block_out_channels=(32,32,32,32),layers_per_block=1,norm_num_groups=8,sample_size=64)
    text=CLIPTextModel(CLIPTextConfig(vocab_size=50,hidden_size=32,intermediate_size=64,
        num_hidden_layers=1,num_attention_heads=4,max_position_embeddings=77))
    empty=torch.tensor([[1,2]+[2]*75])
    return DualDepth(unet,vae,text,DDPMScheduler(num_train_timesteps=1000),empty,
                     DepthHead((32,64,64,64),width=32,filters=8,out=16))


def batch():
    image=torch.rand(1,3,64,128)*2-1
    ids=torch.tensor([[1,4,5,6,2]+[2]*72]); mask=torch.zeros_like(ids); mask[:,:5]=1
    gt=torch.zeros(1,64,128); gt[:,::3,::4]=12.
    target,bounds=log_target(gt[0],torch.full_like(gt[0],15.))
    return image,ids,mask,target[None],bounds[None],gt


class DualTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1); torch.manual_seed(43)

    def test_target_matches_old_dataset_and_keeps_files_unchanged(self):
        # Execute only the existing target method, avoiding unrelated trainer imports.
        source=ast.parse((ROOT/'lotus/utils/ms2_thermal_dataset.py').read_text())
        cls=next(x for x in source.body if isinstance(x,ast.ClassDef) and any(isinstance(y,ast.FunctionDef) and y.name=='_completed_depth' for y in x.body))
        methods=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in ('__getitem__','_get_valid_mask','_get_sky_mask')]
        isolated=ast.Module(body=[ast.ClassDef(name='OldTarget',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
        env=dict(torch=torch,thermal_to_lotus_input=lambda *a,**k:SimpleNamespace(diagnostics={'converted_uint8_std':1},tensor=torch.zeros(1,3,64,128)))
        exec(compile(ast.fix_missing_locations(isolated),'old_target','exec'),env)
        old=env['OldTarget'](); old.rows=[dict(image_path='i',depth_path='d',caption='c')]
        old.transform=None; old.d_min=.001; old.d_max=80.; old.norm_type='log_truncnorm'
        old.truncnorm_min=.02; old.truncnorm_max=.98; old.use_captions=True
        pseudo=torch.linspace(.001,100,64*128).reshape(64,128)
        gt=torch.zeros_like(pseudo); gt[::3,::4]=10.
        before=pseudo.clone(); mask=(gt>.001)&(gt<80)
        dense=torch.where(mask,gt,pseudo).clamp(.001,80)
        old._completed_depth=lambda row:(dense[None],gt[None],mask[None].float())
        expected=old[0]; target,bounds=log_target(gt,pseudo)
        torch.testing.assert_close(target,expected['depth_values'],rtol=0,atol=0)
        torch.testing.assert_close(bounds,expected['norm_bounds'],rtol=0,atol=0)
        torch.testing.assert_close(pseudo,before,rtol=0,atol=0)
        self.assertLess(float(target[0,0,1]),float(target[0,-1,-1]))
        with self.assertRaises(ValueError): log_target(gt,torch.full_like(gt,float('nan')))

    def test_decode_and_original_three_losses(self):
        source=ast.parse((ROOT/'lotus/train_iris_ms2_g.py').read_text())
        funcs=[x for x in source.body if isinstance(x,ast.FunctionDef) and x.name=='decode_image_depth']
        env=dict(torch=torch); exec(compile(ast.Module(body=funcs,type_ignores=[]),'old_decode','exec'),env)
        model=tiny(); z=torch.randn(1,4,8,16,requires_grad=True); bounds=torch.tensor([[1.,3.]])
        actual=decode_log_depth(model.vae,z,bounds)
        expected=env['decode_image_depth'](model.vae,z,bounds,norm_type='log_truncnorm')[1][:,0]
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        gt=torch.zeros_like(actual); gt[:,::4,::4]=10.
        latent=torch.cat((z,z*2)); target=torch.zeros_like(latent)
        direct=torch.full_like(actual,11.,requires_grad=True)
        total,terms,_=four_losses(latent,target,actual,direct,gt)
        mask=gt>0
        old=F.mse_loss(latent[:1],target[:1])+F.mse_loss(latent[1:],target[1:])+.1*(actual[mask]-gt[mask]).abs().mean()
        torch.testing.assert_close(total,old+terms['direct_silog'])
        total.backward(); self.assertGreater(float(z.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in model.vae.parameters()))

    def test_real_modules_four_losses_gradient_owners_and_text(self):
        model=tiny(); model.train(); args=batch()
        frozen=(tensor_hash(model.vae),tensor_hash(model.text_encoder))
        captures=[]
        handle=model.text_encoder.register_forward_pre_hook(lambda m,a,k: captures.append(k['input_ids'].detach().clone()),with_kwargs=True)
        loss,terms,parts=model.training_loss(*args,trace=True)
        handle.remove()
        self.assertEqual(captures[0].shape,(2,77))
        torch.testing.assert_close(captures[0][1],model.empty_ids[0])
        for name,term in terms.items():
            grad=torch.autograd.grad(term,model.unet.conv_in.weight,retain_graph=True)[0]
            self.assertTrue(torch.isfinite(grad).all(),name)
            self.assertGreater(float(grad.abs().sum()),0.,name)
        opt,sched=optimizer_for(model); loss.backward()
        self.assertIsNone(model._capture)
        for module in (model.vae,model.text_encoder): self.assertTrue(all(p.grad is None for p in module.parameters()))
        for module in (model.unet.conv_out,model.unet.class_embedding,model.head):
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in module.parameters()))
        opt.step(); sched.step()
        self.assertEqual(frozen,(tensor_hash(model.vae),tensor_hash(model.text_encoder)))
        self.assertEqual(LOSS_WEIGHTS,dict(latent_depth=1.,reconstruction=1.,image_l1=.1,direct_silog=1.))
        self.assertEqual({g['lr'] for g in opt.param_groups},{3e-5,1e-4})
        self.assertEqual((STEPS,BATCH,ACCUM),(20000,4,3))

    def test_paired_random_draws_and_target_free_eval(self):
        model=tiny(); args=list(batch()); model.train()
        state=torch.get_rng_state(); a=model.training_loss(*args,trace=True)[2]
        args[1]=model.empty_ids.clone(); args[2]=torch.zeros_like(args[1]); args[2][:,:2]=1
        torch.set_rng_state(state); b=model.training_loss(*args,trace=True)[2]
        self.assertEqual(a['stochastic'],b['stochastic'])
        self.assertEqual(a['reconstruction_text'],b['reconstruction_text'])
        model.eval()
        with torch.no_grad():
            state=torch.get_rng_state()
            p=model(*args[:3]); q=model(*args[:3])
        torch.testing.assert_close(p,q,rtol=0,atol=0)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(p.shape,(1,64,128)); self.assertTrue(bool(((p>0)&(p<80)).all()))

    def test_resume_and_checkpoint_contract(self):
        model=tiny(); opt,sched=optimizer_for(model); args=batch()
        model.training_loss(*args)[0].backward(); opt.step(); sched.step(); opt.zero_grad()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'last.pt'; contract={'experiment':'dual','loss_weights':LOSS_WEIGHTS}
            save(path,model,opt,sched,1,contract,True)
            before=tensor_hash(model.unet); expected=torch.rand(4)
            with torch.no_grad(): model.unet.conv_in.weight.add_(1)
            self.assertEqual(restore(path,model,contract,opt,sched),1)
            self.assertEqual(before,tensor_hash(model.unet))
            torch.testing.assert_close(expected,torch.rand(4),rtol=0,atol=0)
            with self.assertRaises(ValueError): restore(path,model,{'wrong':True})

    def test_pair_gate_and_sbatch_dependencies(self):
        self.assertEqual(len(CODE),len(set(CODE)))
        self.assertTrue(all((ROOT/f).is_file() for f in CODE))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for arm in ('frame','empty'):
                out=root/arm; out.mkdir()
                (out/'contract.json').write_text(json.dumps(dict(arm=arm,loss_weights=LOSS_WEIGHTS,pseudo_inventory_sha256='abc')))
                (out/'completed.json').write_text(json.dumps(dict(step=32,mode='smoke',smoke_checks_passed=True)))
                rows=[dict(step=i,data_ids_sha256=str(i),stochastic_sha256=str(i),reconstruction_text_sha256='empty77',loss=3.1,**{k:1. for k in LOSS_WEIGHTS}) for i in range(1,33)]
                (out/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            pair_smoke(root)
            rows[0]['stochastic_sha256']='different'
            (root/'empty/train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError): pair_smoke(root)
            if os.name=='nt': return
            binary=root/'sbatch'; binary.write_text('#!/bin/bash\necho "$*" >> "$CALLS"\necho 12345\n'); binary.chmod(0o755)
            env=dict(os.environ,PATH=str(root)+':'+os.environ['PATH'],IRIS_REPO=str(ROOT),IRIS_RUNS=str(root/'runs'),CALLS=str(root/'calls'))
            subprocess.run(['bash',str(ROOT/'slurm/submit_sd21_dual_pair.sh')],env=env,check=True,capture_output=True)
            calls=(root/'calls').read_text().splitlines(); self.assertEqual(len(calls),3)
            self.assertIn('sd21_dual_pair_pad77_v2',(ROOT/'slurm/submit_sd21_dual_pair.sh').read_text())
            self.assertIn('sd21_dual_pair_pad77_v2',(ROOT/'slurm/sd21_dual_pair.sbatch').read_text())
            for call,arm in zip(calls[1:],('frame','empty')):
                self.assertIn('--dependency=afterok:12345',call); self.assertIn('--kill-on-invalid-dep=yes',call)
                self.assertIn('ARM='+arm,call)

    def test_pad77_reconstruction_invariance_and_mask_policy(self):
        model=tiny(); model.eval(); captured=[]
        def capture(m,a,k):
            self.assertNotIn('attention_mask',k)
            captured.append(k['input_ids'].clone())
        hook=model.text_encoder.register_forward_pre_hook(capture,with_kwargs=True)
        empty=model.empty_ids.expand(2,-1).clone()
        short=empty.clone(); short[:,1:4]=torch.tensor([4,5,2])
        long=empty.clone(); long[:,1:60]=4; long[:,60]=2
        texts=[]
        with torch.no_grad():
            for ids,length in ((short,4),(long,61),(empty,2)):
                mask=torch.zeros_like(ids); mask[:,:length]=1
                texts.append(model._text(ids,mask,True))
            unmasked=model._text(short,torch.ones_like(short),False)
            zero_mask=model._text(short,torch.zeros_like(short),False)
        hook.remove()
        for ids in captured[:3]:
            self.assertEqual(ids.shape,(4,77))
            torch.testing.assert_close(ids[2:],empty,rtol=0,atol=0)
        for text in texts[1:]: torch.testing.assert_close(text[2:],texts[0][2:],rtol=0,atol=0)
        torch.testing.assert_close(unmasked,zero_mask,rtol=0,atol=0)
        self.assertGreater(float((texts[0][:2]-texts[2][:2]).abs().max()),0.)
        with self.assertRaises(ValueError): model._text(short[:,:4],torch.ones(2,4),False)
        seen=[]
        def unet_input(m,a,k):
            self.assertNotIn('attention_mask',k); self.assertNotIn('encoder_attention_mask',k)
            seen.append(k['encoder_hidden_states'].shape[1])
        hook=model.unet.register_forward_pre_hook(unet_input,with_kwargs=True)
        args=batch()
        with torch.no_grad(): model(*args[:3])
        hook.remove(); self.assertEqual(seen,[77])


if __name__=='__main__': unittest.main()
