"""Necessary CPU gates only; tiny stand-ins, no pretrained weights or GPU."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from test_da2_caption import TinyCore, TinyText
from models.da2_residual import ResidualDepth, residual_loss
from tools.train_da2_residual import (BaseOnly, optimizer_for, save, restore, make_tokens,
                                    tensor_hash, pair_smoke, FIXED_TEXT)
from tools.da2_caption_data import ImageStream


class ResidualTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1); torch.manual_seed(43)

    def model(self): return ResidualDepth(TinyCore(),TinyText(),vision_dim=8,text_dim=6,width=8)

    def batch(self):
        return torch.rand(2,3,30,33),torch.randint(1,29,(2,7)),torch.ones(2,7,dtype=torch.bool)

    def test_zero_residual_exact_baseline_and_freeze(self):
        m=self.model().train(); x,ids,mask=self.batch()
        self.assertFalse(m.core.training); self.assertFalse(m.text_encoder.training)
        pred,base,r=m(x,ids,mask,details=True)
        torch.testing.assert_close(pred,BaseOnly(m.core)(x),rtol=0,atol=0)
        torch.testing.assert_close(pred,base,rtol=0,atol=0)
        self.assertEqual(float(r.detach().abs().max()),0.)
        self.assertEqual(pred.shape,(2,30,33))
        self.assertTrue(all(not p.requires_grad for module in (m.core,m.text_encoder) for p in module.parameters()))

    def test_head_learns_and_attention_receives_gradient_frozen_weights_unchanged(self):
        m=self.model().train(); opt,sched=optimizer_for(m); x,ids,mask=self.batch()
        before=(tensor_hash(m.core),tensor_hash(m.text_encoder))
        for _ in range(3):
            opt.zero_grad(); pred,b,r=m(x,ids,mask,details=True)
            loss,_=residual_loss(b,r,torch.ones_like(b)*10); loss.backward()
            opt.step(); sched.step()
        self.assertTrue(any(f.attention.in_proj_weight.grad.abs().sum()>0 for f in m.residual.fusions))
        self.assertTrue(m.residual.combine[-1].weight.abs().sum()>0)
        self.assertEqual(before,(tensor_hash(m.core),tensor_hash(m.text_encoder)))
        self.assertTrue(all(p.grad is None for module in (m.core,m.text_encoder) for p in module.parameters()))

    def test_log_loss_masks_and_multiplier_bounds(self):
        b=torch.tensor([[[2.,4.,8.]],[[4.,5.,6.]]])
        r=torch.tensor([[[.1,.2,.3]],[[.2,.1,0.]]],requires_grad=True)
        gt=torch.tensor([[[3.,0.,float('nan')]],[[4.,6.,80.]]])
        loss,empty=residual_loss(b,r,gt)
        expected=((torch.log((b*r.exp())[0,0,0])-math.log(3)).abs()+
                  (torch.log((b*r.exp())[1,0,:2])-gt[1,0,:2].log()).abs().mean())/2
        torch.testing.assert_close(loss,expected); self.assertEqual(empty,0)
        loss.backward(); self.assertEqual(r.grad[0,0,1].item(),0)
        with self.assertRaises(ValueError): residual_loss(b,r,torch.zeros_like(gt))
        m=self.model(); x,ids,mask=self.batch()
        with torch.no_grad(): m.residual.combine[-1].bias.fill_(100)
        pred,b,r=m(x,ids,mask,details=True)
        self.assertTrue(((pred/b)>=.5).all() and ((pred/b)<=2.000001).all())

    def test_masked_tokens_and_identical_initialization(self):
        m=self.model().eval(); other=ResidualDepth(copy.deepcopy(m.core),copy.deepcopy(m.text_encoder),
                                                 vision_dim=8,text_dim=6,width=8)
        self.assertEqual(tensor_hash(m.residual),tensor_hash(other.residual))
        with torch.no_grad(): m.residual.combine[-1].weight.normal_(0,.1)
        x,ids,mask=self.batch(); mask[:,-2:]=False
        changed=ids.clone(); changed[:,-2:]=(changed[:,-2:]%28)+1
        torch.testing.assert_close(m(x,ids,mask),m(x,changed,mask),rtol=0,atol=0)
        with self.assertRaises(ValueError): m(x,ids,torch.zeros_like(mask))

    def test_resume_restores_optimizer_scheduler_rng_and_data_cursor(self):
        m=self.model(); opt,sched=optimizer_for(m); contract={'base_sha256':'frozen','arm':'frame'}
        def update(model,optimizer,scheduler):
            optimizer.zero_grad(); x,ids,mask=self.batch()
            _,b,r=model(x,ids,mask,details=True)
            loss,_=residual_loss(b,r,torch.rand_like(b)*20+1)
            loss.backward(); optimizer.step(); scheduler.step()
        update(m,opt,sched)
        clone=copy.deepcopy(m); ropt,rsched=optimizer_for(clone)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'last.pt'; save(path,m,opt,sched,1,contract)
            state=torch.load(path,weights_only=False)
            self.assertNotIn('core',state); self.assertEqual(state['image_cursor'],12)
            update(m,opt,sched)
            self.assertEqual(restore(path,clone,ropt,rsched,contract),1)
            update(clone,ropt,rsched)
            self.assertEqual(tensor_hash(m.residual),tensor_hash(clone.residual))
            self.assertEqual(sched.last_epoch,rsched.last_epoch)
            with self.assertRaises(ValueError): restore(path,clone,ropt,rsched,{'arm':'fixed'})
        whole=list(ImageStream(17,43,0,36))
        self.assertEqual(whole[12:],list(ImageStream(17,43,12,24)))

    def test_fixed_text_and_paired_smoke_gate(self):
        class Tokenizer:
            def __call__(self,texts,**kwargs):
                if texts!=[FIXED_TEXT]: raise AssertionError('Fixed text changed')
                return dict(input_ids=torch.ones(1,77,dtype=torch.long),attention_mask=torch.ones(1,77))
        tokens=make_tokens([None]*3,Tokenizer(),{}, {},'fixed',False)
        self.assertEqual(tokens['input_ids'].shape,(3,77))
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            for arm in ('frame','fixed'):
                d=base/arm; d.mkdir()
                (d/'contract.json').write_text(json.dumps(dict(arm=arm,initial_head_sha256='same')))
                (d/'completed.json').write_text(json.dumps(dict(step=32,mode='smoke',frozen_unchanged=True)))
                (d/'train.jsonl').write_text('\n'.join(json.dumps(dict(step=s,data_ids_sha256=str(s))) for s in range(1,33)))
            pair_smoke(base)
            (base/'fixed/completed.json').write_text(json.dumps(dict(step=31,mode='smoke',frozen_unchanged=True)))
            with self.assertRaises(ValueError): pair_smoke(base)
        submit=(ROOT/'slurm/submit_da2_residual.sh').read_text()
        self.assertIn('afterok:$smoke',submit)
        self.assertIn('RESUME=0',submit)

if __name__=='__main__': unittest.main()
