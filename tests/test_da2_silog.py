"""Necessary CPU tests, no real data or pretrained downloads."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from test_da2_caption import TinyCore, TinyText
from models.da2_silog import SiLogDepth, masked_silog
from tools.train_da2_silog import optimizer_for, save, restore, tensor_hash, pair_smoke, repair_log, MILESTONES
from tools.da2_caption_data import ImageStream


class SiLogTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(43)

    def model(self, caption=True):
        return SiLogDepth(TinyCore(),TinyText() if caption else None,vision_dim=8,text_dim=6,width=8)

    def batch(self):
        return torch.rand(2,3,30,33),torch.randint(1,29,(2,7)),torch.ones(2,7,dtype=torch.bool)

    def update(self,m,o,s):
        o.zero_grad(); x,ids,mask=self.batch(); p=m(x,ids,mask)
        loss,_=masked_silog(p,torch.rand_like(p)*15+1)
        loss.backward(); o.step(); s.step()

    def test_official_formula_gradient_and_microbatch_reduction(self):
        p=torch.tensor([[[2.,3.,9.]],[[6.,5.,4.]]],requires_grad=True)
        gt=torch.tensor([[[1.,0.,float('nan')]],[[3.,7.,80.]]])
        loss,empty=masked_silog(p,gt)
        valid=torch.isfinite(gt)&(gt>.001)&(gt<80)
        diff=torch.log(gt[valid])-torch.log(p[valid])
        official=torch.sqrt(diff.square().mean()-.5*diff.mean().square())
        torch.testing.assert_close(loss,official,rtol=0,atol=0)
        a=torch.autograd.grad(loss,p,retain_graph=True)[0]
        b=torch.autograd.grad(official,p)[0]
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(empty,0); self.assertEqual(a[0,0,1].item(),0)
        perframe=sum(masked_silog(p[i:i+1],gt[i:i+1])[0] for i in range(2))/2
        self.assertGreater(abs(float(loss.detach()-perframe.detach())),1e-4)
        perfect=torch.ones(1,2,2,requires_grad=True)
        zero,_=masked_silog(perfect,torch.ones_like(perfect)); zero.backward()
        self.assertTrue(torch.isfinite(perfect.grad).all()); self.assertEqual(zero.item(),0.)
        with self.assertRaises(ValueError): masked_silog(p,torch.zeros_like(gt))
        with self.assertRaises(ValueError): masked_silog(torch.zeros_like(p),gt)

    def test_paired_initial_predictions_and_parameter_groups(self):
        m=self.model(); n=SiLogDepth(copy.deepcopy(m.core)); x,ids,mask=self.batch()
        m.eval(); n.eval()
        torch.testing.assert_close(m(x,ids,mask),n(x),rtol=0,atol=0)
        self.assertEqual(m(x,ids,mask).shape,(2,30,33))
        self.assertIsNone(n.text_encoder); self.assertEqual(len(n.fusions),0)
        self.assertTrue(all(k.startswith('core.') for k,_ in n.named_parameters()))
        for model in (m,n):
            model.train(); opt,_=optimizer_for(model)
            self.assertTrue(model.core.pretrained.training); self.assertTrue(model.core.depth_head.training)
            selected=[id(p) for g in opt.param_groups for p in g['params']]
            self.assertEqual(set(selected),{id(p) for p in model.parameters() if p.requires_grad})
            self.assertEqual(len(selected),len(set(selected)))
            self.assertEqual([g['lr'] for g in opt.param_groups],[5e-6,5e-5])
        self.assertFalse(m.text_encoder.training)

    def test_both_arms_update_encoder_and_dpt_text_stays_frozen(self):
        for caption in (False,True):
            m=self.model(caption).train(); opt,sched=optimizer_for(m)
            enc=tensor_hash(m.core.pretrained); dec=tensor_hash(m.core.depth_head)
            text=tensor_hash(m.text_encoder) if caption else None
            for _ in range(3): self.update(m,opt,sched)
            self.assertNotEqual(enc,tensor_hash(m.core.pretrained))
            self.assertNotEqual(dec,tensor_hash(m.core.depth_head))
            if caption:
                self.assertEqual(text,tensor_hash(m.text_encoder))
                self.assertTrue(all(p.grad is None for p in m.text_encoder.parameters()))
                self.assertTrue(all(f.attention.in_proj_weight.grad.abs().sum()>0 for f in m.fusions))

    def test_padding_mask_and_live_text_path(self):
        m=self.model().eval(); x,ids,mask=self.batch(); mask[:,-2:]=False
        with torch.no_grad():
            for f in m.fusions: f.merge[-1].weight.normal_(0,.1)
        changed=ids.clone(); changed[:,-2:]=(changed[:,-2:]%28)+1
        torch.testing.assert_close(m(x,ids,mask),m(x,changed,mask),rtol=0,atol=0)
        self.assertGreater(float((m(x,ids,mask)-m(x,ids%28+1,mask)).detach().abs().max()),0)
        with self.assertRaises(ValueError): m(x,ids,torch.zeros_like(mask))

    def test_exact_resume_both_arms_and_stale_log_repair(self):
        for caption in (False,True):
            m=self.model(caption); opt,sched=optimizer_for(m); self.update(m,opt,sched)
            clone=copy.deepcopy(m); co,cs=optimizer_for(clone); contract={'arm':str(caption)}
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)/'last.pt'; save(p,m,opt,sched,1,contract)
                state=torch.load(p,weights_only=False)
                self.assertIn('core',state); self.assertIn('pretrained.proj.weight',state['core'])
                self.assertNotIn('text_encoder',state); self.assertEqual(state['image_cursor'],12)
                self.update(m,opt,sched); restore(p,clone,co,cs,contract); self.update(clone,co,cs)
                self.assertEqual(tensor_hash(m),tensor_hash(clone))
                self.assertEqual(sched.state_dict(),cs.state_dict())
                with self.assertRaises(ValueError): restore(p,clone,co,cs,{'arm':'different'})
                log=Path(tmp)/'train.jsonl'
                log.write_text('{"step":1}\n{"step":2}\nBROKEN',encoding='utf-8')
                repair_log(log,1); self.assertEqual([json.loads(l)['step'] for l in log.read_text().splitlines()],[1])
        self.assertEqual(list(ImageStream(17,43,0,36))[12:],list(ImageStream(17,43,12,24)))

    def test_paired_smoke_gate_and_submit_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            for arm in ('no-caption','caption'):
                d=base/arm; d.mkdir()
                (d/'contract.json').write_text(json.dumps(dict(arm=arm,initial_core_sha256='same',initial_fusion_sha256=arm)))
                (d/'completed.json').write_text(json.dumps(dict(step=32,mode='smoke',smoke_checks_passed=True)))
                (d/'train.jsonl').write_text('\n'.join(json.dumps(dict(step=s,data_ids_sha256=str(s))) for s in range(1,33)))
            pair_smoke(base)
            (base/'caption/completed.json').write_text(json.dumps(dict(step=32,mode='smoke',smoke_checks_passed=False)))
            with self.assertRaises(ValueError): pair_smoke(base)
        self.assertEqual(MILESTONES,(10000,20000))
        submit=(ROOT/'slurm/submit_da2_silog.sh').read_text()
        job=(ROOT/'slurm/da2_silog.sbatch').read_text()
        self.assertIn('afterok:$smoke',submit); self.assertIn('--kill-on-invalid-dep=yes',submit)
        self.assertIn('for arm in no-caption caption',submit)
        self.assertIn('RESUME=0',submit); self.assertNotIn('step018000',submit+job)
        self.assertNotIn('--base-checkpoint',job)


if __name__=='__main__': unittest.main()
