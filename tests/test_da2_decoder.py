"""Small CPU gates: trainability, masking, paired setup, and exact resume."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from test_da2_caption import TinyCore, TinyText
from models.da2_decoder import DecoderDepth
from models.da2_caption import masked_log_l1
from tools.train_da2_decoder import (BaseOnly, optimizer_for, save, restore, tensor_hash,
                                     pair_smoke, make_tokens, FIXED_TEXT, MILESTONES)
from tools.da2_caption_data import ImageStream


class DecoderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(43)

    def model(self):
        return DecoderDepth(TinyCore(), TinyText(), vision_dim=8, text_dim=6, width=8)

    def batch(self):
        return torch.rand(2,3,30,33), torch.randint(1,29,(2,7)), torch.ones(2,7,dtype=torch.bool)

    def update(self, m, opt, sched):
        opt.zero_grad()
        x, ids, mask = self.batch()
        p = m(x,ids,mask)
        loss,_ = masked_log_l1(p, torch.rand_like(p)*15+1)
        loss.backward()
        opt.step(); sched.step()

    def test_initial_baseline_and_trainable_partition(self):
        m = self.model().train(); x,ids,mask = self.batch()
        self.assertFalse(m.core.pretrained.training)
        self.assertFalse(m.text_encoder.training)
        self.assertTrue(m.core.depth_head.training)
        torch.testing.assert_close(m(x,ids,mask),BaseOnly(m.core)(x),rtol=0,atol=0)
        self.assertEqual(m(x,ids,mask).shape,(2,30,33))
        opt,_ = optimizer_for(m)
        selected = [id(p) for g in opt.param_groups for p in g['params']]
        expected = {id(p) for p in m.parameters() if p.requires_grad}
        self.assertEqual(set(selected),expected)
        self.assertEqual(len(selected),len(expected))
        self.assertEqual([g['lr'] for g in opt.param_groups],[1e-5,5e-5])
        self.assertFalse(any('alpha' in n for n,_ in m.named_parameters()))

    def test_decoder_and_fusion_learn_encoders_stay_frozen(self):
        m = self.model().train(); opt,sched = optimizer_for(m)
        frozen = (tensor_hash(m.core.pretrained),tensor_hash(m.text_encoder))
        head = tensor_hash(m.core.depth_head); fusion = tensor_hash(m.fusions)
        for _ in range(3): self.update(m,opt,sched)
        self.assertNotEqual(head,tensor_hash(m.core.depth_head))
        self.assertNotEqual(fusion,tensor_hash(m.fusions))
        self.assertTrue(all(f.attention.in_proj_weight.grad.abs().sum()>0 for f in m.fusions))
        self.assertEqual(frozen,(tensor_hash(m.core.pretrained),tensor_hash(m.text_encoder)))
        self.assertTrue(all(p.grad is None for module in (m.core.pretrained,m.text_encoder) for p in module.parameters()))

    def test_paired_initialization_mask_and_active_text_path(self):
        m = self.model().eval()
        other = DecoderDepth(copy.deepcopy(m.core),copy.deepcopy(m.text_encoder),vision_dim=8,text_dim=6,width=8)
        self.assertEqual(tensor_hash(m.fusions),tensor_hash(other.fusions))
        # Nonzero output weights exercise the real text path, not the initial zero output.
        with torch.no_grad():
            for f in m.fusions: f.merge[-1].weight.normal_(0,.1)
        x,ids,mask = self.batch(); mask[:,-2:] = False
        changed=ids.clone(); changed[:,-2:] = (changed[:,-2:]%28)+1
        torch.testing.assert_close(m(x,ids,mask),m(x,changed,mask),rtol=0,atol=0)
        changed=(ids%28)+1
        self.assertGreater(float((m(x,ids,mask)-m(x,changed,mask)).detach().abs().max()),0)
        with self.assertRaises(ValueError): m(x,ids,torch.zeros_like(mask))

    def test_sparse_loss_unchanged(self):
        p=torch.tensor([[[2.,3.,4.]],[[4.,5.,6.]]],requires_grad=True)
        g=torch.tensor([[[1.,0.,float('nan')]],[[2.,10.,80.]]])
        loss,empty=masked_log_l1(p,g)
        torch.testing.assert_close(loss,torch.tensor(2.).log())
        self.assertEqual(empty,0); loss.backward()
        self.assertEqual(p.grad[0,0,1].item(),0.)
        with self.assertRaises(ValueError): masked_log_l1(p,torch.zeros_like(g))

    def test_resume_includes_decoder_optimizer_rng_cursor(self):
        m=self.model(); opt,sched=optimizer_for(m); self.update(m,opt,sched)
        clone=copy.deepcopy(m); co,cs=optimizer_for(clone); contract={'arm':'frame'}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'last.pt'; save(path,m,opt,sched,1,contract)
            state=torch.load(path,weights_only=False)
            self.assertEqual(state['image_cursor'],12)
            self.assertIn('decoder',state); self.assertIn('fusions',state)
            self.assertNotIn('core',state)
            self.update(m,opt,sched)
            self.assertEqual(restore(path,clone,co,cs,contract),1)
            self.update(clone,co,cs)
            self.assertEqual(tensor_hash(m),tensor_hash(clone))
            self.assertEqual(sched.state_dict(),cs.state_dict())
            with self.assertRaises(ValueError): restore(path,clone,co,cs,{'arm':'fixed'})
        self.assertEqual(list(ImageStream(17,43,0,36))[12:],list(ImageStream(17,43,12,24)))

    def test_fixed_tokens_pair_gate_and_submission(self):
        class Tokenizer:
            def __call__(self,texts,**kwargs):
                if texts != [FIXED_TEXT]: raise AssertionError('Unexpected fixed text')
                return dict(input_ids=torch.ones(1,77,dtype=torch.long),attention_mask=torch.ones(1,77))
        t=make_tokens([None]*3,Tokenizer(),{}, {},'fixed',False)
        self.assertEqual(t['input_ids'].shape,(3,77))
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            for arm in ('frame','fixed'):
                d=base/arm; d.mkdir()
                (d/'contract.json').write_text(json.dumps(dict(arm=arm,initial_head_sha256='same')))
                (d/'completed.json').write_text(json.dumps(dict(step=32,mode='smoke',frozen_unchanged=True)))
                (d/'train.jsonl').write_text('\n'.join(json.dumps(dict(step=s,data_ids_sha256=str(s))) for s in range(1,33)))
            pair_smoke(base)
            (base/'fixed/completed.json').write_text(json.dumps(dict(step=32,mode='smoke',frozen_unchanged=False)))
            with self.assertRaises(ValueError): pair_smoke(base)
        self.assertEqual(MILESTONES,(10000,20000))
        script=(ROOT/'slurm/submit_da2_decoder.sh').read_text()
        self.assertIn('afterok:$smoke',script)
        self.assertIn('RESUME=0',script)
        self.assertIn('--kill-on-invalid-dep=yes',script)


if __name__ == '__main__': unittest.main()
