"""Reuse six SiLog core tests, plus matched true/empty caption checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from test_da2_silog import SiLogTests
from test_da2_caption import TinyCore, TinyText
from models.da2_silog import SiLogDepth, masked_silog
from tools.train_da2_silog_pair import make_tokens, pair_smoke, tensor_hash, optimizer_for


class MatchedCaptionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(43)

    def test_empty_encoding_preserves_special_tokens_and_mask(self):
        class Tokenizer:
            def __call__(self, texts, **kwargs):
                if texts != [''] or kwargs != dict(padding='max_length',max_length=77,truncation=True,return_tensors='pt'):
                    raise AssertionError('Empty tokenization recipe changed')
                ids=torch.full((1,77),2,dtype=torch.long); ids[0,0]=1
                mask=torch.zeros(1,77,dtype=torch.long); mask[0,:2]=1
                return dict(input_ids=ids,attention_mask=mask)
        t=make_tokens([None]*3,Tokenizer(),{}, {},'empty',False)
        self.assertEqual(t['input_ids'].shape,(3,77))
        self.assertEqual(t['attention_mask'].sum(1).tolist(),[2,2,2])
        self.assertEqual(t['input_ids'][0,:2].tolist(),[1,2])
        self.assertTrue(torch.equal(t['input_ids'][0],t['input_ids'][2]))
        with self.assertRaises(ValueError): make_tokens([],Tokenizer(),{}, {},'no-caption',False)

    def test_same_modules_initial_predictions_and_empty_arm_gradient(self):
        core=TinyCore(); text=TinyText()
        frame=SiLogDepth(copy.deepcopy(core),copy.deepcopy(text),vision_dim=8,text_dim=6,width=8)
        empty=SiLogDepth(copy.deepcopy(core),copy.deepcopy(text),vision_dim=8,text_dim=6,width=8)
        self.assertEqual(tensor_hash(frame),tensor_hash(empty))
        self.assertEqual([(n,p.shape,p.requires_grad) for n,p in frame.named_parameters()],
                         [(n,p.shape,p.requires_grad) for n,p in empty.named_parameters()])
        x=torch.rand(2,3,30,33)
        ids=torch.randint(1,29,(2,77)); mask=torch.ones_like(ids,dtype=torch.bool)
        ei=torch.ones_like(ids); em=torch.zeros_like(mask); em[:,:2]=True
        frame.eval(); empty.eval()
        torch.testing.assert_close(frame(x,ids,mask),empty(x,ei,em),rtol=0,atol=0)
        empty.train(); opt,sched=optimizer_for(empty)
        frozen=tensor_hash(empty.text_encoder)
        for _ in range(3):
            opt.zero_grad(); p=empty(x,ei,em)
            loss,_=masked_silog(p,torch.ones_like(p)*10)
            loss.backward(); opt.step(); sched.step()
        self.assertEqual(frozen,tensor_hash(empty.text_encoder))
        self.assertTrue(all(f.attention.in_proj_weight.grad.abs().sum()>0 for f in empty.fusions))

    def test_pair_gate_requires_identical_fusion_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            for arm in ('frame','empty'):
                d=base/arm; d.mkdir()
                (d/'contract.json').write_text(json.dumps(dict(arm=arm,initial_core_sha256='core',initial_fusion_sha256='same')))
                (d/'completed.json').write_text(json.dumps(dict(step=32,mode='smoke',smoke_checks_passed=True)))
                (d/'train.jsonl').write_text('\n'.join(json.dumps(dict(step=s,data_ids_sha256=str(s))) for s in range(1,33)))
            pair_smoke(base)
            p=base/'empty/contract.json'; c=json.loads(p.read_text()); c['initial_fusion_sha256']='different'
            p.write_text(json.dumps(c))
            with self.assertRaises(ValueError): pair_smoke(base)
        script=(ROOT/'slurm/submit_da2_silog_pair.sh').read_text()
        job=(ROOT/'slurm/da2_silog_pair.sbatch').read_text()
        self.assertIn('afterok:$smoke',script); self.assertIn('RESUME=0',script)
        self.assertIn('for arm in frame empty',script)
        self.assertIn('--kill-on-invalid-dep=yes',script)
        self.assertIn('da2_caption_silog_pair_v1',script+job)
        self.assertNotIn('no-caption',job)


if __name__=='__main__': unittest.main()
