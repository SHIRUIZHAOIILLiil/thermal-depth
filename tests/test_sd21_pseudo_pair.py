"""Focused CPU checks; run as a file to avoid Aire's `tests` package collision."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from models.da2_silog import masked_silog
from models.sd21_pseudo_loss import mixed_depth_loss
from tools.sd21_pseudo_data import prepare_inventory, load_pseudo, checked_ids, ThermalMixed
from tools.da2_caption_data import TRAIN
from tools.train_sd21_pseudo_pair import pair_smoke, CODE
from test_sd21_tadp_pair import tiny_model


class MixedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(43)

    def test_formula_masks_and_beta_zero(self):
        p = torch.tensor([[[8., 4., 9., 7., 6.]]], requires_grad=True)
        gt = torch.tensor([[[10., 0., 80., float('nan'), 0.]]])
        pseudo = torch.tensor([[[99., 2., 3., 7., float('inf')]]])
        total, parts = mixed_depth_loss(p, gt, pseudo)
        real,_ = masked_silog(p, gt)
        expected = torch.tensor(2.).log()/2
        torch.testing.assert_close(parts['pseudo'], expected)
        torch.testing.assert_close(total, real + .1*expected)
        self.assertEqual(parts['pseudo_pixels'], 2)
        total.backward()
        self.assertEqual(float(p.grad[0,0,2]), 0.)
        self.assertEqual(float(p.grad[0,0,4]), 0.)
        a,_ = mixed_depth_loss(p, gt, pseudo, beta=0.)
        b,_ = masked_silog(p, gt)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(torch.autograd.grad(a,p)[0],torch.autograd.grad(b,p)[0],rtol=0,atol=0)
        changed=pseudo.clone(); changed[0,0,0]=1.; changed[0,0,2]=79.
        torch.testing.assert_close(mixed_depth_loss(p,gt,changed)[0],total)

    def test_invalid_empty_and_normalization(self):
        p=torch.full((1,1,7),4.,requires_grad=True)
        gt=torch.tensor([[[10.,0.,0.,0.,0.,0.,0.]]])
        invalid=torch.tensor([[[2.,0.,-1.,80.,100.,float('nan'),float('inf')]]])
        loss,parts=mixed_depth_loss(p,gt,invalid)
        self.assertEqual(parts['pseudo_pixels'],0)
        torch.testing.assert_close(loss,masked_silog(p,gt)[0])
        one=invalid.clone(); one[0,0,1]=2.
        many=torch.full_like(one,2.)
        torch.testing.assert_close(mixed_depth_loss(p,gt,one)[1]['pseudo'],mixed_depth_loss(p,gt,many)[1]['pseudo'])
        with self.assertRaises(ValueError): mixed_depth_loss(p,gt,one,beta=.2)
        with self.assertRaises(ValueError): mixed_depth_loss(p,torch.zeros_like(gt),one)

    def fixture(self, root):
        folder=root/'calibrated_pseudo_depth'; folder.mkdir()
        (root/'build_config.json').write_text(json.dumps(dict(depth_scale=256.,stored='calibrated depth in metres, no range handling applied')))
        sid=sorted(TRAIN)[0]+'_000001'
        samples=[SimpleNamespace(sample_id=sid,thermal_gt_path='unused')]
        path=folder/(sid+'.npy')
        np.save(path,np.full((256,640),10.,np.float32))
        gt=np.zeros((256,640),np.float32); gt[0,0]=5.
        return folder,samples,path,gt

    def test_inventory_and_changed_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); folder,samples,path,gt=self.fixture(root)
            dest=root/'index.json'
            with patch('tools.sd21_pseudo_data.load_ms2_gt',return_value=(None,gt)):
                index=prepare_inventory(samples,'manifest',folder,dest)
                self.assertEqual(index['stats']['usable_missing_pixels'],gt.size-1)
                self.assertEqual(prepare_inventory(samples,'manifest',folder,dest),index)
                with self.assertRaises(ValueError): prepare_inventory(samples,'other',folder,dest)
            load_pseudo(path,index['files'][samples[0].sample_id])
            np.save(path,np.full((256,640),11.,np.float32))
            with self.assertRaises(ValueError): load_pseudo(path,index['files'][samples[0].sample_id])
            path.unlink()
            with self.assertRaises(FileNotFoundError): prepare_inventory(samples,'manifest',folder,dest)
            with self.assertRaises(ValueError): checked_ids([SimpleNamespace(sample_id='test_000001')])
            with self.assertRaises(ValueError): checked_ids(samples+samples)

    def test_missing_format_and_no_usable_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); folder,samples,path,gt=self.fixture(root)
            path.unlink()
            with self.assertRaises(FileNotFoundError): prepare_inventory(samples,'m',folder,root/'index')
            for array in (np.ones((2,3),np.float32),np.ones((256,640),np.uint16)):
                np.save(path,array)
                with self.assertRaises(ValueError): load_pseudo(path)
            path.write_bytes(b'broken')
            with self.assertRaises(ValueError): load_pseudo(path)
            np.save(path,np.zeros((256,640),np.float32))
            with patch('tools.sd21_pseudo_data.load_ms2_gt',return_value=(None,gt)):
                with self.assertRaises(ValueError): prepare_inventory(samples,'m',folder,root/'index')
            (root/'build_config.json').write_text('{}')
            with self.assertRaises(ValueError): prepare_inventory(samples,'m',folder,root/'index')

    def test_real_model_mixed_backward(self):
        model=tiny_model(); model.train()
        pred=model(torch.rand(2,3,64,128)*2-1,torch.randint(0,50,(2,77)))
        gt=torch.zeros_like(pred); gt[:,::3,::4]=12.
        loss,parts=mixed_depth_loss(pred,gt,torch.full_like(gt,15.))
        loss.backward()
        self.assertGreater(parts['pseudo_pixels'],0)
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for n,p in model.unet.named_parameters() if 'attn2.to_k.weight' in n))
        self.assertTrue(all(p.grad is None for m in (model.vae,model.text_encoder) for p in m.parameters()))

    def test_pair_smoke_accounting_and_order(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for arm in ('frame','empty'):
                out=root/arm; out.mkdir()
                (out/'contract.json').write_text(json.dumps(dict(arm=arm,beta=.1,pseudo_inventory_sha256='abc')))
                (out/'completed.json').write_text(json.dumps(dict(step=32,mode='smoke',smoke_checks_passed=True)))
                rows=[dict(step=i,data_ids_sha256=str(i),pseudo_pixels=2,pseudo_log_l1=.5,real_loss=1.,loss=1.05) for i in range(1,33)]
                (out/'train.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            pair_smoke(root)
            rows[0]['loss']=2.
            (root/'empty/train.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            with self.assertRaises(ValueError): pair_smoke(root)

    def test_submission_dependencies_and_recipe(self):
        self.assertEqual(len(CODE),len(set(CODE)))
        self.assertTrue(all((ROOT/f).is_file() for f in CODE))
        source=(ROOT/'tools/train_sd21_pseudo_pair.py').read_text()
        self.assertIn('ThermalDirect(val, ts[1])',source)
        self.assertIn('ThermalMixed(train, ts[0], inventory)',source)
        if os.name=='nt': self.skipTest('bash mock runs in Linux/WSL')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); binary=root/'sbatch'
            binary.write_text('#!/bin/bash\necho "$*" >> "$CALLS"\necho 12345\n')
            binary.chmod(0o755)
            env=dict(os.environ,PATH=str(root)+':'+os.environ['PATH'],IRIS_REPO=str(ROOT),IRIS_RUNS=str(root/'runs'),CALLS=str(root/'calls'))
            subprocess.run(['bash',str(ROOT/'slurm/submit_sd21_pseudo_pair.sh')],env=env,check=True,capture_output=True)
            calls=(root/'calls').read_text().splitlines()
            self.assertEqual(len(calls),3)
            for call,arm in zip(calls[1:],('frame','empty')):
                self.assertIn('--dependency=afterok:12345',call)
                self.assertIn('--kill-on-invalid-dep=yes',call)
                self.assertIn('ARM='+arm,call)


if __name__=='__main__': unittest.main()
