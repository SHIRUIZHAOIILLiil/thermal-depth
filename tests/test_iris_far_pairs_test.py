import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from tools import test_iris_far_pairs as runner


class SelectedTests(unittest.TestCase):
    def test_stride_winner_falls_back_to_zero_on_full_validation(self):
        z = dict(rmse=3., abs_rel=.08, a1=.94)
        good = dict(z,rmse=2.9)
        summary = dict(curve={'0':z,'500':good,'1000':z,'2000':z},provisional=500,
                       full={'0':z,'500':dict(good,abs_rel=.082),'2000':z},selected=0)
        self.assertEqual(runner.selected_step(summary,0),0)
        summary['full']['500']=good
        summary['selected']=500
        self.assertEqual(runner.selected_step(summary,500),500)
        with self.assertRaises(ValueError):
            runner.selected_step(summary,0)

    def test_each_arm_resets_parent_before_applying_its_own_selected_weights(self):
        net=torch.nn.Module()
        net.up_blocks=torch.nn.ModuleList([torch.nn.Module() for _ in range(4)])
        for block in net.up_blocks:
            block.resnets=torch.nn.Sequential(torch.nn.Conv2d(1,1,1))
        params=runner.train.trainable_convs(net)
        model=SimpleNamespace(unet=net,set_train=net.train)
        def load(_,source,arm):
            with torch.no_grad():
                for p in net.parameters():
                    p.fill_(1. if arm=='frame' else 2.)
        with tempfile.TemporaryDirectory() as d, patch.object(runner.iris,'load_arm',side_effect=load):
            conv=Path(d)/'conv_500.pt'
            torch.save({k:torch.full_like(p,3.) for k,p in params.items()},conv)
            runner.load_selected(model,'frame',None,conv,{'trainable':list(params)})
            self.assertTrue(all((p==3).all() for p in params.values()))
            runner.load_selected(model,'empty',None,None,{})
            self.assertTrue(all((p==2).all() for p in net.parameters()))

    def test_common_fold_uses_same_per_frame_ssi_log_as_full(self):
        samples=[SimpleNamespace(sample_id=s) for s in ('a','b','c')]
        raw=np.arange(12,dtype=np.float32).reshape(3,4)/12
        gt=np.exp(1+raw*2).astype(np.float32)
        preds=[(s.sample_id, raw, gt.copy()) for s in samples]
        preds[1][2][0,1]*=2
        expected=runner.evaluate_sample(raw,preds[1][2],align='ssi_log')['rmse']
        with tempfile.TemporaryDirectory() as d:
            result=runner.score(samples,iter(preds),{'b'},Path(d)/'rows.csv')
            self.assertAlmostEqual(result['official_common']['rmse'],expected)
            self.assertLess(result['full']['rmse'],expected)
            with self.assertRaises(ValueError):
                runner.score(samples,iter(preds),{'unknown'},Path(d)/'bad.csv')


if __name__=='__main__':
    unittest.main()
