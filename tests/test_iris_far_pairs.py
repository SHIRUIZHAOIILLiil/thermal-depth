"""Focused CPU checks for local real-LiDAR pairs, gradient ownership, and validation selection."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from tools.iris_far_pair_loss import local_pairs, signed_gap_loss, trainable_convs
from tools.train_iris_far_pairs import select


class PairTests(unittest.TestCase):
    def test_sparse_native_pairs_and_signed_gradient(self):
        gt = np.zeros((4,40),np.float32)
        gt[1,[0,8,16,24,32]] = [40,60,61,80,np.nan]
        pairs = local_pairs(gt,43)
        np.testing.assert_array_equal(pairs, [[40,48],[40,56]])
        truth = torch.tensor(gt)[None,None]
        pred = torch.nan_to_num(truth).detach().clone()
        pred[0,0,1,8] = 45
        pred.requires_grad_(True)
        loss,n = signed_gap_loss(pred,truth,[pairs],[True])
        self.assertEqual((n, float(loss.detach())), (2, 7.5))
        loss.backward()
        self.assertLess(pred.grad[0,0,1,8].item(), 0.)  # gradient descent moves the underestimated point farther
        np.testing.assert_array_equal(local_pairs(np.zeros_like(gt),43).shape, (0,2))
        self.assertEqual(signed_gap_loss(pred,truth,[pairs],[False])[1],0)

    def test_pair_cap_repeatability_and_frame_weight(self):
        rng = np.random.default_rng(0)
        gt = rng.uniform(1,79,(80,80)).astype('float32')
        a = local_pairs(gt,43)
        self.assertEqual(len(a),256)
        self.assertEqual(len(np.unique(a,axis=0)),256)
        np.testing.assert_array_equal(a, local_pairs(gt,43))
        g = torch.tensor([40.,60.]).view(1,1,1,2).expand(2,-1,-1,-1)
        p = g.clone()
        p[0,0,0,1] -= 10
        loss,n = signed_gap_loss(p,g,[np.array([[0,1]]),np.array([[0,1],[0,1]])],[True,True])
        self.assertEqual((float(loss),n),(5.,3))

    def test_trainable_filter_excludes_attention_norm_and_output(self):
        net = nn.Module()
        net.up_blocks = nn.ModuleList([nn.Module() for _ in range(4)])
        for block in net.up_blocks:
            block.resnets = nn.Sequential(nn.Conv2d(2,2,1),nn.GroupNorm(1,2))
            block.attentions = nn.Sequential(nn.Conv2d(2,2,1))
            block.upsamplers = nn.Sequential(nn.Conv2d(2,2,1))
        net.conv_out = nn.Conv2d(2,1,1)
        params = trainable_convs(net)
        expected = {f'up_blocks.{i}.{part}.0.{key}' for i in (2,3)
                    for part in ('resnets','upsamplers') for key in ('weight','bias')}
        self.assertEqual(set(params),expected)
        self.assertEqual({n for n,p in net.named_parameters() if p.requires_grad},expected)

    def test_selection_prefers_zero_unless_all_gates_pass(self):
        z=dict(abs_rel=.08,rmse=3.,a1=.94)
        curve={0:z,500:dict(z,rmse=2.98),1000:dict(z,rmse=2.90,a1=.938)}
        self.assertEqual(select(curve),0)
        curve[2000]=dict(z,rmse=2.94)
        self.assertEqual(select(curve),2000)


if __name__=='__main__':
    unittest.main()
