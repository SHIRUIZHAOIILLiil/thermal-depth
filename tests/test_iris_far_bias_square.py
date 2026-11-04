import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tools.train_iris_far_bias_square import far_bias_square


class FarBiasTests(unittest.TestCase):
    def test_each_frame_mean_before_squaring_and_equal_frame_weights(self):
        gt = torch.full((3,1,1,100), 60.)
        gt[1,:,:,50:] = 80.  # exactly 50 eligible in frame 1
        pred = gt.clone()
        pred[0,:,:,:50] += 3
        pred[0,:,:,50:] -= 3
        pred[1] -= 4
        pred[2] -= 6
        pred.requires_grad_()
        loss, n = far_bias_square(pred, gt, gt > 0, [True, True, False])
        self.assertEqual((loss.item(), n), (8.,2))
        loss.backward()
        self.assertFalse(pred.grad[0].any())
        self.assertTrue((pred.grad[1,:,:,:50] < 0).all())
        self.assertFalse(pred.grad[1,:,:,50:].any())
        self.assertFalse(pred.grad[2].any())

    def test_small_or_invalid_bands_give_zero(self):
        gt = torch.full((1,1,1,51), 65.)
        gt.flatten()[0] = 59.99
        valid = torch.ones_like(gt, dtype=torch.bool)
        valid.flatten()[1] = False
        pred = gt.clone().requires_grad_()
        loss,n = far_bias_square(pred, gt, valid, [True])
        self.assertEqual((loss.item(), n), (0.,0))
        loss.backward()
        self.assertFalse(pred.grad.any())


if __name__ == '__main__':
    unittest.main()
