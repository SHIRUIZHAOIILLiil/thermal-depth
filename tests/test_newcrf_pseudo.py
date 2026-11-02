"""CPU checks for dense targets, val-only selection and the official test fold."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from tools import newcrf_pseudo as runner


class NewCRFTests(unittest.TestCase):
    def test_target_matches_old_iris_and_optimizer_updates(self):
        gt = np.array([[0, 10, 80, .001, 4]], dtype=np.float32)
        pseudo = np.array([[3, np.nan, 90, -1, 2]], dtype=np.float32)
        target = runner.completed_target(pseudo, gt)
        np.testing.assert_allclose(target, [[3, 10, 80, .001, 4]])
        value = torch.nn.Parameter(torch.tensor(2.))
        opt = torch.optim.SGD([value], lr=.01)
        loss = runner.silog(value.expand(1, 1, 5), torch.from_numpy(target)[None])
        loss.backward()
        self.assertTrue(torch.isfinite(value.grad))
        opt.step()
        self.assertNotEqual(float(value.detach()), 2.)
        depth = torch.ones(1,1,5, requires_grad=True)
        runner.silog(depth, torch.from_numpy(target)[None]).backward()
        self.assertEqual(float(depth.grad[0,0,2]), 0.)
        with self.assertRaises(ValueError):
            runner.completed_target(pseudo, np.zeros_like(gt))

    def test_selection_keeps_original_and_best_nonzero_separate(self):
        def curve(numbers):
            return {s: {'metrics': {'ssi': {'abs_rel': n}}} for s,n in numbers}
        self.assertEqual(runner.selection(curve([(0,.07),(500,.08),(1000,.08)])), (0,500))
        self.assertEqual(runner.selection(curve([(0,.07),(500,.07),(1000,.06)])), (1000,1000))

    def test_train_mode_freezes_bn_statistics_only(self):
        net = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 1), torch.nn.BatchNorm2d(2), torch.nn.Dropout(.5))
        before = runner.bn_stats(net)
        runner.train_mode(net)
        self.assertTrue(net.training and net[0].training and net[2].training)
        self.assertFalse(net[1].training)
        net(torch.randn(2, 1, 3, 3)).sum().backward()
        self.assertTrue(all(torch.equal(v, before[n]) for n, v in runner.bn_stats(net).items()))
        self.assertIsNotNone(net[1].weight.grad)

    def test_official_fold_real_gt_and_no_pseudo_access(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(1.))

            def forward(self, x):
                return self.scale * (x[:, :1]+4)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            thermal = np.tile(np.arange(640, dtype=np.uint16)+3000, (256,1))
            Image.fromarray(thermal).save(d/'thermal.png')
            samples = []
            for i in range(2):
                path = d/f'gt{i}.png'
                Image.fromarray((thermal*(1+i)).astype('uint16')).save(path)
                samples.append(SimpleNamespace(sample_id=str(i), thermal_path=d/'thermal.png', thermal_gt_path=path))
            out = d/'rows.csv'
            result = runner.evaluate(Net(), samples, 0, out, {'1','missing'})
            self.assertEqual(result['frames'], 2)
            common = result['official_common']
            self.assertEqual((common['frames'], common['crawl_count'], common['missing_ids']), (1,2,['missing']))
            with out.open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(common['metrics']['none']['rmse'], float(rows[1]['none_rmse']))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
