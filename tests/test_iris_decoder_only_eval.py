"""CPU contract tests; the real decoder is reproduced on full validation by the GPU job."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from tools import eval_iris_decoder_only as runner

START = dict(abs_rel=.0790, rmse=2.74, a1=.945)
BETTER = dict(abs_rel=.0775, rmse=2.67, a1=.947)
REFERENCE = dict(abs_rel=.085, rmse=2.47, a1=.94)  # synthetic test values, NOT measured NeWCRF results


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.post_quant_conv = nn.Conv2d(1, 1, 1)
        self.decoder = nn.Conv2d(1, 3, 1)


def write_run(root, selected=1000, curve=None):
    (root/'train').mkdir()
    (root/'contract.json').write_text(json.dumps(dict(code={})))
    weights = root/'train'/'decoder_1000.pt'
    torch.save({'decoder.weight': torch.ones(1)}, weights)
    curve = curve or {0: START, 500: BETTER, 1000: BETTER}
    summary = dict(selected=selected, curve={str(k): dict(metrics=v) for k, v in curve.items()},
                   decision=dict(reference=REFERENCE), selected_sha256=runner.pair.sha256_file(weights),
                   contract_sha256=runner.pair.sha256_file(root/'contract.json'))
    (root/'train'/'summary.json').write_text(json.dumps(summary))
    return weights


class Evaluation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_selection_rejects_a_choice_the_rule_did_not_make(self):
        write_run(self.root, selected=1000, curve={0: START, 500: BETTER, 1000: START})
        with self.assertRaises(ValueError):
            runner.selection(self.root)

    def test_retained_start_is_not_tested(self):
        write_run(self.root, selected=0, curve={0: START, 500: START, 1000: START})
        with self.assertRaises(ValueError):
            runner.selection(self.root)

    def test_changed_decoder_file_is_rejected(self):
        curve = {0: START, 500: {**BETTER, 'rmse': 2.68}, 1000: BETTER}
        weights = write_run(self.root, curve=curve)
        self.assertEqual(runner.selection(self.root)[2], 1000)
        torch.save({'decoder.weight': torch.zeros(1)}, weights)
        with self.assertRaises(ValueError):
            runner.selection(self.root)

    def test_reproduction_tolerances(self):
        runner.check_reproduction(dict(abs_rel=.07753, rmse=2.6754, a1=.94693),
                                  dict(abs_rel=.07753, rmse=2.6754, a1=.94693))
        for key, delta in (('abs_rel', .0001), ('rmse', .003), ('a1', .0001)):
            with self.assertRaises(ValueError):
                runner.check_reproduction({**BETTER, key: BETTER[key]+delta}, BETTER)

    def test_installed_weights_reach_the_deployed_decoder(self):
        model = SimpleNamespace(unet=nn.Conv2d(1, 1, 1), gt_vae=TinyVAE(),
                                lotus=SimpleNamespace(vae=nn.Conv2d(1, 1, 1), text_encoder=nn.Linear(1, 1)))
        original, frozen = model.gt_vae, model.gt_vae.post_quant_conv.weight.clone()
        state = {k: p.detach()+1 for k, p in model.gt_vae.named_parameters() if k.startswith('decoder.')}
        path = self.root/'decoder.pt'
        torch.save(state, path)
        runner.install_decoder(model, path)
        self.assertIsNot(model.gt_vae, original)
        torch.testing.assert_close(model.gt_vae.decoder.weight, state['decoder.weight'])
        torch.testing.assert_close(model.gt_vae.post_quant_conv.weight, frozen)
        self.assertFalse(any(p.requires_grad for p in model.gt_vae.parameters()))
        self.assertFalse(model.gt_vae.training)


if __name__ == '__main__':
    unittest.main()
