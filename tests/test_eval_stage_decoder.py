"""CPU checks for tools/eval_stage_decoder.py; the GPU job reproduces full validation before test."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tools import eval_stage_decoder as tool

START = dict(abs_rel=.0746, rmse=2.595, a1=.9516)
BETTER = dict(abs_rel=.0731, rmse=2.560, a1=.9532)
REFERENCE = dict(abs_rel=.085, rmse=2.47, a1=.94)  # synthetic, not measured NeWCRF values


def write_run(root, selected=1000, curve=None):
    (root/'train').mkdir()
    unet = root/'unet.pt'
    torch.save(dict(route='b_thermal_unet', caption_mode='correct', averaged_steps=[12000], state_dicts=dict(unet={})), unet)
    (root/'contract.json').write_text(json.dumps(dict(source=str(unet), source_sha256=tool.pair.sha256_file(unet))))
    weights = root/'train'/'decoder_1000.pt'
    torch.save({'decoder.weight': torch.ones(1)}, weights)
    curve = curve or {0: START, 500: {**BETTER, 'rmse': 2.57}, 1000: BETTER}
    summary = dict(selected=selected, curve={str(k): dict(metrics=v) for k, v in curve.items()},
                   decision=dict(reference=REFERENCE), selected_sha256=tool.pair.sha256_file(weights),
                   contract_sha256=tool.pair.sha256_file(root/'contract.json'))
    (root/'train'/'summary.json').write_text(json.dumps(summary))
    return unet, weights


class FakeModel:
    def __init__(self):
        self.unet = torch.nn.Linear(1, 1)
        self.loaded = None

    def load_state_dicts(self, states, label):
        self.loaded = label

    def set_train(self, flag):
        pass


class Selection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_start_stage_uses_update_zero_and_no_decoder(self):
        write_run(self.root)
        _, expected, step, decoder = tool.selection(self.root, 'start')
        self.assertEqual((expected, step, decoder), (START, 0, None))

    def test_selected_stage_follows_the_frozen_rule(self):
        _, weights = write_run(self.root)
        _, expected, step, decoder = tool.selection(self.root, 'selected')
        self.assertEqual((expected, step, decoder), (BETTER, 1000, weights))

    def test_rule_disagreement_and_changed_files_are_refused(self):
        write_run(self.root, selected=500)
        with self.assertRaises(ValueError):
            tool.selection(self.root, 'selected')

    def test_changed_decoder_is_refused(self):
        _, weights = write_run(self.root)
        torch.save({'decoder.weight': torch.zeros(1)}, weights)
        with self.assertRaises(ValueError):
            tool.selection(self.root, 'selected')

    def test_unet_identity_is_checked(self):
        unet, _ = write_run(self.root)
        contract = json.loads((self.root/'contract.json').read_text())
        model = FakeModel()
        self.assertEqual(tool.load_unet(model, contract), unet)
        self.assertEqual(model.loaded, str(unet))
        torch.save(dict(changed=True), unet)
        with self.assertRaises(ValueError):
            tool.load_unet(FakeModel(), contract)

    def test_stage_folders_are_separate(self):
        self.assertNotEqual(tool.folder(self.root, 'start'), tool.folder(self.root, 'selected'))


if __name__ == '__main__':
    unittest.main()
