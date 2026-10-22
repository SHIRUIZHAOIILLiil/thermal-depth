"""CPU tests with tiny models; no claim of real-checkpoint GPU validation."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from models.da2_caption import CaptionDepth
from tests.test_da2_caption import TinyCore, TinyText
from tools.train_da2_caption import evaluate
from tools.eval_da2_caption_ablation import (AdapterOff, CaptionIntervention, derangement,
                                            validate_checkpoint, check_reproduction, paired_summary)


class AblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(43)

    def test_derangement_reproducible_bijective_no_self_and_local_rng(self):
        np.random.seed(91)
        state = np.random.get_state()
        for n in (2, 3, 100, 14249):
            order = derangement(n, 43)
            self.assertEqual(sorted(order.tolist()), list(range(n)))
            self.assertTrue(np.all(order != np.arange(n)))
            np.testing.assert_array_equal(order, derangement(n, 43))
        np.testing.assert_array_equal(state[1], np.random.get_state()[1])
        self.assertFalse(np.array_equal(derangement(100, 43), derangement(100, 44)))
        with self.assertRaises(ValueError): derangement(1, 43)

    def test_only_tokens_and_mask_change_not_images_or_gt(self):
        base = [dict(id=str(i), sequence='seq', image=torch.full((3, 28, 28), float(i)),
                     gt=torch.full((28, 28), float(i+1))) for i in range(3)]
        tokens = dict(input_ids=torch.tensor([[1, 2], [3, 4], [5, 6]]),
                      attention_mask=torch.tensor([[1, 0], [1, 1], [0, 1]]))
        dataset = CaptionIntervention(base, tokens, [2, 0, 1])
        for i, j in enumerate([2, 0, 1]):
            item = dataset[i]
            self.assertEqual(item['id'], base[i]['id'])
            self.assertIs(item['image'], base[i]['image'])
            self.assertIs(item['gt'], base[i]['gt'])
            torch.testing.assert_close(item['token_ids'], tokens['input_ids'][j])
            torch.testing.assert_close(item['attention_mask'], tokens['attention_mask'][j])
            self.assertNotIn('token_ids', base[i])
        with self.assertRaises(ValueError): CaptionIntervention(base, tokens, [0, 0, 1])

    def test_adapter_off_exact_visual_core_no_text_call_or_weight_mutation(self):
        model = CaptionDepth(TinyCore(), TinyText(), vision_dim=8, text_dim=6).eval()
        for adapter in model.adapters: adapter.alpha.data.fill_(.4)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        def forbidden(*args, **kwargs):
            raise AssertionError('Text encoder must not be called when adapters off')
        model.text_encoder.forward = forbidden
        x = torch.rand(2, 3, 30, 33)
        with torch.no_grad():
            result = AdapterOff(model).eval()(x)
            expected = model.core(torch.nn.functional.pad(x, (0, 9, 0, 12), mode='reflect'))[..., :30, :33]
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        for k, v in model.state_dict().items(): torch.testing.assert_close(v, before[k], rtol=0, atol=0)

    def test_checkpoint_rejects_wrong_arm_step_cursor_contract(self):
        contract = {'arm': 'caption'}
        state = dict(contract=contract, step=8000, image_cursor=96000)
        validate_checkpoint(state, contract)
        for field, value in [('step', 6000), ('image_cursor', 32000), ('contract', {'arm': 'no-caption'})]:
            bad = {**state, field: value}
            with self.assertRaises(ValueError): validate_checkpoint(bad, contract)
        with self.assertRaises(ValueError):
            validate_checkpoint({**state, 'contract': {'arm': 'no-caption'}}, {'arm': 'no-caption'})

    def test_reproduction_checks_precision_sequences_and_counts(self):
        report = dict(step=8000, samples=14249,
                      metrics={m: dict(abs_rel=.07, rmse=2.2, a1=.96) for m in ('ssi', 'none')},
                      sequences={m: {'seq': dict(frames=14249, abs_rel=.07)} for m in ('ssi', 'none')})
        check_reproduction(report, copy.deepcopy(report))
        for kind in ('metric', 'sequence', 'count', 'nan'):
            bad = copy.deepcopy(report)
            if kind == 'metric': bad['metrics']['none']['abs_rel'] += 1e-4
            if kind == 'sequence': bad['sequences']['ssi']['seq']['abs_rel'] += .001
            if kind == 'count': bad['samples'] = 100
            if kind == 'nan': bad['metrics']['ssi']['a1'] = float('nan')
            with self.assertRaises(ValueError): check_reproduction(report, bad)

    def test_three_conditions_use_same_evaluator_and_leave_weights_unchanged(self):
        model = CaptionDepth(TinyCore(), TinyText(), vision_dim=8, text_dim=6).eval()
        for adapter in model.adapters: adapter.alpha.data.fill_(.3)
        model.requires_grad_(False)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        base = [dict(id=str(i), sequence='seq', image=torch.rand(3, 28, 28),
                     gt=torch.linspace(2, 50, 784).reshape(28, 28)) for i in range(2)]
        tokens = dict(input_ids=torch.tensor([[1, 2, 3], [7, 8, 9]]), attention_mask=torch.ones(2, 3, dtype=torch.bool))
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for name, dataset, predictor in [('correct', CaptionIntervention(base, tokens, [0, 1]), model),
                                              ('shuffled', CaptionIntervention(base, tokens, [1, 0]), model),
                                              ('adapter_off', base, AdapterOff(model).eval())]:
                out = Path(tmp) / name; out.mkdir()
                evaluate(predictor, dataset, torch.device('cpu'), False, out, 8000)
                report = json.loads((out / 'val_step008000.json').read_text())
                self.assertEqual(report['samples'], 2)
                self.assertEqual(set(report['metrics']), {'ssi', 'none'})
                paths[name] = out / 'val_step008000_frames.csv'
            result = paired_summary(paths)
            self.assertEqual(result['comparisons']['shuffled']['ssi']['overall']['frames'], 2)
        for k, v in model.state_dict().items(): torch.testing.assert_close(v, before[k], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_paired_delta_sign_and_frame_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for condition, error, accuracy in [('correct', .1, .9), ('shuffled', .2, .8), ('adapter_off', .05, .95)]:
                path = Path(tmp) / f'{condition}.csv'; paths[condition] = path
                with path.open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=['id', 'sequence', 'alignment', 'abs_rel', 'rmse', 'a1'])
                    writer.writeheader()
                    for mode in ('ssi', 'none'):
                        writer.writerow(dict(id='a', sequence='seq', alignment=mode, abs_rel=error, rmse=error*10, a1=accuracy))
            summary = paired_summary(paths)['comparisons']
            self.assertAlmostEqual(summary['shuffled']['ssi']['overall']['metrics']['abs_rel']['mean_delta'], .1)
            self.assertEqual(summary['shuffled']['ssi']['overall']['metrics']['a1']['correct_better_fraction'], 1.)
            self.assertEqual(summary['adapter_off']['ssi']['overall']['metrics']['a1']['correct_better_fraction'], 0.)
            paths['shuffled'].write_text(paths['shuffled'].read_text().replace('a,seq', 'b,seq'))
            with self.assertRaises(ValueError): paired_summary(paths)


if __name__ == '__main__':
    unittest.main()
