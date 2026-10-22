"""CPU gates; tiny stand-ins exercise interfaces without downloading pretrained assets."""
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image
import torch
from torch import nn
from models.da2_caption import CaptionAdapter, CaptionDepth, masked_log_l1
from tools.da2_caption_data import ThermalLidar, ImageStream, checked_manifest
from tools.train_da2_caption import (make_optimizer, save_checkpoint, load_checkpoint,
                                    seed_all, token_cache, evaluate)
from tools.check_da2_caption_pair import check


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 8, 14, 14)

    def get_intermediate_layers(self, image, indices, return_class_token=True):
        tokens = self.proj(image).flatten(2).transpose(1, 2)
        return [(tokens, tokens.mean(dim=1)) for _ in indices]


class TinyHead(nn.Module):
    def __init__(self):
        super().__init__(); self.proj = nn.Linear(8, 1)

    def forward(self, features, h, w):
        tokens = sum(x[0] for x in features) / 4
        x = self.proj(tokens).transpose(1, 2).reshape(-1, 1, h, w)
        return torch.nn.functional.interpolate(x, (h * 14, w * 14), mode='bilinear').sigmoid()


class TinyCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = TinyEncoder(); self.depth_head = TinyHead()
        self.encoder = 'vitb'; self.max_depth = 80.; self.intermediate_layer_idx = {'vitb': [2, 5, 8, 11]}

    def forward(self, image):
        features = self.pretrained.get_intermediate_layers(image, self.intermediate_layer_idx['vitb'])
        return self.depth_head(features, image.shape[-2] // 14, image.shape[-1] // 14).squeeze(1) * 80


class TinyText(nn.Module):
    def __init__(self):
        super().__init__(); self.embedding = nn.Embedding(30, 6); self.drop = nn.Dropout(.5)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.drop(self.embedding(input_ids)))


class CaptionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        seed_all(43)

    def make(self, caption=True):
        return CaptionDepth(TinyCore(), TinyText() if caption else None, vision_dim=8, text_dim=6)

    def test_no_caption_has_no_text_parameters(self):
        model = self.make(False)
        self.assertIsNone(model.text_encoder)
        self.assertEqual(len(model.adapters), 0)
        self.assertTrue(all(n.startswith('core.') for n, _ in model.named_parameters()))

    def test_zero_gate_preserves_core_and_crop(self):
        model = self.make()
        x = torch.rand(2, 3, 30, 33)
        ids = torch.ones(2, 7, dtype=torch.long); mask = torch.ones_like(ids, dtype=torch.bool)
        output = model(x, ids, mask)
        padded = torch.nn.functional.pad(x, (0, 9, 0, 12), mode='reflect')
        torch.testing.assert_close(output, model.core(padded)[..., :30, :33], rtol=0, atol=0)
        self.assertEqual(output.shape, (2, 30, 33))

    def test_frozen_text_and_gradients(self):
        model = self.make(); model.train()
        self.assertFalse(model.text_encoder.training)
        x = torch.rand(2, 3, 28, 28); ids = torch.ones(2, 7, dtype=torch.long)
        out = model(x, ids, ids.bool())
        loss, _ = masked_log_l1(out, torch.ones_like(out) * 10)
        loss.backward()
        self.assertTrue(any(a.alpha.grad.abs() > 0 for a in model.adapters))
        self.assertTrue(all(p.grad is None for p in model.text_encoder.parameters()))
        model.zero_grad()
        for a in model.adapters: a.alpha.data.fill_(.1)
        model(x, ids, ids.bool()).sum().backward()
        self.assertGreater(model.adapters[0].query.weight.grad.abs().sum(), 0)

    def test_masked_tokens_cannot_affect_attention(self):
        adapter = CaptionAdapter(8, 6, width=8, heads=2)
        adapter.alpha.data.fill_(1)
        v = torch.randn(2, 5, 8); t = torch.randn(2, 7, 6)
        mask = torch.tensor([[1, 1, 1, 0, 0, 0, 0]] * 2, dtype=torch.bool)
        before = adapter(v, t, mask)
        t[:, 3:] = 100 * torch.randn_like(t[:, 3:])
        torch.testing.assert_close(before, adapter(v, t, mask), rtol=0, atol=0)
        with self.assertRaises(ValueError): adapter(v, t, torch.zeros_like(mask))

    def test_log_loss_macro_and_empty_mask(self):
        g = torch.tensor([[[1., 1., 1.]], [[1., 0., 80.]]])
        p = torch.tensor([[[2., 2., 2.]], [[4., float('nan'), float('nan')]]], requires_grad=True)
        loss, empty = masked_log_l1(p, g)
        self.assertAlmostEqual(loss.item(), (np.log(2) + np.log(4)) / 2, places=6)
        self.assertEqual(empty, 0)
        loss.backward(); self.assertEqual(p.grad[1, 0, 1].item(), 0)
        with self.assertRaises(ValueError): masked_log_l1(torch.ones_like(g), torch.zeros_like(g))

    def test_stream_no_drop_and_resume(self):
        full = list(ImageStream(17, 43, 0, 100))
        part = list(ImageStream(17, 43, 0, 36)) + list(ImageStream(17, 43, 36, 64))
        self.assertEqual(full, part)
        self.assertEqual(set(full[:17]), set(range(17)))
        self.assertEqual(full, list(ImageStream(17, 43, 0, 100)))

    def test_full_checkpoint_replays_optimizer_scheduler_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.make(); opt, scheduler = make_optimizer(model)
            contract = {'test': 1}
            def update(m, o, s):
                o.zero_grad()
                image = torch.rand(2, 3, 28, 28) * (random.random() + np.random.random())
                ids = torch.randint(0, 30, (2, 7))
                loss = m(image, ids, torch.ones_like(ids).bool()).square().mean()
                loss.backward(); o.step(); s.step()
            update(model, opt, scheduler)
            path = Path(tmp) / 'last.pt'
            save_checkpoint(path, model, opt, scheduler, 1, contract)
            update(model, opt, scheduler)
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            restored = self.make(); opt2, sched2 = make_optimizer(restored)
            self.assertEqual(load_checkpoint(path, restored, opt2, sched2, contract), 1)
            update(restored, opt2, sched2)
            for k, value in expected.items():
                torch.testing.assert_close(value, restored.state_dict()[k], rtol=0, atol=0)
            self.assertEqual(scheduler.state_dict(), sched2.state_dict())
            with self.assertRaisesRegex(ValueError, 'contract differs'):
                load_checkpoint(path, restored, opt2, sched2, {'test': 2})

    def test_uint16_native_gt_and_split_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); seq = '2021-08-06-10-59-33'
            image = root / seq / 'thermal/000001.png'; gt = root / seq / 'thr/000001.png'
            image.parent.mkdir(parents=True); gt.parent.mkdir(parents=True)
            Image.fromarray((np.arange(28*28).reshape(28, 28) + 10000).astype(np.uint16)).save(image)
            raw_gt = np.zeros((28, 28), np.uint16); raw_gt[4, 7] = 2560
            Image.fromarray(raw_gt).save(gt)
            manifest = root / 'train.jsonl'
            manifest.write_text(json.dumps(dict(id=seq+'_000001', thermal_path=str(image),
                                                thermal_depth_path=str(gt), caption='road')), encoding='utf-8')
            samples, _ = checked_manifest(manifest, root, 'train', 1)
            item = ThermalLidar(samples)[0]
            self.assertEqual(int((item['gt'] > 0).sum()), 1)
            self.assertEqual(item['gt'][4, 7], 10)
            self.assertGreater(item['thermal_stats'][2], 0)
            self.assertEqual(item['image'].shape, (3, 28, 28))
            with self.assertRaisesRegex(ValueError, 'Forbidden'):
                checked_manifest(manifest, root, 'val')

    def test_adapter_rng_independent(self):
        core = TinyCore(); text = TinyText()
        state = torch.get_rng_state().clone()
        CaptionDepth(core, text, vision_dim=8, text_dim=6)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_token_cache_and_stale_guard(self):
        class Tokenizer:
            def __call__(self, texts, **kw):
                if not kw.get('truncation'):
                    return {'input_ids': [[1]*80 for _ in texts]}
                return {'input_ids': torch.ones(len(texts), 77, dtype=torch.long),
                        'attention_mask': torch.ones(len(texts), 77, dtype=torch.long)}
        with tempfile.TemporaryDirectory() as tmp:
            assets = {'root': tmp, 'text_fingerprint': 'a'}
            samples = [SimpleNamespace(sample_id='x', caption='road')]
            a = token_cache(samples, Tokenizer(), assets, 'train', True)
            self.assertEqual(a['truncated'], 1)
            b = token_cache(samples, None, assets, 'train', False)
            self.assertTrue(torch.equal(a['input_ids'], b['input_ids']))
            samples[0].caption = 'changed'
            with self.assertRaises(FileNotFoundError): token_cache(samples, None, assets, 'train', False)

    def test_eval_preserves_rng_and_writes_both_protocols(self):
        model = self.make(False)
        dataset = [dict(image=torch.rand(3,28,28), gt=torch.linspace(1,70,28*28).reshape(28,28), id='x', sequence='s')]
        state = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as tmp:
            evaluate(model, dataset, torch.device('cpu'), False, Path(tmp), 6000)
            report = json.loads((Path(tmp)/'val_step006000.json').read_text())
            self.assertEqual(set(report['metrics']), {'ssi', 'none'})
            self.assertEqual(report['samples'], 1)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(model.training)

    def test_pair_audit_rejects_mismatched_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b = Path(tmp)/'a',Path(tmp)/'b'; a.mkdir(); b.mkdir()
            for directory, arm in ((a,'no-caption'), (b,'caption')):
                (directory/'contract.json').write_text(json.dumps(dict(arm=arm, seed=43)))
                (directory/'completed.json').write_text(json.dumps(dict(mode='smoke',step=32)))
                (directory/'train.jsonl').write_text('\n'.join(json.dumps(dict(step=s,data_ids_sha256=str(s))) for s in range(1,33)))
            self.assertTrue(check(a,b,True))
            (b/'train.jsonl').write_text((b/'train.jsonl').read_text().replace('"32"', '"wrong"'))
            with self.assertRaisesRegex(ValueError, 'order differs'): check(a,b,True)

    def test_pilot_decision_is_caption_vs_no_caption(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b = Path(tmp)/'a',Path(tmp)/'b'; a.mkdir(); b.mkdir()
            for directory, arm, error in ((a,'no-caption',.10), (b,'caption',.099)):
                (directory/'contract.json').write_text(json.dumps(dict(arm=arm, seed=43)))
                (directory/'completed.json').write_text(json.dumps(dict(mode='train',step=8000)))
                (directory/'train.jsonl').write_text('\n'.join(json.dumps(dict(step=s,data_ids_sha256=str(s))) for s in range(1,8001)))
                for step in (6000,8000):
                    (directory/f'val_step{step:06d}.json').write_text(json.dumps(dict(samples=14249,
                        metrics={'ssi':dict(abs_rel=error,rmse=3.,a1=.94)},
                        sequences={'ssi':{s:dict(abs_rel=error) for s in ('s1','s2','s3')}})))
            self.assertTrue(check(a,b))
            path = b/'val_step008000.json'
            report = json.loads(path.read_text()); report['metrics']['ssi']['rmse'] = 3.1
            path.write_text(json.dumps(report))
            self.assertFalse(check(a,b))


if __name__ == '__main__':
    unittest.main()
