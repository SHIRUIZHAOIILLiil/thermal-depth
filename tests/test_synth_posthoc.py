"""CPU checks for the synth post-hoc chain (docs/SYNTH_POSTHOC_V1.md); no model downloads."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tools import diagnose_checkpoint_average as avg
from tools import train_iris_synth_decoder as synth
from tools import train_iris_tmax1_decoder as stage


class FakeModel:
    def __init__(self):
        self.loaded = None
        self.unet = torch.nn.Linear(1, 1)

    def load_state_dicts(self, states, label):
        self.loaded = (states, label)

    def set_train(self, flag):
        self.train_flag = flag


class Averaging(unittest.TestCase):
    def test_sources_and_controls(self):
        self.assertEqual(avg.SOURCES[avg.TMAX1_RUN], dict(control_val_rmse=avg.BASELINE_VAL_RMSE, save='gate'))
        self.assertEqual(avg.SOURCES[avg.SYNTH_RUN], dict(control_val_rmse=2.67386, save='not_worse'))

    def test_not_worse_needs_all_three(self):
        base = dict(abs_rel=.0768, rmse=2.674, a1=.9492)
        self.assertTrue(avg.not_worse(dict(base), base))
        self.assertTrue(avg.not_worse(dict(abs_rel=.0760, rmse=2.65, a1=.95), base))
        for worse in (dict(base, abs_rel=.07681), dict(base, rmse=2.6741), dict(base, a1=.94919)):
            self.assertFalse(avg.not_worse(worse, base))

    def test_missing_checkpoints_listed_before_gpu_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp)
            self.assertEqual(len(avg.missing_checkpoints(src)), 9)
            (src/'converted').mkdir()
            (src/'converted/step16000_weights.pt').touch()
            for s in avg.WINDOW:
                if s != 16000:
                    (src/f'checkpoint-{s}').mkdir()
            self.assertEqual(avg.missing_checkpoints(src), [])
            (src/'checkpoint-20000').rmdir()
            self.assertEqual(avg.missing_checkpoints(src), [str(src/'checkpoint-20000')])


class Decoder(unittest.TestCase):
    def summary(self, root, use_average, passed=True, source_run=avg.SYNTH_RUN):
        weights = root/'average_12000_20000.pt'
        torch.save(dict(route='b_thermal_unet', caption_mode='correct', global_step=20000,
                        averaged_steps=list(avg.WINDOW), source_run=source_run,
                        state_dicts=dict(unet=dict(w=torch.ones(1)))), weights)
        path = root/'summary.json'
        path.write_text(json.dumps(dict(source_run=source_run, control=dict(passed=passed), use_average=use_average,
            weights=str(weights) if use_average else None,
            weights_sha256=synth.pair.sha256_file(weights) if use_average else None)))
        return path, weights

    def test_resolve_follows_the_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, weights = self.summary(root, True)
            self.assertEqual(synth.resolve(path, root/'runs'), (weights, 'average'))
            path, _ = self.summary(root, False)
            self.assertEqual(synth.resolve(path, root/'runs'),
                             (root/'runs/iris_ms2'/avg.SYNTH_RUN/'converted/step16000_weights.pt', 'step16000'))

    def test_resolve_refuses_bad_control_source_or_changed_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for kw in (dict(passed=False), dict(source_run=avg.TMAX1_RUN)):
                path, _ = self.summary(root, True, **kw)
                with self.assertRaises(ValueError):
                    synth.resolve(path, root)
            path, weights = self.summary(root, True)
            weights.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                synth.resolve(path, root)

    def test_load_average_checks_the_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, weights = self.summary(root, True)
            model = FakeModel()
            synth.load_average(model, weights)
            self.assertEqual(model.loaded[1], str(weights))
            self.assertFalse(any(p.requires_grad for p in model.unet.parameters()))
            _, wrong = self.summary(root, True, source_run=avg.TMAX1_RUN)
            with self.assertRaises(ValueError):
                synth.load_average(FakeModel(), wrong)

    def test_start_reproduction(self):
        synth.require_16k(synth.EXPECTED_16K)
        synth.require_16k(dict(abs_rel=.07635, rmse=2.6525, a1=.94905))
        with self.assertRaises(ValueError):
            synth.require_16k(dict(abs_rel=.0770, rmse=2.686, a1=.9454))  # synth20b36, not this arm

    def test_verdict_needs_all_three_strictly(self):
        ref = dict(abs_rel=.07286, rmse=2.5625, a1=.9504)
        own = dict(abs_rel=.08455, rmse=2.4710, a1=.9422)
        start = dict(abs_rel=.0763, rmse=2.651, a1=.9491)
        self.assertTrue(synth.verdict(dict(abs_rel=.0728, rmse=2.56, a1=.951), start, ref, own)['passed'])
        for m in (dict(abs_rel=.07286, rmse=2.56, a1=.951), dict(abs_rel=.0728, rmse=2.57, a1=.951),
                  dict(abs_rel=.0728, rmse=2.56, a1=.9504)):
            self.assertFalse(synth.verdict(m, start, ref, own)['passed'])
        r = synth.verdict(dict(abs_rel=.0728, rmse=2.56, a1=.951), start, ref, own)
        self.assertFalse(r['vs_newcrf_ssi']['beats_all_three'])  # RMSE still behind under NeWCRF's own alignment
        self.assertAlmostEqual(r['vs_start']['rmse_m'], 2.56-2.651)

    def test_tmax1_stage_defaults_unchanged(self):
        t = stage.TMAX1
        self.assertEqual((t.experiment, t.source_step, t.code), ('iris_tmax1_decoder_v1', 16000, ()))
        self.assertIs(t.require_start, stage.tmax.require_start)
        source = Path('runs')/stage.tmax.SOURCE_RUN/'converted/step16000_weights.pt'
        self.assertEqual(t.resolve(type('A', (), dict(source=source))()), source)


if __name__ == '__main__':
    unittest.main()
