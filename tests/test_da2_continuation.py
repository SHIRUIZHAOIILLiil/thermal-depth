"""CPU gates for continuation; direct sibling imports avoid third-party tests package."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from test_da2_caption import TinyCore
from models.da2_caption import CaptionDepth
from tools.train_da2_caption import make_optimizer, save_checkpoint, load_checkpoint
from tools.da2_continuation import STEPS, select_step, validate_state, audit_logs
from tools import eval_da2_continuation as evaluator
from ms2_eval.io import sha256_file
from tools.train_da2_caption import json_write


class ContinuationTests(unittest.TestCase):
    def test_select_full_curve_absrel_tie_earlier_not_rmse(self):
        curve = {s: {'metrics': {'ssi': {'abs_rel': .08, 'rmse': 2.}}} for s in STEPS}
        curve[12000]['metrics']['ssi']['abs_rel'] = .07
        curve[18000]['metrics']['ssi']['abs_rel'] = .07
        curve[18000]['metrics']['ssi']['rmse'] = 1.
        self.assertEqual(select_step(curve), 12000)
        bad = copy.deepcopy(curve); del bad[2000]
        with self.assertRaises(ValueError): select_step(bad)
        curve[20000]['metrics']['ssi']['abs_rel'] = float('nan')
        with self.assertRaises(ValueError): select_step(curve)

    def test_checkpoint_guards_full_resume_state(self):
        contract = {'arm': 'caption'}
        state = dict(contract=contract, step=8000, image_cursor=96000, model={}, optimizer={},
                     scheduler={'last_epoch': 8000}, rng={}, amp={})
        validate_state(state, contract, 8000)
        for key in ('model', 'optimizer', 'scheduler', 'rng', 'amp'):
            bad = copy.deepcopy(state); del bad[key]
            with self.assertRaises(ValueError): validate_state(bad, contract)
        for key, value in [('step', 20001), ('image_cursor', 8000), ('scheduler', {'last_epoch': 0})]:
            with self.assertRaises(ValueError): validate_state({**state, key: value}, contract)
        with self.assertRaises(ValueError): validate_state(state, contract, 20000)

    def test_original_scheduler_resume_does_not_restart(self):
        torch.set_num_threads(1)
        model = CaptionDepth(TinyCore())
        opt, scheduler = make_optimizer(model)
        # Advance the original cheap scheduler without any real training workload.
        for _ in range(8000): opt.step(); scheduler.step()
        expected = [5e-6 * .6**.9, 5e-5 * .6**.9]
        for group, lr in zip(opt.param_groups, expected): self.assertAlmostEqual(group['lr'], lr)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'step008000.pt'
            contract = {'arm': 'no-caption'}
            save_checkpoint(path, model, opt, scheduler, 8000, contract)
            state = torch.load(path, weights_only=False)
            validate_state(state, contract, 8000)
            restored = CaptionDepth(TinyCore()); ropt, rsched = make_optimizer(restored)
            self.assertEqual(load_checkpoint(path, restored, ropt, rsched, contract), 8000)
            ropt.step(); rsched.step()
            self.assertEqual(rsched.last_epoch, 8001)
            for group, initial in zip(ropt.param_groups, (5e-6, 5e-5)):
                self.assertAlmostEqual(group['lr'], initial*(1-8001/20000)**.9)

    def test_pair_checks_all_steps_and_image_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            dirs = [Path(tmp)/arm for arm in ('no-caption', 'caption')]
            for directory in dirs:
                directory.mkdir()
                (directory/'contract.json').write_text(json.dumps({'arm': directory.name, 'seed': 43}))
                (directory/'completed.json').write_text(json.dumps({'step': 20000, 'mode': 'train'}))
                rows = [dict(step=s, image_cursor=s*12, data_ids_sha256=str(s)) for s in range(1, 20001)]
                (directory/'train.jsonl').write_text('\n'.join(map(json.dumps, rows)))
            audit_logs(*dirs)
            with (dirs[1]/'train.jsonl').open('a') as handle:
                handle.write('\n'+json.dumps(dict(step=19999, image_cursor=19999*12, data_ids_sha256='wrong')))
            with self.assertRaises(ValueError): audit_logs(*dirs)

    def test_submission_is_resume_only_and_validation_is_afterok(self):
        train = (ROOT/'slurm/da2_continue.sbatch').read_text()
        submit = (ROOT/'slurm/submit_da2_continue.sh').read_text()
        self.assertIn('--resume --stop-step 20000', train)
        self.assertIn('flock -n', train)
        self.assertIn('--dependency="afterok:$job"', submit)
        self.assertNotIn('sbatch slurm/da2_caption.sbatch', submit)

    def test_independent_validation_curve_full_and_safe_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            assets = directory/'assets.json'; assets.write_text('{}')
            info = {'sha256': 'validation-hash'}
            contract = dict(arm='no-caption', seed=43, assets=sha256_file(assets),
                            val=info, caption_val_hash='captions')
            json_write(directory/'contract.json', contract)
            json_write(directory/'completed.json', dict(step=20000, mode='train'))
            json_write(directory/'val_step008000.json', dict(step=8000, samples=14249))
            for step in STEPS:
                torch.save(dict(contract=contract, step=step, image_cursor=step*12, model={},
                                optimizer={}, scheduler={'last_epoch': step}, rng={}, amp={}),
                           directory/f'step{step:06d}.pt')
            json_write(directory/'continuation_anchor.json', dict(
                contract_sha256=sha256_file(directory/'contract.json'),
                checkpoint8000_sha256=sha256_file(directory/'step008000.pt'),
                validation8000_sha256=sha256_file(directory/'val_step008000.json')))
            calls = []
            def fake_evaluate(model, data, device, amp, out, step):
                calls.append((step, len(data)))
                json_write(out/f'val_step{step:06d}.json', dict(step=step, samples=len(data),
                    metrics={'ssi': {'abs_rel': .05 if step == 12000 else .1}}))
            args = SimpleNamespace(run_dir=directory, assets=assets, val_manifest=directory/'val', data_root=directory)
            with ExitStack() as stack:
                for target, value in [
                    ('torch.cuda.is_available', True), ('torch.cuda.is_bf16_supported', True),
                    ('tools.eval_da2_continuation.check_source', None),
                    ('tools.eval_da2_continuation.verify_assets', {}),
                    ('tools.eval_da2_continuation.checked_manifest', ([None]*14249, info)),
                    ('tools.eval_da2_continuation.caption_hash', 'captions'),
                    ('tools.eval_da2_continuation.ThermalLidar', [None]*14249),
                    ('tools.eval_da2_continuation.build_model', MagicMock())]:
                    stack.enter_context(patch(target, return_value=value))
                stack.enter_context(patch('tools.eval_da2_continuation.evaluate', side_effect=fake_evaluate))
                evaluator.run(args)
                self.assertEqual(calls[:10], [(s, 3563) for s in STEPS])
                self.assertEqual(calls[10:], [(20000, 14249), (12000, 14249)])
                report = json.loads((directory/'validation_20000/summary.json').read_text())
                self.assertEqual(report['selected_step'], 12000)
                evaluator.run(args)
                self.assertEqual(len(calls), 12)  # exact-provenance cache, no repeat inference
                (directory/'step010000.pt').write_bytes(b'changed')
                with self.assertRaises(ValueError): evaluator.run(args)


if __name__ == '__main__':
    unittest.main()
