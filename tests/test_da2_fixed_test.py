"""CPU guards for frozen DA2 test runs. No pretrained assets or GPU required."""
from pathlib import Path
import sys
from types import SimpleNamespace
from contextlib import ExitStack
import tempfile
import unittest
from unittest.mock import patch, MagicMock
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.test_da2_fixed import FIXED, TESTS, validate_samples, check_selection
from tools.da2_caption_data import TRAIN, VAL
from tools import test_da2_fixed as runner
from tools.train_da2_caption import json_write
from ms2_eval.io import sha256_file
import torch

class FixedTests(unittest.TestCase):
    def sample(self, **changes):
        seq = '2021-08-06-11-23-45'
        fields = dict(sample_id=seq+'_000001', sequence=seq, split='test', caption='road',
                      thermal_path=f'/data/{seq}/thermal/000001.png',
                      thermal_gt_path=f'/data/{seq}/thr/000001.png')
        return SimpleNamespace(**{**fields, **changes})

    def test_exact_steps_and_selection(self):
        self.assertEqual(FIXED, {'no-caption': 18000, 'caption': 20000})
        for arm, step in FIXED.items():
            report = dict(arm=arm, selected_step=step, selected_full=dict(step=step, samples=14249))
            check_selection(arm, report)
            with self.assertRaises(ValueError): check_selection(arm, {**report, 'selected_step': 8000})
            with self.assertRaises(ValueError): check_selection(arm, {**report, 'selected_full': dict(step=step, samples=100)})

    def test_frozen_conditions_disjoint(self):
        seen = set()
        for _, _, _, seqs in TESTS.values():
            self.assertFalse(seqs & (TRAIN | VAL | seen)); seen |= seqs
        self.assertEqual([x[2] for x in TESTS.values()], [23311, 22916, 25023])

    @patch('pathlib.Path.is_file', return_value=True)
    def test_valid_paths_and_duplicate_guard(self, mock):
        sample = self.sample(); allowed = {sample.sequence}
        self.assertEqual(validate_samples([sample], allowed), {sample.sample_id})
        with self.assertRaises(ValueError): validate_samples([sample, sample], allowed)

    @patch('pathlib.Path.is_file', return_value=True)
    def test_split_field_is_metadata_not_membership(self, mock):
        # The frozen test manifests carry split='train' on every row.
        allowed = {self.sample().sequence}
        for split in ('train', 'val', 'test', 'unknown'):
            with self.subTest(split=split):
                self.assertEqual(validate_samples([self.sample(split=split)], allowed),
                                 {self.sample().sample_id})

    @patch('pathlib.Path.is_file', return_value=True)
    def test_reject_nonthermal_wrong_caption_sequence_and_path(self, mock):
        allowed = {self.sample().sequence}
        for changes in [dict(thermal_gt_path='/data/2021-08-06-11-23-45/rgb/000001.png'),
                        dict(caption=''), dict(sequence='other'),
                        dict(thermal_path='/data/2021-08-06-11-23-45/thermal/000002.png')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_samples([self.sample(**changes)], allowed)

    def test_train_val_and_missing_file_rejected(self):
        with self.assertRaises(ValueError): validate_samples([], TRAIN)
        with self.assertRaises(ValueError): validate_samples([], VAL)
        with patch('pathlib.Path.is_file', return_value=False), self.assertRaises(FileNotFoundError):
            validate_samples([self.sample()], {self.sample().sequence})

    def test_submission_is_test_only(self):
        script = (ROOT/'slurm/da2_fixed_test.sbatch').read_text()
        self.assertIn('tools/test_da2_fixed.py', script)
        self.assertNotIn('tools/train_da2_caption.py', script)
        self.assertIn('flock -n', script)

    def test_three_condition_integration_and_cached_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); directory = base/'no-caption_s43'; directory.mkdir()
            (directory/'validation_20000').mkdir()
            assets = base/'assets.json'; assets.write_text('{}')
            contract = dict(arm='no-caption', seed=43, assets=sha256_file(assets))
            json_write(directory/'contract.json', contract)
            json_write(directory/'validation_20000/summary.json', dict(arm='no-caption', selected_step=18000,
                       selected_full=dict(step=18000, samples=14249)))
            torch.save(dict(contract=contract, step=18000, image_cursor=216000, model={}, optimizer={},
                            scheduler={'last_epoch': 18000}, rng={}, amp={}), directory/'step018000.pt')
            calls = []
            def fake_load(path, *args, **kwargs):
                sample = self.sample(sample_id=path.name)
                return [sample], dict(sha256='abc123', path=str(path), sample_count=1)
            def fake_eval(model, data, device, amp, folder, step):
                calls.append((folder.name, step))
                json_write(folder/f'val_step{step:06d}.json', dict(step=step, samples=len(data),
                    metrics={m: dict(abs_rel=.1, rmse=2., a1=.9) for m in ('ssi','none')}))
            args = SimpleNamespace(arm='no-caption', base=base, assets=assets, manifest_dir=base, data_root=base)
            with ExitStack() as stack:
                for target, value in [('torch.cuda.is_available', True), ('torch.cuda.is_bf16_supported', True),
                    ('tools.test_da2_fixed.check_source', None), ('tools.test_da2_fixed.verify_assets', {}),
                    ('tools.test_da2_fixed.build_model', MagicMock())]:
                    stack.enter_context(patch(target, return_value=value))
                stack.enter_context(patch.object(runner, 'TESTS', {c:(c,'abc',1,set()) for c in TESTS}))
                stack.enter_context(patch.object(runner, 'load_manifest', side_effect=fake_load))
                stack.enter_context(patch.object(runner, 'validate_samples', side_effect=lambda s,a:{s[0].sample_id}))
                stack.enter_context(patch.object(runner, 'ThermalLidar', side_effect=lambda s,t:s))
                stack.enter_context(patch.object(runner, 'evaluate', side_effect=fake_eval))
                runner.run(args)
                self.assertEqual(calls, [(c,18000) for c in TESTS])
                runner.run(args)
                self.assertEqual(len(calls), 3)
                with patch.object(runner, 'caption_hash', return_value='changed'), self.assertRaises(ValueError):
                    runner.run(args)

if __name__ == '__main__': unittest.main()
