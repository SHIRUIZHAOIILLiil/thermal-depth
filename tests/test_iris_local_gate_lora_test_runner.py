import csv
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import test_iris_local_gate_lora as runner


class TestRunner(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.args = SimpleNamespace(base=self.root/'base', parents=self.root/'parents',
                                    run=self.root/'run', arm='frame')
        self.common = dict(experiment=runner.train.NAME, protocol=runner.PROTOCOL, code={},
                           parent_common=dict(base_sha='base', sources={a:dict(path=a) for a in runner.train.PARENTS}),
                           parents={a:dict(sha256=a, summary_sha256=a+'summary') for a in runner.train.PARENTS})
        self.bundles = {c:([SimpleNamespace(sample_id=c)], {c}) for c in runner.TESTS}
        self.manifests = {c:dict(sha256=c) for c in runner.TESTS}
        for obj,name,value in ((runner, 'verify_sources', ({}, self.common['parent_common']['sources'])),
                               (runner.shared, 'load_test_sets', (self.bundles, self.manifests))):
            p = patch.object(obj, name, return_value=value)
            p.start()
            self.addCleanup(p.stop)

    def selection(self, step):
        folder = self.args.run/self.args.arm
        folder.mkdir(parents=True, exist_ok=True)
        curve = {s:dict(ssi=dict(abs_rel=.1 if s==step else .2, rmse=2., a1=.96)) for s in runner.train.STEPS}
        runner.write(folder/'contract.json', dict(arm=self.args.arm, **self.common))
        summary = dict(arm=self.args.arm, common=self.common, samples=14249, curve=curve,
                       selected_step=step, selected_full=curve[step], baseline=curve[0])
        runner.write(folder/'summary.json', summary)
        return folder, summary

    def csv(self, path, condition, step):
        with path.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'step', 'alignment', *runner.train.g.old.OFFICIAL_METRICS])
            for mode in ('ssi', 'none'):
                writer.writerow([condition, step, mode, *range(len(runner.train.g.old.OFFICIAL_METRICS))])

    def test_zero_reuses_own_parent_for_both_arms_and_rejects_wrong_source(self):
        for arm, step in runner.train.PARENTS.items():
            with self.subTest(arm=arm):
                self.args.arm = arm
                self.selection(0)
                source = self.args.parents/'test_selected_v1'/arm
                source.mkdir(parents=True)
                results = {}
                for condition,(samples,wanted) in self.bundles.items():
                    path = source/f'{condition}_frames.csv'
                    self.csv(path, condition, step)
                    results[condition] = dict(**runner.shared.fold(path, samples, wanted, step), step=step,
                                              csv=str(path), csv_sha256=runner.sha256_file(path))
                provenance = dict(arm=arm, step=step, protocol=runner.PROTOCOL, manifests=self.manifests,
                     base_sha='base', checkpoint_sha256=arm, selection_sha256=arm+'summary',
                     iris_source=dict(path=arm), official_ids={c:hashlib.sha256(c.encode()).hexdigest() for c in runner.TESTS})
                runner.write(source/'provenance.json', provenance)
                runner.write(source/'summary.json', dict(arm=arm, step=step, manifests=self.manifests, results=results,
                    reused_empty6k=False, provenance_sha256=runner.sha256_file(source/'provenance.json')))
                with patch.object(runner, 'load_model', side_effect=AssertionError('Must reuse, not infer')):
                    runner.run(self.args)
                result = runner.read(self.args.run/'test_selected_v1'/arm/'summary.json')
                self.assertTrue(result['reused_parent'])
                self.assertEqual(result['step'], 0)
                self.assertEqual(result['parent_step'], step)
                self.assertEqual(set(result['results']), set(runner.TESTS))
                runner.write(source/'provenance.json', dict(provenance, arm='other'))
                with self.assertRaisesRegex(ValueError, 'provenance'):
                    runner.run(self.args)

    def test_nonzero_checkpoint_and_resume_do_not_repeat_inference(self):
        step = 1000
        folder, _ = self.selection(step)
        ck = dict(arm='empty', step=step, common=self.common, lora={'marker':torch.ones(1)}, gate={})
        path = folder/f'step{step:06d}.pt'
        torch.save(ck, path)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            runner.run(self.args)
        ck['arm'] = 'frame'
        torch.save(ck, path)
        calls, restored = [], []
        model = SimpleNamespace(restore=lambda state: restored.append(state['step']))
        def evaluate(model_arg, data, indices, states, output):
            self.assertIs(model_arg, model)
            self.assertEqual(list(states), [step])
            self.assertTrue(torch.equal(states[step]['lora']['marker'], torch.ones(1)))
            condition = data[0].sample_id
            calls.append(condition)
            self.csv(output, condition, step)
        with (patch.object(runner, 'load_model', return_value=(model, object())) as loader,
              patch.object(runner.train.g.old, 'make_tokens', return_value=None),
              patch.object(runner.train.g.old, 'FusionData', side_effect=lambda s,t:s),
              patch.object(runner.train, 'evaluate', side_effect=evaluate)):
            runner.run(self.args)
            runner.run(self.args)
            self.assertEqual(loader.call_count, 1)
        self.assertEqual(calls, list(runner.TESTS))
        self.assertEqual(restored, [step])
        report = runner.read(self.args.run/'test_selected_v1/frame/summary.json')
        self.assertEqual(report['step'], step)
        self.assertFalse(report['reused_parent'])

    def test_selection_honors_rmse_guard(self):
        folder, summary = self.selection(500)
        summary['curve'][500]['ssi']['rmse'] = 3.
        with self.assertRaisesRegex(ValueError, 'Selection'):
            runner.selected_step('frame', runner.read(folder/'contract.json'), summary)

    def test_attach_receipt_afterok_and_no_training_resubmission(self):
        for name,body in (('squeue', 'cat "$QUEUE"'), ('sacct', 'exit 0'),
                           ('sbatch', 'echo "$*" >> "$CALLS"; echo 123')):
            path = self.root/name
            path.write_text('#!/bin/bash\n'+body+'\n')
            path.chmod(0o755)
        queue, calls = self.root/'queue', self.root/'calls'
        queue.write_text('81|iris_ll_frame\n82|iris_ll_empty\n83|iris_ll_empty\n')
        env = dict(os.environ, PATH=str(self.root)+':'+os.environ['PATH'], QUEUE=str(queue), CALLS=str(calls),
                   IRIS_REPO=str(ROOT), IRIS_RUNS=str(self.root/'runs'))
        command = ['bash', str(ROOT/'slurm/attach_iris_local_gate_lora_tests.sh')]
        self.assertNotEqual(subprocess.run(command, env=env, capture_output=True).returncode, 0)
        self.assertFalse(calls.exists())
        receipt = self.root/'runs/outputs/lotus_line_v2/iris_local_gate_lora_kv_r4_v1/submission.txt'
        receipt.parent.mkdir(parents=True)
        receipt.write_text('smoke 80\nframe 81 afterok:80\nempty 82 afterok:80\n')
        for cmd in (command, command+['81','82']):
            subprocess.run(cmd, env=env, check=True, capture_output=True)
        frame, empty = calls.read_text().splitlines()
        for line,arm,parent in ((frame,'frame',81), (empty,'empty',82)):
            self.assertIn(f'--dependency=afterok:{parent}', line)
            self.assertIn('--kill-on-invalid-dep=yes', line)
            self.assertIn(f'ARM={arm}', line)
            self.assertTrue(line.endswith('slurm/iris_local_gate_lora_test.sbatch'))


if __name__ == '__main__':
    unittest.main()
