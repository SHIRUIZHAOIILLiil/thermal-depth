import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import far_target_pipeline as tool

BASH = """
set -euo pipefail
echo 0 > "$TMP/count"
sbatch() { n=$(( $(cat "$TMP/count") + 1 )); echo "$n" > "$TMP/count"; echo "$n sbatch $* NO_CAPTIONS=${NO_CAPTIONS:-} PSEUDO=${PSEUDO_DIR:-} TAG=${RUN_TAG:-} RUN=${RUN:-} ONLY=${VAL_ONLY:-} PROMPT=${SEL_PROMPT:-} TEST=$(basename "${TEST_MANIFEST:-none}")" >> "$TMP/calls"; echo "$n"; }
squeue() { echo RUNNING; }
sha256sum() { echo "abcdef12  $1"; }
export -f sbatch squeue sha256sum
bash slurm/submit_far_target_pipeline.sh 777
"""


def report(status='clean_source_passed', chosen='bts', control=True, passed=True, protocol=tool.PROTOCOL):
    return dict(protocol=protocol, control=dict(passed=control),
                decision=dict(status=status, chosen=chosen, sources=dict(bts=dict(passed=passed))))


class Choice(unittest.TestCase):
    def test_only_clean_source_starts_a_retrain(self):
        self.assertEqual(tool.chosen_source(report()), 'bts')
        for status in ('control_failed', 'only_newcrf_passed_needs_user_decision', 'no_source_passed'):
            self.assertIsNone(tool.chosen_source(report(status, None)))

    def test_contradictory_summary_is_refused(self):
        for bad in (report(control=False), report(passed=False), report(chosen='newcrf'), report(protocol='old')):
            with self.assertRaises((ValueError, KeyError)):
                tool.chosen_source(bad)

    def test_choose_writes_nothing_when_stopping(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, out = Path(tmp)/'s.json', Path(tmp)/'choice.json'
            summary.write_text(json.dumps(report('no_source_passed', None)))
            self.assertEqual(tool.choose(SimpleNamespace(summary=summary, out=out)), tool.STOP)
            self.assertFalse(out.exists())
            summary.write_text(json.dumps(report()))
            self.assertEqual(tool.choose(SimpleNamespace(summary=summary, out=out)), 0)
            self.assertEqual(json.loads(out.read_text())['source'], 'bts')


class Gate(unittest.TestCase):
    start = dict(abs_rel=.08137, rmse=2.7571, a1=.93937)

    def test_boundaries(self):
        good = dict(abs_rel=.08337, rmse=2.6571, a1=.93637)
        self.assertTrue(tool.retrain_gate(good, self.start)['passed'])
        for key, value in (('rmse', 2.6572), ('abs_rel', .08338), ('a1', .93636)):
            self.assertFalse(tool.retrain_gate(dict(good, **{key: value}), self.start)['passed'])


    def test_gate_exit_status_controls_the_test_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference, caption = Path(tmp)/'r.json', Path(tmp)/'c.json'
            reference.write_text(json.dumps(self.start))
            args = SimpleNamespace(reference=reference, caption=caption)
            caption.write_text(json.dumps(dict(abs_rel=.082, rmse=2.6, a1=.94)))
            self.assertEqual(tool.gate(args), 0)
            caption.write_text(json.dumps(dict(abs_rel=.082, rmse=2.7, a1=.94)))
            self.assertEqual(tool.gate(args), tool.STOP)


class Infer(unittest.TestCase):
    def test_resume_skips_only_finished_frames(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tool.save_atomic(out/'a.npy', np.zeros((2, 2), np.float32))
            (out/'.b.tmp.npy').write_bytes(b'partial')
            samples = [SimpleNamespace(sample_id=x) for x in 'ab']
            self.assertEqual([s.sample_id for s in tool.pending(samples, out)], ['b'])


@unittest.skipIf(sys.platform == 'win32', 'bash stubs run under WSL or on Aire')
class Submit(unittest.TestCase):
    def test_tests_wait_for_the_gate_and_the_report_survives_a_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scratch = root/'scratch'
            for path in ('baselines/supdepth4thermal/MS2_MD_BTS_THR_ckpt.ckpt',
                         'baselines/supdepth4thermal/MS2_MD_AdaBins_THR_ckpt.ckpt'):
                (scratch/path).parent.mkdir(parents=True, exist_ok=True)
                (scratch/path).write_text('x')
            source = (ROOT/'slurm/submit_far_target_pipeline.sh').read_text()
            repo = root/'repo'
            (repo/'slurm').mkdir(parents=True)
            source = source.replace('/mnt/scratch/dwdh0363/from_sc23sz', str(root/'from'))
            (repo/'slurm/submit_far_target_pipeline.sh').write_text(source)
            manifests = root/'from/manifests/sequence_level_internvl3_8b'
            manifests.mkdir(parents=True)
            for name in ('ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl',
                         'ms2_val_official3_thermalcap_20260821.jsonl',
                         'ms2_test_day3_common_thermalcap_20260821.jsonl',
                         'ms2_test_night3_thermalcap_20260821.jsonl',
                         'ms2_test_rainy3_thermalcap_20260821.jsonl'):
                (manifests/name).write_text('x\n')
            runs = root/'runs'
            reference = runs/'eval/sd2_cap_pct_logtn_img_l0100_tmax1_s43_val_abcdef12_stride1_correct_s16000'
            reference.mkdir(parents=True)
            (reference/'eval_eval_affine_invariant_log_space.json').write_text('{}')
            env = dict(os.environ, TMP=str(root), SCRATCH=str(scratch), IRIS_REPO=str(repo), IRIS_RUNS=str(runs))
            done = subprocess.run(['bash', '-c', BASH], cwd=repo, env=env, text=True, capture_output=True)
            self.assertEqual(done.returncode, 0, done.stderr)
            calls = (root/'calls').read_text().splitlines()
            self.assertEqual(len(calls), 16)
            self.assertIn('afterok:777', calls[0])
            self.assertTrue(all('afterok:' in c for c in calls[:15]))
            self.assertIn('afterany:', calls[15])
            self.assertIn('slurm/far_target_report.sbatch', calls[15])
            self.assertNotIn('kill-on-invalid-dep', calls[15])
            self.assertEqual(sum('slurm/iris_ms2.sbatch' in c for c in calls), 4)
            vals = [c for c in calls if 'ftr_val_' in c]
            self.assertTrue(all('ONLY=1' in c for c in vals))
            gate = next(c for c in calls if 'ftr_gate' in c)
            self.assertIn(f"afterok:{vals[0].split()[0]}:{vals[1].split()[0]}", gate)
            tests = [c for c in calls if 'ftr_test_' in c]
            self.assertEqual(len(tests), 6)
            self.assertTrue(all(f"afterok:{gate.split()[0]}" in c and 'ONLY=0' in c for c in tests))
            self.assertEqual(len({(c.split('RUN=')[1].split()[0], c.split('TEST=')[1]) for c in tests}), 6)
            self.assertTrue(all('_caption_' in c.split('RUN=')[1].split()[0] and 'PROMPT=correct' in c for c in tests[:3]))
            self.assertTrue(all('_empty_' in c.split('RUN=')[1].split()[0] and 'PROMPT=empty' in c for c in tests[3:]))
            trains = [c for c in calls if 'ftr_tr_' in c]
            self.assertIn('NO_CAPTIONS=0', trains[0])
            self.assertIn('NO_CAPTIONS=1', trains[1])
            self.assertEqual(len({c.split('PSEUDO=')[1].split()[0] for c in trains}), 1)


if __name__ == '__main__':
    unittest.main()
