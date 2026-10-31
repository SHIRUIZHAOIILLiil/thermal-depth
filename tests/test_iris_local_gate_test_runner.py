import csv
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
from tools import test_iris_local_gate as runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.args = SimpleNamespace(base=root/"fusion", run=root/"gate", arm="frame")
        self.previous = self.args.base/"empty_s43"
        self.previous.mkdir(parents=True)
        (self.previous/"step006000.pt").write_bytes(b"base")
        self.sources = {a:dict(path=a) for a in ("frame","empty")}
        self.common = dict(experiment=runner.train.NAME, base_sha=runner.sha256_file(self.previous/"step006000.pt"),
                           code={}, sources=self.sources, snapshot="cached")
        runner.write(self.previous/"contract.json", dict(arm="empty", experiment=runner.train.old.EXPERIMENT,
                                                         iris_sources=self.sources, code={}))
        runner.write(self.previous/"summary.json", dict(selected_step=6000))
        self.arm = self.args.run/"frame"
        self.arm.mkdir(parents=True)
        runner.write(self.arm/"contract.json", dict(**self.common, arm="frame"))
        self.set_selection(0)
        self.bundles = {c:([SimpleNamespace(sample_id=c)], {c}) for c in runner.TESTS}
        self.manifests = {c:dict(sha256=c) for c in runner.TESTS}
        data = patch.object(runner, "load_test_sets", return_value=(self.bundles, self.manifests))
        data.start()
        self.addCleanup(data.stop)

    def set_selection(self, step):
        curve = {s:dict(ssi=dict(abs_rel=.1 if s==step else .2)) for s in runner.train.STEPS}
        runner.write(self.arm/"summary.json", dict(common=self.common, arm="frame", samples=14249, selected_step=step,
                     curve=curve, baseline=curve[0], selected_full=curve[step]))

    def csv(self, path, condition, step=None):
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", *(["step"] if step is not None else []), "alignment", *runner.train.old.OFFICIAL_METRICS])
            for mode in ("ssi","none"):
                writer.writerow([condition, *([step] if step is not None else []), mode, *range(len(runner.train.old.OFFICIAL_METRICS))])

    def test_zero_requires_matching_source_and_reuses_all_three_without_model(self):
        src = self.args.base/"test_selected_20261030/empty"
        results = {}
        for c, (samples, wanted) in self.bundles.items():
            (src/c).mkdir(parents=True)
            csv_path = src/c/"val_step006000_frames.csv"
            self.csv(csv_path, c)
            results[c] = dict(**runner.fold(csv_path, samples, wanted, None), step=6000)
        runner.write(src/"summary.json", dict(arm="empty", step=6000, manifests=self.manifests, results=results))
        prov = dict(arm="empty", step=6000, checkpoint_sha256=self.common["base_sha"], iris_source=self.sources["empty"],
                    protocol=runner.PROTOCOL, manifests=self.manifests, selection_sha256=runner.sha256_file(self.previous/"summary.json"),
                    contract_sha256=runner.sha256_file(self.previous/"contract.json"))
        runner.write(src/"provenance.json", prov)
        with patch.object(runner.train, "load_inputs", side_effect=AssertionError), patch.object(runner,"restore_gate", side_effect=AssertionError):
            runner.run(self.args)
            report = runner.read(self.args.run/"test_selected_v1/frame/summary.json")
            self.assertTrue(report["reused_empty6k"])
            self.assertEqual(report["step"], 0)
            self.assertEqual(set(report["results"]), set(runner.TESTS))
            runner.write(src/"provenance.json", dict(prov, arm="frame"))
            with self.assertRaisesRegex(ValueError, "provenance"):
                runner.run(self.args)

    def test_nonzero_uses_own_selected_checkpoint_and_evaluates_each_split_once(self):
        step = 2000
        self.set_selection(step)
        path = self.arm/"step002000.pt"
        state = dict(common=self.common, gate=runner.train.LocalGate().state_dict(), step=step, arm="empty")
        torch.save(state, path)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            runner.restore_gate(path, "frame", self.common, step)
        torch.save(dict(state, arm="frame"), path)
        calls = []
        def evaluate(inputs, data, indices, gates, output):
            self.assertEqual(list(gates), [step])
            self.assertIsInstance(gates[step], runner.train.LocalGate)
            cond = data[0].sample_id
            calls.append(cond)
            self.csv(output, cond, step)
        with (patch.object(torch.cuda, "is_available", return_value=True),
              patch.object(torch.cuda, "is_bf16_supported", return_value=True),
              patch.object(runner.train.LocalGate,"cuda",lambda self:self),
              patch.object(runner.train.old,"matched_sources",return_value=self.sources),
              patch.object(runner.train,"load_inputs",return_value=(object(),object())) as loader,
              patch.object(runner.train.old,"make_tokens",return_value=None),
              patch.object(runner.train.old,"FusionData",side_effect=lambda samples, tokens: samples),
              patch.object(runner.train,"evaluate",side_effect=evaluate)):
            runner.run(self.args)
            runner.run(self.args)
            self.assertEqual(loader.call_args.args[0].arm, "frame")
        self.assertEqual(calls, list(runner.TESTS))
        self.assertFalse(runner.read(self.args.run/"test_selected_v1/frame/summary.json")["reused_empty6k"])

    def test_attach_only_tests_and_refuse_ambiguous_parent(self):
        tmp = self.args.run.parent
        for name, body in (("squeue", 'cat "$QUEUE"'), ("sacct", "exit 0"),
                           ("sbatch", 'echo "$*" >> "$CALLS"; echo 111')):
            path = tmp/name
            path.write_text("#!/bin/bash\n"+body+"\n")
            path.chmod(0o755)
        queue, calls = tmp/"queue", tmp/"calls"
        env = dict(os.environ, PATH=str(tmp)+":"+os.environ["PATH"], QUEUE=str(queue), CALLS=str(calls),
                   IRIS_REPO=str(ROOT), IRIS_RUNS=str(tmp/"runs"))
        cmd = ["bash", str(ROOT/"slurm/attach_iris_local_gate_tests.sh")]
        queue.write_text("81|iris_lg_frame\n82|iris_lg_empty\n83|iris_lg_empty\n")
        self.assertNotEqual(subprocess.run(cmd, env=env, capture_output=True).returncode, 0)
        self.assertFalse(calls.exists())
        queue.write_text("81|iris_lg_frame\n82|iris_lg_empty\n")
        for argv in (cmd, cmd+["81","82"]):
            subprocess.run(argv, env=env, check=True, capture_output=True)
        frame, empty = calls.read_text().splitlines()
        for line, arm, parent in ((frame, "frame",81), (empty,"empty",82)):
            self.assertIn(f"--dependency=afterok:{parent}", line)
            self.assertIn(f"ARM={arm}", line)
            self.assertTrue(line.endswith("slurm/iris_local_gate_test.sbatch"))


if __name__ == "__main__":
    unittest.main()
