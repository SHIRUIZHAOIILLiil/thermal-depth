import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import test_iris_far_stretch as runner
from tools import iris_far_stretch as s


class FrozenTest(unittest.TestCase):
    def parent(self, path, passes):
        zero = {'rmse':3., 'abs_rel':.08, 'a1':.94}
        corrected = dict(zero, rmse=2.94 if passes else 2.96)
        results = {}
        betas = {'frame':.25, 'empty':0.}
        for arm,b in betas.items():
            row = corrected if b else zero
            gate = s.gates(zero, row)
            results[arm] = dict(beta=b, gates=gate, pass_self=all(gate.values()), zero=zero, corrected=row)
            (path/f'{arm}_fit.json').write_text(json.dumps(dict(beta=b, train_scores={0.:zero, b:row})))
        (path/'contract.json').write_text(json.dumps(dict(runner_sha256=runner.sha256_file(s.__file__))))
        (path/'frozen_betas.json').write_text(json.dumps(betas))
        (path/'summary.json').write_text(json.dumps(dict(validation_frames=14249, results=results,
                                                        adopt_frame_stretch=passes)))

    def test_skip_unsuccessful_validation_without_touching_test_data(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            self.parent(path, False)
            runner.run(SimpleNamespace(parent=path, condition='day'))
            self.assertTrue((path/'test_selected_v1/day/skipped.json').is_file())

    def test_reject_mismatched_frozen_beta(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            self.parent(path, True)
            self.assertTrue(runner.selection(path)[0])
            (path/'frozen_betas.json').write_text('{"frame":0,"empty":0}')
            with self.assertRaisesRegex(ValueError, 'Inconsistent frozen'):
                runner.selection(path)

    def test_official_subset_is_paired_for_both_settings(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'rows.csv'
            with path.open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['id','beta',*s.OFFICIAL_METRICS])
                for b in (0.,.25):
                    for i in range(3):
                        w.writerow([str(i),b,*([i+b]*len(s.OFFICIAL_METRICS))])
            sums = runner.official_fold(path, {'1','2'}, [0.,.25])
            self.assertEqual(sums[.25]['rmse']-sums[0.]['rmse'], .25)
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                runner.official_fold(path, {'3'}, [0., .25])


if __name__ == '__main__':
    unittest.main()
