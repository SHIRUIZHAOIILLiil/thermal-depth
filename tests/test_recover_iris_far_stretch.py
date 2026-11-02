import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools import recover_iris_far_stretch as repair


class RecoveryTest(unittest.TestCase):
    def test_actual_numpy_gate_serializes_to_boolean(self):
        zero = dict(rmse=np.float64(2.945714),abs_rel=np.float64(.079359),a1=np.float64(.942265))
        corrected = dict(rmse=np.float64(2.894992),abs_rel=np.float64(.079126),a1=np.float64(.943257))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'summary.json'
            repair.json_write(p,{'gates':repair.stretch.gates(zero,corrected)})
            self.assertIs(json.loads(p.read_text())['gates']['rmse'],True)

    def test_fold_rejects_interrupted_or_unpaired_csv(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'arm.csv'
            with p.open('w',newline='') as f:
                w=csv.writer(f)
                w.writerow(('id','beta',*repair.stretch.OFFICIAL_METRICS))
                for b in (0.,.05):
                    w.writerow(('a',b,*([2+b]*len(repair.stretch.OFFICIAL_METRICS))))
            self.assertEqual(repair.fold(p,['a'],.05)[.05]['rmse'],2.05)
            with self.assertRaises(ValueError):
                repair.fold(p,['a','b'],.05)


if __name__=='__main__':
    unittest.main()
