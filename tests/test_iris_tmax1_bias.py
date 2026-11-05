import sys
import unittest
import csv
import json
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import extend_iris_far_pairs as ext
from tools.train_iris_tmax1_bias import EXPECTED, SOURCE_RUN, UPPER_QUANTILE, check_source, require_start
from tools.iris_tmax1_selection import POLICY, comparison, read_reference, select

REFERENCE = dict(abs_rel=.085, rmse=2.47, a1=.94)  # synthetic test values, NOT measured NeWCRF results


class Tmax1Bias(unittest.TestCase):
    def test_only_the_tmax1_caption_weights_are_accepted(self):
        good = Path('/runs')/SOURCE_RUN/'converted'/'step16000_weights.pt'
        self.assertEqual(check_source(good), good)
        self.assertEqual(UPPER_QUANTILE, 1.)
        for bad in ('/runs/sd2_cap_pct_logtn_img_s43/converted/step16000_weights.pt',
                    f'/runs/{SOURCE_RUN}/converted/step18000_weights.pt'):
            with self.assertRaises(ValueError):
                check_source(bad)

    def test_start_must_reproduce_recorded_validation(self):
        require_start(dict(EXPECTED))
        for key, delta in (('rmse', .01), ('abs_rel', .001), ('a1', .001)):
            with self.assertRaises(ValueError):
                require_start({**EXPECTED, key: EXPECTED[key]+delta})

    def test_relaxed_absrel_and_a1_allow_meaningful_rmse_improvement(self):
        start = dict(EXPECTED)
        full = {ext.START: start, 3000: {**start, 'rmse': start['rmse']-.01},
                4000: dict(abs_rel=start['abs_rel']+.001, rmse=2.60, a1=start['a1']-.002)}
        chosen, report = select(full, REFERENCE)
        self.assertEqual(chosen, 4000)
        self.assertEqual(report['status'], 'narrowed_gap_on_val')
        self.assertFalse(report['candidates'][4000]['matched_primary3'])
        full[4000]['abs_rel'] = start['abs_rel']+.00201
        self.assertEqual(select(full, REFERENCE)[0], ext.START)
        full[4000]['abs_rel'] = start['abs_rel']
        full[4000]['a1'] = start['a1']-.00301
        self.assertEqual(select(full, REFERENCE)[0], ext.START)

    def test_rmse_only_win_is_not_overall_progress(self):
        start = dict(abs_rel=.09, rmse=2.50, a1=.94)
        # RMSE improves, but relative AbsRel/threshold-error deficits grow more.
        worse = dict(abs_rel=.092, rmse=2.479, a1=.937)
        chosen, report = select({2000:start, 3000:worse, 4000:worse}, REFERENCE)
        self.assertEqual(chosen, 2000)
        self.assertFalse(report['candidates'][3000]['gates']['overall'])

    def test_primary3_match_and_deterministic_tie_break(self):
        better = dict(abs_rel=.082, rmse=2.46, a1=.941)
        chosen, report = select({2000:EXPECTED, 3000:better, 4000:better}, REFERENCE)
        self.assertEqual(chosen, 3000)
        self.assertEqual(report['status'], 'matched_newcrf_primary3_on_val')
        self.assertEqual(report['candidates'][chosen]['deficit_score'], 0)
        only_rmse = dict(abs_rel=.087, rmse=2.46, a1=.938)
        self.assertFalse(comparison(only_rmse, REFERENCE)['matched_primary3'])
        with self.assertRaises(ValueError):
            select({2000:EXPECTED, 3000:{**better,'rmse':float('nan')}, 4000:better}, REFERENCE)
        with self.assertRaises(ValueError):
            require_start({**EXPECTED,'a1':float('nan')})

    def test_reference_cache_identity_and_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root/'attempt'/'newcrf'/'raw_predictions'/'newcrf'
            report_path = root/'attempt'/'analysis'/'summary.json'
            report_path.parent.mkdir(parents=True)
            archive_dir = root/'archive'
            archive_dir.mkdir()
            val = dict(path='val.jsonl',sha256='frozen',sample_count=2)
            ids = ['seq_1', 'seq_2']
            archive = dict(validation=val, align={'newcrf':'ssi'}, paths={'newcrf_cache':str(raw)},
                           sha256={'newcrf':'weights'},official_rmse_m={'newcrf':REFERENCE['rmse']})
            (archive_dir/'summary.json').write_text(json.dumps(archive))
            np.savez(archive_dir/'newcrf_moments.npz', ids=ids, official_rmse=[REFERENCE['rmse']]*2)
            report = dict(manifest=val, frames=2, skipped_missing_prediction=0,
                checkpoints={'newcrf':{'sha256':'weights'}},
                models=[dict(name='newcrf',prediction_dir=str(raw),align='ssi')],
                settings=dict(min_depth=.001,max_depth=80.,depth_scale=256.,limit=0),
                headline={'newcrf':REFERENCE})
            report_path.write_text(json.dumps(report))
            with report_path.with_name('per_frame.csv').open('w',newline='') as f:
                w = csv.DictWriter(f,['id', *['newcrf_'+k for k in REFERENCE]])
                w.writeheader()
                w.writerows(dict(id=s, **{'newcrf_'+k:v for k,v in REFERENCE.items()}) for s in ids)
            self.assertEqual(read_reference(archive_dir,archive,val,ids)['metrics'], REFERENCE)
            with self.assertRaises(ValueError):
                read_reference(archive_dir,archive,val,ids[::-1])
            report['models'][0]['align'] = 'ssi_log'
            report_path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                read_reference(archive_dir,archive,val,ids)


if __name__ == '__main__':
    unittest.main()
