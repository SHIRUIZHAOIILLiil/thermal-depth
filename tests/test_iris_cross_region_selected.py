import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import test_iris_cross_region_pairs as runner


class CrossRegionTestSelection(unittest.TestCase):
    def test_fallback_and_improved_checkpoint_with_overlapping_absolute_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            start = run/'parent.pt'
            start.write_bytes(b'parent')
            child = run/'conv_4000.pt'
            child.write_bytes(b'new')
            base = dict(rmse=2.8, abs_rel=.075, a1=.94)
            parent = dict(original=base, full={'3000': base})
            curve = {'0': base, '1000': base, '2000': dict(base, rmse=2.7, a1=.92)}
            for added, weight in [(0,start), (1000,child)]:
                if added:
                    curve['1000'] = dict(base, rmse=2.77)
                step = 3000+added
                record = dict(start=3000, full_by_added_steps=curve, added_steps=added, selected=step,
                              selected_conv=str(weight), selected_sha256=runner.sha256_file(weight))
                (run/'summary.json').write_text(json.dumps(record))
                self.assertEqual(runner.checkpoint(run, 3000, start, parent), (added,step,weight))
            child.write_bytes(b'wrong')
            with self.assertRaises(ValueError):
                runner.checkpoint(run, 3000, start, parent)

    def test_cache_checks_inherited_csv_and_both_aggregates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'summary.json'
            frames = Path(tmp)/'older_frames.csv'
            keys = runner.pilot.OFFICIAL_METRICS
            rows = [('id', *keys), ['a'] + ['1']*len(keys), ['b'] + ['3']*len(keys)]
            frames.write_text('\n'.join(','.join(row) for row in rows))
            samples = [SimpleNamespace(sample_id=s) for s in ('a','b')]
            expected = dict(selection_sha256='parent', frame=dict(conv_sha256='selected'))
            result = dict(full=dict.fromkeys(keys, 2.), official_common=dict.fromkeys(keys, 3.))
            summary = dict(provenance=expected, frames=2, common_frames=1, missing_official_ids=['c'],
                           frame_csv=str(frames), results={'frame':result})
            path.write_text(json.dumps(summary))
            args = (path, frames, samples, {'b'}, ['c'])
            self.assertEqual(runner.verify_cache(*args, expected), result)
            with self.assertRaises(ValueError):
                runner.verify_cache(*args, dict(expected, selection_sha256='other'))
            result['official_common']['rmse'] = 2.
            path.write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                runner.verify_cache(*args, expected)


if __name__ == '__main__':
    unittest.main()
