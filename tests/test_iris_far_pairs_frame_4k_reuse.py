import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import test_iris_far_pairs_frame_4k as runner


class FrameTests(unittest.TestCase):
    def test_recomputed_selection_and_weight_digest(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            start = run/'parent.pt'
            start.write_bytes(b'parent')
            base = dict(rmse=2.8, abs_rel=.08, a1=.94)
            previous = {'0': base, '2000': base}
            settings = dict(parent=d, parent_sha256=runner.sha256_file(start), original_sha256='16k')
            (run/'contract.json').write_text(json.dumps(settings))
            candidate = run/'conv_3000.pt'
            candidate.write_bytes(b'new')
            s = dict(full={'2000': base, '3000': dict(base, rmse=2.77), '4000': base}, original=base,
                     selected=3000, selected_conv=str(candidate), selected_sha256=runner.sha256_file(candidate))
            (run/'summary.json').write_text(json.dumps(s))
            with patch.object(runner.ext, 'parent_inputs', return_value=({'source_sha256':'16k'}, previous, None, start)):
                self.assertEqual(runner.selection(run)[-2:], (candidate, 3000))
                candidate.write_bytes(b'wrong')
                with self.assertRaisesRegex(ValueError, 'checkpoint'):
                    runner.selection(run)
                s.update(selected=2000, selected_conv=str(start), selected_sha256=settings['parent_sha256'])
                (run/'summary.json').write_text(json.dumps(s))
                with self.assertRaisesRegex(ValueError, 'checkpoint'):
                    runner.selection(run)

    def test_reuse_requires_matching_frame_scores_and_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'summary.json'
            keys = runner.pilot.OFFICIAL_METRICS
            protocol = dict(alignment='ssi_log')
            record = dict(conv_sha256='2k')
            scores = dict(full=dict.fromkeys(keys, 2.), official_common=dict.fromkeys(keys, 3.))
            prev = dict(provenance=dict(**protocol, arms={'frame': record}), results={'frame': scores},
                        frames=2, common_frames=1, missing_official_ids=[])
            path.write_text(json.dumps(prev))
            samples = [SimpleNamespace(sample_id=k) for k in ('a','b')]
            csv_path = path.parent/'frame_frames.csv'
            contents = '\n'.join(','.join(row) for row in [('id',*keys), ['a']+['1']*len(keys), ['b']+['3']*len(keys)])
            csv_path.write_text(contents)
            args = (path, samples, {'b'}, [], protocol)
            self.assertEqual(runner.reuse(*args, record), scores)
            with self.assertRaises(ValueError):
                runner.reuse(*args, dict(conv_sha256='other'))
            csv_path.write_text(contents.replace('b,3', 'b,4'))
            with self.assertRaisesRegex(ValueError, 'per-frame'):
                runner.reuse(*args, record)


if __name__ == '__main__':
    unittest.main()
