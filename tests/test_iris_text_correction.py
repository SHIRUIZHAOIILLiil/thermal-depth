import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import iris_text_correction as corr


class CorrectionTests(unittest.TestCase):
    def test_frozen_training_ids_and_validation_permutation(self):
        samples = [SimpleNamespace(sample_id=f"{s}_{i}", caption=f"{s} caption {i}")
                   for s in sorted(corr.TRAIN) for i in range(6)]
        refs, fit = corr.pick_train(samples, per_sequence=3)
        self.assertEqual((refs, fit), corr.pick_train(samples, per_sequence=3))
        self.assertEqual(len(set(refs)), 4)
        self.assertEqual(len(set(fit)), 24)
        self.assertFalse(set(refs) & set(fit))
        with self.assertRaises(ValueError):
            corr.pick_train(samples + [SimpleNamespace(sample_id="val_0")], per_sequence=3)
        val = samples[:3]
        pairs = [[s.sample_id, val[(i+1)%3].sample_id] for i, s in enumerate(val)]
        digest = hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(corr.donors_for(val, {"pairs":pairs}, digest), [s.caption for s in val[1:]+val[:1]])
        with self.assertRaises(ValueError):
            corr.donors_for(val[::-1], {"pairs":pairs}, digest)

    def test_only_one_frame_balanced_scalar_on_valid_lidar(self):
        raw = np.arange(12, dtype=np.float32).reshape(3, 4)
        delta = corr.unit_map(raw)
        np.testing.assert_allclose(delta, corr.unit_map(2*raw+13))
        base = np.full_like(raw, 4.)
        self.assertIs(corr.corrected(base, delta, 0), base)
        gt = corr.corrected(base, delta, .2)
        gt[0] = [0., np.nan, 80., 100.]
        terms = [corr.fit_terms(base, delta, gt)]
        self.assertAlmostEqual(corr.fit_coefficient(terms), .2, places=6)
        self.assertEqual(corr.fit_coefficient([(-1., 2.)]), 0.)
        self.assertIsNone(corr.fit_terms(base, delta, np.zeros_like(gt)))
        # Equal-frame objective even though one frame contains only one return.
        sparse = np.zeros_like(gt)
        sparse[0, 0] = 4*np.exp(.4)
        ones = np.ones_like(delta)
        mixed = [corr.fit_terms(base, ones, sparse), corr.fit_terms(base, ones, base)]
        self.assertAlmostEqual(corr.fit_coefficient(mixed), .2, places=6)

    def test_only_the_prompt_changes_in_the_decoded_difference(self):
        raw = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        class Iris:
            def encode_prompt(self, text): return text
            def predict_disparity(self, row, image, prompt):
                calls.append((row["manifest_index"], image.data_ptr(), prompt))
                return raw.T if prompt == "true" else raw
        class Data:
            samples = [SimpleNamespace(manifest_line=22, caption="true")]
            def __getitem__(self, i):
                return dict(iris_image=raw[None], gt=raw)
        from unittest.mock import patch
        calls = []
        p = corr.Predictor(None, Iris(), ["ref1", "ref2"])
        with patch.object(corr, "forward_batch", return_value=(raw+1)[None]):
            _, frame, perm, _ = p.maps(Data(), 0, "donor")
        self.assertEqual([call[2] for call in calls], ["ref1", "ref2", "true", "donor"])
        self.assertEqual(len({call[:2] for call in calls}), 1)
        self.assertEqual(calls[0][0], 21)
        np.testing.assert_allclose(frame, corr.unit_map(raw.T)-corr.unit_map(raw))
        np.testing.assert_array_equal(perm, 0)

    def test_paired_scoring_resume_and_zero_reproduction(self):
        base = np.linspace(1., 12., 36, dtype=np.float32).reshape(6, 6)
        delta = np.sin(base)
        gt = corr.corrected(base, delta, .1)
        class P:
            def maps(self, data, i, donor):
                calls.append(donor)
                return base, delta, -delta, gt
        class D:
            samples = [SimpleNamespace(sample_id=s) for s in ("A", "B")]
            def __len__(self): return 2
        rows = corr.paired_scores(base, delta, -delta, gt, .1)
        expected = {r["alignment"]: {k:r[k] for k in corr.OFFICIAL_METRICS}
                    for r in rows if r["condition"]=="zero"}
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            args = (P(), D(), ["B", "A"], .1, Path(directory), expected)
            report = corr.validate(*args)
            self.assertTrue(report["passes"])
            self.assertEqual(report, corr.validate(*args))
            self.assertEqual(calls, ["B", "A"])
            expected["ssi"]["abs_rel"] += .01
            with self.assertRaisesRegex(ValueError, "reproduce"):
                corr.validate(*args)


if __name__ == "__main__":
    unittest.main()
