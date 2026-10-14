"""Unit tests for evaluation-time ensembling in tools/train_route_suite.py.

Run:
    python -m pytest tests/test_ensemble_prediction.py -q
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

# Pull just the helper out of the route suite, which otherwise imports diffusers
# and the Lotus stack.
source = Path(__file__).resolve().parents[1] / "tools/train_route_suite.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
nodes = [n for n in tree.body
         if (isinstance(n, ast.FunctionDef) and n.name == "ensemble_prediction")
         or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "ENSEMBLE_SEED_STRIDE"
                                               for t in n.targets))]
namespace = {"torch": torch}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
ensemble_prediction = namespace["ensemble_prediction"]
STRIDE = namespace["ENSEMBLE_SEED_STRIDE"]


class FakeModel:
    """Prediction = image row-gradient + a seed-dependent offset; records calls."""

    def __init__(self):
        self.calls = []

    def predict_disparity(self, row, image, prompt, donor_row=None, donor_tensor=None,
                          noise_offset=0):
        self.calls.append((noise_offset, bool(getattr(image, "_flipped", False))))
        seed = 7 + row["manifest_index"] + noise_offset
        return image[0, 0] + torch.full_like(image[0, 0], float(seed % 97) / 97.0)


def args(size=1, flip=False, reduce="mean"):
    return SimpleNamespace(ensemble_size=size, ensemble_flip=flip, ensemble_reduce=reduce)


class EnsembleTest(unittest.TestCase):
    def setUp(self):
        self.row = {"manifest_index": 3}
        self.image = torch.arange(2 * 5, dtype=torch.float32).reshape(1, 1, 2, 5).repeat(1, 3, 1, 1)

    def test_single_member_is_the_plain_call(self):
        model = FakeModel()
        out = ensemble_prediction(model, self.row, self.image, None, None, None, args())
        self.assertEqual(model.calls, [(0, False)])
        self.assertTrue(torch.equal(out, model.predict_disparity(self.row, self.image, None)))

    def test_members_use_distinct_offsets_and_mean(self):
        model = FakeModel()
        out = ensemble_prediction(model, self.row, self.image, None, None, None, args(size=3))
        self.assertEqual([c[0] for c in model.calls], [0, STRIDE, 2 * STRIDE])
        expected = torch.stack([self.image[0, 0] + ((7 + 3 + k * STRIDE) % 97) / 97.0
                                for k in range(3)]).mean(0)
        self.assertTrue(torch.allclose(out, expected))

    def test_flip_is_undone(self):
        model = FakeModel()
        # The flipped member sees the mirrored image and is mirrored back, so for a
        # prediction that is just the image, both members agree pixel for pixel.
        out = ensemble_prediction(model, self.row, self.image, None, None, None, args(size=1, flip=True))
        self.assertEqual(len(model.calls), 2)
        self.assertTrue(torch.allclose(out, model.predict_disparity(self.row, self.image, None)))

    def test_median(self):
        model = FakeModel()
        out = ensemble_prediction(model, self.row, self.image, None, None, None, args(size=3, reduce="median"))
        offsets = sorted(((7 + 3 + k * STRIDE) % 97) / 97.0 for k in range(3))
        self.assertTrue(torch.allclose(out, self.image[0, 0] + offsets[1]))

    def test_stride_exceeds_manifest_lengths(self):
        self.assertGreater(STRIDE, 100_000)


if __name__ == "__main__":
    unittest.main()
