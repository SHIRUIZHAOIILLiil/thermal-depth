from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ms2_eval.ablations import assert_caption_controls, training_ablation
from ms2_eval.io import audit_gt_samples, load_manifest


class PipelineIOTests(unittest.TestCase):
    def test_manifest_hash_thermal_view_and_gt_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = root / "manifest.jsonl"; rows = []
            for index in range(8):
                thermal = root / f"thermal_{index}.png"; depth = root / f"depth_{index}.png"
                Image.fromarray(np.full((3, 4), 100 + index, np.uint16)).save(thermal)
                raw = np.asarray([[0, 256, 512, 1024]] * 3, np.uint16); Image.fromarray(raw).save(depth)
                rows.append({"id": f"s{index}", "thermal_path": thermal.name, "thermal_depth_path": depth.name,
                             "sequence": "seq", "condition": "day", "split": "val", "gt_view": "thermal"})
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            samples, info = load_manifest(manifest, root)
            self.assertEqual(info["sample_count"], 8); self.assertEqual(len(info["sha256"]), 64)
            audit = audit_gt_samples(samples, depth_scale=256.0, min_depth=0.1, max_depth=80.0, count=8)
            self.assertEqual(len(audit), 8); self.assertEqual(audit[0]["decoded_max_m"], 4.0)
            self.assertEqual(audit[0]["zero_count"], 3); self.assertEqual(audit[0]["valid_pixel_count"], 9)

    def test_rgb_gt_is_not_a_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({"id": "x", "thermal_path": "t.png", "rgb_depth_path": "rgb.png"}) + "\n")
            with self.assertRaisesRegex(ValueError, "thermal-view"):
                load_manifest(manifest, root)

    def test_caption_controls_and_training_contract(self):
        base = {"image_ids_hash": "i", "checkpoint_sha256": "c", "seed": 1, "scheduler": "s", "denoising_steps": 4, "initial_noise_hash": "n"}
        assert_caption_controls({"correct": base, "empty": dict(base), "hard-wrong": dict(base)})
        changed = dict(base); changed["seed"] = 2
        with self.assertRaisesRegex(ValueError, "changed seed"):
            assert_caption_controls({"correct": base, "empty": changed, "hard-wrong": dict(base)})
        with self.assertRaisesRegex(ValueError, "validation manifest"):
            training_ablation([], [], manifest_sha_a="a", manifest_sha_b="b", checkpoint_rule_a="x", checkpoint_rule_b="x", metrics=[])


if __name__ == "__main__": unittest.main()
