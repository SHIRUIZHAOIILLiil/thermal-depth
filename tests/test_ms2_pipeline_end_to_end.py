from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from ms2_eval.pipeline import run


class EndToEndPipelineTest(unittest.TestCase):
    def test_full_output_contract_on_synthetic_metric_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); pred_dir = root / "raw_predictions"; pred_dir.mkdir()
            manifest = root / "manifest.jsonl"; rows = []
            for index in range(8):
                thermal = root / f"t{index}.png"; gt_path = root / f"g{index}.png"; sample_id = f"sample_{index}"
                Image.fromarray(np.arange(16, dtype=np.uint16).reshape(4, 4) * 100).save(thermal)
                gt_m = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], np.float32)
                Image.fromarray((gt_m * 256).astype(np.uint16)).save(gt_path)
                np.save(pred_dir / f"{sample_id}.npy", np.maximum(gt_m, 0.2).astype(np.float32))
                rows.append({"id": sample_id, "thermal_path": thermal.name, "thermal_depth_path": gt_path.name,
                             "sequence": "synthetic", "condition": "day_clear", "split": "val", "gt_view": "thermal"})
            manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output = root / "output"
            config = {
                "output_dir": str(output), "gt_audit_samples": 8, "visualization_samples": 1,
                "dataset": {"manifest": str(manifest), "root": str(root), "depth_scale": 256.0, "allow_legacy_depth_path": False},
                "evaluation": {"min_depth_m": 0.1, "max_depth_m": 80.0, "spearman": True},
                "model": {"route": "sp-dit", "checkpoint": None, "prediction_dir": str(pred_dir)},
                "visualization": {"depth_range_m": [0.1, 80.0], "colormap": "magma_r", "error_max_m": 10.0},
            }
            result = run(config)
            self.assertEqual(result["status"], "complete")
            for relative in ("config_resolved.yaml", "run_metadata.json", "checkpoint_info.json",
                             "metrics/per_image.csv", "metrics/summary.json", "metrics/summary_by_condition.json",
                             "metrics/bootstrap_comparison.json", "logs/gt_audit.json", "logs/resolutions.json",
                             "visualizations/sample_0/comparison_panel.png"):
                self.assertTrue((output / relative).is_file(), relative)
            summary = json.loads((output / "metrics" / "summary.json").read_text())
            self.assertEqual(summary["sample_count"], 8)
            self.assertEqual(summary["global_pixel"]["raw_metric"]["delta1"], 1.0)


if __name__ == "__main__": unittest.main()
