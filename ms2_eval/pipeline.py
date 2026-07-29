"""Manifest-driven unified MS2 evaluation over precomputed raw predictions.

Checkpoint inference remains route-specific. Each route must export raw arrays
with the same sample IDs; this module owns every operation after inference.
"""

from __future__ import annotations

import csv
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .adapters import create_output_adapter
from .aggregate import summarize_by_condition, summarize_rows
from .core import compute_depth_metrics, evaluate_depth_pair
from .io import audit_gt_samples, checkpoint_info, load_manifest, load_ms2_gt
from .resize import resize_dense_prediction
from .visualize import save_shared_visualization


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


class PixelAccumulator:
    def __init__(self): self.values = {k: 0.0 for k in ("n", "abs", "sqrel", "sq", "sqlog", "d1", "d2", "d3")}

    def update(self, pred, gt, valid):
        p, g = pred[valid].astype(np.float64), gt[valid].astype(np.float64)
        diff, ratio = p - g, np.maximum(p / g, g / p); n = p.size
        self.values["n"] += n; self.values["abs"] += np.sum(np.abs(diff) / g)
        self.values["sqrel"] += np.sum(diff**2 / g); self.values["sq"] += np.sum(diff**2)
        self.values["sqlog"] += np.sum((np.log(p) - np.log(g))**2)
        self.values["d1"] += np.sum(ratio < 1.25); self.values["d2"] += np.sum(ratio < 1.25**2); self.values["d3"] += np.sum(ratio < 1.25**3)

    def result(self):
        v, n = self.values, self.values["n"]
        if not n: return None
        return {"abs_rel": v["abs"] / n, "sq_rel": v["sqrel"] / n, "rmse_m": float(np.sqrt(v["sq"] / n)),
                "rmse_log": float(np.sqrt(v["sqlog"] / n)), "delta1": v["d1"] / n,
                "delta2": v["d2"] / n, "delta3": v["d3"] / n, "valid_pixels": int(n)}


def prepare_output_tree(output: Path) -> None:
    for relative in ("predictions", "metrics", "visualizations", "logs"):
        (output / relative).mkdir(parents=True, exist_ok=True)


def prediction_path(directory: Path, sample_id: str) -> Path:
    direct, safe = directory / f"{sample_id}.npy", directory / f"{safe_id(sample_id)}.npy"
    if direct.is_file(): return direct
    if safe.is_file(): return safe
    raise FileNotFoundError(f"No raw prediction for sample {sample_id!r}; tried {direct} and {safe}")


def run(config: dict[str, Any]) -> dict[str, Any]:
    output = Path(config["output_dir"]).resolve(); prepare_output_tree(output)
    dataset = config["dataset"]; evaluation = config["evaluation"]; model = config["model"]
    samples, manifest_info = load_manifest(dataset["manifest"], dataset["root"], allow_legacy_depth_path=bool(dataset.get("allow_legacy_depth_path", False)))
    audit = audit_gt_samples(samples, depth_scale=float(dataset["depth_scale"]), min_depth=float(evaluation["min_depth_m"]),
                             max_depth=float(evaluation["max_depth_m"]), count=max(8, int(config.get("gt_audit_samples", 8))))
    write_json(output / "logs" / "gt_audit.json", audit)
    metadata = {
        "protocol": "MS2-unified-depth-evaluation-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "platform": platform.platform(), "manifest": manifest_info,
        "gt_view": "thermal-view-filtered-lidar", "input_view": "left-thermal",
        "resize": {"prediction": "bilinear", "mask": "nearest", "sparse_gt": "never resized"},
    }
    write_json(output / "run_metadata.json", metadata)
    write_json(output / "checkpoint_info.json", checkpoint_info(model.get("checkpoint")))
    try:
        import yaml
        (output / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except ImportError as error:
        raise RuntimeError("PyYAML is required to save config_resolved.yaml") from error

    if config.get("audit_only", False):
        write_json(output / "metrics" / "bootstrap_comparison.json", {"status": "not_run_audit_only"})
        return {"status": "audit-only", "sample_count": len(samples), "output_dir": str(output)}

    adapter = create_output_adapter(model["route"], epsilon=float(model.get("epsilon", 1e-6)),
                                    clip_min=model.get("clip_min"), clip_max=model.get("clip_max"))
    write_json(output / "logs" / "output_adapter.json", adapter.metadata())
    pred_dir = Path(model["prediction_dir"]).resolve(); rows = []
    raw_global, aligned_global = PixelAccumulator(), PixelAccumulator()
    resolution_log = []; visualize_count = int(config.get("visualization_samples", 8))
    for index, sample in enumerate(samples):
        raw = np.load(prediction_path(pred_dir, sample.sample_id), allow_pickle=False)
        adapted = adapter.adapt(raw); _, gt = load_ms2_gt(sample.thermal_gt_path, float(dataset["depth_scale"]))
        valid = np.isfinite(gt) & (gt > float(evaluation["min_depth_m"])) & (gt < float(evaluation["max_depth_m"]))
        native = resize_dense_prediction(adapted.depth, tuple(gt.shape))
        metrics, aligned = evaluate_depth_pair(native, gt, valid, raw_is_metric=adapter.declaration.metric_scale_exists,
                                                include_spearman=bool(evaluation.get("spearman", True)))
        row = {"sample_id": sample.sample_id, "sequence": sample.sequence, "condition": sample.condition,
               "split": sample.split, "route": adapter.declaration.route, **metrics}
        rows.append(row); aligned_global.update(aligned, gt, valid)
        if adapter.declaration.metric_scale_exists: raw_global.update(native, gt, valid)
        name = safe_id(sample.sample_id)
        np.save(output / "predictions" / f"{name}__native_depth.npy", native.astype(np.float32))
        np.save(output / "predictions" / f"{name}__affine_aligned_depth_m.npy", aligned.astype(np.float32))
        resolution_log.append({"sample_id": sample.sample_id, "raw_prediction_hw": list(adapted.depth.shape),
                               "gt_hw": list(gt.shape), "final_prediction_hw": list(native.shape)})
        if index < visualize_count:
            save_shared_visualization(output / "visualizations" / name, sample_id=sample.sample_id,
                thermal_path=sample.thermal_path, gt_m=gt, valid=valid, native_depth=native,
                aligned_depth_m=aligned, raw_is_metric=adapter.declaration.metric_scale_exists,
                depth_range_m=tuple(config["visualization"]["depth_range_m"]),
                colormap=config["visualization"].get("colormap", "magma_r"),
                error_max_m=float(config["visualization"].get("error_max_m", 10.0)))

    fields = list(rows[0].keys())
    with (output / "metrics" / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    summary = {"image_wise": summarize_rows(rows), "global_pixel": {"raw_metric": raw_global.result(), "affine_aligned": aligned_global.result()},
               "sample_count": len(rows), "manifest": manifest_info, "adapter": adapter.metadata()}
    write_json(output / "metrics" / "summary.json", summary)
    write_json(output / "metrics" / "summary_by_condition.json", summarize_by_condition(rows))
    write_json(output / "metrics" / "bootstrap_comparison.json", {"status": "single-run; use paired comparison for ablations"})
    write_json(output / "logs" / "resolutions.json", resolution_log)
    return {"status": "complete", "sample_count": len(rows), "output_dir": str(output), "summary": summary}
