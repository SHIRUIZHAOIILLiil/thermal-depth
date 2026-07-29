"""Canonical image/condition summaries and paired bootstrap comparisons."""

from __future__ import annotations

from collections import defaultdict
import numpy as np


IDENTITY_COLUMNS = {"sample_id", "sequence", "condition", "split", "route", "mode"}


def condition_bucket(value: str) -> str:
    text = str(value or "unknown").lower()
    if "rain" in text: return "rain"
    if "night" in text: return "night"
    if "day" in text: return "day"
    return "unknown"


def bootstrap_mean_ci(values, *, iterations=2000, confidence=0.95, seed=20260630):
    values = np.asarray(values, np.float64); values = values[np.isfinite(values)]
    if not values.size: raise ValueError("Cannot bootstrap an empty finite array")
    rng = np.random.default_rng(seed); means = np.empty(iterations, np.float64)
    for start in range(0, iterations, 256):
        batch = min(256, iterations - start); indices = rng.integers(0, values.size, (batch, values.size))
        means[start:start + batch] = values[indices].mean(1)
    alpha = 1 - confidence
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "ci_low": float(np.quantile(means, alpha / 2)), "ci_high": float(np.quantile(means, 1 - alpha / 2)),
            "confidence": confidence, "bootstrap_iterations": iterations, "sample_count": int(values.size), "seed": seed}


def numeric_metric_names(rows):
    if not rows: return []
    result = []
    for key in rows[0]:
        if key in IDENTITY_COLUMNS or key == "raw_metric_available": continue
        if any(isinstance(row.get(key), (int, float, np.number)) and row.get(key) is not None for row in rows): result.append(key)
    return result


def summarize_rows(rows, metric_names=None):
    if not rows: raise ValueError("Cannot summarize an empty per-image table")
    result = {"image_count": len(rows), "statistics": {}, "total_valid_pixels": int(sum(int(r.get("valid_pixels", 0)) for r in rows))}
    for offset, name in enumerate(metric_names or numeric_metric_names(rows)):
        values = np.asarray([r[name] for r in rows if r.get(name) is not None and np.isfinite(r[name])], np.float64)
        if not values.size: continue
        ci = bootstrap_mean_ci(values, seed=20260630 + offset)
        result["statistics"][name] = {"mean": float(values.mean()), "std": float(values.std(ddof=0)),
            "median": float(np.median(values)), "count": int(values.size), "mean_ci95_low": ci["ci_low"], "mean_ci95_high": ci["ci_high"]}
    return result


def summarize_by_condition(rows):
    grouped = defaultdict(list)
    for row in rows: grouped[condition_bucket(row.get("condition"))].append(row)
    output = {}
    for condition in ("day", "night", "rain", "unknown"):
        if grouped[condition]: output[condition] = summarize_rows(grouped[condition])
    return output


def paired_comparison(rows_a, rows_b, *, label_a, label_b, metrics, lower_is_better=None, iterations=2000, seed=20260630):
    lower_is_better = set(lower_is_better or metrics)
    a, b = {r["sample_id"]: r for r in rows_a}, {r["sample_id"]: r for r in rows_b}
    ids = sorted(set(a) & set(b))
    if not ids: raise ValueError(f"No paired sample IDs for {label_a} vs {label_b}")
    result = {"left": label_a, "right": label_b, "paired_sample_count": len(ids), "metrics": {}}
    for offset, metric in enumerate(metrics):
        pairs = [(a[i].get(metric), b[i].get(metric)) for i in ids]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
        if not pairs: continue
        improvement = np.asarray([(y - x) if metric in lower_is_better else (x - y) for x, y in pairs], np.float64)
        stats = bootstrap_mean_ci(improvement, iterations=iterations, seed=seed + offset)
        stats.update({"win_rate": float(np.mean(improvement > 0)), "tie_rate": float(np.mean(improvement == 0)),
                      "difference_definition": f"improvement of {label_a} over {label_b}; positive means {label_a} wins"})
        result["metrics"][metric] = stats
    return result


def caption_ablation_summary(rows_by_mode, *, metrics, iterations=2000, seed=20260630):
    missing = {"correct", "empty", "hard-wrong"} - set(rows_by_mode)
    if missing: raise ValueError(f"Caption ablation is missing modes: {sorted(missing)}")
    return {"correct_vs_empty": paired_comparison(rows_by_mode["correct"], rows_by_mode["empty"], label_a="correct", label_b="empty", metrics=metrics, iterations=iterations, seed=seed),
            "correct_vs_hard_wrong": paired_comparison(rows_by_mode["correct"], rows_by_mode["hard-wrong"], label_a="correct", label_b="hard-wrong", metrics=metrics, iterations=iterations, seed=seed + 100),
            "mechanism_diagnostics_are_depth_metrics": False}


def training_ablation_summary(adapter_only, adapter_unet, *, metrics, checkpoint_rule_a, checkpoint_rule_b, iterations=2000):
    if checkpoint_rule_a != checkpoint_rule_b: raise ValueError("Training ablation requires the same checkpoint selection rule")
    return {"checkpoint_selection_rule": checkpoint_rule_a,
            "geometry": paired_comparison(adapter_unet, adapter_only, label_a="adapter+u-net", label_b="adapter-only", metrics=metrics, iterations=iterations),
            "diffusion_loss": {"reported_separately": True, "geometry_inference_allowed": False}}
