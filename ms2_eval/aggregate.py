"""Image-wise summaries and paired bootstrap comparisons."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np


IDENTITY_COLUMNS = {"sample_id", "sequence", "condition", "split", "route", "mode"}


def numeric_metric_names(rows: list[dict[str, Any]]) -> list[str]:
    if not rows: return []
    names = []
    for key in rows[0]:
        if key in IDENTITY_COLUMNS or key in {"raw_metric_available"}: continue
        values = [row.get(key) for row in rows]
        if any(isinstance(value, (int, float, np.number)) and value is not None for value in values): names.append(key)
    return names


def summarize_rows(rows: list[dict[str, Any]], metric_names: Iterable[str] | None = None) -> dict[str, Any]:
    if not rows: raise ValueError("Cannot summarize an empty per-image table")
    metrics = list(metric_names or numeric_metric_names(rows))
    result: dict[str, Any] = {"image_count": len(rows), "statistics": {}}
    for name in metrics:
        values = np.asarray([row[name] for row in rows if row.get(name) is not None and np.isfinite(row[name])], np.float64)
        if values.size:
            result["statistics"][name] = {"mean": float(values.mean()), "std": float(values.std(ddof=0)),
                                            "median": float(np.median(values)), "count": int(values.size)}
    result["total_valid_pixels"] = int(sum(int(row.get("valid_pixels", 0)) for row in rows))
    return result


def summarize_by_condition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows: grouped[str(row.get("condition") or "unknown").lower()].append(row)
    return {condition: summarize_rows(items) for condition, items in sorted(grouped.items())}


def bootstrap_mean_ci(values, *, iterations: int = 2000, confidence: float = 0.95, seed: int = 20260630):
    values = np.asarray(values, np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0: raise ValueError("Cannot bootstrap an empty finite array")
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, np.float64)
    for start in range(0, iterations, 256):
        batch = min(256, iterations - start)
        indices = rng.integers(0, values.size, size=(batch, values.size))
        means[start:start + batch] = values[indices].mean(axis=1)
    alpha = 1.0 - confidence
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "ci_low": float(np.quantile(means, alpha / 2)), "ci_high": float(np.quantile(means, 1 - alpha / 2)),
            "confidence": confidence, "bootstrap_iterations": iterations, "sample_count": int(values.size), "seed": seed}


def paired_comparison(rows_a, rows_b, *, label_a: str, label_b: str, metrics, lower_is_better=None,
                      iterations: int = 2000, seed: int = 20260630):
    """Paired image-level A-B differences. Positive improvement always means A wins."""
    lower_is_better = set(lower_is_better or metrics)
    a_by_id, b_by_id = {r["sample_id"]: r for r in rows_a}, {r["sample_id"]: r for r in rows_b}
    ids = sorted(set(a_by_id) & set(b_by_id))
    if not ids: raise ValueError(f"No paired sample IDs for {label_a} vs {label_b}")
    output = {"left": label_a, "right": label_b, "paired_sample_count": len(ids), "metrics": {}}
    for offset, metric in enumerate(metrics):
        pairs = [(a_by_id[i].get(metric), b_by_id[i].get(metric)) for i in ids]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
        if not pairs: continue
        differences = np.asarray([(b - a) if metric in lower_is_better else (a - b) for a, b in pairs], np.float64)
        stats = bootstrap_mean_ci(differences, iterations=iterations, seed=seed + offset)
        stats.update({"win_rate": float(np.mean(differences > 0)), "tie_rate": float(np.mean(differences == 0)),
                      "difference_definition": f"improvement of {label_a} over {label_b}; positive means {label_a} wins"})
        output["metrics"][metric] = stats
    return output


def caption_ablation_summary(rows_by_mode, *, metrics, iterations=2000, seed=20260630):
    required = {"correct", "empty", "hard-wrong"}
    missing = required - set(rows_by_mode)
    if missing: raise ValueError(f"Caption ablation is missing modes: {sorted(missing)}")
    return {
        "controlled_variables_required": ["image", "checkpoint", "seed", "scheduler", "denoising_steps", "initial_latent_or_noise"],
        "correct_vs_empty": paired_comparison(rows_by_mode["correct"], rows_by_mode["empty"], label_a="correct", label_b="empty", metrics=metrics, iterations=iterations, seed=seed),
        "correct_vs_hard_wrong": paired_comparison(rows_by_mode["correct"], rows_by_mode["hard-wrong"], label_a="correct", label_b="hard-wrong", metrics=metrics, iterations=iterations, seed=seed + 100),
        "mechanism_diagnostics_are_depth_metrics": False,
    }


def training_ablation_summary(adapter_only, adapter_unet, *, metrics, checkpoint_rule_a, checkpoint_rule_b, iterations=2000):
    if checkpoint_rule_a != checkpoint_rule_b:
        raise ValueError("Training ablation requires the same checkpoint selection rule")
    return {"checkpoint_selection_rule": checkpoint_rule_a,
            "geometry": paired_comparison(adapter_unet, adapter_only, label_a="adapter+u-net", label_b="adapter-only", metrics=metrics, iterations=iterations),
            "diffusion_loss": {"reported_separately": True, "geometry_inference_allowed": False}}
