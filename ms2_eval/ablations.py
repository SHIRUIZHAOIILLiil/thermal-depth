"""Controlled caption and training ablation validation."""

from __future__ import annotations

from typing import Any

from .aggregate import paired_comparison


CAPTION_CONTROL_KEYS = ("image_ids_hash", "checkpoint_sha256", "seed", "scheduler", "denoising_steps", "initial_noise_hash")


def assert_caption_controls(metadata_by_mode: dict[str, dict[str, Any]]) -> None:
    required_modes = ("correct", "empty", "hard-wrong")
    for mode in required_modes:
        if mode not in metadata_by_mode: raise ValueError(f"Missing caption mode metadata: {mode}")
    reference = metadata_by_mode["correct"]
    for key in CAPTION_CONTROL_KEYS:
        if key not in reference: raise ValueError(f"Caption control metadata lacks {key}")
        for mode in required_modes[1:]:
            if metadata_by_mode[mode].get(key) != reference[key]:
                raise ValueError(f"Caption ablation changed {key}: correct={reference[key]!r}, {mode}={metadata_by_mode[mode].get(key)!r}")


def caption_ablation(rows_by_mode, metadata_by_mode, *, metrics, iterations=2000, seed=20260630):
    assert_caption_controls(metadata_by_mode)
    return {
        "controls": {key: metadata_by_mode["correct"][key] for key in CAPTION_CONTROL_KEYS},
        "correct_vs_empty": paired_comparison(rows_by_mode["correct"], rows_by_mode["empty"], label_a="correct", label_b="empty", metrics=metrics, iterations=iterations, seed=seed),
        "correct_vs_hard_wrong": paired_comparison(rows_by_mode["correct"], rows_by_mode["hard-wrong"], label_a="correct", label_b="hard-wrong", metrics=metrics, iterations=iterations, seed=seed + 100),
        "mechanism_diagnostics": {"attention_entropy": "separate", "prediction_change_ratio": "separate", "depth_quality_metrics": False},
    }


def training_ablation(adapter_only_rows, adapter_unet_rows, *, manifest_sha_a, manifest_sha_b,
                      checkpoint_rule_a, checkpoint_rule_b, metrics, iterations=2000):
    if manifest_sha_a != manifest_sha_b: raise ValueError("Training ablation changed validation manifest")
    if checkpoint_rule_a != checkpoint_rule_b: raise ValueError("Training ablation changed checkpoint selection rule")
    return {
        "validation_manifest_sha256": manifest_sha_a, "checkpoint_selection_rule": checkpoint_rule_a,
        "geometry": paired_comparison(adapter_unet_rows, adapter_only_rows, label_a="adapter+u-net", label_b="adapter-only", metrics=metrics, iterations=iterations),
        "diffusion_loss": {"reported_separately": True, "may_imply_geometry_quality": False},
    }
