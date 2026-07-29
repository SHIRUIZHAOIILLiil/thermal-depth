"""Paired per-sample analysis of the 20-cell caption ablation (task 2).

Pairs каждой caption cell with its empty counterpart on the lotus evaluator's
per_sample_metrics.csv (identical filename keys, identical per-row seeds), and
reports mean diff, win rate, and bootstrap CI via ms2_eval.aggregate.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ms2_eval.aggregate import paired_comparison  # noqa: E402

OUT = ROOT / "outputs" / "lotus_line_v2"
METRICS = ("abs_relative_difference", "rmse_linear", "delta1_acc")
LOWER_IS_BETTER = {"abs_relative_difference", "rmse_linear"}

# (label, caption_run_dir, empty_run_dir)
PAIRS = (
    ("a_frozen: correct vs empty", "route_a_rgb_frozen_val_full_caption", "route_a_rgb_frozen_val_full"),
    ("b_emptytrain: correct vs empty", "route_b_rgb_unet_val_full_caption", "route_b_rgb_unet_val_full"),
    ("b_capttrain: correct vs empty", "route_b_capttrain_val_full_correct", "route_b_capttrain_val_full_empty"),
    ("b_train_effect(empty-inf): capttrain vs emptytrain", "route_b_capttrain_val_full_empty", "route_b_rgb_unet_val_full"),
    ("c_frozen: correct vs empty", "route_c_thermal_frozen_val_full_caption", "route_c_thermal_frozen_val_full"),
    ("d_emptytrain: correct vs empty", "route_d_fp32dec_val_full_caption", "route_d_fp32dec_val_full_empty"),
    ("d_capttrain: correct vs empty", "route_d_capttrain_val_full_correct", "route_d_capttrain_val_full_empty"),
    ("d_train_effect(empty-inf): capttrain vs emptytrain", "route_d_capttrain_val_full_empty", "route_d_fp32dec_val_full_empty"),
    ("d_slope: fp32dec-emptytrain vs frozen-c", "route_d_fp32dec_val_full_empty", "route_c_thermal_frozen_val_full"),
    ("e_adapter: correct vs empty", "route_e_vae_adapter_val_full_caption", "route_e_vae_adapter_val_full"),
    ("e_capttrain: correct vs empty", "route_e_capttrain_val_full_correct", "route_e_capttrain_val_full_empty"),
    ("e_train_effect(empty-inf): capttrain vs emptytrain", "route_e_capttrain_val_full_empty", "route_e_vae_adapter_val_full"),
    ("f_adapter: correct vs empty", "route_f_adapter_only_val_full_caption", "route_f_adapter_only_val_full"),
    ("f_capttrain: correct vs empty", "route_f_capttrain_val_full_correct", "route_f_capttrain_val_full_empty"),
    ("f_train_effect(empty-inf): capttrain vs emptytrain", "route_f_capttrain_val_full_empty", "route_f_adapter_only_val_full"),
    ("f_train_effect(correct-inf): capttrain vs emptytrain", "route_f_capttrain_val_full_correct", "route_f_adapter_only_val_full_caption"),
)


def load_rows(run_dir: str):
    path = OUT / run_dir / "per_sample_metrics.csv"
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            entry = {"sample_id": row["filename"]}
            for metric in METRICS:
                entry[metric] = float(row[metric])
            rows.append(entry)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def main() -> int:
    report = {}
    print(f"{'comparison':52s} {'metric':24s} {'mean_diff':>10s} {'ci':>22s} {'win%':>6s}")
    for label, left_dir, right_dir in PAIRS:
        result = paired_comparison(
            load_rows(left_dir), load_rows(right_dir),
            label_a=left_dir, label_b=right_dir,
            metrics=list(METRICS), lower_is_better=LOWER_IS_BETTER,
        )
        report[label] = result
        for metric, stats in result["metrics"].items():
            # positive improvement = left arm wins
            sig = "*" if stats["ci_low"] > 0 or stats["ci_high"] < 0 else " "
            print(f"{label:52s} {metric:24s} {stats['mean']:+.5f}{sig} "
                  f"[{stats['ci_low']:+.5f},{stats['ci_high']:+.5f}] {stats['win_rate']*100:5.1f}")
        print()
    out_path = OUT / "caption_ablation_20cell_paired.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
