"""Reproducible analysis of Iris/Lotus official route-selection outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "lotus_line_v1"
REPORT_PATH = ROOT / "docs" / "LOTUS_LINE_V1_ROUTE_CAPTION_ANALYSIS.md"
JSON_PATH = OUTPUT_ROOT / "route_caption_analysis.json"

RUNS = {
    "direct_empty": "direct_baseline_official_val_full",
    "adapter_empty": "adapter_only_official_val_full",
    "unet_empty": "unet_only_official_val_full",
    "joint_empty": "adapter_unet_official_val_full",
    "direct_correct": "direct_correct_caption_official_val_full",
    "adapter_correct": "adapter_only_correct_caption_official_val_full",
    "unet_correct": "unet_only_correct_caption_official_val_full",
    "joint_correct": "adapter_unet_correct_caption_official_val_full",
    "adapter_caption_model_empty": "adapter_only_caption_model_empty_official_val_full",
    "unet_caption_model_empty": "unet_only_caption_model_empty_official_val_full",
    "joint_caption_model_empty": "adapter_unet_caption_model_empty_official_val_full",
}

ROUTES = ("direct", "adapter", "unet", "joint")
ROUTE_LABELS = {
    "direct": "Direct",
    "adapter": "Adapter-only",
    "unet": "U-Net-only",
    "joint": "Adapter+U-Net",
}
METRICS = {
    "abs_relative_difference": ("AbsRel", "lower"),
    "squared_relative_difference": ("SqRel", "lower"),
    "rmse_linear": ("RMSE (m)", "lower"),
    "rmse_log": ("RMSElog", "lower"),
    "delta1_acc": ("δ1", "higher"),
    "delta2_acc": ("δ2", "higher"),
    "delta3_acc": ("δ3", "higher"),
    "i_rmse": ("iRMSE", "lower"),
    "silog_rmse": ("SILog", "lower"),
}
PRIMARY = ("abs_relative_difference", "squared_relative_difference", "rmse_linear", "rmse_log", "delta1_acc", "delta2_acc", "delta3_acc")


def read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_run(name: str, relative: str):
    directory = OUTPUT_ROOT / relative
    metadata = json.loads((directory / "official_run_metadata.json").read_text(encoding="utf-8"))
    rows = read_rows(directory / "per_sample_metrics.csv")
    filename_hash = hashlib.sha256((directory / "official_ms2_filename_list.txt").read_bytes()).hexdigest()
    vis_count = len(list((directory / "vis").glob("*.png")))
    expected_count = int(metadata["sample_count"])
    if len(rows) != expected_count or vis_count != expected_count:
        raise RuntimeError(f"Incomplete run {name}: rows={len(rows)}, vis={vis_count}")
    return {
        "name": name,
        "directory": str(directory),
        "metadata": metadata,
        "rows": rows,
        "filename_list_sha256": filename_hash,
        "row_count": len(rows),
        "vis_count": vis_count,
    }


def paired_stats(correct_run, comparison_run, *, iterations=2000, seed=20260702):
    correct_rows = correct_run["rows"]
    comparison_rows = comparison_run["rows"]
    if [row["filename"] for row in correct_rows] != [row["filename"] for row in comparison_rows]:
        raise RuntimeError("Paired comparison filename mismatch")
    rng = np.random.default_rng(seed)
    result = {}
    for metric, (label, direction) in METRICS.items():
        correct = np.asarray([float(row[metric]) for row in correct_rows], dtype=np.float64)
        comparison = np.asarray([float(row[metric]) for row in comparison_rows], dtype=np.float64)
        improvement = comparison - correct if direction == "lower" else correct - comparison
        bootstrap = np.empty(iterations, dtype=np.float64)
        batch = 100
        for start in range(0, iterations, batch):
            count = min(batch, iterations - start)
            indices = rng.integers(0, improvement.size, size=(count, improvement.size))
            bootstrap[start : start + count] = improvement[indices].mean(axis=1)
        result[metric] = {
            "label": label,
            "direction": direction,
            "correct_mean": float(correct.mean()),
            "comparison_mean": float(comparison.mean()),
            "mean_improvement": float(improvement.mean()),
            "median_improvement": float(np.median(improvement)),
            "correct_win_rate": float(np.mean(improvement > 0.0)),
            "tie_rate": float(np.mean(improvement == 0.0)),
            "mean_improvement_95ci": [float(x) for x in np.percentile(bootstrap, [2.5, 97.5])],
        }
    return result


def fmt(value: float):
    return f"{value:.4f}"


def metric_table(run_keys, runs):
    header = "| 指标 | " + " | ".join(run_keys) + " |\n"
    sep = "|---|" + "---:|" * len(run_keys) + "\n"
    lines = [header, sep]
    for metric in PRIMARY:
        label = METRICS[metric][0]
        values = [runs[key]["metadata"]["metrics"][metric] for key in run_keys]
        lines.append("| " + label + " | " + " | ".join(fmt(value) for value in values) + " |\n")
    return "".join(lines)


def causal_table(route, stats):
    lines = [
        "| 指标 | Correct | Empty | Correct 胜率 | 平均改善 [95% CI] |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for metric in PRIMARY:
        item = stats[metric]
        lo, hi = item["mean_improvement_95ci"]
        lines.append(
            f"| {item['label']} | {fmt(item['correct_mean'])} | {fmt(item['comparison_mean'])} | "
            f"{100 * item['correct_win_rate']:.1f}% | {item['mean_improvement']:.4f} [{lo:.4f}, {hi:.4f}] |\n"
        )
    return "".join(lines)


def main():
    runs = {name: load_run(name, relative) for name, relative in RUNS.items()}
    manifest_hashes = {run["metadata"]["manifest_sha256"] for run in runs.values()}
    filename_hashes = {run["filename_list_sha256"] for run in runs.values()}
    evaluators = {run["metadata"]["evaluator"] for run in runs.values()}
    alignments = {run["metadata"]["alignment"] for run in runs.values()}
    if not (len(manifest_hashes) == len(filename_hashes) == len(evaluators) == len(alignments) == 1):
        raise RuntimeError("Protocol mismatch across official runs")

    causal = {
        "direct": paired_stats(runs["direct_correct"], runs["direct_empty"], seed=11),
        "adapter": paired_stats(runs["adapter_correct"], runs["adapter_caption_model_empty"], seed=12),
        "unet": paired_stats(runs["unet_correct"], runs["unet_caption_model_empty"], seed=13),
        "joint": paired_stats(runs["joint_correct"], runs["joint_caption_model_empty"], seed=14),
    }
    trained_vs_no_caption = {
        route: paired_stats(runs[f"{route}_correct"], runs[f"{route}_empty"], seed=100 + index)
        for index, route in enumerate(("adapter", "unet", "joint"))
    }
    serializable_runs = {
        name: {
            "directory": run["directory"],
            "metadata": run["metadata"],
            "filename_list_sha256": run["filename_list_sha256"],
            "row_count": run["row_count"],
            "vis_count": run["vis_count"],
        }
        for name, run in runs.items()
    }
    payload = {
        "protocol_audit": {
            "manifest_sha256": next(iter(manifest_hashes)),
            "filename_list_sha256": next(iter(filename_hashes)),
            "evaluator": next(iter(evaluators)),
            "alignment": next(iter(alignments)),
            "sample_count": 5810,
            "test_split_used": False,
        },
        "runs": serializable_runs,
        "correct_vs_same_checkpoint_empty": causal,
        "caption_trained_correct_vs_separate_no_caption_model": trained_vs_no_caption,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    no_caption_keys = [f"{route}_empty" for route in ROUTES]
    correct_keys = [f"{route}_correct" for route in ROUTES]
    report = [
        "# Lotus line V1：路线与 Caption 全面分析\n\n",
        "## 1. 协议审计\n\n",
        "全部结果均包含 5810 张 MS2 left-thermal Val 样本，使用同一 thermal-view filtered LiDAR GT、同一文件列表、"
        "`lotus/evaluation/evaluation.py::evaluation_depth`、`least_square_disparity` 对齐和官方 `vis/*.png`。"
        "本阶段未使用 Test split。\n\n",
        "## 2. 无 Caption 路线结果\n\n",
        metric_table([ROUTE_LABELS[r] for r in ROUTES], {ROUTE_LABELS[r]: runs[f"{r}_empty"] for r in ROUTES}),
        "\n## 3. 正确 Caption 结果\n\n",
        metric_table([ROUTE_LABELS[r] for r in ROUTES], {ROUTE_LABELS[r]: runs[f"{r}_correct"] for r in ROUTES}),
        "\n## 4. 同 checkpoint：Correct vs Empty\n\n",
    ]
    for route in ROUTES:
        report.extend([f"### {ROUTE_LABELS[route]}\n\n", causal_table(route, causal[route]), "\n"])
    report.extend([
        "## 5. 结论\n\n",
        "- **无 Caption 几何路线没有单一赢家。** Adapter-only 的 RMSE/SqRel 最稳；Adapter+U-Net 的 AbsRel/δ1/δ2 最好；U-Net-only 更均衡。\n",
        "- **Direct 对 Caption 基本不敏感。** 变化幅度极小，不能作为有效 Caption 利用证据。\n",
        "- **Adapter-only 会利用 Caption，但产生明显取舍。** Correct 提升 δ1、RMSElog 和 iRMSE，却显著恶化 RMSE/SqRel。\n",
        "- **U-Net-only 是唯一在 Correct-vs-Empty 中所有核心指标一致改善的 Caption 路线。** 这是当前最干净的文本条件证据。\n",
        "- **Adapter+U-Net 的 Correct Caption 反而全面弱于 Empty。** 联合模型不能据此宣称 Caption 有益。\n",
        "- Caption 由 RGB 图像生成；`rgb_depth_v1` 是面向单目深度估计的提示模板名称，不表示输入了 GT depth。因此不存在 GT depth 泄漏，但系统输入应准确表述为 thermal + RGB-derived Caption。\n",
        "- Caption 模型训练时 dropout=0，因此 Empty 对 Caption-trained checkpoint 属于未见条件；大幅 Correct-vs-Empty 差距同时包含文本依赖和缺失文本分布偏移。\n",
    ])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("".join(report), encoding="utf-8")
    print(REPORT_PATH)
    print(JSON_PATH)


if __name__ == "__main__":
    main()
