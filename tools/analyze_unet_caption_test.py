"""Create the frozen U-Net-only Caption Test report from official Iris outputs."""

from __future__ import annotations

import json
from pathlib import Path

from analyze_lotus_line_v1 import METRICS, load_run, paired_stats


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "UNET_ONLY_CAPTION_TEST_ANALYSIS.md"
JSON_OUTPUT = ROOT / "outputs" / "lotus_line_v1" / "unet_caption_test_paired_analysis.json"


def main():
    correct = load_run("test_correct", "unet_only_correct_caption_official_test_full")
    empty = load_run("test_empty", "unet_only_caption_model_empty_official_test_full")
    if correct["metadata"]["checkpoint_sha256"] != empty["metadata"]["checkpoint_sha256"]:
        raise RuntimeError("Correct and Empty Test did not use the same checkpoint")
    if correct["filename_list_sha256"] != empty["filename_list_sha256"]:
        raise RuntimeError("Correct and Empty Test did not use the same sample list")
    stats = paired_stats(correct, empty, iterations=2000, seed=20260703)
    payload = {
        "protocol": {
            "checkpoint_sha256": correct["metadata"]["checkpoint_sha256"],
            "manifest_sha256": correct["metadata"]["manifest_sha256"],
            "sample_count": 9508,
            "evaluator": correct["metadata"]["evaluator"],
            "alignment": correct["metadata"]["alignment"],
            "bootstrap_iterations": 2000,
        },
        "correct_vs_empty": stats,
    }
    JSON_OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# U-Net-only Caption：冻结 Test 结果\n\n",
        "同一 step-1000 checkpoint 在 9508 张独立 Test 序列上分别输入 Correct 与 Empty Caption；图像、seed、噪声、GT、官方 evaluator 和对齐方式完全一致。\n\n",
        "| 指标 | Correct | Empty | Correct 胜率 | 平均改善 [95% CI] |\n",
        "|---|---:|---:|---:|---:|\n",
    ]
    for metric in METRICS:
        item = stats[metric]
        lo, hi = item["mean_improvement_95ci"]
        lines.append(
            f"| {item['label']} | {item['correct_mean']:.4f} | {item['comparison_mean']:.4f} | "
            f"{100 * item['correct_win_rate']:.1f}% | {item['mean_improvement']:.4f} [{lo:.4f}, {hi:.4f}] |\n"
        )
    lines.extend([
        "\n## 结论\n\n",
        "- Correct Caption 在 AbsRel、RMSElog、δ1/δ2/δ3、iRMSE 和 SILog 上稳定优于 Empty，且配对 95% CI 不跨零。\n",
        "- Correct Caption 在 SqRel 和米制 RMSE 上更差，说明文本改善相对结构与多数像素，但放大了一部分远距离或大误差尾部。\n",
        "- Caption 的正向贡献从 Val 泛化到了独立 Test 序列，但不能表述为所有几何指标全面提升。\n",
        "- Test 已冻结用于最终报告，不应再根据这些结果调参或重新选择 checkpoint。\n",
    ])
    REPORT.write_text("".join(lines), encoding="utf-8")
    print(REPORT)
    print(JSON_OUTPUT)


if __name__ == "__main__":
    main()
