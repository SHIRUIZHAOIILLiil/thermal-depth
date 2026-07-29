#!/usr/bin/env python3
"""Print the last line of a training_metrics.jsonl in one readable row.

Usage:  watch -n 30 python3 tools/watch_training.py <run_output_dir>
"""

import json
import sys
from pathlib import Path

directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
log = directory / "training_metrics.jsonl"
if not log.is_file():
    raise SystemExit(f"no training_metrics.jsonl in {directory}")

last = ""
with log.open("r", encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            last = line
record = json.loads(last)

step = record["step"]
rate = record["elapsed_seconds"] / max(step, 1)
fields = [f"step {step}", f"{rate:.1f}s/step"]
for key, fmt in (
    ("adapter_grad_norm", "grad={:.2f}"),
    ("unet_grad_norm", "grad={:.2f}"),
    ("gt_abs_rel", "gt_absrel={:.4f}"),
    ("condition_delta_l1", "delta={:.5f}"),
    ("total", "loss={:.4f}"),
    ("samples_seen", "samples={:.0f}"),
):
    if key in record:
        fields.append(fmt.format(record[key]))
print("  ".join(fields))
