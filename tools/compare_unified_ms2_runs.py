"""Create paired bootstrap summaries from completed unified MS2 runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from ms2_eval.ablations import caption_ablation, training_ablation

DEFAULT_METRICS = ["abs_rel", "rmse_m", "rmse_log", "delta1", "aligned_abs_rel", "aligned_rmse_m", "aligned_rmse_log", "aligned_delta1"]


def read_rows(run_dir: Path):
    with (run_dir / "metrics" / "per_image.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key in {"sample_id", "sequence", "condition", "split", "route", "mode"} or value in ("", "None"): continue
            try: row[key] = float(value)
            except ValueError: pass
    return rows


def manifest_sha(run_dir: Path) -> str:
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("gt_view") != "thermal-view-filtered-lidar": raise ValueError(f"Wrong GT view in {run_dir}")
    return metadata["manifest"]["sha256"]


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="kind", required=True)
    caption = sub.add_parser("caption")
    caption.add_argument("--correct", type=Path, required=True); caption.add_argument("--empty", type=Path, required=True)
    caption.add_argument("--hard-wrong", type=Path, required=True); caption.add_argument("--controls", type=Path, required=True)
    caption.add_argument("--output", type=Path, required=True)
    training = sub.add_parser("training")
    training.add_argument("--adapter-only", type=Path, required=True); training.add_argument("--adapter-unet", type=Path, required=True)
    training.add_argument("--checkpoint-rule", required=True); training.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.kind == "caption":
        runs = {"correct": args.correct, "empty": args.empty, "hard-wrong": args.hard_wrong}
        hashes = {manifest_sha(path) for path in runs.values()}
        if len(hashes) != 1: raise ValueError("Caption runs use different manifests")
        controls = json.loads(args.controls.read_text(encoding="utf-8"))
        result = caption_ablation({mode: read_rows(path) for mode, path in runs.items()}, controls, metrics=DEFAULT_METRICS)
    else:
        hash_a, hash_b = manifest_sha(args.adapter_only), manifest_sha(args.adapter_unet)
        result = training_ablation(read_rows(args.adapter_only), read_rows(args.adapter_unet), manifest_sha_a=hash_a,
            manifest_sha_b=hash_b, checkpoint_rule_a=args.checkpoint_rule, checkpoint_rule_b=args.checkpoint_rule, metrics=DEFAULT_METRICS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__": main()
