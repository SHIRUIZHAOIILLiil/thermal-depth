"""Collect every evaluation result under one root into a single CSV.

One file, one row per (arm, split, checkpoint, prompt, exam), each row naming the
`eval_eval.json` it came from. Figures, tables and analysis all read this, so a
number cannot be current in one place and stale in another.

The `manifest_fp` column is the first eight hex of the manifest's sha256, taken
from the directory name that `iris_ms2_pipeline.sbatch` writes. It identifies the
exam. Two rows that agree on everything but this are two different exams and must
not be tabulated together -- which is a mistake this project has made, so the
column exists to make it visible rather than to be trusted to memory.

Results written before the fingerprint was introduced sit in directories without
one and are skipped; they belong to a superseded split and should not be mixed in.

    python tools/collect_eval_grid.py --eval-root $IRIS_RUNS/eval --output grid.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

TAG_TEST = re.compile(r"^(?P<arm>.*)_test_(?P<fp>[0-9a-f]{8})_s(?P<step>\d+)_(?P<prompt>[a-z]+)$")
TAG_VAL = re.compile(r"^(?P<arm>.*)_val_(?P<fp>[0-9a-f]{8}|dflt)_s(?P<step>\d+)$")
FIELDS = ["arm", "split", "step", "prompt", "manifest_fp", "abs_rel", "rmse", "a1",
          "sequence_manifest", "caption_mode", "perm_seed", "self_assign", "frames", "source"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    skipped = 0
    for path in sorted(args.eval_root.glob("*/eval_eval.json")):
        tag = path.parent.name
        match = TAG_TEST.match(tag)
        split, prompt = "test", None
        if match:
            prompt = match["prompt"]
        else:
            match = TAG_VAL.match(tag)
            if not match:
                skipped += 1
                continue
            split, prompt = "val", ""
        payload = json.loads(path.read_text(encoding="utf-8"))
        rotation = payload.get("caption_rotation") or {}
        rows.append({
            "arm": match["arm"], "split": split, "step": match["step"], "prompt": prompt,
            "manifest_fp": match["fp"],
            "abs_rel": payload.get("abs_rel"), "rmse": payload.get("rmse"),
            "a1": payload.get("a1"),
            "sequence_manifest": Path(payload.get("val_manifest") or "").name,
            "caption_mode": payload.get("val_caption_mode", ""),
            "perm_seed": rotation.get("permutation_seed", ""),
            "self_assign": rotation.get("self_assignments", ""),
            "frames": rotation.get("frames", ""),
            "source": str(path),
        })

    if not rows:
        raise SystemExit(f"no fingerprinted eval results under {args.eval_root}")

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] {len(rows)} rows -> {args.output}")
    if skipped:
        print(f"[note] skipped {skipped} directories with no manifest fingerprint "
              f"(pre-fingerprint results, superseded split)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
