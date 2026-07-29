"""Sequence-level train/test split for the cleaned RGBDT500 manifest.

RGBDT500 is a video dataset: frames inside one sequence share the scene, the
objects and the camera pose. Splitting by FRAME would put near-duplicates on
both sides and measure memorisation instead of generalisation. This tool only
ever splits by SEQUENCE, and asserts the two sides are disjoint before writing.

    python tools/split_rgbdt500_manifest.py \\
        --manifest /mnt/e/dataset/RGBDT500/clean_train/rgbdt500_train_manifest.jsonl \\
        --test-sequences 80
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="Cleaned RGBDT500 manifest JSONL")
    parser.add_argument("--test-sequences", type=int, default=80,
                        help="How many sequences go to the held-out side")
    parser.add_argument("--seed", type=int, default=20260703, help="Same seed convention as the MS2 runs")
    parser.add_argument("--out-prefix", type=Path, default=None,
                        help="Defaults to the manifest path without its .jsonl suffix")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = args.manifest.resolve()
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"Empty manifest: {manifest}")

    by_sequence: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_sequence[str(row["sequence"])].append(row)
    sequences = sorted(by_sequence)
    if not 0 < args.test_sequences < len(sequences):
        raise SystemExit(f"--test-sequences must be in (0, {len(sequences)}), got {args.test_sequences}")

    shuffled = list(sequences)
    random.Random(args.seed).shuffle(shuffled)
    test_sequences = set(shuffled[: args.test_sequences])
    train_sequences = set(shuffled[args.test_sequences:])

    overlap = test_sequences & train_sequences
    if overlap:
        raise SystemExit(f"Sequence overlap between splits: {sorted(overlap)[:5]}")

    prefix = args.out_prefix or manifest.with_suffix("")
    written = {}
    for name, keep in (("train", train_sequences), ("test", test_sequences)):
        subset = [dict(row, split=name) for row in rows if row["sequence"] in keep]
        path = Path(f"{prefix}_{name}.jsonl")
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset), encoding="utf-8")
        fractions = [r.get("gt_valid_fraction", 0.0) for r in subset]
        written[name] = {
            "path": str(path), "sequences": len(keep), "frames": len(subset),
            "gt_valid_fraction_mean": round(sum(fractions) / max(len(fractions), 1), 4),
        }

    report = {
        "source_manifest": str(manifest),
        "split_level": "sequence (never frame -- video frames are near-duplicates)",
        "seed": args.seed,
        "total_sequences": len(sequences),
        "disjoint_verified": True,
        **written,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
