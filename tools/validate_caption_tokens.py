"""Validate regenerated captions against the CLIP 77-token budget.

Tokenizes every caption in a manifest with the Lotus CLIP tokenizer and
reports the token-length distribution, the over-budget list, and spatial-
keyword coverage (near/middle/far usage), so a regeneration batch can be
accepted or sent back before any training run consumes it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=75)
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--show-worst", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(
        args.lotus_model_path, subfolder="tokenizer", local_files_only=args.local_files_only
    )
    rows = [json.loads(l) for l in args.manifest.open(encoding="utf-8") if l.strip()]
    lengths = []
    over = []
    spatial_hits = 0
    for row in rows:
        caption = str(row.get("caption", "")).strip()
        if not caption:
            over.append((row.get("id"), 0, "<EMPTY>"))
            continue
        n_tokens = len(tokenizer(caption)["input_ids"])
        lengths.append(n_tokens)
        if n_tokens > args.token_budget:
            over.append((row.get("id"), n_tokens, caption[:80]))
        lowered = caption.lower()
        if any(k in lowered for k in ("near", "middle", "far", " m.", " m,", "meter")):
            spatial_hits += 1

    lengths.sort()
    n = len(lengths)
    print(f"captions: {len(rows)}")
    if n:
        print(
            f"tokens: min {lengths[0]}  median {lengths[n // 2]}  "
            f"p95 {lengths[int(n * 0.95)]}  max {lengths[-1]}"
        )
    print(f"over budget (> {args.token_budget}): {len(over)} ({len(over) / max(len(rows), 1) * 100:.1f}%)")
    print(f"spatial keyword coverage: {spatial_hits / max(len(rows), 1) * 100:.1f}%")
    for item in over[: args.show_worst]:
        print("  OVER:", item)
    if over:
        raise SystemExit(f"REJECT: {len(over)} captions exceed the budget — regenerate them.")
    print("ACCEPT: all captions within budget.")


if __name__ == "__main__":
    main()
