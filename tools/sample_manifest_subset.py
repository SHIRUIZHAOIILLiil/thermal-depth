"""Draw a small, sequence-balanced subset manifest for a pre-flight check.

A subset taken off the front of a manifest is one stretch of one drive, and MS2's
training split is two sequences recorded on different routes. So frames are
allocated across sequences in proportion to their size and then taken at even
spacing within each, which keeps a few hundred rows representative of the whole
corpus rather than of whichever sequence happens to be listed first.

Where the check is about the sky, a frame with no sky in it contributes nothing.
``--sky-mask-dir`` with ``--min-sky-pixels`` restricts the pool to frames that
can actually answer the question -- a quarter of this corpus is tunnel, canopy
or tall building and carries no sky at all.

    python tools/sample_manifest_subset.py \
        --manifest <train.jsonl> --output <subset.jsonl> --count 400 \
        --sky-mask-dir <runs>/sky_masks/skymask_train_full/masks --min-sky-pixels 2000
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--group-field", default="sequence", help="Rows are balanced across its values.")
    parser.add_argument("--sky-mask-dir", type=Path, default=None)
    parser.add_argument("--min-sky-pixels", type=int, default=2000,
                        help="Only applies with --sky-mask-dir.")
    return parser.parse_args()


def sky_pixels(mask_dir: Path, sample_id: str) -> int:
    path = mask_dir / f"{sample_id}.png"
    if not path.is_file():
        return 0
    return int((np.asarray(Image.open(path), dtype=np.uint8) > 127).sum())


def evenly_spaced(items: list, count: int) -> list:
    """Deterministic even spacing, so a rerun draws the same frames."""
    if count >= len(items):
        return list(items)
    return [items[int(round(i * (len(items) - 1) / max(1, count - 1)))] for i in range(count)]


def main() -> int:
    args = parse_args()
    rows = [json.loads(line) for line in args.manifest.open(encoding="utf-8") if line.strip()]
    if not rows:
        raise SystemExit(f"{args.manifest} is empty")

    groups: "OrderedDict[str, list]" = OrderedDict()
    for row in rows:
        groups.setdefault(str(row.get(args.group_field, "unknown")), []).append(row)
    print(f"[pool] {len(rows)} rows in {len(groups)} groups: "
          + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))

    if args.sky_mask_dir is not None:
        kept: "OrderedDict[str, list]" = OrderedDict()
        for name, members in groups.items():
            eligible = [r for r in members if sky_pixels(args.sky_mask_dir, str(r["id"])) >= args.min_sky_pixels]
            kept[name] = eligible
            print(f"[sky]  {name}: {len(eligible)}/{len(members)} frames carry "
                  f">= {args.min_sky_pixels} sky pixels")
        if sum(len(v) for v in kept.values()) < args.count:
            raise SystemExit(
                f"Only {sum(len(v) for v in kept.values())} frames clear --min-sky-pixels; "
                f"lower it rather than letting the subset silently shrink")
        groups = OrderedDict((k, v) for k, v in kept.items() if v)

    # Proportional allocation, largest remainder, so the subset mirrors the corpus.
    sizes = {name: len(members) for name, members in groups.items()}
    total = sum(sizes.values())
    exact = {name: args.count * size / total for name, size in sizes.items()}
    quota = {name: int(value) for name, value in exact.items()}
    for name in sorted(exact, key=lambda n: exact[n] - quota[n], reverse=True):
        if sum(quota.values()) >= args.count:
            break
        quota[name] += 1

    selected = []
    for name, members in groups.items():
        take = min(quota[name], len(members))
        chosen = evenly_spaced(members, take)
        selected.extend(chosen)
        print(f"[take] {name}: {len(chosen)} of {len(members)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8")
    print(f"[done] {len(selected)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
