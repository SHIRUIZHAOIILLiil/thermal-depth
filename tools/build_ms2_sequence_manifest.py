"""Build a full-field MS2 manifest for one sequence that has no manifest yet.

The train/val manifests came out of the thermal-depth captioning pipeline, so
every sequence we had already carried captions.  The independent test sequence
(2021-08-13-16-08-46, official *val* split, daytime, 2543 frames) only exists as
raw data on disk, and the caption regenerator is manifest-driven -- it needs an
input manifest with `id` and `rgb_path` before it can write captions.  Hence
this tool: it walks the sequence on disk and emits rows in exactly the field
layout that `train_route_suite.read_manifest`, the official evaluation runner
and `regenerate_manifest_captions.py` all expect.

A frame is kept only when all four files exist -- thermal image, RGB image,
thermal-view GT and RGB-view GT -- so the RGB route (conclusion 15: RGB is
scored against RGB-view GT) stays available on the same manifest.

`caption` is written empty; run the caption pass afterwards and point
`--output-manifest` at a new file:

    python tools/build_ms2_sequence_manifest.py --sequence 2021-08-13-16-08-46 \
        --ms2-root /mnt/e/dataset/ms2 --split test --condition day_clear \
        --output /mnt/e/dataset/ms2/ms2_test_16-08-46.jsonl --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, help="Sequence name without the leading underscore.")
    parser.add_argument("--ms2-root", type=Path, required=True, help="Root the relative paths resolve against.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth frame (1 = all).")
    parser.add_argument("--limit", type=int, default=None, help="Keep at most N frames after striding.")
    parser.add_argument("--split", default="test", help="Split label written on every row.")
    parser.add_argument("--condition", default="", help="e.g. day_clear / night / rainy. Recorded, not inferred.")
    parser.add_argument("--scene", default="", help="e.g. urban / campus / residential.")
    parser.add_argument("--official-split", default="", help="The sequence's split in the official MS2 lists.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Stat every referenced file again after assembly (paranoia, cheap).",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    seq_dir = f"_{args.sequence}"
    relative = {
        "thermal_path": f"sync_data/{seq_dir}/thr/img_left",
        "rgb_path": f"sync_data/{seq_dir}/rgb/img_left",
        "thermal_depth_path": f"proj_depth/{seq_dir}/thr/depth_filtered",
        "rgb_depth_path": f"proj_depth/{seq_dir}/rgb/depth_filtered",
    }
    for field, rel in relative.items():
        if not (args.ms2_root / rel).is_dir():
            raise SystemExit(f"Missing directory for {field}: {args.ms2_root / rel}")

    # The thermal GT is the scarcest of the four, so it drives the frame list.
    stems = sorted(p.stem for p in (args.ms2_root / relative["thermal_depth_path"]).glob("*.png"))
    if not stems:
        raise SystemExit(f"No GT frames under {args.ms2_root / relative['thermal_depth_path']}")

    rows = []
    dropped: dict[str, int] = {}
    for index, stem in enumerate(stems):
        if index % args.stride:
            continue
        paths = {field: f"{rel}/{stem}.png" for field, rel in relative.items()}
        missing = [field for field, rel in paths.items() if not (args.ms2_root / rel).is_file()]
        if missing:
            for field in missing:
                dropped[field] = dropped.get(field, 0) + 1
            continue
        rows.append(
            {
                "id": f"{args.sequence}_{stem}",
                "image_id": stem,
                "dataset": "MS2",
                "sequence": args.sequence,
                "split": args.split,
                "official_split": args.official_split,
                "condition": args.condition,
                "scene": args.scene,
                "thermal_path": paths["thermal_path"],
                "rgb_path": paths["rgb_path"],
                "depth_path": paths["thermal_depth_path"],
                "thermal_depth_path": paths["thermal_depth_path"],
                "rgb_depth_path": paths["rgb_depth_path"],
                "caption": "",
                "caption_status": "missing",
                "pairing_method": "sequence_stem_match",
                "gt_variant": "depth_filtered",
                "split_method": "sequence_level",
            }
        )
        if args.limit is not None and len(rows) >= args.limit:
            break

    if not rows:
        raise SystemExit("No frame had all four files; nothing written.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.verify:
        absent = [
            f"{row['id']}:{field}"
            for row in rows
            for field in ("thermal_path", "rgb_path", "thermal_depth_path", "rgb_depth_path")
            if not (args.ms2_root / row[field]).is_file()
        ]
        if absent:
            raise SystemExit(f"[verify] {len(absent)} files vanished, first: {absent[:3]}")
        print(f"[verify] all {len(rows) * 4} referenced files exist")

    metadata = {
        "sequence": args.sequence,
        "ms2_root": str(args.ms2_root),
        "split": args.split,
        "official_split": args.official_split,
        "condition": args.condition,
        "scene": args.scene,
        "stride": args.stride,
        "frames_on_disk": len(stems),
        "rows": len(rows),
        "dropped_by_missing_file": dropped,
        "captions": "none (caption field empty; run regenerate_manifest_captions.py next)",
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] {len(rows)} rows (of {len(stems)} GT frames) -> {args.output}")
    if dropped:
        print(f"[note] dropped frames by missing file: {dropped}")
    print(f"[done] metadata -> {metadata_path}")
    print(f"       sha256 {metadata['output_sha256']}")


if __name__ == "__main__":
    main()
