"""Write a caption-input manifest straight from the thermal image directories.

Why this exists: build_ms2_sequence_manifest.py enumerates frames from the
**GT** directory (proj_depth/<seq>/thr/depth_filtered) and hard-fails when it is
empty, so no training manifest can be built until the depth GT has finished
downloading. Caption generation needs none of that -- it only reads the thermal
image -- and it is the longest step in the pipeline (162k frames, ~73 GPU-hours).
Making it wait on GT would idle the critical path for hours.

The rows carry only what the captioner reads: an id and a thermal_path. The
caption output is later joined back on the image path (last 5 components, the
same key tools/build_thermal_caption_manifest.py uses), never on image_id --
image_id is the bare frame number and every MS2 sequence has a 000000.

    python tools/build_capinput_from_images.py \
        --ms2-root /mnt/scratch/sc23sz/data/ms2 \
        --split-list /mnt/scratch/sc23sz/data/ms2/train_list.txt \
        --output /mnt/scratch/sc23sz/manifests/capinput/ms2_official_train_capinput.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ms2-root", type=Path, required=True,
                        help="Dataset root; sync_data/<seq>/thr/img_left resolves under it.")
    parser.add_argument("--split-list", type=Path, required=True,
                        help="Official MS2 list file, one sequence name per line (e.g. train_list.txt).")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modality-dir", default="thr/img_left",
                        help="Which image directory to enumerate under sync_data/<seq>/.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.ms2_root.resolve()

    seqs = [s.strip() for s in args.split_list.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not seqs:
        raise SystemExit(f"No sequences in {args.split_list}")

    rows: list[dict] = []
    per_seq: list[tuple[str, int]] = []
    for seq in seqs:
        img_dir = root / "sync_data" / seq / args.modality_dir
        if not img_dir.is_dir():
            raise SystemExit(f"Missing image directory: {img_dir}\n"
                             f"  The sync_data transfer for {seq} has not landed.")
        frames = sorted(p for p in img_dir.glob("*.png") if p.is_file())
        if not frames:
            raise SystemExit(f"No PNGs under {img_dir}")
        for p in frames:
            # Paths absolute: the captioner resolves relative paths against the
            # manifest's own directory, not the dataset root, so a relative path
            # here would make every frame report as a read failure.
            rows.append({
                "id": f"{seq.lstrip('_')}_{p.stem}",
                "image_id": p.stem,
                "sequence": seq.lstrip("_"),
                "thermal_path": p.as_posix(),
            })
        per_seq.append((seq, len(frames)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    for seq, n in per_seq:
        print(f"  {seq:<24} {n:>7d}")
    print(f"  {'合计':<24} {len(rows):>7d}  -> {args.output}")

    probe = Path(rows[0]["thermal_path"])
    if not probe.is_file():
        raise SystemExit(f"!! First frame does not resolve: {probe}")
    print(f"  首帧已验证: {probe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
