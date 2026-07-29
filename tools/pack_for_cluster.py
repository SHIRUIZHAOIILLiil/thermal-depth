"""Work out the minimal MS2 payload for a cluster run, straight from the manifests.

The extracted MS2 tree is ~143 GB, but the routes only ever open four things per
frame: the thermal image, the RGB image (route a and captioning only), and the
matching depth_filtered maps. Right-eye images, LiDAR point clouds, NIR, and the
depth_multi / intensity variants are never touched.

Manifest paths are relative to --ms2-root, so nothing has to be rewritten on the
other side: copy the listed files preserving their relative structure, then point
--ms2-root at the new location.

Output is an rsync file list, so no second copy is made locally:

    python tools/pack_for_cluster.py --manifests <train.jsonl> <val.jsonl> --modalities thermal depth --output ms2_files.txt
    rsync -av --files-from=ms2_files.txt /mnt/e/dataset/ms2/ user@aire:/scratch/ms2/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GB = 1024**3
FIELDS = {
    "thermal": ("thermal_path",),
    "rgb": ("rgb_path",),
    "depth": ("thermal_depth_path", "rgb_depth_path", "depth_path"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["thermal", "depth"],
        choices=sorted(FIELDS),
        help="thermal+depth covers routes b/c/d; add rgb for route a and caption generation.",
    )
    parser.add_argument(
        "--depth-views",
        nargs="+",
        default=["thr"],
        choices=("thr", "rgb"),
        help="Which GT view to ship. Route a needs the rgb view, the thermal routes need thr.",
    )
    parser.add_argument("--ms2-root", type=Path, default=Path("/mnt/e/dataset/ms2"))
    parser.add_argument("--output", type=Path, required=True, help="rsync --files-from list.")
    parser.add_argument(
        "--extras",
        nargs="*",
        default=["calib"],
        help="Substrings of extra per-sequence files to include (calib.npy carries the intrinsics).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wanted: set[str] = set()
    sequences: set[str] = set()

    for manifest in args.manifests:
        if not manifest.exists():
            raise SystemExit(f"Manifest not found: {manifest}")
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                sequences.add(str(row.get("sequence", "")))
                for modality in args.modalities:
                    for field in FIELDS[modality]:
                        value = row.get(field)
                        if not value:
                            continue
                        normalised = str(value).replace("\\", "/")
                        if modality == "depth":
                            # keep only the requested GT views
                            if not any(f"/{view}/" in normalised for view in args.depth_views):
                                continue
                        wanted.add(normalised)
        print(f"[src] {manifest.name}", flush=True)

    # per-sequence extras (calibration etc.) live outside the manifest rows
    for sequence in sorted(s for s in sequences if s):
        directory = args.ms2_root / "sync_data" / f"_{sequence}"
        if not directory.is_dir():
            continue
        for entry in directory.iterdir():
            if entry.is_file() and any(token in entry.name for token in args.extras):
                wanted.add(str(entry.relative_to(args.ms2_root)).replace("\\", "/"))

    listed = sorted(wanted)
    total = 0
    missing = []
    for relative in listed:
        path = args.ms2_root / relative
        if path.is_file():
            total += path.stat().st_size
        else:
            missing.append(relative)

    args.output.write_text("\n".join(listed) + "\n", encoding="utf-8")

    print(f"\nsequences : {', '.join(sorted(s for s in sequences if s))}")
    print(f"modalities: {', '.join(args.modalities)}  (depth views: {', '.join(args.depth_views)})")
    print(f"files     : {len(listed):,}")
    print(f"size      : {total / GB:.2f} GB")
    if missing:
        print(f"[warn] {len(missing)} listed files are absent locally (first: {missing[:3]})")
    print(f"\nlist -> {args.output}")
    print(f"rsync -av --files-from={args.output} {args.ms2_root}/ user@aire:/path/to/ms2/")


if __name__ == "__main__":
    main()
