"""Merge several MS2 sequences into one training manifest.

The 1-epoch era trained on a single sequence (2021-08-06-10-59-33, 10441
frames).  For the 20-epoch suite that is not enough route diversity -- a single
continuous drive gets memorised.  This tool concatenates any number of source
manifests into one manifest, optionally sub-sampling each sequence by stride so
the per-epoch frame budget stays under control.

Default recipe (2026-07-25): the two *daytime* sequences of the official MS2
train split that are already extracted on disk with depth GT and captions:

    2021-08-06-10-59-33   campus, 6.95 km, 10441 frames  (current train)
    2021-08-06-11-37-46   urban,  5.28 km,  9508 frames  (official train; our
                                                          old "test" before the
                                                          split correction)

Their GPS tracks do not overlap (0% of one within 50 m of the other), so this
is genuinely two different routes, not the same road twice.

Usage:

    python tools/build_ms2_multiseq_manifest.py \\
        --output /mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/ms2_train_day2seq_20260725.jsonl

Add ``--stride 2`` (or per-source ``--source path:stride``) to halve the epoch.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

METRES_PER_LATITUDE = 110574.0

MANIFEST_DIR = Path(
    "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b"
)

# source manifest -> which sequence we take out of it
DEFAULT_SOURCES = [
    (
        MANIFEST_DIR
        / "ms2_fixed_sequence_internvl3_8b_filtered_caption_train_rgb_depth_v1_clip75_rerun_20260714.jsonl",
        "2021-08-06-10-59-33",
    ),
    (
        MANIFEST_DIR / "ms2_fixed_sequence_internvl3_8b_filtered_caption_test.jsonl",
        "2021-08-06-11-37-46",
    ),
]

REQUIRED_FIELDS = ("id", "sequence", "thermal_path", "rgb_path", "thermal_depth_path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        metavar="MANIFEST[:SEQUENCE[:STRIDE]]",
        help=(
            "Source manifest, optionally restricted to one sequence and "
            "sub-sampled by stride. Repeatable. Defaults to the two daytime "
            "official-train sequences."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Global frame stride applied to every source that has no explicit stride.",
    )
    parser.add_argument(
        "--min-spacing-m",
        type=float,
        default=0.0,
        help=(
            "Keep at most one frame per this many metres of travel, measured from "
            "the per-frame GPS track. MS2 runs at ~10 Hz, so a plain index stride "
            "keeps whatever the car happened to be doing -- at a red light that is "
            "hundreds of identical frames. 0 disables distance sampling."
        ),
    )
    parser.add_argument(
        "--ms2-root",
        type=Path,
        default=Path("/mnt/e/dataset/ms2"),
        help="Used by --verify to check that every referenced file exists.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Stat every thermal/rgb/depth file (slow, but catches a missing sequence early).",
    )
    parser.add_argument(
        "--require-captions",
        action="store_true",
        help="Fail if any row has an empty caption (only needed for the caption arm).",
    )
    return parser.parse_args()


def parse_source(spec: str, default_stride: int) -> tuple[Path, str | None, int]:
    """``path``, ``path:sequence`` or ``path:sequence:stride``.

    Windows-style drive letters are not expected here (the manifests live under
    /mnt/e in WSL), but split from the right so a leading ``C:`` would survive.
    """
    parts = spec.rsplit(":", 2)
    if len(parts) == 3 and parts[2].isdigit():
        return Path(parts[0]), parts[1] or None, int(parts[2])
    if len(parts) >= 2 and not parts[-1].isdigit() and "/" not in parts[-1]:
        return Path(":".join(parts[:-1])), parts[-1] or None, default_stride
    return Path(spec), None, default_stride


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gps_track(ms2_root: Path, sequence: str) -> dict[str, tuple[float, float]]:
    """Frame id -> (east, north) metres, from the sequence's per-frame GPS."""
    directory = ms2_root / "sync_data" / f"_{sequence}" / "gps_imu" / "data"
    if not directory.is_dir():
        raise SystemExit(f"--min-spacing-m needs GPS, but {directory} does not exist")
    latitudes = []
    raw: list[tuple[str, float, float]] = []
    for entry in sorted(directory.iterdir()):
        if entry.suffix != ".txt":
            continue
        values = entry.read_text().split()
        longitude, latitude = float(values[0]), float(values[1])
        raw.append((entry.stem, longitude, latitude))
        latitudes.append(latitude)
    if not raw:
        raise SystemExit(f"No GPS samples under {directory}")
    metres_per_longitude = 111320.0 * math.cos(math.radians(sum(latitudes) / len(latitudes)))
    return {
        stem: (longitude * metres_per_longitude, latitude * METRES_PER_LATITUDE)
        for stem, longitude, latitude in raw
    }


def distance_sample(rows: list[dict], track: dict[str, tuple[float, float]], spacing: float) -> list[dict]:
    """Greedy: walk the sequence in order, keep a frame once `spacing` metres accrued."""
    kept: list[dict] = []
    travelled = 0.0
    previous: tuple[float, float] | None = None
    for row in rows:
        position = track.get(str(row.get("image_id", "")))
        if position is None:
            raise SystemExit(f"Row {row.get('id')} has no GPS sample for image_id {row.get('image_id')!r}")
        if previous is not None:
            travelled += math.dist(previous, position)
        previous = position
        if not kept or travelled >= spacing:
            kept.append(row)
            travelled = 0.0
    return kept


def load_rows(path: Path, sequence: str | None, stride: int) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Source manifest not found: {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if sequence is not None and row.get("sequence") != sequence:
                continue
            rows.append(row)
    if not rows:
        raise SystemExit(f"No rows matched sequence={sequence!r} in {path}")
    if stride > 1:
        rows = rows[::stride]
    return rows


def main() -> None:
    args = parse_args()

    if args.source:
        sources = [parse_source(spec, args.stride) for spec in args.source]
    else:
        sources = [(path, sequence, args.stride) for path, sequence in DEFAULT_SOURCES]

    merged: list[dict] = []
    provenance = []
    seen_ids: set[str] = set()

    for path, sequence, stride in sources:
        rows = load_rows(path, sequence, stride)
        before = len(rows)
        if args.min_spacing_m > 0:
            if sequence is None:
                raise SystemExit("--min-spacing-m requires each source to name a single sequence")
            rows = distance_sample(rows, gps_track(args.ms2_root, sequence), args.min_spacing_m)
        for row in rows:
            missing = [field for field in REQUIRED_FIELDS if not row.get(field)]
            if missing:
                raise SystemExit(f"{path.name}: row {row.get('id')} is missing {missing}")
            if row["id"] in seen_ids:
                raise SystemExit(f"Duplicate row id across sources: {row['id']}")
            seen_ids.add(row["id"])
            # every row in the merged file is training data regardless of the
            # split label it carried in its source manifest
            row["split"] = "train"
            row["source_manifest"] = path.name
            merged.append(row)
        provenance.append(
            {
                "manifest": str(path),
                "manifest_sha256": sha256(path),
                "sequence_filter": sequence,
                "stride": stride,
                "min_spacing_m": args.min_spacing_m,
                "rows_before_spacing": before,
                "rows_taken": len(rows),
            }
        )
        spacing = f" spacing={args.min_spacing_m}m ({before}->{len(rows)})" if args.min_spacing_m > 0 else ""
        print(f"[src] {path.name} seq={sequence} stride={stride}{spacing} -> {len(rows)} rows")

    per_sequence = collections.Counter(row["sequence"] for row in merged)
    missing_captions = sum(1 for row in merged if not str(row.get("caption", "")).strip())

    if args.require_captions and missing_captions:
        raise SystemExit(
            f"--require-captions but {missing_captions} of {len(merged)} rows have no caption"
        )

    if args.verify:
        absent = []
        for row in merged:
            for field in ("thermal_path", "rgb_path", "thermal_depth_path"):
                if not (args.ms2_root / row[field]).exists():
                    absent.append(f"{row['id']}:{field}")
                    if len(absent) > 20:
                        break
            if len(absent) > 20:
                break
        if absent:
            raise SystemExit(f"Missing files on disk (first 20): {absent[:20]}")
        print(f"[verify] all {len(merged) * 3} referenced files exist")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "total_rows": len(merged),
        "rows_per_sequence": dict(sorted(per_sequence.items())),
        "rows_without_caption": missing_captions,
        "sources": provenance,
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[done] {len(merged)} rows -> {args.output}")
    for sequence, count in sorted(per_sequence.items()):
        print(f"       {sequence}: {count}")
    if missing_captions:
        print(f"[note] {missing_captions} rows have no caption (fine for the empty-prompt arm)")
    print(f"[done] metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
