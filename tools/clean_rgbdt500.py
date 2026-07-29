"""RGBDT500 cleaning: crop away the registration borders, unify size/format.

WHY THIS SHAPE (measured 2026-07-18 on the train archive, 400 seqs x 10 frames):

* The authors already registered the three modalities into a common frame --
  the infrared frames carry warp borders, which is the fingerprint of that
  warp. So we must NOT re-register; doing so would warp already-registered
  data twice. We only remove the invalid regions.
* Those borders are almost purely horizontal and per-sequence:
  left median 52px (0-172), right median 126px (0-329), top/bottom ~0.
  That is a side-by-side (horizontal-baseline) rig registered with a
  per-sequence horizontal shift -- which aligns one depth plane exactly and
  leaves residual parallax at other depths. Treat alignment as good-but-not-
  exact: fine for paired ablations, not for absolute accuracy claims.
* Depth additionally has a small structural left border (median 16px).
  Depth *holes* (sky / beyond the 20 m cap) are content, NOT borders, and are
  deliberately preserved as invalid-mask pixels rather than cropped away.

Cropping to the common valid box keeps ~90% of the frame area and leaves depth
validity at ~77% (vs 75.8% uncropped) -- i.e. still ~2.6x denser than MS2's
~29%, which is the whole reason this dataset was chosen.

Output mirrors the input layout under --out-root, plus a manifest JSONL whose
fields match what the MS2 tooling expects (thermal_path / rgb_path /
depth_path / sequence / split), so captioning and evaluation can reuse it.

    python tools/clean_rgbdt500.py --src /mnt/e/dataset/RGBDT500/train/Train \\
        --out-root /mnt/e/dataset/RGBDT500/clean_train --split train
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

THERMAL_BLACK_THRESHOLD = 6   # >this counts as valid thermal signal
MIN_KEEP_FRACTION = 0.40      # refuse a sequence whose valid box is tiny


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True,
                        help="RGBDT500 split root containing <seq>/{color,infrared,depth}")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--split", default="train", help="Value written into the manifest 'split' field")
    parser.add_argument("--target-width", type=int, default=768)
    parser.add_argument("--target-height", type=int, default=480)
    parser.add_argument("--limit-sequences", type=int, default=0, help="0 = all; else first N sequences")
    parser.add_argument(
        "--frames-per-sequence",
        type=int,
        default=0,
        help=(
            "0 = keep every frame (right for the train archive, which the authors "
            "already subsampled to 10 non-consecutive frames per sequence). The test "
            "archive instead ships full ~432-frame videos whose consecutive frames are "
            "near-duplicates; pick N evenly spaced frames per sequence so the kept "
            "frames stay de-correlated and captioning stays affordable."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Report crop boxes without writing images")
    return parser.parse_args()


def valid_box(thermal_gray: np.ndarray, depth: np.ndarray) -> tuple[int, int, int, int] | None:
    """Common valid box (top, bottom, left, right, exclusive-end) of thermal & depth.

    Thermal validity = non-black (the warp border). Depth validity here means
    *structural* columns/rows that are entirely zero -- interior holes are kept.
    """
    tm = thermal_gray > THERMAL_BLACK_THRESHOLD
    rows = np.where(tm.any(axis=1))[0]
    cols = np.where(tm.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    top, bottom, left, right = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1

    dm = depth > 0
    drows = np.where(dm.any(axis=1))[0]
    dcols = np.where(dm.any(axis=0))[0]
    if drows.size and dcols.size:
        # only tighten by *structural* edges, never by interior sky holes
        left = max(left, int(dcols[0]))
        right = min(right, int(dcols[-1]) + 1)
        bottom = min(bottom, int(drows[-1]) + 1)
        # deliberately NOT tightening `top`: depth holes reach the top edge on
        # sky-heavy frames and cropping them would delete real thermal content.
    if right - left < 64 or bottom - top < 64:
        return None
    return int(top), int(bottom), int(left), int(right)


def fixed_aspect_box(box: tuple[int, int, int, int], aspect: float) -> tuple[int, int, int, int]:
    """Largest centred sub-box of `box` with the given width/height ratio.

    Keeps geometry consistent across sequences whose borders differ, so the
    resize step does not stretch each sequence by a different amount.
    """
    top, bottom, left, right = box
    height, width = bottom - top, right - left
    if width / height > aspect:
        new_width = int(round(height * aspect))
        offset = (width - new_width) // 2
        left, right = left + offset, left + offset + new_width
    else:
        new_height = int(round(width / aspect))
        offset = (height - new_height) // 2
        top, bottom = top + offset, top + offset + new_height
    return top, bottom, left, right


def process_sequence(seq_dir: Path, out_root: Path, args, aspect: float) -> list[dict]:
    color_dir, ir_dir, depth_dir = seq_dir / "color", seq_dir / "infrared", seq_dir / "depth"
    if not (color_dir.is_dir() and ir_dir.is_dir() and depth_dir.is_dir()):
        return []
    frames = sorted(p.name for p in ir_dir.glob("*.png"))
    if args.frames_per_sequence > 0 and len(frames) > args.frames_per_sequence:
        picks = np.linspace(0, len(frames) - 1, args.frames_per_sequence)
        frames = [frames[int(round(i))] for i in picks]
    rows: list[dict] = []
    for name in frames:
        color_path, ir_path, depth_path = color_dir / name, ir_dir / name, depth_dir / name
        if not (color_path.is_file() and depth_path.is_file()):
            continue
        thermal_rgb = np.asarray(Image.open(ir_path).convert("RGB"))
        thermal_gray = np.asarray(Image.open(ir_path).convert("L"))
        depth = np.asarray(Image.open(depth_path))
        if depth.ndim == 3:
            depth = depth[..., 0]

        box = valid_box(thermal_gray, depth)
        if box is None:
            continue
        keep = ((box[1] - box[0]) * (box[3] - box[2])) / float(thermal_gray.size)
        if keep < MIN_KEEP_FRACTION:
            continue
        top, bottom, left, right = fixed_aspect_box(box, aspect)

        cropped_depth = depth[top:bottom, left:right]
        valid_fraction = float((cropped_depth > 0).mean())

        rel = Path(seq_dir.name)
        record = {
            "id": f"rgbdt500_{seq_dir.name}_{Path(name).stem}",
            "dataset": "RGBDT500",
            "sequence": seq_dir.name,
            "frame": Path(name).stem,
            "split": args.split,
            "thermal_path": str(rel / "infrared" / name),
            "rgb_path": str(rel / "color" / name),
            "depth_path": str(rel / "depth" / name),
            "depth_scale": 1000.0,
            "max_depth": 20.0,
            "crop_box_tblr": [top, bottom, left, right],
            "source_hw": list(thermal_gray.shape),
            "target_hw": [args.target_height, args.target_width],
            "gt_valid_fraction": round(valid_fraction, 4),
        }
        rows.append(record)

        if args.dry_run:
            continue

        size = (args.target_width, args.target_height)
        # thermal: 3-channel storage is grayscale-with-compression-noise -> take L
        Image.fromarray(thermal_gray[top:bottom, left:right]).resize(size, Image.BILINEAR).save(
            _prepare(out_root / rel / "infrared" / name))
        Image.open(color_path).convert("RGB").crop((left, top, right, bottom)).resize(
            size, Image.BILINEAR).save(_prepare(out_root / rel / "color" / name))
        # depth: NEAREST keeps holes crisp and preserves the uint16 mm encoding
        Image.fromarray(cropped_depth.astype(np.uint16)).resize(size, Image.NEAREST).save(
            _prepare(out_root / rel / "depth" / name))
    return rows


def _prepare(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    args = parse_args()
    src, out_root = args.src.resolve(), args.out_root.resolve()
    if not src.is_dir():
        raise SystemExit(f"--src is not a directory: {src}")
    aspect = args.target_width / args.target_height

    sequences = sorted(p for p in src.iterdir() if p.is_dir())
    if args.limit_sequences > 0:
        sequences = sequences[: args.limit_sequences]

    all_rows: list[dict] = []
    for index, seq_dir in enumerate(sequences, 1):
        rows = process_sequence(seq_dir, out_root, args, aspect)
        all_rows.extend(rows)
        if index % 25 == 0 or index == len(sequences):
            print(f"[{index}/{len(sequences)}] {seq_dir.name}: {len(all_rows)} frames so far", flush=True)

    if not all_rows:
        raise SystemExit("No frames survived cleaning; check --src layout.")

    fractions = np.asarray([r["gt_valid_fraction"] for r in all_rows])
    areas = np.asarray([
        (r["crop_box_tblr"][1] - r["crop_box_tblr"][0]) * (r["crop_box_tblr"][3] - r["crop_box_tblr"][2])
        / float(r["source_hw"][0] * r["source_hw"][1]) for r in all_rows
    ])
    summary = {
        "source": str(src),
        "out_root": str(out_root),
        "split": args.split,
        "sequences": len({r["sequence"] for r in all_rows}),
        "frames": len(all_rows),
        "frames_per_sequence": args.frames_per_sequence or "all",
        "target_hw": [args.target_height, args.target_width],
        "gt_valid_fraction": {"mean": float(fractions.mean()), "median": float(np.median(fractions)),
                              "min": float(fractions.min()), "max": float(fractions.max())},
        "kept_area_fraction": {"mean": float(areas.mean()), "median": float(np.median(areas))},
        "ms2_reference_valid_fraction": 0.29,
        "notes": [
            "Modalities were already registered by the dataset authors; this script only crops.",
            "Per-sequence horizontal registration aligns one depth plane; residual parallax remains.",
            "Use for paired ablations; do not quote absolute depth accuracy from this set.",
        ],
    }
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        manifest = out_root / f"rgbdt500_{args.split}_manifest.jsonl"
        manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows),
                            encoding="utf-8")
        (out_root / f"cleaning_summary_{args.split}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nmanifest: {manifest}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
