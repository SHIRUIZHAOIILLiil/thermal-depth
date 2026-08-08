"""Where four sources put the sky, on the same frames under the same mask.

Every sky number this project has quoted came from the top 32 rows, which are
canopy, gantries and buildings as much as sky: measured under a segmentation
mask the same pseudo depth reads 25.8 m at the median where the row band reads
16.0 m. Comparing a masked figure against a banded one is comparing two
different regions, so this puts all sources through one region and one
alignment.

Sources declare what they hold. Lotus routes emit relative disparity and are
aligned exactly as the official protocol aligns them -- `align_to_metric_depth`
is imported from `analyze_sky_band`, not restated, so these metres and the
frozen sky-band metres are the same quantity. Calibrated pseudo depth is already
in metres and is passed through.

All sources are then clamped to the same evaluation range, so the top bucket
reads "this source said at least the ceiling" for every one of them alike rather
than meaning something different per source.

    python tools/compare_sky_sources.py --manifest <subset.jsonl> --data-root <ms2> \
        --sky-mask-dir <masks> --output-dir <out> \
        --sources <dir>:zero_training:disparity <dir>:b_line:disparity \
                  <dir>:b_sky_loss:disparity <dir>:pseudo_gt:depth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_sky_band import align_to_metric_depth, read_manifest, resize_to  # noqa: E402
from ms2_eval.official_protocol import official_valid_mask  # noqa: E402

KINDS = ("disparity", "depth")
PERCENTILES = (10, 50, 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sky-mask-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sources", nargs="+", required=True, metavar="DIR[:LABEL[:KIND]]",
                        help="KIND is disparity (Lotus routes, aligned here) or depth "
                             "(calibrated pseudo depth, already metric). Default disparity.")
    parser.add_argument("--gt-view", default="thermal", choices=("thermal", "rgb"))
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--top-rows", type=int, default=32,
                        help="Reported alongside the mask, so the two regions can be "
                             "seen to differ rather than quietly substituted.")
    return parser.parse_args()


def parse_sources(values: list[str]) -> list[tuple[Path, str, str]]:
    sources = []
    for value in values:
        kind = "disparity"
        head, _, tail = value.rpartition(":")
        if tail in KINDS and ":" in head[2:]:
            value, kind = head, tail
        if ":" in value[2:]:
            head, _, label = value.rpartition(":")
            directory = Path(head)
        else:
            directory, label = Path(value), Path(value).name
        if not directory.is_dir():
            raise SystemExit(f"{label}: not a directory: {directory}")
        sources.append((directory, label, kind))
    labels = [label for _, label, _ in sources]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"Source labels must be unique, got {labels}")
    return sources


def band_shares(values: np.ndarray, ceiling: float) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "under_20": float(np.mean(values < 20.0)),
        "20_to_40": float(np.mean((values >= 20.0) & (values < 40.0))),
        "40_to_ceiling": float(np.mean((values >= 40.0) & (values < ceiling))),
        "at_ceiling": float(np.mean(values >= ceiling)),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest, args.gt_view)
    sources = parse_sources(args.sources)
    print(f"[data] {len(rows)} frames, {len(sources)} sources, mask {args.sky_mask_dir}", flush=True)
    for _, label, kind in sources:
        print(f"       {label}: {kind}")

    regions = ("sky", "sky_pseudo", "top_rows")
    stats: dict = {label: {r: {"percentiles": [], "bands": []} for r in regions} for _, label, _ in sources}
    missing: dict[str, int] = {label: 0 for _, label, _ in sources}
    no_mask = 0
    frames = 0

    for row in rows:
        gt = np.asarray(Image.open(args.data_root / row["depth_path"]), dtype=np.float32) / args.depth_scale
        mask_path = args.sky_mask_dir / f"{row['id']}.png"
        if not mask_path.is_file():
            no_mask += 1
            continue
        sky = np.asarray(Image.open(mask_path), dtype=np.uint8) > 127
        valid = official_valid_mask(gt, args.min_depth, args.max_depth)
        views = {
            "sky": sky,
            # Where lidar spoke, lidar wins -- this is the part completion would supply.
            "sky_pseudo": sky & ~valid,
            "top_rows": np.zeros_like(sky),
        }
        views["top_rows"][: args.top_rows] = True
        frames += 1

        for directory, label, kind in sources:
            npy = directory / f"{row['id']}.npy"
            if not npy.is_file():
                missing[label] += 1
                continue
            prediction = np.load(npy, allow_pickle=False).astype(np.float32)
            if prediction.ndim == 3:
                prediction = prediction.mean(axis=0 if prediction.shape[0] <= 4 else -1)
            prediction = resize_to(prediction, gt.shape)
            if kind == "disparity":
                depth, _ = align_to_metric_depth(prediction, gt, args.min_depth, args.max_depth)
            else:
                depth = np.clip(prediction.astype(np.float64), args.min_depth, args.max_depth)
            for region, selector in views.items():
                values = depth[selector]
                if values.size == 0:
                    continue
                stats[label][region]["percentiles"].append(np.percentile(values, PERCENTILES))
                stats[label][region]["bands"].append(band_shares(values, args.max_depth))

    if not frames:
        raise SystemExit("No frame had a sky mask; check --sky-mask-dir")

    report: dict = {"frames": frames, "frames_without_mask": no_mask, "missing_per_source": missing,
                    "ceiling_m": args.max_depth, "top_rows": args.top_rows, "regions": {}}
    for region in regions:
        report["regions"][region] = {}
        for _, label, _ in sources:
            entry = stats[label][region]
            if not entry["percentiles"]:
                continue
            percentiles = np.median(np.stack(entry["percentiles"]), axis=0)
            bands = {k: float(np.mean([b[k] for b in entry["bands"]])) for k in entry["bands"][0]}
            report["regions"][region][label] = {
                "frames": len(entry["percentiles"]),
                **{f"p{p}_m": float(v) for p, v in zip(PERCENTILES, percentiles)},
                **bands,
            }
    (args.output_dir / "sky_sources.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    ceiling = f"{args.max_depth:g}"
    for region in regions:
        table = report["regions"].get(region)
        if not table:
            continue
        print(f"\n=== {region} ===")
        print(f"{'source':>18}{'p10':>9}{'p50':>9}{'p90':>9}{'<20m':>9}{'20-40m':>9}"
              f"{f'40-{ceiling}m':>10}{f'>={ceiling}m':>10}")
        for label, values in table.items():
            print(f"{label:>18}{values['p10_m']:>9.2f}{values['p50_m']:>9.2f}{values['p90_m']:>9.2f}"
                  f"{values['under_20']:>9.1%}{values['20_to_40']:>9.1%}"
                  f"{values['40_to_ceiling']:>10.1%}{values['at_ceiling']:>10.1%}")

    if any(missing.values()):
        print(f"\n[missing] {missing}")
    print(f"\n{frames} frames scored, {no_mask} skipped for having no sky mask.")
    print("sky_pseudo is the region completion would supply; top_rows is the band the"
          "\nearlier numbers used, kept here only so the two are not confused again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
