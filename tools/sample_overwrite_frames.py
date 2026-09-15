"""How much does the LiDAR overwrite change the target, and on which pixels.

One frame showed the overwrite stamping scan lines into an otherwise smooth
surface, with a median change of 0.72 m. Whether that frame was typical is a
question about the distribution, so this scans a sample of the training split
and reports it -- lidar coverage, and the size of the change on the pixels the
overwrite touches -- rather than inviting another eyeball on another frame.

It also copies the raw inputs for a few frames so the side-by-side figure can be
drawn somewhere with a display. Those are chosen by coverage quantile, not by
hand: picking the frame that makes the point is how a figure stops being
evidence.

CPU only, no GPU. A few minutes for a couple of hundred frames.

    python tools/sample_overwrite_frames.py \
        --manifest $SCRATCH/manifests/.../ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl \
        --ms2-root $SCRATCH/data/ms2 \
        --pseudo-dir $SCRATCH/runs/pseudo_gt/official_train/calibrated_pseudo_depth \
        --out-dir $SCRATCH/runs/analysis/overwrite_frames
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

D_MIN, D_MAX = 1e-3, 80.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ms2-root", type=Path, required=True)
    # Optional so the thermal-side statistics can be measured on the test
    # split, which has no pseudo depth by design -- calibrating it would
    # mean fitting to the test GT.
    parser.add_argument("--pseudo-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=200,
                        help="How many frames to measure.")
    parser.add_argument("--export", type=int, default=3,
                        help="How many frames to copy out, taken at evenly "
                             "spaced coverage quantiles.")
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def rows_from(manifest: Path, sample: int, seed: int) -> list[dict]:
    rows = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # Spread the sample over sequences rather than taking a prefix: consecutive
    # frames in a drive are nearly the same scene, so a prefix would measure one
    # stretch of road and call it the split.
    random.Random(seed).shuffle(rows)
    return rows[:sample]


def measure(row: dict, args: argparse.Namespace) -> dict | None:
    gt_path = args.ms2_root / row["thermal_depth_path"]
    if not gt_path.is_file():
        return None
    gt = np.asarray(Image.open(gt_path), dtype=np.float32) / args.depth_scale
    pseudo = None
    if args.pseudo_dir is not None:
        pseudo_path = args.pseudo_dir / f"{row['id']}.npy"
        if not pseudo_path.is_file():
            return None
        pseudo = np.load(pseudo_path, allow_pickle=False).astype(np.float32)
        if pseudo.shape != gt.shape:
            return None

    # How much of the 8-bit range the frame survives into. The thermal PNG is
    # 16-bit and AnyThermalEncoder._array_to_uint8 stretches it by min-max with
    # no percentile clip, so a single hot pixel -- a streetlight, an exhaust --
    # compresses the whole scene into the bottom of the range before it is
    # quantised. The existing guard only rejects constant and all-white frames,
    # so this passes silently, and AnyThermal used the same conversion when the
    # pseudo depth for this frame was built.
    raw = np.asarray(Image.open(args.ms2_root / row["thermal_path"]), dtype=np.float32)
    lo, hi = float(raw.min()), float(raw.max())
    u8 = (np.clip((raw - lo) / (hi - lo) * 255.0, 0, 255).round().astype(np.uint8)
          if hi > lo else np.zeros_like(raw, np.uint8))
    p1, p99 = np.percentile(u8, [1, 99])
    # The direct cost, rather than a proxy for it: of the neighbouring pixel
    # pairs that differ in the raw 16-bit frame, what fraction end up equal
    # after quantisation -- a difference that is gone, not merely reduced.
    # `clip` is the same measure under a 1%/99% clip before quantising, i.e.
    # what the official baseline preprocessing would have left, so the gap
    # between the two is what our choice of min-max costs on this frame.
    lo_c, hi_c = np.percentile(raw, [1, 99])
    clipped = (np.clip((raw - lo_c) / max(hi_c - lo_c, 1e-6) * 255.0, 0, 255)
               .round().astype(np.uint8))
    raw_step = np.abs(np.diff(raw, axis=1))
    differs = raw_step > 0
    def flattened(quantised):
        step = np.abs(np.diff(quantised.astype(np.int16), axis=1))
        return float(np.mean(step[differs] == 0)) if differs.any() else float("nan")
    thermal_stats = {
        "thermal_raw_max_over_p99": float(hi / max(np.percentile(raw, 99), 1e-6)),
        "thermal_used_range": float((p99 - p1) / 255.0),
        "thermal_levels": int(len(np.unique(u8))),
        "flattened_now": flattened(u8),
        "flattened_clipped": flattened(clipped),
    }

    real = np.isfinite(gt) & (gt > D_MIN) & (gt < D_MAX)
    if not real.any():
        return None
    if pseudo is None:
        return {"id": row["id"], "sequence": row.get("sequence", ""),
                "coverage": float(real.mean()), **thermal_stats}
    pure = np.clip(pseudo, D_MIN, D_MAX)
    completed = np.clip(np.where(real, gt, pseudo), D_MIN, D_MAX)
    change = completed[real] - pure[real]
    absolute = np.abs(change)
    return {
        "id": row["id"],
        "sequence": row.get("sequence", ""),
        "coverage": float(real.mean()),
        "change_median": float(np.median(absolute)),
        "change_mean": float(absolute.mean()),
        "change_p95": float(np.percentile(absolute, 95)),
        "change_max": float(absolute.max()),
        "target_rmse": float(np.sqrt(np.mean((completed - pure) ** 2))),
        "pseudo_negative_fraction": float(np.mean(pseudo <= 0)),
        "pseudo_min": float(pseudo.min()),
        "pseudo_max": float(pseudo.max()),
        **thermal_stats,
    }


def report(values: list[float], name: str, unit: str = "") -> str:
    array = np.array(values)
    return (f"{name:<22}" + "  ".join(
        f"{label} {value:.4g}{unit}" for label, value in (
            ("最小", array.min()), ("p25", np.percentile(array, 25)),
            ("中位", np.median(array)), ("p75", np.percentile(array, 75)),
            ("最大", array.max()))))


def main() -> None:
    args = parse_args()
    rows = rows_from(args.manifest, args.sample, args.seed)
    print(f"[data] 抽 {len(rows)} 帧", flush=True)

    records = [r for r in (measure(row, args) for row in rows) if r]
    if not records:
        raise SystemExit("没有一帧可测；给了 --pseudo-dir 就检查它，否则检查 --ms2-root")
    print(f"[data] {len(records)} 帧同时有激光与伪深度\n")

    print(report([r["coverage"] * 100 for r in records], "激光覆盖", "%"))
    if "change_median" in records[0]:
        print(report([r["change_median"] for r in records], "改动量中位（米）"))
        print(report([r["change_p95"] for r in records], "改动量 p95（米）"))
        print(report([r["target_rmse"] for r in records], "两目标整体 RMSE（米）"))
        print(report([r["pseudo_negative_fraction"] * 100 for r in records],
                     "伪深度非正像素", "%"))
    print(report([r["thermal_used_range"] * 100 for r in records],
                 "热像用掉的灰度范围", "%"))
    print(report([r["thermal_levels"] for r in records], "热像不同灰阶数"))
    print(report([r["flattened_now"] * 100 for r in records],
                 "相邻差别被抹平 现状", "%"))
    print(report([r["flattened_clipped"] * 100 for r in records],
                 "相邻差别被抹平 若裁剪", "%"))
    cost = np.array([r["flattened_now"] - r["flattened_clipped"] for r in records])
    print(report(list(cost * 100), "其中归因于 min-max", "%"))
    starved = [r for r in records if r["thermal_used_range"] < 0.25]
    print("")
    print(f"热像动态范围 <25% 的帧：{len(starved)} / {len(records)} "
          f"（{len(starved)/len(records):.1%}）")
    for r in sorted(starved, key=lambda r: r["thermal_used_range"])[:5]:
        print(f"    {r['id']}  用掉 {r['thermal_used_range']:.1%}  "
              f"{r['thermal_levels']} 个灰阶  max/p99 = {r['thermal_raw_max_over_p99']:.2f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = args.out_dir / "overwrite_stats.json"
    summary.write_text(json.dumps(records, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    print(f"\n逐帧统计 -> {summary}")

    # Export at evenly spaced coverage quantiles, so the figure carries a sparse
    # frame and a dense one rather than whichever looked best.
    if args.export == 0 or args.pseudo_dir is None:
        return
    ordered = sorted(records, key=lambda r: r["coverage"])
    picks = [ordered[round(q * (len(ordered) - 1))]
             for q in np.linspace(0.05, 0.95, args.export)]
    by_id = {row["id"]: row for row in rows}
    for rank, pick in enumerate(picks):
        row = by_id[pick["id"]]
        target = args.out_dir / f"frame{rank}_{pick['id']}"
        target.mkdir(exist_ok=True)
        shutil.copy(args.ms2_root / row["thermal_path"], target / "thermal.png")
        shutil.copy(args.ms2_root / row["thermal_depth_path"], target / "lidar.png")
        shutil.copy(args.pseudo_dir / f"{pick['id']}.npy", target / "pseudo.npy")
        (target / "meta.json").write_text(
            json.dumps(pick, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  导出 {target.name}  覆盖 {pick['coverage']:.1%}  "
              f"改动中位 {pick['change_median']:.2f} m")


if __name__ == "__main__":
    main()
