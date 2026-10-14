"""Where does one model lose to another? Depth range, depth boundaries, and bad frames.

    python tools/diagnose_error_sources.py \
        --manifest $M/ms2_val_official3_thermalcap_20260821.jsonl --data-root $IRIS_MS2_ROOT \
        --model ours=$OUT/ours/raw_predictions:ssi_log \
        --model newcrf=$OUT/newcrf/raw_predictions/newcrf:ssi \
        --out $OUT/analysis

DIAGNOSTIC ONLY. Run it on validation frames: its answers decide what to change in
the method, so running it on test would let test steer the method.

Every frame is aligned exactly as the official evaluator aligns it (per-image
two-parameter fit in each model's own output space, then the [min, max] clamp), and
the per-frame AbsRel is checked against `ms2_eval.official_protocol.evaluate_sample`
on every frame -- a mismatch aborts, so nothing below is measured on a different exam.

Three questions, all on the same aligned pixels:

1. Depth range. Pixels are binned by GT depth. The benchmark's AbsRel is a mean over
   frames of per-frame pixel means, so with weight 1/n_f per pixel it splits exactly
   into per-bin parts that add up to the headline number; the same holds for the mean
   per-frame MSE (not for RMSE itself, which takes a square root per frame). The
   "gap share" columns say which bins the first model's deficit comes from.
2. Boundaries. A valid GT pixel is an *edge* when the valid GT inside its window spans
   more than --edge-ratio in depth, *interior* when it spans less than --flat-ratio,
   and *sparse* when it has fewer than --min-neighbors valid neighbours (LiDAR cannot
   tell us there whether it is an edge). Same exact decomposition.
3. Bad frames. Paired per-frame differences, how much of the total gap the worst
   frames carry, and the gap per sequence.

Outputs: summary.json, per_frame.csv, and the tables printed to stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ms2_eval.io import load_manifest, load_ms2_gt  # noqa: E402
from ms2_eval.official_protocol import (  # noqa: E402
    collapse_channels,
    evaluate_sample,
    fit_scale_shift,
    official_valid_mask,
)
from ms2_eval.resize import resize_dense_prediction  # noqa: E402

ALIGNS = ("ssi", "ssi_log", "ssi_disparity", "median", "none")
STATS = ("w", "w_absrel", "w_sq", "w_d1", "pixels")


def aligned_depth(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, align: str) -> np.ndarray:
    """The official evaluator's alignment, returning the dense aligned depth."""
    pred64 = pred.astype(np.float64)
    if align == "ssi":
        scale, shift = fit_scale_shift(pred, gt, valid)
        return pred64 * scale + shift
    if align == "ssi_disparity":
        target = np.zeros_like(gt, np.float64)
        target[valid] = 1.0 / gt[valid].astype(np.float64)
        scale, shift = fit_scale_shift(pred, target.astype(np.float32), valid)
        return 1.0 / np.clip(pred64 * scale + shift, 1e-3, None)
    if align == "ssi_log":
        target = np.zeros_like(gt, np.float64)
        target[valid] = np.log(np.maximum(gt[valid].astype(np.float64), 1e-6))
        scale, shift = fit_scale_shift(pred, target.astype(np.float32), valid)
        return np.exp(np.clip(pred64 * scale + shift, -9.0, 9.0))
    if align == "median":
        p = np.sort(pred64[valid])
        g = np.sort(gt[valid].astype(np.float64))
        return pred64 * (g[(g.size - 1) // 2] / p[(p.size - 1) // 2])
    return pred64


def boundary_regions(gt: np.ndarray, valid: np.ndarray, window: int, edge_ratio: float,
                     flat_ratio: float, min_neighbors: int) -> dict[str, np.ndarray]:
    """Edge / interior / mid / sparse labels for valid pixels, from valid GT only."""
    half = window // 2
    log_gt = np.where(valid, np.log(np.maximum(gt, 1e-6)), np.nan)
    padded = np.pad(log_gt, half, constant_values=np.nan)
    view = np.lib.stride_tricks.sliding_window_view(padded, (window, window))
    present = np.isfinite(view)
    neighbours = present.sum(axis=(2, 3)) - 1
    hi = np.where(present, view, -np.inf).max(axis=(2, 3))
    lo = np.where(present, view, np.inf).min(axis=(2, 3))
    span = hi - lo
    supported = valid & (neighbours >= min_neighbors)
    edge = supported & (span > math.log(edge_ratio))
    interior = supported & (span < math.log(flat_ratio))
    return {
        "edge": edge,
        "interior": interior,
        "between": supported & ~edge & ~interior,
        "sparse": valid & ~supported,
    }


def add(acc: dict, key: str, weight: float, absrel: np.ndarray, sq: np.ndarray, d1: np.ndarray) -> None:
    bucket = acc[key]
    bucket["w"] += weight * absrel.size
    bucket["w_absrel"] += weight * float(absrel.sum())
    bucket["w_sq"] += weight * float(sq.sum())
    bucket["w_d1"] += weight * float(d1.sum())
    bucket["pixels"] += int(absrel.size)


def table(accs: dict[str, dict], keys: list[str], names: list[str], frames: int) -> list[dict]:
    """Per-bucket stats per model, plus each bucket's share of the first model's deficit."""
    first, second = names
    totals = {
        n: {s: sum(accs[n][k][s] for k in keys) for s in STATS} for n in names
    }
    gap_absrel = totals[first]["w_absrel"] - totals[second]["w_absrel"]
    gap_sq = totals[first]["w_sq"] - totals[second]["w_sq"]
    rows = []
    for key in keys:
        row = {"bucket": key,
               "pixel_share": accs[first][key]["pixels"] / max(totals[first]["pixels"], 1),
               "frame_weight_share": accs[first][key]["w"] / max(frames, 1)}
        for n in names:
            b = accs[n][key]
            row[f"{n}_absrel"] = b["w_absrel"] / b["w"] if b["w"] else float("nan")
            row[f"{n}_rmse"] = math.sqrt(b["w_sq"] / b["w"]) if b["w"] else float("nan")
            row[f"{n}_d1"] = b["w_d1"] / b["w"] if b["w"] else float("nan")
            # contribution to the headline macro AbsRel / mean per-frame MSE
            row[f"{n}_absrel_part"] = b["w_absrel"] / frames
            row[f"{n}_mse_part"] = b["w_sq"] / frames
        da = accs[first][key]["w_absrel"] - accs[second][key]["w_absrel"]
        ds = accs[first][key]["w_sq"] - accs[second][key]["w_sq"]
        row["absrel_gap_share"] = da / gap_absrel if gap_absrel else float("nan")
        row["mse_gap_share"] = ds / gap_sq if gap_sq else float("nan")
        rows.append(row)
    return rows


def print_table(title: str, rows: list[dict], names: list[str]) -> None:
    a, b = names
    print(f"\n=== {title} ===")
    print(f"{'bucket':>12} {'pix%':>6} | {a + ' AbsRel':>13} {b + ' AbsRel':>13} | "
          f"{a + ' RMSE':>11} {b + ' RMSE':>11} | {a + ' d1':>9} {b + ' d1':>9} | "
          f"{'AbsRel gap%':>11} {'MSE gap%':>9}")
    for r in rows:
        print(f"{r['bucket']:>12} {100 * r['pixel_share']:6.1f} | "
              f"{100 * r[f'{a}_absrel']:13.2f} {100 * r[f'{b}_absrel']:13.2f} | "
              f"{r[f'{a}_rmse']:11.3f} {r[f'{b}_rmse']:11.3f} | "
              f"{100 * r[f'{a}_d1']:9.2f} {100 * r[f'{b}_d1']:9.2f} | "
              f"{100 * r['absrel_gap_share']:11.1f} {100 * r['mse_gap_share']:9.1f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True,
                    help="name=<raw prediction dir>:<align>; exactly two, the one being diagnosed first")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--depth-scale", type=float, default=256.0)
    ap.add_argument("--min-depth", type=float, default=1e-3)
    ap.add_argument("--max-depth", type=float, default=80.0)
    ap.add_argument("--bins", default="0,5,10,20,30,40,60,80", help="GT depth bin edges in metres")
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--edge-ratio", type=float, default=1.25)
    ap.add_argument("--flat-ratio", type=float, default=1.05)
    ap.add_argument("--min-neighbors", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N frames only")
    args = ap.parse_args()

    models = []
    for spec in args.model:
        name, rest = spec.split("=", 1)
        directory, align = rest.rsplit(":", 1)
        if align not in ALIGNS:
            raise SystemExit(f"unknown align {align!r} for {name}")
        models.append((name, Path(directory), align))
    if len(models) != 2:
        raise SystemExit("give exactly two --model entries")
    names = [m[0] for m in models]
    edges = [float(x) for x in args.bins.split(",")]
    bin_keys = [f"{edges[i]:g}-{edges[i + 1]:g}m" for i in range(len(edges) - 1)]
    region_keys = ["edge", "between", "interior", "sparse"]

    samples, info = load_manifest(args.manifest, args.data_root, allow_legacy_depth_path=True)
    if args.limit:
        samples = samples[: args.limit]
    zero = lambda: {s: 0.0 for s in STATS}  # noqa: E731
    by_bin = {n: defaultdict(zero) for n in names}
    by_region = {n: defaultdict(zero) for n in names}
    per_frame: list[dict] = []
    missing = 0
    worst_check = 0.0

    for index, sample in enumerate(samples):
        paths = [d / f"{sample.sample_id}.npy" for _, d, _ in models]
        if not all(p.is_file() for p in paths):
            missing += 1
            continue
        _, gt = load_ms2_gt(sample.thermal_gt_path, args.depth_scale)
        valid = official_valid_mask(gt, args.min_depth, args.max_depth)
        if not valid.any():
            continue
        g = gt[valid].astype(np.float64)
        weight = 1.0 / g.size
        bins = np.clip(np.digitize(g, edges) - 1, 0, len(bin_keys) - 1)
        regions = boundary_regions(gt, valid, args.window, args.edge_ratio, args.flat_ratio,
                                   args.min_neighbors)
        region_of = {k: regions[k][valid] for k in region_keys}
        row = {"id": sample.sample_id, "sequence": sample.sequence, "condition": sample.condition,
               "valid_pixels": int(g.size)}
        for (name, _, align), path in zip(models, paths):
            pred = resize_dense_prediction(collapse_channels(np.load(path, allow_pickle=False)), gt.shape)
            p = np.clip(aligned_depth(pred, gt, valid, align)[valid], args.min_depth, args.max_depth)
            diff = p - g
            absrel = np.abs(diff) / g
            sq = diff * diff
            d1 = (np.maximum(g / p, p / g) < 1.25).astype(np.float64)
            official = evaluate_sample(pred, gt, align=align, min_depth=args.min_depth,
                                       max_depth=args.max_depth)["abs_rel"]
            mismatch = abs(float(absrel.mean()) - official)
            worst_check = max(worst_check, mismatch)
            if mismatch > 1e-9:
                raise SystemExit(f"{name} {sample.sample_id}: AbsRel {absrel.mean()} != official {official}")
            for b, key in enumerate(bin_keys):
                sel = bins == b
                if sel.any():
                    add(by_bin[name], key, weight, absrel[sel], sq[sel], d1[sel])
            for key in region_keys:
                sel = region_of[key]
                if sel.any():
                    add(by_region[name], key, weight, absrel[sel], sq[sel], d1[sel])
            row[f"{name}_abs_rel"] = float(absrel.mean())
            row[f"{name}_rmse"] = float(math.sqrt(sq.mean()))
            row[f"{name}_a1"] = float(d1.mean())
        per_frame.append(row)
        if (index + 1) % 500 == 0:
            print(f"[diag] {index + 1}/{len(samples)}", flush=True)

    frames = len(per_frame)
    if not frames:
        raise SystemExit("no frame had predictions from both models")
    print(f"[diag] {frames} frames scored, {missing} skipped (prediction missing for a model); "
          f"largest |AbsRel - official| = {worst_check:.2e}")

    bin_rows = table(by_bin, bin_keys, names, frames)
    region_rows = table(by_region, region_keys, names, frames)
    a, b = names
    print_table("1. by GT depth", bin_rows, names)
    print_table("2. edge / interior (window %d, edge > x%.2f, interior < x%.2f, sparse < %d neighbours)"
                % (args.window, args.edge_ratio, args.flat_ratio, args.min_neighbors), region_rows, names)

    # 3. paired frames
    frame_summary = {}
    print("\n=== 3. paired frames (difference = %s - %s; positive = %s worse) ===" % (a, b, a))
    for metric, sign in (("abs_rel", 1), ("rmse", 1), ("a1", -1)):
        diff = np.array([sign * (r[f"{a}_{metric}"] - r[f"{b}_{metric}"]) for r in per_frame])
        total = float(diff.sum())
        order = np.sort(diff)[::-1]
        tail = {}
        for pct in (1, 5, 10, 25):
            k = max(1, int(round(frames * pct / 100)))
            tail[f"top{pct}pct_share_of_gap"] = float(order[:k].sum() / total) if total else float("nan")
        frame_summary[metric] = {"mean_diff": float(diff.mean()), "median_diff": float(np.median(diff)),
                                 "frames_worse": int((diff > 0).sum()), "frames": frames, **tail}
        print(f"{metric:>8}: mean diff {diff.mean():+.5f}, median {np.median(diff):+.5f}, "
              f"{a} worse on {(diff > 0).sum()}/{frames} frames; worst 1/5/10/25% of frames carry "
              + " / ".join(f"{100 * tail[f'top{p}pct_share_of_gap']:.0f}%" for p in (1, 5, 10, 25))
              + " of the total gap")
    by_seq: dict[str, list[dict]] = defaultdict(list)
    for r in per_frame:
        by_seq[f"{r['sequence']} ({r['condition']})"].append(r)
    seq_rows = []
    print(f"\n{'sequence':>40} {'frames':>7} | {a + ' AbsRel':>12} {b + ' AbsRel':>12} | "
          f"{a + ' RMSE':>10} {b + ' RMSE':>10}")
    for seq, rows in sorted(by_seq.items()):
        s = {"sequence": seq, "frames": len(rows)}
        for n in names:
            s[f"{n}_abs_rel"] = float(np.mean([r[f"{n}_abs_rel"] for r in rows]))
            s[f"{n}_rmse"] = float(np.mean([r[f"{n}_rmse"] for r in rows]))
        seq_rows.append(s)
        print(f"{seq:>40} {len(rows):7d} | {100 * s[f'{a}_abs_rel']:12.2f} {100 * s[f'{b}_abs_rel']:12.2f} | "
              f"{s[f'{a}_rmse']:10.3f} {s[f'{b}_rmse']:10.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "per_frame.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_frame[0]))
        writer.writeheader()
        writer.writerows(per_frame)
    summary = {
        "diagnostic_only": True,
        "manifest": info,
        "models": [{"name": n, "prediction_dir": str(d), "align": al} for n, d, al in models],
        "frames": frames, "skipped_missing_prediction": missing,
        "largest_absrel_mismatch_vs_official": worst_check,
        "settings": {k: getattr(args, k) for k in ("bins", "window", "edge_ratio", "flat_ratio",
                                                   "min_neighbors", "min_depth", "max_depth")},
        "headline": {n: {"abs_rel": float(np.mean([r[f"{n}_abs_rel"] for r in per_frame])),
                         "rmse": float(np.mean([r[f"{n}_rmse"] for r in per_frame])),
                         "a1": float(np.mean([r[f"{n}_a1"] for r in per_frame]))} for n in names},
        "by_depth": bin_rows, "by_region": region_rows, "paired_frames": frame_summary,
        "by_sequence": seq_rows,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nheadline (mean over frames): " + "; ".join(
        f"{n} AbsRel {100 * summary['headline'][n]['abs_rel']:.2f} RMSE {summary['headline'][n]['rmse']:.3f} "
        f"d1 {100 * summary['headline'][n]['a1']:.2f}" for n in names))
    print(f"[diag] wrote {args.out}/summary.json and per_frame.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
