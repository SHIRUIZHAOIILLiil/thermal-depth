"""Far-field bias by sequence, and under three per-frame fits. CPU only, val only.

DIAGNOSTIC ONLY. Reads raw predictions already saved by slurm/diagnose_val.sbatch;
no inference, never test.

Two questions:

1. Is the far under-estimate common to every val sequence and most frames, or carried
   by a few? Per sequence x GT-depth bin, and the per-frame distribution of the far
   (>= --split-depth) median log error.
2. Does the per-frame alignment trade the far field away? Each frame is aligned three
   ways, all in the model's own official space (ssi_log for ours, ssi for NeWCRF),
   and the same pixels are scored:
     all   -- the official fit on every valid pixel (the reporting protocol);
     near  -- fit only on GT < split-depth, applied to the whole frame;
     far   -- fit only on GT >= split-depth, applied to the whole frame.
   `near` and `far` are diagnostics, never a reporting protocol: `far` in particular
   uses the far GT to fit the far prediction it is then scored on.

Per frame, the `all` fit's AbsRel is checked against the official evaluator and a
mismatch aborts. Frames with fewer than --min-fit-pixels pixels on a side are left out
of that side's fit and recorded with IDs and reasons. The fit-comparison table uses
only the shared successful cohort across ALL models and fits. Full all-fit results
are retained separately as cohort=available. A failed all fit aborts.

Medians are taken over at most --cap pixels per frame and bin (evenly spaced), so every
frame contributes; counts, too-near shares and AbsRel use every pixel.
Both pixel-pooled and equal-frame AbsRel are exported. Far-error tails rank frames
by far MSE and report pixel-SSE and frame-MSE contribution separately.

    python tools/far_bias_breakdown.py --manifest $VAL --data-root $IRIS_MS2_ROOT \
        --model l100=$A1/ours/raw_predictions:ssi_log --model tmax1=$A2/ours/raw_predictions:ssi_log \
        --model newcrf=$A1/newcrf/raw_predictions/newcrf:ssi --out $OUT
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    OfficialProtocolError,
    collapse_channels,
    evaluate_sample,
    official_valid_mask,
)
from ms2_eval.resize import resize_dense_prediction  # noqa: E402
from tools.diagnose_error_sources import aligned_depth  # noqa: E402
from tools.far_structure import structure, paired_summary, structure_summary  # noqa: E402

FITS = ("all", "near", "far")


def new_cell():
    return {"pixels": 0, "too_near": 0, "abs_rel_sum": 0.0, "log_err": [],
            "frames": 0, "frame_abs_rel_sum": 0.0}


def add(cell, log_err, abs_rel, cap):
    cell["pixels"] += int(log_err.size)
    cell["too_near"] += int((log_err < 0).sum())
    cell["abs_rel_sum"] += float(abs_rel.sum())
    cell["frames"] += 1
    cell["frame_abs_rel_sum"] += float(abs_rel.mean())
    sample = log_err if log_err.size <= cap else log_err[np.linspace(0, log_err.size - 1, cap).astype(int)]
    cell["log_err"].append(sample)


def finish(cell):
    if not cell["pixels"]:
        return None
    e = np.concatenate(cell["log_err"])
    return {"pixels": cell["pixels"], "median_ratio": float(math.exp(np.median(e))),
            "median_log_err": float(np.median(e)), "too_near_share": cell["too_near"] / cell["pixels"],
            "frames": cell["frames"],
            "pixel_abs_rel": cell["abs_rel_sum"] / cell["pixels"],
            "frame_mean_abs_rel": cell["frame_abs_rel_sum"] / cell["frames"]}


def error_tail(records):
    """Rank by far per-frame MSE; report both pixel SSE and frame-MSE shares."""
    ranked = sorted(records, key=lambda r: (-r[6], r[1]))
    k = max(1, math.ceil(0.1 * len(ranked)))
    selected = ranked[:k]
    def share(column):
        total = sum(r[column] for r in ranked)
        return sum(r[column] for r in selected) / total if total > 0 else None
    return {"ranking": "far per-frame MSE descending; ID ascending for ties",
            "frames": len(ranked), "tail_frames": len(selected),
            "pixel_sse_share": share(5), "frame_mse_share": share(6),
            "ids": [r[1] for r in selected]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True, help="name=<raw prediction dir>:<align>")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bins", default="0,10,20,30,40,60,80")
    ap.add_argument("--split-depth", type=float, default=40.0)
    ap.add_argument("--min-fit-pixels", type=int, default=50)
    ap.add_argument("--frame-min-far", type=int, default=50,
                    help="frames need this many far pixels to enter the per-frame distribution")
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--depth-scale", type=float, default=256.0)
    ap.add_argument("--min-depth", type=float, default=1e-3)
    ap.add_argument("--max-depth", type=float, default=80.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--structure-reference", help="enable structure and paired MSE-gap diagnostics against this model")
    args = ap.parse_args()
    if min(args.cap, args.min_fit_pixels, args.frame_min_far) < 1 or args.limit < 0:
        ap.error("cap and pixel thresholds must be positive; limit must be nonnegative")

    models = []
    for spec in args.model:
        name, rest = spec.split("=", 1)
        directory, align = rest.rsplit(":", 1)
        if align not in ("ssi", "ssi_log"):
            ap.error("this diagnostic supports only ssi / ssi_log")
        models.append((name, Path(directory), align))
    if len({n for n, _, _ in models}) != len(models):
        ap.error("model names must be unique")
    if args.structure_reference and args.structure_reference not in {n for n, _, _ in models}:
        ap.error("structure-reference must name one of the supplied models")
    edges = [float(x) for x in args.bins.split(",")]
    if (len(edges) < 2 or not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0)
            or edges[0] > args.min_depth or edges[-1] < args.max_depth
            or not args.min_depth < args.split_depth < args.max_depth):
        ap.error("bins must increase and cover the evaluation range; split-depth must be inside it")
    bin_keys = [f"{edges[i]:g}-{edges[i + 1]:g}m" for i in range(len(edges) - 1)]
    far_key = f">={args.split_depth:g}m"

    samples, info = load_manifest(args.manifest, args.data_root, allow_legacy_depth_path=True)
    if args.limit:
        samples = samples[: args.limit]
    if not samples or len({s.sample_id for s in samples}) != len(samples):
        ap.error("manifest must contain nonempty, unique sample IDs")
    for name, directory, _ in models:
        missing = [s.sample_id for s in samples if not (directory / f"{s.sample_id}.npy").is_file()]
        if missing:
            raise SystemExit(f"{name}: missing {len(missing)}/{len(samples)} predictions, e.g. {missing[:5]}")

    # cells[model][fit][sequence][bin]; sequence "ALL" pools every sequence
    cells = {n: {f: defaultdict(lambda: defaultdict(new_cell)) for f in FITS} for n, _, _ in models}
    paired_cells = {n: {f: defaultdict(lambda: defaultdict(new_cell)) for f in FITS} for n, _, _ in models}
    exclusions = []
    paired_ids = []
    structure_rows = []
    skipped = {n: {f: 0 for f in FITS} for n, _, _ in models}
    frame_far = {n: [] for n, _, _ in models}   # (sequence, id, far pixels, median log err, too-near pixels)
    worst = 0.0

    for index, sample in enumerate(samples):
        _, gt = load_ms2_gt(sample.thermal_gt_path, args.depth_scale)
        valid = official_valid_mask(gt, args.min_depth, args.max_depth)
        if not valid.any():
            raise SystemExit(f"{sample.sample_id}: no valid GT pixels")
        g = gt[valid].astype(np.float64)
        bins = np.clip(np.digitize(g, edges) - 1, 0, len(bin_keys) - 1)
        near_mask = valid & (gt < args.split_depth)
        far_mask = valid & (gt >= args.split_depth)
        far_sel = g >= args.split_depth
        pending = []
        for name, directory, align in models:
            raw = resize_dense_prediction(collapse_channels(np.load(directory / f"{sample.sample_id}.npy")), gt.shape)
            if not np.isfinite(raw[valid]).all():
                raise SystemExit(f"{name} {sample.sample_id}: non-finite raw prediction")
            for fit in FITS:
                mask = {"all": valid, "near": near_mask, "far": far_mask}[fit]
                if fit != "all" and int(mask.sum()) < args.min_fit_pixels:
                    skipped[name][fit] += 1
                    exclusions.append(dict(model=name, fit=fit, sequence=sample.sequence,
                                           id=sample.sample_id, reason="insufficient_fit_pixels",
                                           fit_pixels=int(mask.sum())))
                    continue
                try:
                    dense = aligned_depth(raw, gt, mask, align)
                except OfficialProtocolError as exc:
                    if fit == "all":
                        raise SystemExit(f"{name} {sample.sample_id}: all fit failed: {exc}") from exc
                    skipped[name][fit] += 1
                    exclusions.append(dict(model=name, fit=fit, sequence=sample.sequence,
                                           id=sample.sample_id, reason=str(exc), fit_pixels=int(mask.sum())))
                    continue
                p = np.clip(dense[valid], args.min_depth, args.max_depth)
                log_err = np.log(p) - np.log(g)
                abs_rel = np.abs(p - g) / g
                pending.append((name, fit, log_err, abs_rel))
                if fit == "all":
                    official = evaluate_sample(raw, gt, align=align, min_depth=args.min_depth,
                                               max_depth=args.max_depth)["abs_rel"]
                    worst = max(worst, abs(float(abs_rel.mean()) - official))
                    if worst > 1e-9:
                        raise SystemExit(f"{name} {sample.sample_id}: AbsRel differs from the official evaluator")
                    if args.structure_reference:
                        seed = int.from_bytes(hashlib.sha256(sample.sample_id.encode()).digest()[:8], 'little')
                        for lower in sorted({args.split_depth, 60.0}):
                            sel = g >= lower
                            if int(sel.sum()) < args.frame_min_far:
                                continue
                            stats = structure(g[sel], dense[valid][sel], raw[valid][sel], seed=seed)
                            sq = np.square(p[sel] - g[sel])
                            structure_rows.append(dict(model=name, id=sample.sample_id, sequence=sample.sequence,
                                                       band=f'>={lower:g}m', pixels=int(sel.sum()),
                                                       mse=float(sq.mean()),
                                                       full_frame_mse_contribution=float(sq.sum() / g.size),
                                                       clipped_share=float(np.mean((dense[valid][sel] < args.min_depth)
                                                                                    | (dense[valid][sel] > args.max_depth))),
                                                       **stats))
                    if int(far_sel.sum()) >= args.frame_min_far:
                        e = log_err[far_sel]
                        sq = np.square(p[far_sel] - g[far_sel])
                        frame_far[name].append((sample.sequence, sample.sample_id, int(e.size),
                                                float(np.median(e)), int((e < 0).sum()),
                                                float(sq.sum()), float(sq.mean())))
        common = len(pending) == len(models) * len(FITS)
        if common:
            paired_ids.append(sample.sample_id)
        for name, fit, log_err, abs_rel in pending:
            for destination in ([cells, paired_cells] if common else [cells]):
                for seq in (sample.sequence, "ALL"):
                    for b, key in enumerate(bin_keys):
                        sel = bins == b
                        if sel.any():
                            add(destination[name][fit][seq][key], log_err[sel], abs_rel[sel], args.cap)
                    if far_sel.any():
                        add(destination[name][fit][seq][far_key], log_err[far_sel], abs_rel[far_sel], args.cap)
        if (index + 1) % 2000 == 0:
            print(f"[far-bias] {index + 1}/{len(samples)}", flush=True)

    print(f"[far-bias] {len(samples)} frames; largest |AbsRel - official| for the 'all' fit = {worst:.2e}")
    for name, _, _ in models:
        print(f"[far-bias] {name}: frames left out of the near / far fit: "
              f"{skipped[name]['near']} / {skipped[name]['far']}")

    rows = []
    for cohort, source in (("available", cells), ("paired", paired_cells)):
        for name, _, align in models:
            for fit in FITS:
                for seq, by_bin in source[name][fit].items():
                    for key in bin_keys + [far_key]:
                        stats = finish(by_bin[key]) if key in by_bin else None
                        if stats:
                            rows.append({"cohort": cohort, "model": name, "align": align,
                                         "fit": fit, "sequence": seq, "bin": key, **stats})

    def lookup(name, fit, seq, key, cohort="paired"):
        for r in rows:
            if (r["model"], r["fit"], r["sequence"], r["bin"], r["cohort"]) == (name, fit, seq, key, cohort):
                return r
        return None

    print(f"\n=== 1. PAIRED {len(paired_ids)}/{len(samples)} frames: median pred/gt | too-near % | frame-mean AbsRel %, per fit ===")
    if not paired_ids:
        print("WARNING: no shared successful frames; no fit-to-fit comparison is available.")
    for name, _, align in models:
        print(f"\n[{name}] ({align})  " + "  ".join(f"{fit:>26}" for fit in FITS))
        for key in bin_keys + [far_key]:
            cells_txt = []
            for fit in FITS:
                r = lookup(name, fit, "ALL", key)
                cells_txt.append(f"{r['median_ratio']:6.3f} | {100 * r['too_near_share']:5.1f} | "
                                 f"{100 * r['frame_mean_abs_rel']:6.2f}" if r else f"{'-':>26}")
            print(f"[{name}] {key:>8}  " + "  ".join(f"{c:>26}" for c in cells_txt))

    sequences = sorted({r["sequence"] for r in rows if r["sequence"] != "ALL"})
    print(f"\n=== 2. per sequence, official fit, far ({far_key}) and 60-80 m: median pred/gt | too-near % ===")
    print(f"{'sequence':>22} " + " ".join(f"{n + ' ' + far_key:>22} {n + ' 60-80m':>20}" for n, _, _ in models))
    for seq in sequences:
        parts = []
        for name, _, _ in models:
            for key, width in ((far_key, 22), ("60-80m", 20)):
                r = lookup(name, "all", seq, key, "available")
                parts.append(f"{r['median_ratio']:.3f} | {100 * r['too_near_share']:4.1f}".rjust(width) if r else "-".rjust(width))
        print(f"{seq:>22} " + " ".join(parts))

    print(f"\n=== 3. per frame (official fit, frames with >= {args.frame_min_far} far pixels): "
          f"far median pred/gt ===")
    distribution = {}
    for name, _, _ in models:
        recs = frame_far[name]
        if not recs:
            continue
        med = np.array([r[3] for r in recs])
        near_px = np.array([r[4] for r in recs], dtype=np.float64)
        order = np.argsort(med)                       # most under-estimated first
        k = max(1, int(round(0.1 * len(recs))))
        quant = {f"p{q}": float(math.exp(np.percentile(med, q))) for q in (10, 25, 50, 75, 90)}
        d = {"frames": len(recs), **quant,
             "frames_below_0.9": int((med < math.log(0.9)).sum()),
             "frames_below_0.8": int((med < math.log(0.8)).sum()),
             "worst10pct_share_of_too_near_far_pixels": float(near_px[order[:k]].sum() / max(near_px.sum(), 1))}
        per_seq = {}
        for seq in sequences:
            m = np.array([r[3] for r in recs if r[0] == seq])
            if m.size:
                per_seq[seq] = {"frames": int(m.size), "median": float(math.exp(np.median(m))),
                                "frames_below_0.9": int((m < math.log(0.9)).sum())}
        d["per_sequence"] = per_seq
        d["squared_error_tail"] = error_tail(recs)
        distribution[name] = d
        print(f"[{name}] {len(recs)} frames; frame-median pred/gt p10/p25/p50/p75/p90 = "
              + " / ".join(f"{quant[f'p{q}']:.3f}" for q in (10, 25, 50, 75, 90))
              + f"; below 0.9: {d['frames_below_0.9']}, below 0.8: {d['frames_below_0.8']}; "
              f"worst 10% of frames hold {100 * d['worst10pct_share_of_too_near_far_pixels']:.0f}% "
              f"of too-near far pixels")
        for seq, s in per_seq.items():
            print(f"[{name}]   {seq}: {s['frames']} frames, median {s['median']:.3f}, below 0.9: {s['frames_below_0.9']}")
        tail_display = {k: v for k, v in d['squared_error_tail'].items() if k != 'ids'}
        print(f"[{name}] far MSE-ranked top 10%: {json.dumps(tail_display)} (IDs in JSON)")

    args.out.mkdir(parents=True, exist_ok=True)
    if args.structure_reference:
        gaps = paired_summary(structure_rows, args.structure_reference)
        contributions = paired_summary(structure_rows, args.structure_reference, 'full_frame_mse_contribution')
        structural = structure_summary(structure_rows)
        refs = {(r['id'], r['band']): r for r in structure_rows if r['model'] == args.structure_reference}
        for row in structure_rows:
            ref = refs[(row['id'], row['band'])]
            row['mse_gap_vs_reference'] = row['mse'] - ref['mse']
            row['full_frame_mse_contribution_gap'] = row['full_frame_mse_contribution'] - ref['full_frame_mse_contribution']
        if structure_rows:
            with (args.out / 'far_structure_per_frame.csv').open('w', encoding='utf-8', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(structure_rows[0]))
                writer.writeheader()
                writer.writerows(structure_rows)
        report = dict(diagnostic_only=True, reference=args.structure_reference,
                      settings=dict(seed='SHA256(frame ID), first 8 bytes little-endian', cap=4096,
                                    pair_min_gt_gap_m=5., min_pixels=args.frame_min_far,
                                    alignment='whole-frame all fit; no local refit',
                                    structure='unclipped aligned metres; ties averaged in Spearman; undefined=null',
                                    error='official clipped metres; per-frame MSE, not RMSE or whole-frame gap',
                                    gap_shares='signed shares may be negative or >1; positive-excess shares avoid cancellation'),
                      paired_gaps=gaps, full_frame_contribution_gaps=contributions, structure=structural)
        (args.out / 'far_structure_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print('\n=== 4. paired far MSE gaps vs reference (m^2, equal-frame) ===')
        for item in gaps:
            display = {**item, 'top10pct': {k: v for k, v in item['top10pct'].items() if k != 'ids'}}
            print(json.dumps(display))
        print('\n=== 4b. far contribution to whole-frame MSE gap (same eligible frames) ===')
        for item in contributions:
            display = {**item, 'top10pct': {k: v for k, v in item['top10pct'].items() if k != 'ids'}}
            print(json.dumps(display))
        print('\n=== 5. far ordering / range, per-frame medians ===')
        for item in structural:
            print(json.dumps(item))
    with (args.out / "far_bias_by_bin.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.out / "far_bias_per_frame.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "sequence", "id", "far_pixels", "far_median_log_err", "far_too_near_pixels", "far_sse", "far_mse"])
        for name, recs in frame_far.items():
            for r in recs:
                writer.writerow([name, *r])
    with (args.out / "excluded_fits.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "fit", "sequence", "id", "reason", "fit_pixels"])
        writer.writeheader()
        writer.writerows(exclusions)
    (args.out / "far_bias_summary.json").write_text(json.dumps({
        "diagnostic_only": True, "manifest": info,
        "models": [{"name": n, "prediction_dir": str(d), "align": a} for n, d, a in models],
        "settings": {k: getattr(args, k) for k in ("bins", "split_depth", "min_fit_pixels", "frame_min_far", "cap",
                                                   "limit", "depth_scale", "min_depth", "max_depth")},
        "largest_absrel_mismatch_vs_official": worst, "skipped_fits": skipped,
        "paired_ids": paired_ids, "selected_ids": [s.sample_id for s in samples],
        "aggregation": {"pixel_abs_rel": "pooled pixels", "frame_mean_abs_rel": "equal weight per nonempty frame/bin",
                        "median_ratio": "exp(median of capped per-frame log-error samples)); approximate, not equal-frame median",
                        "paired": "intersection of successful all/near/far fits across every model",
                        "available": "each fit uses its own successful frames; not for fit-to-fit comparisons"},
        "per_frame_far": distribution,
    }, indent=2), encoding="utf-8")
    print(f"\n[far-bias] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
