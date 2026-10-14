"""Is the far field cut off by the training target? Two CPU-only checks.

DIAGNOSTIC ONLY (val + train, never test).

The val breakdown (tools/diagnose_error_sources.py) put ~88% of our RMSE deficit to
NeWCRF on the 1.2% of LiDAR pixels at 60-80 m. One candidate: `log_truncnorm`
normalises each frame by the 2%/98% quantiles of log depth and then clips to [-1, 1]
(lotus/utils/ms2_thermal_dataset.py), so every pixel beyond the frame's 98th
percentile is taught the same depth. Two checks decide whether that is what happens:

A. val, our saved predictions: per GT-depth bin, after the official alignment, the
   signed log error and the share of pixels predicted too near. Truncation shows as
   a systematic under-estimate that grows with depth. Optionally the same for a
   second model, and for ours the share of pixels sitting at the top of the frame's
   raw output range (saturation).
B. train, the actual targets: rebuild the completed target with the training
   dataset class itself, take the same quantiles, and count per GT bin how many
   real LiDAR pixels land above the upper bound -- for the current truncnorm_min and
   for alternatives, so a fix can be sized before anything is trained.

    python tools/check_far_truncation.py \
        --val-manifest $M/ms2_val_official3_thermalcap_20260821.jsonl --data-root $IRIS_MS2_ROOT \
        --pred ours=$A/ours/raw_predictions:ssi_log --pred newcrf=$A/newcrf/raw_predictions/newcrf:ssi \
        --train-manifest $M/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl \
        --pseudo-dir $F/pseudo_gt/official_train/calibrated_pseudo_depth --out $A/far_check
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ms2_eval.io import load_manifest, load_ms2_gt  # noqa: E402
from ms2_eval.official_protocol import collapse_channels, official_valid_mask  # noqa: E402
from ms2_eval.resize import resize_dense_prediction  # noqa: E402
from tools.diagnose_error_sources import aligned_depth  # noqa: E402


def bins_of(depth: np.ndarray, edges: list[float]) -> np.ndarray:
    return np.clip(np.digitize(depth, edges) - 1, 0, len(edges) - 2)


def check_val(args, edges, keys) -> dict:
    samples, _ = load_manifest(args.val_manifest, args.data_root, allow_legacy_depth_path=True)
    samples = samples[:: args.val_stride]
    models = []
    for spec in args.pred:
        name, rest = spec.split("=", 1)
        directory, align = rest.rsplit(":", 1)
        models.append((name, Path(directory), align))
    acc = {n: {k: {"n": 0, "log_err": [], "under": 0, "sat": 0} for k in keys} for n, _, _ in models}
    frames = 0
    for index, sample in enumerate(samples):
        paths = [d / f"{sample.sample_id}.npy" for _, d, _ in models]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise SystemExit(f"missing prediction(s) for {sample.sample_id}: {missing}")
        _, gt = load_ms2_gt(sample.thermal_gt_path, args.depth_scale)
        valid = official_valid_mask(gt, args.min_depth, args.max_depth)
        if not valid.any():
            continue
        g = gt[valid].astype(np.float64)
        b = bins_of(g, edges)
        for (name, _, align), path in zip(models, paths):
            raw = resize_dense_prediction(collapse_channels(np.load(path, allow_pickle=False)), gt.shape)
            p = np.clip(aligned_depth(raw, gt, valid, align)[valid], args.min_depth, args.max_depth)
            log_err = np.log(p) - np.log(g)
            # Saturation in the raw output: within 1% of the frame's raw range of
            # its maximum. For a log/disparity/depth output alike, a far region
            # pressed against a ceiling shows up here.
            r = raw[valid].astype(np.float64)
            top = r.max() - 0.01 * (r.max() - r.min())
            far_end = r >= top if align != "ssi_disparity" else r <= r.min() + 0.01 * (r.max() - r.min())
            for i, key in enumerate(keys):
                sel = b == i
                if not sel.any():
                    continue
                a = acc[name][key]
                a["n"] += int(sel.sum())
                # Subsample per frame so the median stays cheap and every frame counts.
                e = log_err[sel]
                a["log_err"].append(e if e.size <= 200 else e[np.linspace(0, e.size - 1, 200).astype(int)])
                a["under"] += int((e < 0).sum())
                a["sat"] += int(far_end[sel].sum())
        frames += 1
        if (index + 1) % 2000 == 0:
            print(f"[A] {index + 1}/{len(samples)}", flush=True)

    out = {"frames": frames, "stride": args.val_stride, "models": {}}
    print(f"\n=== A. val: signed error per GT bin ({frames} frames, stride {args.val_stride}) ===")
    print("    log err = log(pred/gt) after the official alignment; negative = predicted too near")
    for name, _, _ in models:
        print(f"\n[{name}]  {'bin':>8} {'pixels':>10} {'median log err':>15} {'x(median)':>10} "
              f"{'too near %':>11} {'raw at top %':>13}")
        rows = []
        for key in keys:
            a = acc[name][key]
            if not a["n"]:
                continue
            e = np.concatenate(a["log_err"])
            row = {"bin": key, "pixels": a["n"], "median_log_err": float(np.median(e)),
                   "mean_log_err": float(np.mean(e)), "too_near_share": a["under"] / a["n"],
                   "raw_at_top_share": a["sat"] / a["n"]}
            rows.append(row)
            print(f"[{name}]  {key:>8} {a['n']:10d} {row['median_log_err']:+15.3f} "
                  f"{math.exp(row['median_log_err']):10.3f} {100 * row['too_near_share']:11.1f} "
                  f"{100 * row['raw_at_top_share']:13.1f}")
        out["models"][name] = rows
    return out


def load_dataset_class():
    path = ROOT / "lotus" / "utils" / "ms2_thermal_dataset.py"
    spec = importlib.util.spec_from_file_location("ms2_thermal_dataset_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MS2ThermalDataset


def check_train(args, edges, keys) -> dict:
    import torch

    dataset_cls = load_dataset_class()
    ds = dataset_cls(args.train_manifest, args.data_root, args.pseudo_dir, transform=None,
                     norm_type="log_truncnorm", truncnorm_min=args.truncnorm_min[0],
                     sky_mask_dir=args.sky_mask_dir)
    rng = np.random.default_rng(args.seed)
    picks = sorted(rng.choice(len(ds), size=min(args.train_frames, len(ds)), replace=False).tolist())
    q_list = args.truncnorm_min
    count = {q: {k: [0, 0] for k in keys} for q in q_list}       # [clipped high, total lidar]
    target_clip = {q: [0, 0] for q in q_list}                    # [clipped high, valid target px]
    upper_m = {q: [] for q in q_list}
    for index, idx in enumerate(picks):
        dense, gt_depth, gt_mask = ds._completed_depth(ds.rows[idx])
        valid = ds._get_valid_mask(dense) & (dense > 0)
        log_depth = torch.log(dense.clamp(min=1e-6))
        lidar = gt_mask.bool()
        g = gt_depth[lidar].numpy().astype(np.float64)
        lidar_log = log_depth[lidar].numpy()
        b = bins_of(g, edges)
        for q in q_list:
            hi = float(torch.quantile(log_depth[valid], 1 - q)) if q > 0 else float(log_depth[valid].max())
            upper_m[q].append(math.exp(hi))
            target_clip[q][0] += int((log_depth[valid] > hi).sum())
            target_clip[q][1] += int(valid.sum())
            above = lidar_log > hi
            for i, key in enumerate(keys):
                sel = b == i
                count[q][key][0] += int(above[sel].sum())
                count[q][key][1] += int(sel.sum())
        if (index + 1) % 500 == 0:
            print(f"[B] {index + 1}/{len(picks)}", flush=True)

    print(f"\n=== B. train targets ({len(picks)} frames, seed {args.seed}"
          f"{', sky rule on' if args.sky_mask_dir else ''}): LiDAR pixels above the upper bound ===")
    print(f"{'truncnorm_min':>14} {'upper bound m (p10/p50/p90)':>30} {'target px clipped %':>20} | "
          + " ".join(f"{k:>9}" for k in keys))
    out = {"frames": len(picks), "seed": args.seed, "sky_rule": bool(args.sky_mask_dir), "by_q": {}}
    for q in q_list:
        u = np.asarray(upper_m[q])
        shares = {k: (count[q][k][0] / count[q][k][1] if count[q][k][1] else float("nan")) for k in keys}
        out["by_q"][str(q)] = {
            "upper_bound_m": {"p10": float(np.percentile(u, 10)), "p50": float(np.median(u)),
                              "p90": float(np.percentile(u, 90))},
            "target_pixels_clipped_high": target_clip[q][0] / max(target_clip[q][1], 1),
            "lidar_clipped_high_by_bin": shares,
            "lidar_pixels_by_bin": {k: count[q][k][1] for k in keys},
        }
        print(f"{q:>14g} {np.percentile(u, 10):9.1f} / {np.median(u):5.1f} / {np.percentile(u, 90):5.1f}"
              f"{100 * target_clip[q][0] / max(target_clip[q][1], 1):21.2f} | "
              + " ".join(f"{100 * shares[k]:8.1f}%" for k in keys))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--val-manifest", type=Path)
    ap.add_argument("--pred", action="append", default=[], help="name=<raw prediction dir>:<align>")
    ap.add_argument("--val-stride", type=int, default=1)
    ap.add_argument("--train-manifest", type=Path)
    ap.add_argument("--pseudo-dir", type=Path)
    ap.add_argument("--sky-mask-dir", type=Path, default=None)
    ap.add_argument("--train-frames", type=int, default=2000)
    ap.add_argument("--truncnorm-min", type=float, nargs="+", default=[0.02, 0.01, 0.005, 0.0],
                    help="first value = what training used; the rest are alternatives to size a fix")
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--bins", default="0,10,20,30,40,60,80")
    ap.add_argument("--depth-scale", type=float, default=256.0)
    ap.add_argument("--min-depth", type=float, default=1e-3)
    ap.add_argument("--max-depth", type=float, default=80.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    edges = [float(x) for x in args.bins.split(",")]
    keys = [f"{edges[i]:g}-{edges[i + 1]:g}m" for i in range(len(edges) - 1)]
    result = {"diagnostic_only": True, "bins": keys}
    if args.pred:
        if args.val_manifest is None:
            raise SystemExit("--pred needs --val-manifest")
        result["val"] = check_val(args, edges, keys)
    if args.train_manifest is not None:
        if args.pseudo_dir is None:
            raise SystemExit("--train-manifest needs --pseudo-dir")
        result["train"] = check_train(args, edges, keys)
    if len(result) == 2:
        raise SystemExit("nothing to do: give --pred and/or --train-manifest")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "far_truncation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[far] wrote {args.out}/far_truncation.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
