"""Isolate GT DENSITY as the cause of the RGBDT500 caption result.

RGBDT500 differs from MS2 on many axes at once (scene domain, depth range,
registration, model strength, steps, resolution), so a cross-dataset comparison
cannot attribute the flipped caption verdict to GT density. This script removes
every one of those confounds by re-scoring the SAME predictions against the SAME
scenes, changing only how many GT pixels the evaluator is allowed to see.

Method: keep a random fraction of the valid GT pixels (per-sample seeded, and
identical across the four cells so the pairing stays exact), zero the rest so
`official_valid_mask` drops them, then run the unchanged official protocol.

Reading the sweep:
  * caption effect shrinks / reverses as density falls toward MS2's ~29%
        -> dense GT is what enabled the effect; the MS2 null was a GT artefact
  * caption effect survives at MS2 density
        -> GT density is NOT the explanation; look at the other six differences

    python tools/rgbdt500_gt_density_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ms2_eval.aggregate import paired_comparison  # noqa: E402
from ms2_eval.official_protocol import (  # noqa: E402
    OfficialProtocolError,
    evaluate_sample,
    official_valid_mask,
)
from ms2_eval.resize import resize_dense_prediction  # noqa: E402

DATA_ROOT = Path("/mnt/e/dataset/RGBDT500/clean_test")
OUT = ROOT / "outputs" / "lotus_line_v2"
CELLS = {
    "emptytrain_empty": OUT / "rgbdt500_eval_emptytrain_empty",
    "emptytrain_correct": OUT / "rgbdt500_eval_emptytrain_correct",
    "capttrain_empty": OUT / "rgbdt500_eval_capttrain_empty",
    "capttrain_correct": OUT / "rgbdt500_eval_capttrain_correct",
}
DENSITIES = (1.0, 0.60, 0.29, 0.15, 0.05)  # 0.29 = MS2 filtered-LiDAR density
MIN_DEPTH, MAX_DEPTH, DEPTH_SCALE = 0.1, 20.0, 1000.0
MIN_VALID_PIXELS = 500
SEED = 20260703

COMPARISONS = (
    ("注入价值(empty训)", "emptytrain_correct", "emptytrain_empty"),
    ("注入价值(capt训)", "capttrain_correct", "capttrain_empty"),
    ("caption训练净收益", "capttrain_correct", "emptytrain_correct"),
    ("总效果", "capttrain_correct", "emptytrain_empty"),
)


def main() -> int:
    manifest = CELLS["emptytrain_empty"] / "selected_manifest.jsonl"
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"样本数 {len(rows)}   密度扫描 {DENSITIES}   (0.29 = MS2 密度)")

    # results[density][cell] -> list of per-sample metric dicts
    results = {d: {c: [] for c in CELLS} for d in DENSITIES}
    rng_base = np.random.default_rng(SEED)
    seeds = rng_base.integers(0, 2**31 - 1, size=len(rows))

    for index, row in enumerate(rows):
        gt_path = DATA_ROOT / row["thermal_depth_path"]
        with Image.open(gt_path) as image:
            gt = np.asarray(image).astype(np.float32) / DEPTH_SCALE
        base_valid = official_valid_mask(gt, MIN_DEPTH, MAX_DEPTH)
        n_valid = int(base_valid.sum())
        if n_valid < MIN_VALID_PIXELS:
            continue

        preds = {}
        for cell, directory in CELLS.items():
            raw = np.load(directory / "raw_predictions" / f"{row['id']}.npy", allow_pickle=False)
            pred = np.squeeze(raw).astype(np.float32)
            preds[cell] = resize_dense_prediction(pred, gt.shape)

        # one mask per (sample, density), shared by all four cells -> pairing exact
        rng = np.random.default_rng(int(seeds[index]))
        keep_draw = rng.random(gt.shape)
        for density in DENSITIES:
            if density >= 1.0:
                gt_sparse = gt
            else:
                keep = base_valid & (keep_draw < density)
                if int(keep.sum()) < MIN_VALID_PIXELS:
                    continue
                gt_sparse = np.where(keep, gt, 0.0).astype(np.float32)
            for cell, pred in preds.items():
                try:
                    metrics = evaluate_sample(pred, gt_sparse, align="ssi_disparity",
                                              min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)
                except OfficialProtocolError:
                    continue
                results[density][cell].append({
                    "sample_id": row["id"], "abs_rel": metrics["abs_rel"], "a1": metrics["a1"],
                })
        if (index + 1) % 500 == 0:
            print(f"  {index + 1}/{len(rows)}", flush=True)

    report = {}
    print(f"\n{'密度':>6} {'有效GT':>8} | " + " | ".join(f"{n:>22}" for n, _, _ in COMPARISONS))
    for density in DENSITIES:
        cells = results[density]
        counts = {c: len(v) for c, v in cells.items()}
        if min(counts.values()) < 100:
            print(f"{density*100:5.0f}%  样本不足 {counts}")
            continue
        line = f"{density*100:5.0f}% {counts['emptytrain_empty']:8d} | "
        entry = {"n": counts["emptytrain_empty"], "abs_rel_mean": {
            c: float(np.mean([r["abs_rel"] for r in v])) for c, v in cells.items()}}
        parts = []
        for name, a, b in COMPARISONS:
            res = paired_comparison(cells[a], cells[b], label_a=a, label_b=b,
                                    metrics=["abs_rel"], lower_is_better={"abs_rel"})
            s = res["metrics"]["abs_rel"]
            sig = "*" if (s["ci_low"] > 0 or s["ci_high"] < 0) else " "
            parts.append(f"{s['mean']:+.5f}{sig} ({s['win_rate']*100:4.1f}%)")
            entry[name] = {"mean": s["mean"], "ci": [s["ci_low"], s["ci_high"]],
                           "win_rate": s["win_rate"], "significant": sig == "*"}
        report[f"{density:.2f}"] = entry
        print(line + " | ".join(f"{p:>22}" for p in parts))

    print("\n各格 abs_rel 均值:")
    for density in DENSITIES:
        if f"{density:.2f}" not in report:
            continue
        m = report[f"{density:.2f}"]["abs_rel_mean"]
        print(f"  {density*100:5.0f}%  " + "  ".join(f"{c}={m[c]:.4f}" for c in CELLS))

    path = OUT / "rgbdt500_gt_density_sweep.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n保存: {path}")
    print("正值 = caption 占优 | * = bootstrap 95% CI 排除零 | 括号内为逐样本胜率")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
