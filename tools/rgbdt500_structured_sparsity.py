"""Does LiDAR-SHAPED sparsity hide the RGBDT500 caption effect?

The random-density sweep showed the effect is flat from 100% down to 5% of GT
pixels -- but random subsampling is an unbiased estimator, so that only refutes
"too few pixels". MS2's LiDAR sparsity is *structured*: scan-line bands, and no
returns at all in sky / far / dark regions. If the caption helps precisely where
LiDAR is blind, structured masking would hide the effect while random masking
would not. Two tests:

A) MS2 MASK TRANSFER -- take the validity pattern of real MS2 filtered-LiDAR
   depth maps, resize to the RGBDT500 frame, and let the evaluator see only
   those pixels. Fit and metric both use the reduced set, i.e. exactly what an
   evaluator holding sparse LiDAR GT would compute.

B) DEPTH-BAND DECOMPOSITION -- keep the protocol's global affine fit (all valid
   pixels, as the real evaluator does) but report the error separately for near
   / mid / far pixels. This says WHERE the caption earns its gain; if the gain
   sits in the far band, MS2's far-blindness is a plausible mechanism.

    python tools/rgbdt500_structured_sparsity.py
"""

from __future__ import annotations

import glob
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
    fit_scale_shift,
    official_valid_mask,
)
from ms2_eval.resize import resize_dense_prediction  # noqa: E402

DATA_ROOT = Path("/mnt/e/dataset/RGBDT500/clean_test")
MS2_DEPTH_GLOB = "/mnt/e/dataset/ms2/proj_depth/_2021-08-06-11-23-45/thr/depth_filtered/*.png"
OUT = ROOT / "outputs" / "lotus_line_v2"
CELLS = {
    "emptytrain_empty": OUT / "rgbdt500_eval_emptytrain_empty",
    "emptytrain_correct": OUT / "rgbdt500_eval_emptytrain_correct",
    "capttrain_empty": OUT / "rgbdt500_eval_capttrain_empty",
    "capttrain_correct": OUT / "rgbdt500_eval_capttrain_correct",
}
MIN_DEPTH, MAX_DEPTH, DEPTH_SCALE = 0.1, 20.0, 1000.0
BANDS = (("near <5m", 0.1, 5.0), ("mid 5-12m", 5.0, 12.0), ("far >12m", 12.0, 20.0))
MIN_PIXELS = 500
COMPARISONS = (
    ("注入(empty训)", "emptytrain_correct", "emptytrain_empty"),
    ("注入(capt训)", "capttrain_correct", "capttrain_empty"),
    ("总效果", "capttrain_correct", "emptytrain_empty"),
)


def load_ms2_masks(limit: int = 400) -> list[np.ndarray]:
    """Validity patterns of real MS2 filtered-LiDAR depth maps."""
    files = sorted(glob.glob(MS2_DEPTH_GLOB))[::7][:limit]
    if not files:
        raise SystemExit(f"找不到 MS2 深度图: {MS2_DEPTH_GLOB}")
    masks = []
    for path in files:
        with Image.open(path) as image:
            raw = np.asarray(image)
        masks.append(raw > 0)
    density = float(np.mean([m.mean() for m in masks]))
    print(f"MS2 掩码: {len(masks)} 张, 平均密度 {density*100:.1f}%")
    return masks


def band_errors(pred_depth, gt, valid):
    """abs_rel per depth band under the protocol's global affine fit."""
    out = {}
    for name, low, high in BANDS:
        band = valid & (gt >= low) & (gt < high)
        if int(band.sum()) < MIN_PIXELS:
            out[name] = None
            continue
        p = np.clip(pred_depth[band], MIN_DEPTH, MAX_DEPTH)
        g = gt[band]
        out[name] = float(np.mean(np.abs(p - g) / g))
    return out


def main() -> int:
    manifest = CELLS["emptytrain_empty"] / "selected_manifest.jsonl"
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    ms2_masks = load_ms2_masks()

    transfer = {c: [] for c in CELLS}
    bands = {c: {b[0]: [] for b in BANDS} for c in CELLS}

    for index, row in enumerate(rows):
        with Image.open(DATA_ROOT / row["thermal_depth_path"]) as image:
            gt = np.asarray(image).astype(np.float32) / DEPTH_SCALE
        valid = official_valid_mask(gt, MIN_DEPTH, MAX_DEPTH)
        if int(valid.sum()) < MIN_PIXELS:
            continue

        # --- A) MS2 mask transfer: resize a real LiDAR validity pattern onto this frame
        raw_mask = ms2_masks[index % len(ms2_masks)]
        mask = np.asarray(
            Image.fromarray(raw_mask.astype(np.uint8) * 255).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST)) > 127
        gt_lidar = np.where(valid & mask, gt, 0.0).astype(np.float32)
        enough = int(official_valid_mask(gt_lidar, MIN_DEPTH, MAX_DEPTH).sum()) >= MIN_PIXELS

        for cell, directory in CELLS.items():
            raw = np.load(directory / "raw_predictions" / f"{row['id']}.npy", allow_pickle=False)
            pred = resize_dense_prediction(np.squeeze(raw).astype(np.float32), gt.shape)

            if enough:
                try:
                    m = evaluate_sample(pred, gt_lidar, align="ssi_disparity",
                                        min_depth=MIN_DEPTH, max_depth=MAX_DEPTH)
                    transfer[cell].append({"sample_id": row["id"], "abs_rel": m["abs_rel"]})
                except OfficialProtocolError:
                    pass

            # --- B) depth bands under the protocol's global fit
            gt_disp = np.zeros_like(gt, np.float64)
            gt_disp[valid] = 1.0 / gt[valid]
            try:
                scale, shift = fit_scale_shift(pred, gt_disp.astype(np.float32), valid)
            except OfficialProtocolError:
                continue
            aligned = 1.0 / np.clip(pred.astype(np.float64) * scale + shift, 1e-3, None)
            for name, value in band_errors(aligned, gt, valid).items():
                if value is not None:
                    bands[cell][name].append({"sample_id": row["id"], "abs_rel": value})

        if (index + 1) % 700 == 0:
            print(f"  {index + 1}/{len(rows)}", flush=True)

    report = {}
    print("\n=== A) 真实 MS2 LiDAR 掩码迁移（评估器只看 LiDAR 形状的像素）===")
    n = min(len(v) for v in transfer.values())
    print(f"配对样本 {n}")
    for name, a, b in COMPARISONS:
        res = paired_comparison(transfer[a], transfer[b], label_a=a, label_b=b,
                                metrics=["abs_rel"], lower_is_better={"abs_rel"})
        s = res["metrics"]["abs_rel"]
        sig = "*" if (s["ci_low"] > 0 or s["ci_high"] < 0) else " "
        print(f"  {name:16s} {s['mean']:+.5f}{sig} [{s['ci_low']:+.5f},{s['ci_high']:+.5f}] 胜率 {s['win_rate']*100:.1f}%")
        report[f"ms2_mask/{name}"] = {"mean": s["mean"], "ci": [s["ci_low"], s["ci_high"]],
                                      "significant": sig == "*"}

    print("\n=== B) caption 增益按深度分层（全局对齐，分段计误差）===")
    for band_name, _, _ in BANDS:
        counts = [len(bands[c][band_name]) for c in CELLS]
        if min(counts) < 100:
            print(f"  {band_name:12s} 样本不足 ({min(counts)})")
            continue
        base = float(np.mean([r["abs_rel"] for r in bands["emptytrain_empty"][band_name]]))
        print(f"  {band_name:12s} (n={min(counts)}, 基线 abs_rel={base:.4f})")
        for name, a, b in COMPARISONS:
            res = paired_comparison(bands[a][band_name], bands[b][band_name],
                                    label_a=a, label_b=b, metrics=["abs_rel"],
                                    lower_is_better={"abs_rel"})
            s = res["metrics"]["abs_rel"]
            sig = "*" if (s["ci_low"] > 0 or s["ci_high"] < 0) else " "
            rel = 100 * s["mean"] / base if base else float("nan")
            print(f"      {name:16s} {s['mean']:+.5f}{sig} (相对基线 {rel:+.2f}%) 胜率 {s['win_rate']*100:.1f}%")
            report[f"band_{band_name}/{name}"] = {"mean": s["mean"], "relative_pct": rel,
                                                 "significant": sig == "*", "baseline": base}

    path = OUT / "rgbdt500_structured_sparsity.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n保存: {path}")
    print("对照: 全稠密 GT 下 注入(empty训)=+0.00186* 注入(capt训)=+0.00583* 总效果=+0.00729*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
