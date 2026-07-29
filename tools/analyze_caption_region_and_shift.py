"""Two diagnostics for the surviving 'GT quality hides the caption benefit' story.

A. REGION. If the caption helps precisely at the fine edges and small objects
   that a semi-dense or misregistered GT cannot score, then its benefit must
   concentrate on high-depth-gradient (boundary) pixels. If the benefit is the
   same on boundary and interior pixels, that story fails regardless of how
   clean the GT is -- and no new dataset can rescue it.

B. SHIFT. RGBDT500's depth sits ~28 px right of the cameras (measured
   2026-07-21; a horizontal-baseline rig registered with one per-sequence shift,
   so one depth plane is exact and the rest carry residual parallax). If that
   misregistration is masking the caption benefit, then scoring against a
   shift-corrected GT should make the measured caption effect LARGER. Same
   models, same predictions, pure rescoring -- no new data, no retraining.

Alignment reuses ms2_eval.official_protocol so region metrics stay on the same
footing as the headline table: per-image ssi_disparity fit on the official valid
mask, then errors recomputed on the region subset.

CPU only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from ms2_eval.official_protocol import (  # noqa: E402
    collapse_channels,
    fit_scale_shift,
    official_depth_errors,
    official_valid_mask,
)

PAIRS = {
    "c 线 冻结热像(零训练)": ("rgbdt500_thermal_frozen_empty", "rgbdt500_thermal_frozen_capt"),
    "d 线 端到端": ("rgbdt500_eval_emptytrain_empty", "rgbdt500_eval_capttrain_correct"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path,
                   default=Path("E:/dataset/RGBDT500/clean_test/rgbdt500_test_manifest.jsonl"))
    p.add_argument("--root", type=Path, default=Path("E:/dataset/RGBDT500/clean_test"))
    p.add_argument("--runs-root", type=Path, default=Path("outputs/lotus_line_v2"))
    p.add_argument("--output", type=Path,
                   default=Path("outputs/lotus_line_v2/caption_region_and_shift.json"))
    p.add_argument("--shifts", type=int, nargs="+", default=[0, -28])
    p.add_argument("--boundary-quantile", type=float, default=0.80)
    p.add_argument("--interior-quantile", type=float, default=0.50)
    p.add_argument("--min-depth", type=float, default=0.1)
    p.add_argument("--max-depth", type=float, default=20.0)
    p.add_argument("--depth-scale", type=float, default=1000.0)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def shift_gt(gt: np.ndarray, dx: int) -> np.ndarray:
    if dx == 0:
        return gt
    out = np.zeros_like(gt)
    if dx > 0:
        out[:, dx:] = gt[:, :-dx]
    else:
        out[:, :dx] = gt[:, -dx:]
    return out


def region_masks(gt: np.ndarray, valid: np.ndarray, args):
    """Split valid pixels by local GT depth gradient (boundary vs interior).

    The gradient is only defined where both neighbours are valid, so hole
    borders -- which are an artefact of the sensor, not a real depth edge --
    cannot masquerade as boundaries.
    """
    grad = np.zeros_like(gt, np.float64)
    both_x = valid[:, 1:] & valid[:, :-1]
    both_y = valid[1:, :] & valid[:-1, :]
    dx = np.zeros_like(gt, np.float64)
    dy = np.zeros_like(gt, np.float64)
    dx[:, 1:][both_x] = np.abs(gt[:, 1:] - gt[:, :-1])[both_x]
    dy[1:, :][both_y] = np.abs(gt[1:, :] - gt[:-1, :])[both_y]
    grad = dx + dy
    measurable = valid & ((dx > 0) | (dy > 0) | (grad == 0))
    pool = grad[valid]
    if pool.size < 100:
        return None, None
    hi = np.quantile(pool, args.boundary_quantile)
    lo = np.quantile(pool, args.interior_quantile)
    boundary = valid & (grad >= hi)
    interior = valid & (grad <= lo)
    if boundary.sum() < 50 or interior.sum() < 50:
        return None, None
    return boundary, interior


def abs_rel_on(pred_raw: np.ndarray, gt: np.ndarray, fit_mask: np.ndarray,
               score_mask: np.ndarray, args) -> float:
    """Official ssi_disparity alignment fitted on the full valid mask,
    then abs_rel recomputed on the region subset only."""
    gt_disparity = np.zeros_like(gt, np.float64)
    gt_disparity[fit_mask] = 1.0 / gt[fit_mask].astype(np.float64)
    scale, shift = fit_scale_shift(pred_raw, gt_disparity.astype(np.float32), fit_mask)
    aligned = 1.0 / np.clip(pred_raw.astype(np.float64) * scale + shift, 1e-3, None)
    errors = official_depth_errors(aligned, gt, score_mask,
                                   min_depth=args.min_depth, max_depth=args.max_depth)
    return float(errors["abs_rel"])


def main() -> int:
    args = parse_args()
    from PIL import Image

    rows = [json.loads(l) for l in args.manifest.open(encoding="utf-8") if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    report = {}
    for label, (empty_run, capt_run) in PAIRS.items():
        dir_e = args.runs_root / empty_run / "raw_predictions"
        dir_c = args.runs_root / capt_run / "raw_predictions"
        acc = {dx: {r: {"empty": [], "capt": []} for r in ("all", "boundary", "interior")}
               for dx in args.shifts}
        used = 0
        for row in rows:
            pe, pc = dir_e / f"{row['id']}.npy", dir_c / f"{row['id']}.npy"
            if not pe.is_file() or not pc.is_file():
                continue
            pred_e = collapse_channels(np.load(pe))
            pred_c = collapse_channels(np.load(pc))
            gt0 = np.asarray(Image.open(args.root / row["depth_path"])).astype(np.float32) / args.depth_scale
            if gt0.shape != pred_e.shape:
                continue
            ok = False
            for dx in args.shifts:
                gt = shift_gt(gt0, dx)
                valid = official_valid_mask(gt, args.min_depth, args.max_depth)
                if valid.sum() < 100:
                    continue
                boundary, interior = region_masks(gt, valid, args)
                if boundary is None:
                    continue
                for name, mask in (("all", valid), ("boundary", boundary), ("interior", interior)):
                    acc[dx][name]["empty"].append(abs_rel_on(pred_e, gt, valid, mask, args))
                    acc[dx][name]["capt"].append(abs_rel_on(pred_c, gt, valid, mask, args))
                ok = True
            used += int(ok)
            if used % 400 == 0 and ok:
                print(f"  {label}: {used} images")

        entry = {}
        print(f"\n=== {label}   n={used}")
        print(f"{'GT位移':>7} {'区域':>9} {'empty':>9} {'caption':>9} {'caption效应':>12} {'胜率':>7}")
        for dx in args.shifts:
            entry[str(dx)] = {}
            for name in ("all", "boundary", "interior"):
                e = np.array(acc[dx][name]["empty"])
                c = np.array(acc[dx][name]["capt"])
                if e.size == 0:
                    continue
                d = e - c
                se = d.std(ddof=1) / np.sqrt(len(d))
                entry[str(dx)][name] = {
                    "n": int(len(d)), "empty": float(e.mean()), "caption": float(c.mean()),
                    "effect": float(d.mean()), "ci_lo": float(d.mean() - 1.96 * se),
                    "ci_hi": float(d.mean() + 1.96 * se), "win_rate": float((d > 0).mean()),
                }
                v = entry[str(dx)][name]
                star = "*" if v["ci_lo"] * v["ci_hi"] > 0 else " "
                print(f"{dx:>+7} {name:>9} {v['empty']:>9.4f} {v['caption']:>9.4f} "
                      f"{v['effect']:>+11.4f}{star} {v['win_rate']:>6.1%}")
        report[label] = entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.output}")
    print("\n读法:")
    print("  A(区域): 'GT 打不准细边界' 要成立 => boundary 的 caption 效应必须明显大于 interior")
    print("  B(位移): '错位掩盖收益' 要成立 => dx=-28 的 caption 效应必须明显大于 dx=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
