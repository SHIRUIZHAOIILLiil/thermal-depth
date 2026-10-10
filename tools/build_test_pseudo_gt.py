"""DIAGNOSTIC ONLY: write the AnyThermal+LiDAR dense reference for the official test subset.

Same construction as our training target and as `score_on_pseudo_gt.py` (which builds
it on the fly): AnyThermal's raw prediction fitted per frame, in depth space, to that
frame's LiDAR with `build_anythermal_pseudo_gt.calibrate_pseudo_depth`; LiDAR wherever
LiDAR exists. Outside the LiDAR this *is* AnyThermal, and it is calibrated with test
labels -- it is not ground truth and nothing scored on it is an accuracy.

Saved in MS2's own GT format (16-bit PNG, metres x 256, 0 = no value) so the existing
evaluators read it unchanged, in two variants per frame:

  dense/<id>.png   LiDAR where LiDAR exists, calibrated AnyThermal elsewhere
  filled/<id>.png  only the pixels the completion filled (LiDAR pixels set to 0)

Values outside (1e-3, 80) m are written as 0, the official validity range. Next to
them, manifests that are copies of the official-subset manifests with the GT path
pointing here (absolute), so `--manifest` is the only thing to change.

    python tools/build_test_pseudo_gt.py --ms2-root <root> --manifest-dir <official manifests> \
        --anythermal-root <runs>/anythermal --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_anythermal_pseudo_gt import calibrate_pseudo_depth  # noqa: E402

CONDS = ("day3_common", "night3", "rainy3")
D_MIN, D_MAX = 1e-3, 80.0


def to_png(depth: np.ndarray, path: Path) -> None:
    keep = np.isfinite(depth) & (depth > D_MIN) & (depth < D_MAX)
    arr = np.where(keep, np.round(depth * 256.0), 0).astype(np.uint16)
    Image.fromarray(arr).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ms2-root", type=Path, required=True)
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--anythermal-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    (args.out / "manifests").mkdir(parents=True, exist_ok=True)
    (args.out / "README.txt").write_text(
        "DIAGNOSTIC reference, not ground truth: LiDAR + AnyThermal calibrated on test LiDAR.\n"
        "Outside the LiDAR it is AnyThermal. Built by tools/build_test_pseudo_gt.py.\n", encoding="utf-8")

    for m in CONDS:
        src = args.manifest_dir / f"ms2_test_{m}_thermalcap_20260821_official.jsonl"
        rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        for variant in ("dense", "filled"):
            (args.out / m / variant).mkdir(parents=True, exist_ok=True)
        out_rows = {"dense": [], "filled": []}
        skipped = 0
        for row in rows:
            fid = row["id"]
            lidar = np.asarray(Image.open(args.ms2_root / row["thermal_depth_path"]), np.float64) / 256.0
            lmask = (lidar > D_MIN) & (lidar < D_MAX)
            raw = np.load(args.anythermal_root / f"official_{m}" / "raw_predictions" / f"{fid}.npy").astype(np.float64)
            raw = raw.reshape(raw.shape[-2:]) if raw.ndim > 2 else raw
            try:
                pseudo, _ = calibrate_pseudo_depth(raw, lidar, lmask, min_fit_pixels=100)
            except RuntimeError:
                skipped += 1
                continue
            filled = np.where(lmask, 0.0, pseudo)
            dense = np.where(lmask, lidar, pseudo)
            for variant, depth in (("dense", dense), ("filled", filled)):
                path = args.out / m / variant / f"{fid}.png"
                to_png(depth, path)
                new = dict(row)
                new["thermal_depth_path"] = str(path)
                new["depth_path"] = str(path)
                new["gt_variant"] = f"pseudo_{variant}_DIAGNOSTIC"
                out_rows[variant].append(json.dumps(new, ensure_ascii=False))
        for variant, lines in out_rows.items():
            (args.out / "manifests" / f"ms2_test_{m}_thermalcap_20260821_official_pseudo_{variant}.jsonl").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
        print(f"[{m}] {len(rows) - skipped} frames written ({skipped} skipped: too little LiDAR to calibrate)")
    print(f"[done] {args.out}  -- DIAGNOSTIC reference, not ground truth")


if __name__ == "__main__":
    main()
