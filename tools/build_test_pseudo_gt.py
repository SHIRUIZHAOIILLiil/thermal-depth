"""DIAGNOSTIC ONLY: write the AnyThermal+LiDAR dense reference for test frames.

Same construction as our training target and as `score_on_pseudo_gt.py` (which builds
it on the fly): AnyThermal's raw prediction fitted per frame, in depth space, to that
frame's LiDAR with `build_anythermal_pseudo_gt.calibrate_pseudo_depth`; LiDAR wherever
LiDAR exists. Outside the LiDAR this *is* AnyThermal, and it is calibrated with test
labels -- it is not ground truth and nothing scored on it is an accuracy.

Every frame of each manifest given (no subsampling). Saved in MS2's own GT format
(16-bit PNG, metres x 256, 0 = no value) so the existing evaluators read it
unchanged, in two variants per frame:

  <cond>/dense/<id>.png   LiDAR where LiDAR exists, calibrated AnyThermal elsewhere
  <cond>/filled/<id>.png  only the pixels the completion filled (LiDAR pixels set to 0)

Values outside (1e-3, 80) m are written as 0, the official validity range. Beside
them, copies of the input manifests with the GT path pointing here (absolute), so
`--manifest` is the only thing an evaluation has to change.

`<cond>` is read from the manifest name (ms2_test_<cond>_thermalcap_...), and
AnyThermal's raw predictions are expected at <anythermal-root>/official_<cond>/raw_predictions.

    python tools/build_test_pseudo_gt.py --ms2-root <root> --anythermal-root <runs>/anythermal \
        --out <dir> --workers 4 <manifest> [<manifest> ...]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_anythermal_pseudo_gt import calibrate_pseudo_depth  # noqa: E402

D_MIN, D_MAX = 1e-3, 80.0


def to_png(depth: np.ndarray, path: Path) -> None:
    keep = np.isfinite(depth) & (depth > D_MIN) & (depth < D_MAX)
    Image.fromarray(np.where(keep, np.round(depth * 256.0), 0).astype(np.uint16)).save(path)


def one_frame(line: str, *, ms2_root: Path, raw_dir: Path, out_dir: Path) -> tuple[str | None, str | None]:
    row = json.loads(line)
    fid = row["id"]
    lidar = np.asarray(Image.open(ms2_root / row["thermal_depth_path"]), np.float64) / 256.0
    lmask = (lidar > D_MIN) & (lidar < D_MAX)
    raw = np.load(raw_dir / f"{fid}.npy").astype(np.float64)
    raw = raw.reshape(raw.shape[-2:]) if raw.ndim > 2 else raw
    try:
        pseudo, _ = calibrate_pseudo_depth(raw, lidar, lmask, min_fit_pixels=100)
    except RuntimeError:
        return None, None
    out = {}
    for variant, depth in (("dense", np.where(lmask, lidar, pseudo)), ("filled", np.where(lmask, 0.0, pseudo))):
        path = out_dir / variant / f"{fid}.png"
        to_png(depth, path)
        new = dict(row, thermal_depth_path=str(path), depth_path=str(path),
                   gt_variant=f"pseudo_{variant}_DIAGNOSTIC")
        out[variant] = json.dumps(new, ensure_ascii=False)
    return out["dense"], out["filled"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ms2-root", type=Path, required=True)
    ap.add_argument("--anythermal-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("manifests", type=Path, nargs="+")
    args = ap.parse_args()
    (args.out / "manifests").mkdir(parents=True, exist_ok=True)
    (args.out / "README.txt").write_text(
        "DIAGNOSTIC reference, not ground truth: LiDAR + AnyThermal calibrated on test LiDAR.\n"
        "Outside the LiDAR it is AnyThermal. Built by tools/build_test_pseudo_gt.py.\n", encoding="utf-8")

    for src in args.manifests:
        match = re.match(r"ms2_test_(.+?)_thermalcap", src.name)
        if not match:
            raise SystemExit(f"!! cannot read the condition from {src.name}")
        cond = match.group(1)
        raw_dir = args.anythermal_root / f"official_{cond}" / "raw_predictions"
        out_dir = args.out / cond
        for variant in ("dense", "filled"):
            (out_dir / variant).mkdir(parents=True, exist_ok=True)
        lines = [l for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        work = partial(one_frame, ms2_root=args.ms2_root, raw_dir=raw_dir, out_dir=out_dir)
        with Pool(args.workers) as pool:
            results = pool.map(work, lines, chunksize=64)
        kept = [r for r in results if r[0] is not None]
        for k, variant in enumerate(("dense", "filled")):
            (args.out / "manifests" / f"{src.stem}_pseudo_{variant}.jsonl").write_text(
                "\n".join(r[k] for r in kept) + "\n", encoding="utf-8")
        print(f"[{cond}] {len(kept)} of {len(lines)} frames written "
              f"({len(lines) - len(kept)} skipped: too little LiDAR to calibrate)", flush=True)
    print(f"[done] {args.out}  -- DIAGNOSTIC reference, not ground truth")


if __name__ == "__main__":
    main()
