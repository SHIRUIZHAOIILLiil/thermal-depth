"""DIAGNOSTIC ONLY: score models against a dense target built from AnyThermal + LiDAR.

Not a benchmark result and not for the paper's tables. The dense target is built
exactly as our training target is -- AnyThermal's raw prediction, fitted per frame
in depth space to that frame's LiDAR (`build_anythermal_pseudo_gt.calibrate_pseudo_depth`,
the function that built the training corpus), LiDAR wherever LiDAR exists -- so
outside the LiDAR it *is* AnyThermal. Scoring there measures agreement with
AnyThermal, which our model was trained to imitate. Calibrating against test LiDAR
also means the target is built from test labels. Read it with that in mind; the
AnyThermal row shows the ceiling the construction hands to its own source.

Three regions per frame, each scored with the official protocol (per-image affine
fit in the model's own space on that region, 1e-3 < d < 80 m, then per-frame metrics
averaged over frames):

  lidar   the LiDAR pixels only -- must reproduce the published tables (self-check)
  filled  only the pixels the completion filled (no LiDAR there)
  dense   both

    python tools/score_on_pseudo_gt.py --ms2-root <root> --manifest-dir <dir> --anythermal-root <dir> \
        --model ours=<dir_with_{cond}>:ssi_log --model newcrf=<dir>:ssi --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from build_anythermal_pseudo_gt import calibrate_pseudo_depth  # noqa: E402
from ms2_eval.official_protocol import OfficialProtocolError, collapse_channels, evaluate_sample  # noqa: E402
from ms2_eval.resize import resize_dense_prediction  # noqa: E402

CONDS = {"day": ("day3_common", "test_day"), "night": ("night3", "test_night"), "rainy": ("rainy3", "test_rain")}
MANIFEST = "ms2_test_{m}_thermalcap_20260821_official.jsonl"
D_MIN, D_MAX = 1e-3, 80.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ms2-root", type=Path, required=True)
    ap.add_argument("--manifest-dir", type=Path, required=True, help="Where the *_official.jsonl live.")
    ap.add_argument("--anythermal-root", type=Path, required=True,
                    help="Holds official_{day3_common,night3,rainy3}/raw_predictions/<id>.npy")
    ap.add_argument("--model", action="append", required=True,
                    help="name=dir:align; '{m}' and '{env}' in dir expand per condition")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    models = {}
    for spec in args.model:
        name, rest = spec.split("=", 1)
        path, align = rest.rsplit(":", 1)
        models[name] = (path, align)
    models["anythermal"] = (str(args.anythermal_root / "official_{m}" / "raw_predictions"), "ssi")

    rows, summary = [], {}
    for cond, (m, env) in CONDS.items():
        frames = [json.loads(l) for l in (args.manifest_dir / MANIFEST.format(m=m)).read_text(encoding="utf-8").splitlines() if l.strip()]
        filled_share, skipped = [], 0
        acc = {(n, r): {"abs_rel": [], "rmse": [], "a1": []} for n in models for r in ("lidar", "filled", "dense")}
        for row in frames:
            fid = row["id"]
            lidar = np.asarray(Image.open(args.ms2_root / row["thermal_depth_path"]), np.float64) / 256.0
            lmask = (lidar > D_MIN) & (lidar < D_MAX)
            raw_at = np.load(Path(str(models["anythermal"][0]).format(m=m, env=env)) / f"{fid}.npy").astype(np.float64)
            try:
                pseudo, _ = calibrate_pseudo_depth(collapse_channels(raw_at), lidar, lmask, min_fit_pixels=100)
            except RuntimeError:
                skipped += 1
                continue
            fmask = ~lmask & (pseudo > D_MIN) & (pseudo < D_MAX)
            filled_share.append(float(fmask.mean()))
            targets = {
                "lidar": np.where(lmask, lidar, 0.0),
                "filled": np.where(fmask, pseudo, 0.0),
                "dense": np.where(lmask, lidar, np.where(fmask, pseudo, 0.0)),
            }
            for name, (path, align) in models.items():
                pred_path = Path(path.format(m=m, env=env)) / f"{fid}.npy"
                if not pred_path.is_file():
                    continue
                pred = resize_dense_prediction(collapse_channels(np.load(pred_path)), lidar.shape)
                for region, gt in targets.items():
                    try:
                        r = evaluate_sample(pred, gt.astype(np.float32), align=align)
                    except OfficialProtocolError:
                        continue
                    for k in ("abs_rel", "rmse", "a1"):
                        acc[(name, region)][k].append(r[k])
                    rows.append({"cond": cond, "id": fid, "model": name, "region": region,
                                 **{k: r[k] for k in ("abs_rel", "rmse", "a1")}})
        print(f"\n===== {cond}: {len(frames)} frames ({skipped} skipped, too little LiDAR to calibrate); "
              f"completion fills {statistics.fmean(filled_share):.1%} of the frame on average")
        print(f"{'model':12s} {'region':7s} {'n':>5s} {'AbsRel%':>8s} {'RMSE m':>8s} {'d1 %':>7s}")
        for (name, region), v in acc.items():
            if not v["abs_rel"]:
                continue
            s = (statistics.fmean(v["abs_rel"]) * 100, statistics.fmean(v["rmse"]), statistics.fmean(v["a1"]) * 100)
            summary[f"{cond}/{name}/{region}"] = {"n": len(v["abs_rel"]), "abs_rel": s[0], "rmse": s[1], "a1": s[2]}
            print(f"{name:12s} {region:7s} {len(v['abs_rel']):5d} {s[0]:8.2f} {s[1]:8.3f} {s[2]:7.2f}")

    with (args.out / "per_frame.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[done] {args.out}  -- DIAGNOSTIC: the 'filled' target is AnyThermal itself; not a benchmark.")


if __name__ == "__main__":
    main()
