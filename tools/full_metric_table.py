"""The full MS2 metric set on the official subset, read from per-frame CSVs already on disk.

MS2 papers report seven numbers per condition -- AbsRel, SqRel, RMSE, RMSE_log, d1, d2,
d3 -- and our tables so far carry three. This reads the per-frame results every model
already wrote (no inference), restricts them to the official evaluation subset, and
prints all seven side by side, so the question "do the other four change the picture"
is answered for every model at once and none of them chosen afterwards.

Each --model is name=<csv pattern>[:prefix]. The pattern may use {fp} (our manifest
fingerprint), {env} (test_day/...), {cond} (day/night/rainy) and {m} (day3_common/...);
`**` globs are allowed. prefix is the column prefix (NeWCRF's bench CSV writes ssi_abs_rel).

    python tools/full_metric_table.py --ms2-root <root> --model ours=...:  --model newcrf=...:ssi_
"""

from __future__ import annotations

import argparse
import csv
import glob
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_qualitative_frames import CONDITIONS, official_frames  # noqa: E402

KEYS = {"day": dict(fp="f50d4506", env="test_day", m="day3_common"),
        "night": dict(fp="273310de", env="test_night", m="night3"),
        "rainy": dict(fp="5a63743f", env="test_rain", m="rainy3")}
METRICS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")
SCALE = {"abs_rel": 100, "sq_rel": 1, "rmse": 1, "rmse_log": 1, "a1": 100, "a2": 100, "a3": 100}
BETTER_LOW = {"abs_rel", "sq_rel", "rmse", "rmse_log"}


def read(pattern: str, prefix: str, keep: set[str]) -> dict[str, list[float]]:
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        return {}
    out = {k: [] for k in METRICS}
    seen = set()
    for p in paths:
        with open(p, newline="", encoding="utf-8") as h:
            for r in csv.DictReader(h):
                fid = r.get("sample_id") or r.get("id")
                if fid not in keep or fid in seen:
                    continue
                seen.add(fid)
                for k in METRICS:
                    v = r.get(prefix + k)
                    if v not in (None, ""):
                        out[k].append(float(v))
    out["_n"] = [len(seen)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ms2-root", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True)
    args = ap.parse_args()
    models = []
    for spec in args.model:
        name, rest = spec.split("=", 1)
        pattern, _, prefix = rest.rpartition(":") if rest.count(":") else (rest, "", "")
        models.append((name, pattern, prefix))

    for cond, list_file in CONDITIONS.items():
        keep = {f["id"] for f in official_frames(args.ms2_root, list_file)}
        print(f"\n===== {cond}  (official subset, {len(keep)} frames; AbsRel/d in %)")
        print(f"{'model':16s} {'n':>5s} " + " ".join(f"{k:>9s}" for k in METRICS))
        table = {}
        for name, pattern, prefix in models:
            vals = read(pattern.format(**KEYS[cond], cond=cond), prefix, keep)
            if not vals or not vals["abs_rel"]:
                print(f"{name:16s}  -- no CSV for {pattern.format(**KEYS[cond], cond=cond)}")
                continue
            table[name] = {k: statistics.fmean(vals[k]) * SCALE[k] for k in METRICS if vals[k]}
            row = " ".join(f"{table[name].get(k, float('nan')):9.3f}" for k in METRICS)
            print(f"{name:16s} {vals['_n'][0]:5d} {row}")
        if table:
            best = {k: (min if k in BETTER_LOW else max)(table, key=lambda n: table[n].get(k, float('nan'))) for k in METRICS}
            print(f"{'best':16s} {'':5s} " + " ".join(f"{best[k][:9]:>9s}" for k in METRICS))


if __name__ == "__main__":
    main()
