"""How typical are the figure's frames? Per-frame AbsRel of every model, read
from the CSVs the test runs already wrote -- no inference.

The qualitative frames were chosen for scene content alone. This reports, per
condition, each model's median per-frame AbsRel over the official subset, the
median of ours-minus-each-model, and the same numbers on the eight structure-
ranked candidates, so a frame can be judged representative or not before it
goes into the paper. Raw AbsRel only; nothing here is for the paper itself.

    python tools/qual_frame_gaps.py --ms2-root <root> --candidates <qual_candidates/candidates.json>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_qualitative_frames import CONDITIONS, official_frames  # noqa: E402

S = "/mnt/scratch/sc23sz/runs"
SOURCES = {
    "ours":   f"{S}/eval/sd2_cap_pct_logtn_s43_test_*_s18000_correct/eval_*log_space*per_sample.csv",
    "lotus0": f"{S}/analysis/lotus_zeroshot_{{c}}/official/metrics/per_image.csv",
    "ppd":    f"{S}/analysis/ppdv2_test_{{c}}/**/per_image.csv",
    "da2":    f"{S}/analysis/da2_test_s16000_{{c}}/**/per_image.csv",
}
CHOSEN = {"day": "2021-08-06-11-23-45_005720", "night": "2021-08-13-22-03-03_006139",
          "rainy": "2021-08-06-16-45-28_005762"}


def read(pattern: str) -> dict[str, float]:
    got = {}
    for path in glob.glob(pattern, recursive=True):
        with open(path, newline="", encoding="utf-8") as h:
            for r in csv.DictReader(h):
                fid = r.get("sample_id") or r.get("id")
                key = "abs_rel" if "abs_rel" in r else next((k for k in r if "abs_rel" in k), None)
                if fid and key:
                    got[fid] = float(r[key])
    return got


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    args = parser.parse_args()
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    ours_all = read(SOURCES["ours"])

    for cond, list_file in CONDITIONS.items():
        subset = {f["id"] for f in official_frames(args.ms2_root, list_file)}
        tables = {"ours": ours_all}
        for name in ("lotus0", "ppd", "da2"):
            tables[name] = read(SOURCES[name].format(c=cond))
        tables = {k: {i: v for i, v in t.items() if i in subset} for k, t in tables.items()}
        print(f"\n===== {cond}  (official subset {len(subset)} frames) =====")
        for name, t in tables.items():
            print(f"  {name:7s} frames {len(t):5d}   median AbsRel {np.median(list(t.values())):.4f}")
        others = [k for k in tables if k != "ours"]
        for other in others:
            common = tables["ours"].keys() & tables[other].keys()
            gaps = np.array([tables["ours"][i] - tables[other][i] for i in common])
            print(f"  ours - {other:7s} median gap {np.median(gaps):+.4f}   "
                  f"(25-75%: {np.percentile(gaps, 25):+.4f} .. {np.percentile(gaps, 75):+.4f})")

        print(f"  {'candidate':30s} " + " ".join(f"{k:>8s}" for k in tables))
        for rank, cand in enumerate(candidates[cond]["candidates"], 1):
            fid = cand["id"]
            mark = "  <- chosen" if fid == CHOSEN[cond] else ""
            cells = " ".join(f"{tables[k][fid]:8.4f}" if fid in tables[k] else f"{'-':>8s}" for k in tables)
            print(f"  #{rank} {fid:27s} {cells}{mark}")


if __name__ == "__main__":
    main()
