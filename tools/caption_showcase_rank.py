"""Pick frames for a caption showcase: where captions helped most, said so plainly.

The figure this feeds shows frames on which the caption-trained model beats the
caption-free one, which is a best-case illustration, and its caption must say how
the frames were chosen and what the average gain is. This makes the choice
mechanical so that sentence is true:

  * both seed-43 arms at step 18000, each scored under its training prompt, read
    from the per-frame CSVs their test runs already wrote -- no inference here;
  * official evaluation subset only, the frame set of Table main;
  * ranked by nocap AbsRel minus cap AbsRel, largest first;
  * greedy, at most one frame per 300-frame stretch of a sequence, so the list is
    not eight views of one junction.

Writes, per condition, the ranked top K with both AbsRel values; copies their
thermal and GT; and writes the manifests the showcase job runs the two models on
(one compact, and one per condition padded with blank lines so each frame keeps
the line number -- and so the noise seed -- it had in the full test run).

    python tools/caption_showcase_rank.py --ms2-root <root> --manifest-dir <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import shutil
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_qualitative_frames import official_frames  # noqa: E402

EVAL = "/mnt/scratch/sc23sz/runs/eval"
CONDITIONS = {  # condition: (official list, test manifest, manifest fingerprint)
    "day": ("test_day_list.txt", "ms2_test_day3_common_thermalcap_20260821.jsonl", "f50d4506"),
    "night": ("test_night_list.txt", "ms2_test_night3_thermalcap_20260821.jsonl", "273310de"),
    "rainy": ("test_rainy_list.txt", "ms2_test_rainy3_thermalcap_20260821.jsonl", "5a63743f"),
}
ARMS = {"cap": ("sd2_cap_pct_logtn_s43", "correct"), "nocap": ("sd2_nocap_pct_logtn_s43", "empty")}


def per_frame(run: str, fingerprint: str, prompt: str) -> dict[str, float]:
    paths = glob.glob(f"{EVAL}/{run}_test_{fingerprint}_s18000_{prompt}/eval_eval_affine_invariant_log_space_per_sample.csv")
    if len(paths) != 1:
        raise SystemExit(f"!! expected one CSV for {run} {fingerprint} {prompt}, found {paths}")
    with open(paths[0], newline="", encoding="utf-8") as h:
        return {r["id"]: float(r["abs_rel"]) for r in csv.DictReader(h)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-condition", type=int, default=8)
    parser.add_argument("--min-gap", type=int, default=300)
    args = parser.parse_args()
    (args.out / "inputs").mkdir(parents=True, exist_ok=True)

    compact = []
    for cond, (list_file, manifest_name, fp) in CONDITIONS.items():
        subset = {f["id"] for f in official_frames(args.ms2_root, list_file)}
        cap = per_frame(*ARMS["cap"][:1], fp, ARMS["cap"][1])
        nocap = per_frame(*ARMS["nocap"][:1], fp, ARMS["nocap"][1])
        common = sorted(subset & cap.keys() & nocap.keys())
        gains = {i: nocap[i] - cap[i] for i in common}
        ranked = sorted(common, key=lambda i: -gains[i])

        picked: list[str] = []
        for fid in ranked:
            seq, stem = fid.rsplit("_", 1)
            if any(p.rsplit("_", 1)[0] == seq and abs(int(p.rsplit("_", 1)[1]) - int(stem)) < args.min_gap
                   for p in picked):
                continue
            picked.append(fid)
            if len(picked) == args.per_condition:
                break

        mean_gain = statistics.fmean(gains.values())
        lines = [f"{cond}: official frames scored by both arms {len(common)}; "
                 f"mean gain (nocap - cap AbsRel) {mean_gain:+.5f}; "
                 f"frames where captions help {sum(g > 0 for g in gains.values())}",
                 f"{'rank':>4s} {'frame':28s} {'nocap':>8s} {'cap':>8s} {'gain':>8s}"]
        lines += [f"{k:4d} {fid:28s} {nocap[fid]:8.4f} {cap[fid]:8.4f} {gains[fid]:+8.4f}"
                  for k, fid in enumerate(picked, 1)]
        (args.out / f"rank_{cond}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines) + "\n", flush=True)

        # Manifests: rows by original line number, so the padded copy restores seeds.
        at = {}
        with (args.manifest_dir / manifest_name).open(encoding="utf-8") as h:
            for index, line in enumerate(h):
                if line.strip():
                    row = json.loads(line)
                    if row.get("id") in picked:
                        at[index] = row
        if len(at) != len(picked):
            raise SystemExit(f"!! {cond}: found {len(at)} of {len(picked)} frames in {manifest_name}")
        padded: list[str] = []
        for index in sorted(at):
            padded += [""] * (index - len(padded))
            padded.append(json.dumps(at[index]))
        (args.out / f"frames_ours_{cond}.jsonl").write_text("\n".join(padded) + "\n", encoding="utf-8")
        for row in at.values():
            compact.append(row)
            for kind, key in (("thermal", "thermal_path"), ("gt", "thermal_depth_path")):
                shutil.copy2(args.ms2_root / row[key], args.out / "inputs" / f"{cond}_{row['id']}_{kind}.png")
    (args.out / "frames.jsonl").write_text("".join(json.dumps(r) + "\n" for r in compact), encoding="utf-8")
    print(f"[done] {len(compact)} frames -> {args.out}")


if __name__ == "__main__":
    main()
