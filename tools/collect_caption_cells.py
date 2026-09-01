"""Collect one backbone's caption 2x2 from whatever the pipeline has written.

Iris reports four rows per backbone -- Baseline, Train Only, Infer Only, Train
& Infer -- and the pipeline writes each of those four cells into its own
directory. This walks them into one table per condition, so the numbers reach
the paper without being copied by hand out of six job logs.

Conditions are told apart by frame count (day 23311, night 22916, rain 25023)
rather than by the manifest fingerprint in the directory name: the counts are
distinct and are recorded inside the result itself, so a renamed manifest
cannot silently reassign a row.

Both alignment families are read. The file is eval_eval.json under the
disparity-space affine that Lotus uses and
eval_eval_affine_invariant_depth_space.json under the depth-space one that
Marigold and E2E-FT need; mixing the two in one table would be comparing
different exams, so the alignment is printed with the numbers.
"""
import argparse
import json
import os
from pathlib import Path

FRAMES_TO_CONDITION = {23311: "day", 22916: "night", 25023: "rain"}
CONDITION_ORDER = ("day", "night", "rain")

RESULT_FILES = {
    "eval_eval.json": "ssi_disparity",
    "eval_eval_affine_invariant_depth_space.json": "ssi",
}


def load_cells(root: Path, run: str, step: int):
    """Every (condition, prompt) result this run has produced."""
    out = {}
    for directory in sorted(root.glob(f"{run}_test_*_s{step}_*")):
        prompt = directory.name.rsplit("_", 1)[-1]
        for filename, alignment in RESULT_FILES.items():
            path = directory / filename
            if not path.exists():
                continue
            try:
                metrics = json.loads(path.read_text())
            except Exception as exc:  # a half-written file should say so
                print(f"  !! unreadable {path}: {exc}")
                continue
            frames = int(metrics.get("val_samples", 0))
            condition = FRAMES_TO_CONDITION.get(frames)
            if condition is None:
                print(f"  !! {directory.name}: {frames} frames matches no condition")
                continue
            out[(condition, prompt)] = (metrics, alignment)
    return out


def fmt(metrics):
    return (f"{metrics['abs_rel'] * 100:6.2f}  {metrics['rmse']:6.3f}  "
            f"{metrics['a1'] * 100:6.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-run", required=True, help="caption-trained arm")
    parser.add_argument("--nocap-run", required=True, help="caption-free arm")
    parser.add_argument("--step", type=int, default=20000)
    parser.add_argument("--label", default=None, help="backbone name for the header")
    args = parser.parse_args()

    root = Path(os.environ["IRIS_RUNS"]) / "eval"
    cap = load_cells(root, args.cap_run, args.step)
    nocap = load_cells(root, args.nocap_run, args.step)

    label = args.label or args.cap_run
    print(f"=== {label}, step {args.step} "
          f"(AbsRel and delta1 in percent, RMSE in metres) ===")

    for condition in CONDITION_ORDER:
        rows = [
            ("Baseline      (nocap, empty)  ", nocap.get((condition, "empty"))),
            ("Infer Only    (nocap, caption)", nocap.get((condition, "correct"))),
            ("Train Only    (cap,   empty)  ", cap.get((condition, "empty"))),
            ("Train & Infer (cap,   caption)", cap.get((condition, "correct"))),
        ]
        if not any(cell for _, cell in rows):
            continue
        alignments = {a for _, cell in rows if cell for a in (cell[1],)}
        note = f"  [align {'/'.join(sorted(alignments))}]" if alignments else ""
        print(f"\n-- {condition}{note}")
        print(f"   {'':30s} AbsRel    RMSE      d1")
        for name, cell in rows:
            print(f"   {name} " + (fmt(cell[0]) if cell else "     --      --      --"))

        base = rows[0][1]
        both = rows[3][1]
        if base and both:
            delta = (both[0]["abs_rel"] - base[0]["abs_rel"]) * 100
            sign = "helps" if delta < 0 else "hurts"
            print(f"   {'Train & Infer - Baseline':30s} {delta:+6.2f}  ({sign})")

    missing = [f"{c}/{p}" for c in CONDITION_ORDER for p in ("empty", "correct")
               if (c, p) not in cap or (c, p) not in nocap]
    if missing:
        print(f"\nstill missing: {', '.join(missing)}")


if __name__ == "__main__":
    main()
