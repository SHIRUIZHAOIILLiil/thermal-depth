"""Collect the SD2-start evaluation results into tables that can go in a report.

Two naming schemes end up under $IRIS_RUNS/eval and both are read here:
`iris_ms2_pipeline.sbatch` writes `sd2_cap[_s43]_{val,test}_<manifest fp>_s<step>[_<prompt>]`,
and the fixed-step runs submitted through `eval.sbatch` write
`fix_s<seed>_<arm>_<scene>_s<step>_<prompt>`.

The wide table puts the arms side by side with raw AbsRel / RMSE / delta1 and
nothing else -- no differences, no win rates, no intervals. Those belong in the
prose that reads the table, not in the table, and a column of deltas invites
exactly the comparison the selection protocol cannot support.

  python tools/export_sd2_results.py --out-dir docs/data
"""

import argparse
import csv
import glob
import json
import os
import re

SCENES = {"f50d4506": "day3", "273310de": "night3", "5a63743f": "rainy3"}
ORDER = {"day3": 0, "night3": 1, "rainy3": 2, "": 9}
PROMPTS = ["correct", "shuffled", "empty"]

PIPE = re.compile(
    r"^sd2_(?P<arm>cap|nocap)(?:_s(?P<seed>\d+))?_(?P<split>val|test)_"
    r"(?P<fp>[0-9a-f]{8}|dflt)_s(?P<step>\d+)(?:_(?P<prompt>[a-z]+))?$")
FIXED = re.compile(
    r"^fix_s(?P<seed>\d+)_(?P<arm>cap|nocap)_(?P<scene>[a-z0-9]+)_"
    r"s(?P<step>\d+)_(?P<prompt>[a-z]+)$")


def parse(name):
    m = FIXED.match(name)
    if m:
        d = m.groupdict()
        return dict(seed=d["seed"], arm=d["arm"], split="test", scene=d["scene"],
                    step=int(d["step"]), prompt=d["prompt"], fixed_step=True)
    m = PIPE.match(name)
    if m:
        d = m.groupdict()
        return dict(seed=d["seed"] or "42", arm=d["arm"], split=d["split"],
                    scene=SCENES.get(d["fp"], "") if d["split"] == "test" else "",
                    step=int(d["step"]), prompt=d["prompt"] or "", fixed_step=False)
    return None


def collect(root):
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "*"))):
        rec = parse(os.path.basename(path))
        if rec is None:
            continue
        for f in glob.glob(os.path.join(path, "eval_eval*.json")):
            try:
                m = json.load(open(f))
            except Exception:
                continue
            # The evaluator renames the file when it aligns in depth space, so the
            # space a number came from is recoverable from the filename alone.
            rec = dict(rec)
            rec["align"] = ("ssi" if "affine_invariant_depth" in f
                            else "ssi_disparity")
            rec.update(abs_rel=m.get("abs_rel"), rmse=m.get("rmse"), d1=m.get("a1"),
                       dir=os.path.basename(path))
            rows.append(rec)
    rows.sort(key=lambda r: (r["seed"], r["split"], ORDER.get(r["scene"], 9),
                             r["step"], PROMPTS.index(r["prompt"])
                             if r["prompt"] in PROMPTS else 9))
    return rows


def write_long(rows, path):
    cols = ["seed", "split", "scene", "arm", "step", "prompt", "fixed_step",
            "align", "abs_rel", "rmse", "d1", "dir"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_test_wide(rows, path):
    """One row per seed/scene/step, the arms and prompts side by side."""
    arms = []
    for r in rows:
        if r["split"] != "test":
            continue
        key = (r["arm"], r["prompt"])
        if key not in arms:
            arms.append(key)
    arms.sort(key=lambda k: (k[0] != "cap", PROMPTS.index(k[1])
                             if k[1] in PROMPTS else 9))

    table = {}
    for r in rows:
        if r["split"] != "test":
            continue
        table.setdefault((r["seed"], r["scene"], r["step"]), {})[
            (r["arm"], r["prompt"])] = r

    head = ["seed", "scene", "step"]
    for a, p in arms:
        head += [f"{a}/{p} AbsRel", f"{a}/{p} RMSE", f"{a}/{p} d1"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(head)
        for k in sorted(table, key=lambda k: (k[0], ORDER.get(k[1], 9), k[2])):
            row = [k[0], k[1], k[2]]
            for a in arms:
                r = table[k].get(a)
                row += ["", "", ""] if r is None else [
                    f"{r['abs_rel']:.5f}", f"{r['rmse']:.3f}", f"{r['d1']:.4f}"]
            w.writerow(row)


def write_val_wide(rows, path):
    arms = sorted({(r["seed"], r["arm"]) for r in rows if r["split"] == "val"})
    steps = sorted({r["step"] for r in rows if r["split"] == "val"})
    table = {(r["seed"], r["arm"], r["step"]): r
             for r in rows if r["split"] == "val"}
    head = ["step"]
    for s, a in arms:
        head += [f"s{s}/{a} AbsRel", f"s{s}/{a} RMSE", f"s{s}/{a} d1"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(head)
        for st in steps:
            row = [st]
            for s, a in arms:
                r = table.get((s, a, st))
                row += ["", "", ""] if r is None else [
                    f"{r['abs_rel']:.5f}", f"{r['rmse']:.3f}", f"{r['d1']:.4f}"]
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root",
                    default=os.path.expandvars("$IRIS_RUNS/eval"))
    ap.add_argument("--out-dir", default="docs/data")
    ap.add_argument("--xlsx", action="store_true",
                    help="also write one workbook with the three sheets")
    args = ap.parse_args()

    rows = collect(os.path.expandvars(args.eval_root))
    if not rows:
        raise SystemExit(f"没有结果：{args.eval_root}")
    os.makedirs(args.out_dir, exist_ok=True)

    paths = {
        "long": os.path.join(args.out_dir, "sd2_results_long.csv"),
        "test": os.path.join(args.out_dir, "sd2_results_test.csv"),
        "val": os.path.join(args.out_dir, "sd2_results_val.csv"),
    }
    write_long(rows, paths["long"])
    write_test_wide(rows, paths["test"])
    write_val_wide(rows, paths["val"])
    for k, p in paths.items():
        print(f"{k:<5} -> {p}")
    print(f"\n共 {len(rows)} 条评估结果，"
          f"{len({r['seed'] for r in rows})} 个种子")

    if args.xlsx:
        try:
            import openpyxl  # noqa: F401
            import csv as _csv
            from openpyxl import Workbook
            wb = Workbook()
            wb.remove(wb.active)
            for name, p in (("test", paths["test"]), ("val", paths["val"]),
                            ("long", paths["long"])):
                ws = wb.create_sheet(name)
                with open(p, encoding="utf-8-sig") as f:
                    for r in _csv.reader(f):
                        ws.append(r)
                ws.freeze_panes = "A2"
            out = os.path.join(args.out_dir, "sd2_results.xlsx")
            wb.save(out)
            print(f"xlsx  -> {out}")
        except ImportError:
            print("\n(没装 openpyxl，只写了 csv)")


if __name__ == "__main__":
    main()
