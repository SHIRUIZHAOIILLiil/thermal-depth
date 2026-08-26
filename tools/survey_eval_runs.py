"""Every evaluation result on this machine, in one table.

    python tools/survey_eval_runs.py                     # everything
    python tools/survey_eval_runs.py --filter full8      # one line of work
    python tools/survey_eval_runs.py --filter full8 --split val --sort abs_rel

Written because "which checkpoint is best" turned out not to have an obvious
answer.  Each arm's step was chosen by a rule (lowest val AbsRel, prompt matching
that arm's training condition) applied inside `iris_ms2_pipeline.sbatch`, and the
curve it was applied to lives only in the eval directories -- it was printed to a
job log and never written anywhere durable.  This reads it back out of the result
JSONs, which carry everything needed: the checkpoint, its step, the prompt, the
manifest, the frame count and the metrics.

⚠️ The eval directory name carries the manifest *fingerprint*, not the manifest
name, and two different exams can otherwise look identical.  The manifest column
is therefore printed from inside the JSON, not parsed out of the path.

⚠️ `metric_no_test_alignment` results are listed separately and never sorted
beside the affine-invariant ones.  They answer a different question and a table
that ranks them together would be wrong in a way that reads as fine.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    default_root = os.environ.get("IRIS_RUNS") or (
        os.environ.get("SCRATCH", "") + "/runs" if os.environ.get("SCRATCH") else "outputs"
    )
    parser.add_argument("--runs-root", type=Path, default=Path(default_root),
                        help=f"Directory holding eval/. Default: {default_root}")
    parser.add_argument("--filter", default="", help="Substring the eval directory name must contain.")
    parser.add_argument("--split", choices=("all", "val", "test"), default="all",
                        help="Matched against the eval directory name.")
    parser.add_argument("--sort", choices=("name", "abs_rel", "step"), default="name")
    return parser.parse_args()


def collect(root: Path, wanted: str, split: str) -> tuple[list[dict], list[dict]]:
    affine, metric = [], []
    eval_root = root / "eval"
    if not eval_root.is_dir():
        raise SystemExit(f"No eval directory under {root}. Set --runs-root or $IRIS_RUNS.")
    for path in sorted(eval_root.glob("*/eval_*.json")):
        name = path.parent.name
        if wanted and wanted not in name:
            continue
        if split != "all" and f"_{split}_" not in name:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:                      # a job killed mid-write
            print(f"[warn] unreadable {path}: {error}")
            continue
        if "abs_rel" not in payload:
            continue
        calibration = payload.get("metric_calibration") or {}
        row = {
            "dir": name,
            "file": path.name,
            "step": payload.get("checkpoint_epoch"),
            "prompt": payload.get("val_caption_mode"),
            "frames": payload.get("val_samples"),
            "stride": payload.get("val_stride"),
            "manifest": Path(str(payload.get("val_manifest", ""))).name,
            "checkpoint": payload.get("checkpoint"),
            "abs_rel": payload.get("abs_rel"),
            "rmse": payload.get("rmse"),
            "a1": payload.get("a1"),
            "sq_rel": payload.get("sq_rel"),
            "rmse_log": payload.get("rmse_log"),
            "mode": payload.get("evaluation_mode", "affine_invariant"),
            "source": calibration.get("metric_source", ""),
            "condition_latent": payload.get("condition_latent"),
        }
        (metric if row["mode"] != "affine_invariant" else affine).append(row)
    return affine, metric


def show(rows: list[dict], title: str, sort: str, metric_columns: bool) -> None:
    if not rows:
        return
    key = {
        "name": lambda r: r["dir"],
        "abs_rel": lambda r: r["abs_rel"],
        "step": lambda r: (r["step"] is None, r["step"]),
    }[sort]
    rows = sorted(rows, key=key)
    width = min(60, max(len(r["dir"]) for r in rows))
    print(f"\n=== {title} ({len(rows)} results) ===")
    header = (
        f"{'eval dir':<{width}} {'step':>6} {'prompt':>9} {'frames':>7} "
        f"{'AbsRel':>8} {'RMSE':>7} {'d1':>7}"
    )
    if metric_columns:
        header += f" {'SqRel':>7} {'RMSElog':>8} {'source':>14}"
    header += "  manifest"
    print(header)
    print("-" * len(header))
    for r in rows:
        line = (
            f"{r['dir'][:width]:<{width}} {str(r['step']):>6} {str(r['prompt']):>9} "
            f"{str(r['frames']):>7} {r['abs_rel']:>8.5f} {r['rmse']:>7.3f} {r['a1']:>7.4f}"
        )
        if metric_columns:
            sq = r["sq_rel"] if r["sq_rel"] is not None else float("nan")
            rl = r["rmse_log"] if r["rmse_log"] is not None else float("nan")
            line += f" {sq:>7.4f} {rl:>8.4f} {r['source']:>14}"
        line += f"  {r['manifest']}"
        print(line)


def main() -> int:
    args = parse_args()
    affine, metric = collect(args.runs_root, args.filter, args.split)
    show(affine, "affine_invariant  (per-frame affine fitted to the evaluated split's GT)",
         args.sort, metric_columns=False)
    show(metric, "metric_no_test_alignment  (no fit; NOT comparable with the table above)",
         args.sort, metric_columns=True)

    # The selection rule, applied here rather than left to the eye. Grouped by
    # (arm, manifest, prompt) because a step is only comparable with the steps
    # that sat the same exam.
    val = [r for r in affine if "_val_" in r["dir"]]
    if val:
        groups: dict[tuple, list[dict]] = {}
        for r in val:
            arm = r["dir"].split("_val_")[0]
            groups.setdefault((arm, r["manifest"], r["prompt"]), []).append(r)
        print("\n=== lowest val AbsRel per (arm, manifest, prompt) ===")
        for (arm, manifest, prompt), rows in sorted(groups.items()):
            best = min(rows, key=lambda r: r["abs_rel"])
            steps = sorted(r["step"] for r in rows if r["step"] is not None)
            edge = ""
            if len(rows) > 1 and best["step"] in (steps[:1] + steps[-1:]):
                edge = "  <-- at the sampled edge; the minimum may not be bracketed"
            print(
                f"  {arm} / prompt {prompt} / {manifest}\n"
                f"    steps seen {steps}\n"
                f"    best step {best['step']}  AbsRel {best['abs_rel']:.5f}  "
                f"RMSE {best['rmse']:.3f}  d1 {best['a1']:.4f}{edge}"
            )
    if not affine and not metric:
        print("No results matched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
