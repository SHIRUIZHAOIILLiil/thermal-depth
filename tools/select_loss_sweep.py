#!/usr/bin/env python3
"""Compare fixed-step validation results without consulting test metrics.

Example:
  python tools/select_loss_sweep.py \
    --control l050=/path/to/eval_eval_affine_invariant_log_space.json \
    --candidate l025=/path/to/result.json l100=/path/to/result.json

The control is always eligible. A candidate must preserve AbsRel and delta1
within the frozen tolerances, then candidates are ranked by RMSE, SqRel, and
AbsRel. The last stdout line is only the selected label, so a shell can capture
it; the audit table is written to stderr.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


METRICS = ("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3")


def labelled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=JSON_PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("expected non-empty LABEL=JSON_PATH")
    return label, Path(raw_path)


def load_record(spec: tuple[str, Path]) -> tuple[str, dict[str, float], Path]:
    label, path = spec
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"cannot read {label} from {path}: {exc}") from exc
    missing = [key for key in METRICS if key not in payload]
    if missing:
        raise SystemExit(f"{path} is missing metrics: {', '.join(missing)}")
    record = {key: float(payload[key]) for key in METRICS}
    bad = [key for key, value in record.items() if not math.isfinite(value)]
    if bad:
        raise SystemExit(f"{path} has non-finite metrics: {', '.join(bad)}")
    return label, record, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, type=labelled_path)
    parser.add_argument(
        "--candidate", required=True, nargs="+", type=labelled_path,
        help="One or more LABEL=JSON_PATH validation results.",
    )
    parser.add_argument(
        "--max-abs-rel-regression", type=float, default=0.0005,
        help="Maximum candidate AbsRel increase relative to control.",
    )
    parser.add_argument(
        "--max-a1-regression", type=float, default=0.001,
        help="Maximum candidate delta1 decrease relative to control.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_abs_rel_regression < 0 or args.max_a1_regression < 0:
        raise SystemExit("regression tolerances must be non-negative")

    control = load_record(args.control)
    candidates = [load_record(spec) for spec in args.candidate]
    labels = [control[0], *(item[0] for item in candidates)]
    if len(labels) != len(set(labels)):
        raise SystemExit("labels must be unique")

    control_metrics = control[1]
    eligible = [control]
    decisions: dict[str, str] = {control[0]: "control"}
    for item in candidates:
        label, metrics, _ = item
        reasons = []
        if metrics["abs_rel"] > (
            control_metrics["abs_rel"] + args.max_abs_rel_regression
        ):
            reasons.append("AbsRel gate")
        if metrics["a1"] < control_metrics["a1"] - args.max_a1_regression:
            reasons.append("a1 gate")
        if reasons:
            decisions[label] = "reject: " + ", ".join(reasons)
        else:
            decisions[label] = "eligible"
            eligible.append(item)

    winner = min(
        eligible,
        key=lambda item: (
            item[1]["rmse"],
            item[1]["sq_rel"],
            item[1]["abs_rel"],
            item[0],
        ),
    )

    print(
        "label             AbsRel    SqRel     RMSE  RMSElog       a1       a2       a3  decision",
        file=sys.stderr,
    )
    for label, metrics, _ in [control, *candidates]:
        print(
            f"{label:<16} "
            f"{metrics['abs_rel']:>8.5f} {metrics['sq_rel']:>8.5f} "
            f"{metrics['rmse']:>8.4f} {metrics['rmse_log']:>8.5f} "
            f"{metrics['a1']:>8.5f} {metrics['a2']:>8.5f} {metrics['a3']:>8.5f}  "
            f"{decisions[label]}",
            file=sys.stderr,
        )
    print(f"selected: {winner[0]}", file=sys.stderr)
    print(winner[0])


