"""Inventory and clean the derived artefacts under outputs/.

Read-only by default: it prints what each category costs and what deleting it
would mean.  Deletion happens only for the categories named with `--delete`,
and every category enforces its own safety gate.

Categories
----------
raw_predictions   Per-image `.npy` prediction dumps under `*/raw_predictions/`.
                  Pure derived data: the metrics were already computed from them
                  (`per_sample_metrics.csv` / `*_run_metadata.json`).  Gate: the
                  run must have its metrics file, otherwise the dump is the only
                  record and is kept.
vis               Visualisation PNGs under `*/vis/`. Always regenerable.
intermediate      Mid-training checkpoints (`*_step_*.pt`). Gate (2026-07-20
                  rule): the directory has `summary.json` AND at least one
                  surviving `*_end.pt`.
gate_runs         `smoke_*` / `overfit_*` output directories. Throwaway.

    python tools/archive_outputs.py                              # inventory only
    python tools/archive_outputs.py --delete raw_predictions vis  # actually remove
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

GB = 1024**3
CATEGORIES = ("raw_predictions", "vis", "intermediate", "gate_runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="*",
        type=Path,
        default=[Path("outputs/lotus_line_v2"), Path("outputs/route_suite"), Path("outputs/ms2_official")],
    )
    parser.add_argument("--delete", nargs="*", default=[], choices=CATEGORIES)
    parser.add_argument("--report", type=Path, default=None, help="Write the inventory as JSON.")
    parser.add_argument(
        "--full-inventory",
        action="store_true",
        help=(
            "Scan every category even when --delete names only some. Sizing "
            "raw_predictions/vis means stat-ing ~740k files, which takes a long "
            "time over WSL's /mnt/e; by default --delete only scans what it needs."
        ),
    )
    return parser.parse_args()


def directory_size(path: Path) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                pass
    return total


def has_metrics(run: Path) -> bool:
    return any(
        (run / name).exists()
        for name in ("per_sample_metrics.csv", "official_run_metadata.json", "run_metadata.json")
    )


def collect(roots, wanted: set[str]) -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = {name: [] for name in CATEGORIES}
    for root in roots:
        if not root.exists():
            continue
        for run in sorted(p for p in root.iterdir() if p.is_dir()):
            name = run.name
            if name.startswith(("smoke_", "overfit_")):
                if "gate_runs" in wanted:
                    found["gate_runs"].append(
                        {"path": str(run), "bytes": directory_size(run), "safe": True, "why": "gate run"}
                    )
                continue

            raw = run / "raw_predictions"
            if raw.is_dir() and "raw_predictions" in wanted:
                keep = not has_metrics(run)
                found["raw_predictions"].append(
                    {
                        "path": str(raw),
                        "bytes": directory_size(raw),
                        "safe": not keep,
                        "why": "metrics already computed" if not keep else "NO metrics file - dump is the only record",
                    }
                )

            vis = run / "vis"
            if vis.is_dir() and "vis" in wanted:
                found["vis"].append(
                    {"path": str(vis), "bytes": directory_size(vis), "safe": True, "why": "regenerable"}
                )

            checkpoints = sorted(run.glob("*.pt")) if "intermediate" in wanted else []
            if checkpoints:
                finished = [p for p in checkpoints if p.name.endswith(("_end.pt", "end_weights.pt", "best_weights.pt"))]
                intermediate = [p for p in checkpoints if p not in finished]
                if intermediate:
                    gate_ok = (run / "summary.json").exists() and bool(finished)
                    found["intermediate"].append(
                        {
                            "path": str(run),
                            "files": [str(p) for p in intermediate],
                            "bytes": sum(p.stat().st_size for p in intermediate),
                            "safe": gate_ok,
                            "why": "summary.json + surviving _end.pt"
                            if gate_ok
                            else "missing summary.json or no _end.pt survivor",
                        }
                    )
    return found


def main() -> None:
    args = parse_args()
    wanted = set(CATEGORIES) if (args.full_inventory or not args.delete) else set(args.delete)
    if wanted != set(CATEGORIES):
        print(f"[scan] only {', '.join(sorted(wanted))} (use --full-inventory for everything)", flush=True)
    found = collect(args.roots, wanted)

    print(f"{'category':16s} {'entries':>8s} {'safe GB':>10s} {'held GB':>10s}")
    print("-" * 48)
    for category in sorted(wanted):
        entries = found[category]
        safe = sum(entry["bytes"] for entry in entries if entry["safe"])
        held = sum(entry["bytes"] for entry in entries if not entry["safe"])
        print(f"{category:16s} {len(entries):8d} {safe / GB:10.1f} {held / GB:10.1f}")
    grand = sum(entry["bytes"] for entries in found.values() for entry in entries if entry["safe"])
    print("-" * 48)
    print(f"{'TOTAL SAFE':16s} {'':8s} {grand / GB:10.1f}")

    held_entries = [
        (category, entry)
        for category in sorted(wanted)
        for entry in found[category]
        if not entry["safe"] and entry["bytes"] > GB
    ]
    if held_entries:
        print("\nHeld back (gate not satisfied):")
        for category, entry in sorted(held_entries, key=lambda item: -item[1]["bytes"]):
            print(f"  {entry['bytes'] / GB:7.1f} GB  [{category}] {entry['path']}  <- {entry['why']}")

    if args.report:
        args.report.write_text(json.dumps(found, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nInventory written to {args.report}")

    if not args.delete:
        print("\nRead-only run. Pass --delete <category> [...] to remove the safe entries.")
        return

    removed = 0
    for category in args.delete:
        for entry in found[category]:
            if not entry["safe"]:
                print(f"[hold] {entry['path']}: {entry['why']}")
                continue
            target = Path(entry["path"])
            if category == "intermediate":
                for file_path in entry["files"]:
                    size = os.path.getsize(file_path)
                    os.remove(file_path)
                    removed += size
                    print(f"[del] {file_path} ({size / GB:.2f} GB)")
            else:
                size = entry["bytes"]
                shutil.rmtree(target)
                removed += size
                print(f"[del] {target} ({size / GB:.2f} GB)")
    print(f"\nFreed {removed / GB:.1f} GB")


if __name__ == "__main__":
    main()
