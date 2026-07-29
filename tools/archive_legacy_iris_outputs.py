"""Archive research evidence and remove regenerable legacy Iris outputs.

Dry-run is the default. Execution requires an explicit confirmation token.
Every resolved source and destination is verified to remain below repo/outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path


CONFIRMATION = "ARCHIVE_AND_DELETE_LEGACY_IRIS_OUTPUTS"
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = (ROOT / "outputs").resolve()
ARCHIVE = (OUTPUTS / "_legacy_archive_20260701").resolve()

BIG_RUNS = {
    "adapter_v0_thermal_only_short_run_v2": ["checkpoint_best.pt"],
    "adapter_v0_unet_joint_short_run": ["checkpoint_best.pt"],
    "adapter_v0_unet_joint_conservative_short_run": [],
    "adapter_v0_full_caption_attn_short_run": ["checkpoint_best_real_aligned_rmse.pt"],
    "adapter_v0_caption_attn_short_run": ["checkpoint_best_aligned_rmse.pt"],
    "adapter_v0_caption_attn_dryrun": [],
    "adapter_v0_caption_attn_dryrun_metrics": [],
}

MOVE_WHOLE = [
    "adapter_v0_overfit32",
    "adapter_v0_caption_sensitivity_v2",
    "iris_rgb_grounding_audit",
    "ms2_000000_caption_ablation",
]

DELETE_AFTER_MANIFEST = [
    "adapter_v0_caption_sensitivity",
    "adapter_v0",
    "anythermal_lotus_direct_smoke",
    "adapter_v0_caption_interface_audit",
]

PRESERVE_EXTENSIONS = {".json", ".jsonl", ".csv", ".txt", ".md", ".yaml", ".yml", ".png", ".jpg", ".jpeg"}


def ensure_below_outputs(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == OUTPUTS or not resolved.is_relative_to(OUTPUTS):
        raise RuntimeError(f"Unsafe path outside outputs or outputs root itself: {resolved}")
    return resolved


def size_bytes(path: Path) -> int:
    if path.is_file(): return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def preserved_files(run: Path, checkpoint_names: list[str]) -> list[Path]:
    selected = []
    checkpoints = set(checkpoint_names)
    for path in run.rglob("*"):
        if not path.is_file(): continue
        relative = path.relative_to(run)
        if relative.as_posix() in checkpoints or path.name in checkpoints:
            selected.append(path)
        elif path.suffix.lower() in PRESERVE_EXTENSIONS:
            selected.append(path)
    return sorted(set(selected))


def build_plan() -> dict:
    actions = []
    for name, checkpoints in BIG_RUNS.items():
        source = ensure_below_outputs(OUTPUTS / name)
        if not source.is_dir(): continue
        kept = preserved_files(source, checkpoints)
        kept_bytes = sum(path.stat().st_size for path in kept)
        total = size_bytes(source)
        actions.append({"action": "selective_archive_then_delete", "name": name, "source": str(source),
                        "total_bytes": total, "archive_bytes": kept_bytes, "delete_bytes": total - kept_bytes,
                        "selected_checkpoints": checkpoints, "preserved_file_count": len(kept)})
    for name in MOVE_WHOLE:
        source = ensure_below_outputs(OUTPUTS / name)
        if source.is_dir():
            actions.append({"action": "move_whole_to_archive", "name": name, "source": str(source),
                            "total_bytes": size_bytes(source), "archive_bytes": size_bytes(source), "delete_bytes": 0})
    for name in DELETE_AFTER_MANIFEST:
        source = ensure_below_outputs(OUTPUTS / name)
        if source.is_dir():
            actions.append({"action": "delete_after_inventory", "name": name, "source": str(source),
                            "total_bytes": size_bytes(source), "archive_bytes": 0, "delete_bytes": size_bytes(source)})
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "outputs_root": str(OUTPUTS),
            "archive_root": str(ARCHIVE), "actions": actions,
            "estimated_released_bytes": sum(a["delete_bytes"] for a in actions),
            "estimated_archived_bytes": sum(a["archive_bytes"] for a in actions)}


def inventory_tree(path: Path) -> list[dict]:
    records = []
    for item in path.rglob("*"):
        if item.is_file():
            records.append({"path": str(item), "relative": item.relative_to(OUTPUTS).as_posix(),
                            "size_bytes": item.stat().st_size})
    return records


def remove_tree(path: Path) -> None:
    """Remove a verified outputs subtree, clearing Windows read-only bits."""
    ensure_below_outputs(path)

    def clear_readonly(function, target, excinfo):
        os.chmod(target, stat.S_IWRITE)
        function(target)

    shutil.rmtree(path, onexc=clear_readonly)


def execute(plan: dict) -> dict:
    ensure_below_outputs(ARCHIVE)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    pre_inventory = []
    for action in plan["actions"]:
        source = ensure_below_outputs(Path(action["source"]))
        if source.exists(): pre_inventory.extend(inventory_tree(source))
    (ARCHIVE / "pre_cleanup_inventory.json").write_text(json.dumps(pre_inventory, indent=2), encoding="utf-8")

    checkpoint_hashes = []
    for action in plan["actions"]:
        source = ensure_below_outputs(Path(action["source"]))
        if not source.exists(): continue
        destination = ensure_below_outputs(ARCHIVE / action["name"])
        if destination.exists() and action["action"] != "selective_archive_then_delete":
            raise FileExistsError(f"Archive destination already exists: {destination}")
        if action["action"] == "move_whole_to_archive":
            shutil.move(str(source), str(destination))
        elif action["action"] == "selective_archive_then_delete":
            destination.mkdir(parents=True, exist_ok=True)
            selected = preserved_files(source, action["selected_checkpoints"])
            for item in selected:
                relative = item.relative_to(source); target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                is_checkpoint = item.name in action["selected_checkpoints"]
                digest = sha256(item) if is_checkpoint else None
                if target.exists():
                    if target.stat().st_size != item.stat().st_size:
                        raise FileExistsError(f"Archive conflict: {target}")
                    item.unlink()
                else:
                    shutil.move(str(item), str(target))
                if is_checkpoint:
                    checkpoint_hashes.append({"original": str(item), "archived": str(target),
                                              "size_bytes": target.stat().st_size, "sha256": digest})
            ensure_below_outputs(source)
            remove_tree(source)
        elif action["action"] == "delete_after_inventory":
            ensure_below_outputs(source)
            remove_tree(source)

    report = {**plan, "executed_utc": datetime.now(timezone.utc).isoformat(),
              "checkpoint_hashes": checkpoint_hashes,
              "archive_size_bytes": size_bytes(ARCHIVE)}
    (ARCHIVE / "cleanup_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ARCHIVE / "README.md").write_text(
        "# Legacy Iris outputs archived 2026-07-01\n\n"
        "These runs predate the frozen Lotus/MS2 comparison protocol. Selected best checkpoints, "
        "configuration, metrics, logs, and visual evidence were retained. Repeated step checkpoints, "
        "tensor dumps, dry-run weights, and superseded smoke results were removed. These results are "
        "development evidence and must not be mixed with the new Direct/Adapter/Adapter+U-Net table.\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation", default="")
    parser.add_argument("--plan-output", type=Path, default=ROOT / "docs" / "legacy_outputs_cleanup_plan_20260701.json")
    args = parser.parse_args()
    plan = build_plan()
    args.plan_output.parent.mkdir(parents=True, exist_ok=True)
    args.plan_output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if args.execute:
        if args.confirmation != CONFIRMATION:
            raise SystemExit(f"Execution requires --confirmation {CONFIRMATION}")
        report = execute(plan)
        print(json.dumps({"status": "executed", "released_gib_estimate": report["estimated_released_bytes"] / 1024**3,
                          "archive_gib": report["archive_size_bytes"] / 1024**3}, indent=2))
    else:
        print(json.dumps({"status": "dry-run", "actions": len(plan["actions"]),
                          "release_gib_estimate": plan["estimated_released_bytes"] / 1024**3,
                          "archive_gib_estimate": plan["estimated_archived_bytes"] / 1024**3,
                          "plan": str(args.plan_output.resolve())}, indent=2))


if __name__ == "__main__": main()
