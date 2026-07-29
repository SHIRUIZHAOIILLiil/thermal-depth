"""Write the final audit report after the 2026-07-01 legacy output cleanup."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = (ROOT / "outputs").resolve()
ARCHIVE = (OUTPUTS / "_legacy_archive_20260701").resolve()

REPRESENTATIVE_CHECKPOINTS = [
    "adapter_v0_thermal_only_short_run_v2/checkpoint_best.pt",
    "adapter_v0_unet_joint_short_run/checkpoint_best.pt",
    "adapter_v0_full_caption_attn_short_run/checkpoint_best_real_aligned_rmse.pt",
    "adapter_v0_caption_attn_short_run/checkpoint_best_aligned_rmse.pt",
]

REMOVED_LOW_VALUE = [
    "adapter_v0_unet_joint_conservative_short_run checkpoints/tensors (metadata/images archived)",
    "adapter_v0_caption_attn_dryrun checkpoints/tensors (metadata/images archived)",
    "adapter_v0_caption_attn_dryrun_metrics checkpoints/tensors (metadata/images archived)",
    "intermediate checkpoints and tensors from retained representative runs",
    "adapter_v0_caption_sensitivity (superseded by v2)",
    "adapter_v0 smoke",
    "anythermal_lotus_direct_smoke",
    "adapter_v0_caption_interface_audit",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    if ARCHIVE == OUTPUTS or not ARCHIVE.is_relative_to(OUTPUTS): raise RuntimeError("Unsafe archive path")
    checkpoints = []
    for relative in REPRESENTATIVE_CHECKPOINTS:
        path = ARCHIVE / relative
        if not path.is_file(): raise FileNotFoundError(path)
        checkpoints.append({"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    archive_dirs = []
    for directory in sorted(item for item in ARCHIVE.iterdir() if item.is_dir()):
        archive_dirs.append({"name": directory.name, "size_bytes": tree_size(directory),
                             "file_count": sum(1 for item in directory.rglob("*") if item.is_file())})
    archive_bytes = tree_size(ARCHIVE)
    report = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "outputs_root": str(OUTPUTS), "archive_root": str(ARCHIVE),
        "pre_cleanup_estimated_gib": 83.01634758058935,
        "archive_gib": archive_bytes / 1024**3,
        "estimated_released_gib": 83.01634758058935 - archive_bytes / 1024**3,
        "remaining_active_output_directories": ["ms2_unified"],
        "archive_directories": archive_dirs,
        "representative_checkpoint_hashes": checkpoints,
        "removed_low_value_material": REMOVED_LOW_VALUE,
        "research_status": "Legacy development evidence only; do not mix with the new Direct/Adapter/Adapter+U-Net comparison.",
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    (ARCHIVE / "cleanup_report.json").write_text(text, encoding="utf-8")
    (ROOT / "docs" / "legacy_outputs_cleanup_report_20260701.json").write_text(text, encoding="utf-8")
    (ARCHIVE / "README.md").write_text(
        "# Legacy Iris outputs archived 2026-07-01\n\n"
        "These runs predate the frozen Lotus/MS2 comparison protocol. Selected best checkpoints, "
        "configuration, metrics, logs, and visual evidence were retained. Repeated checkpoints, "
        "tensor dumps, dry-run weights, and superseded smoke results were removed.\n\n"
        "Do not mix these development results with the new Direct/Adapter/Adapter+U-Net table.\n",
        encoding="utf-8",
    )
    print(json.dumps({"archive_gib": report["archive_gib"], "released_gib": report["estimated_released_gib"],
                      "checkpoint_count": len(checkpoints)}, indent=2))


if __name__ == "__main__": main()
