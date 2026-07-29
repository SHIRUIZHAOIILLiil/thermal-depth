"""Manifest, GT, checkpoint, and reproducibility metadata I/O."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .core import build_valid_mask


THERMAL_KEYS = ("thermal_path", "thermal", "thr_path")
THERMAL_GT_KEYS = ("thermal_depth_path", "thermal_filtered_depth_path", "depth_filtered_path")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def first_value(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if row.get(key): return str(row[key])
    return None


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    thermal_path: str
    thermal_gt_path: str
    sequence: str
    condition: str
    split: str
    manifest_line: int
    caption: str | None = None


def load_manifest(path: str | Path, data_root: str | Path, *, allow_legacy_depth_path: bool = False):
    """Load JSONL and enforce thermal-view inputs and GT."""
    manifest, root = Path(path).resolve(), Path(data_root).resolve()
    samples: list[ManifestSample] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, text in enumerate(handle, 1):
            if not text.strip(): continue
            row = json.loads(text)
            thermal_value = first_value(row, THERMAL_KEYS)
            depth_value = first_value(row, THERMAL_GT_KEYS)
            if depth_value is None and allow_legacy_depth_path:
                depth_value = row.get("depth_path") or row.get("depth")
            if not thermal_value or not depth_value:
                raise ValueError(
                    f"Manifest line {line_number} lacks explicit thermal input/thermal-view filtered GT; "
                    "RGB-view depth is deliberately not a fallback"
                )
            if row.get("gt_view") not in (None, "thermal", "thermal-view"):
                raise ValueError(f"Manifest line {line_number} declares non-thermal GT view: {row.get('gt_view')!r}")
            thermal, depth = resolve_path(root, thermal_value), resolve_path(root, str(depth_value))
            samples.append(ManifestSample(
                sample_id=str(row.get("id") or row.get("sample_id") or Path(thermal_value).stem),
                thermal_path=str(thermal), thermal_gt_path=str(depth),
                sequence=str(row.get("sequence") or row.get("sequence_id") or "unknown"),
                condition=str(row.get("condition") or row.get("weather") or "unknown").lower(),
                split=str(row.get("split") or "unknown"), manifest_line=line_number,
                caption=None if row.get("caption") is None else str(row["caption"]),
            ))
    if not samples: raise ValueError(f"Manifest contains no samples: {manifest}")
    return samples, {"path": str(manifest), "sha256": sha256_file(manifest), "sample_count": len(samples)}


def load_ms2_gt(path: str | Path, depth_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Decode official filtered MS2 depth: uint16 / depth_scale metres."""
    with Image.open(path) as image: raw = np.asarray(image)
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError(f"Expected 2-D uint16 filtered MS2 depth, got {raw.shape} {raw.dtype} at {path}")
    return raw, raw.astype(np.float32) / float(depth_scale)


def audit_gt_samples(samples, *, depth_scale: float, min_depth: float, max_depth: float, count: int = 8):
    if len(samples) < count: raise ValueError(f"GT audit requires at least {count} manifest samples, found {len(samples)}")
    positions = np.linspace(0, len(samples) - 1, count, dtype=int)
    records = []
    for index in positions:
        sample = samples[int(index)]
        raw, depth = load_ms2_gt(sample.thermal_gt_path, depth_scale)
        valid = build_valid_mask(depth, min_depth, max_depth)
        records.append({
            "sample_id": sample.sample_id, "path": sample.thermal_gt_path,
            "dtype": str(raw.dtype), "raw_min": int(raw.min()), "raw_max": int(raw.max()),
            "decoded_min_m": float(depth.min()), "decoded_max_m": float(depth.max()),
            "zero_count": int((raw == 0).sum()), "invalid_count": int((~valid).sum()),
            "valid_pixel_count": int(valid.sum()), "valid_ratio": float(valid.mean()),
            "depth_scale_uint16_per_metre": float(depth_scale),
            "validity_rule": f"finite(gt) & gt > {min_depth} & gt < {max_depth}",
        })
    return records


def checkpoint_info(path: str | Path | None) -> dict[str, Any]:
    if path is None: return {"path": None, "sha256": None, "exists": False}
    value = Path(path).resolve()
    return {"path": str(value), "sha256": sha256_file(value) if value.is_file() else None,
            "exists": value.is_file(), "size_bytes": value.stat().st_size if value.is_file() else None}


def sample_to_dict(sample: ManifestSample) -> dict[str, Any]:
    return asdict(sample)
