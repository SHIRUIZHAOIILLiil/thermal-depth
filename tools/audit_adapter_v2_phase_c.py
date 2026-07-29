"""Audit eight fixed MS2 Train samples for Adapter V2 Phase C.

This is inference/data auditing only.  It never constructs an optimizer and
never reads Val/Test manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.anythermal_lotus_v2 import (  # noqa: E402
    encode_seeded_condition_latent,
    thermal_to_lotus_input,
)
from models.lotus_target_v2 import trunc_disparity_target  # noqa: E402


DEFAULT_MANIFEST = Path(
    "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/"
    "ms2_fixed_sequence_internvl3_8b_filtered_caption_train.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ms2-root", type=Path, default=Path("/mnt/e/dataset/ms2"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lotus_line_v2/phase_c_audit_8"),
    )
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--processing-res", type=int, default=768)
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Audit thermal/depth inputs without loading the VAE.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_train_rows(path: Path) -> list[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError(f"Non-Train sample found in Train manifest: {row.get('id')}")
            depth_path = row.get("thermal_depth_path") or row.get("depth_path")
            if not depth_path or "/thr/" not in str(depth_path).replace("\\", "/"):
                raise ValueError(f"Sample lacks thermal-view GT: {row.get('id')}")
            rows.append(
                {
                    "id": str(row["id"]),
                    "thermal_path": str(row["thermal_path"]),
                    "depth_path": str(depth_path),
                }
            )
    if len(rows) < 8:
        raise ValueError(f"Need at least 8 Train samples, found {len(rows)}.")
    return rows


def choose_fixed_eight(rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    indices = np.linspace(0, len(rows) - 1, 8, dtype=int)
    return [dict(rows[int(index)], manifest_index=int(index)) for index in indices]


def tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    values = tensor.detach().float().cpu()
    return {
        "shape": list(values.shape),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "channel_mean": values.mean((0, 2, 3)).tolist(),
        "channel_std": values.std((0, 2, 3), unbiased=False).tolist(),
        "finite": bool(torch.isfinite(values).all()),
    }


def load_vae(args: argparse.Namespace):
    from diffusers import AutoencoderKL

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    vae = AutoencoderKL.from_pretrained(
        args.lotus_model_path,
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(torch.device(args.device))
    return vae.requires_grad_(False).eval()


def main() -> None:
    args = parse_args()
    if args.depth_scale <= 0:
        raise ValueError("depth_scale must be positive.")
    manifest = args.train_manifest.resolve()
    ms2_root = args.ms2_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = choose_fixed_eight(read_train_rows(manifest))
    vae = None if args.data_only else load_vae(args)
    records = []
    for sample_index, row in enumerate(selected):
        thermal_path = ms2_root / row["thermal_path"]
        depth_path = ms2_root / row["depth_path"]
        if not thermal_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError(
                f"Missing sample {row['id']}: thermal={thermal_path}, depth={depth_path}"
            )

        thermal = thermal_to_lotus_input(
            thermal_path,
            processing_res=args.processing_res,
        )
        with Image.open(depth_path) as image:
            raw_depth = np.asarray(image).copy()
        if raw_depth.ndim == 3:
            raw_depth = raw_depth[..., 0]
        depth_m = torch.from_numpy(raw_depth.astype(np.float32) / args.depth_scale)
        valid_mask = depth_m > 0
        disparity = trunc_disparity_target(depth_m, valid_mask)
        valid_depth = depth_m[valid_mask]
        valid_disparity = disparity.disparity[disparity.valid_mask]

        record: Dict[str, Any] = {
            **row,
            "thermal_path": str(thermal_path),
            "depth_path": str(depth_path),
            "thermal": thermal.diagnostics,
            "depth": {
                "raw_dtype": str(raw_depth.dtype),
                "raw_shape": list(raw_depth.shape),
                "depth_scale": float(args.depth_scale),
                "valid_ratio": float(valid_mask.float().mean()),
                "valid_min_m": float(valid_depth.min()),
                "valid_max_m": float(valid_depth.max()),
                "valid_mean_m": float(valid_depth.mean()),
                "zero_count": int((depth_m == 0).sum()),
                "invalid_policy": "excluded before reciprocal/quantile/normalization",
                "invalid_fill_before_vae": "not applicable: sparse depth is never VAE-encoded",
            },
            "trunc_disparity": {
                "valid_min": float(valid_disparity.min()),
                "valid_max": float(valid_disparity.max()),
                "q02": float(disparity.disparity_min),
                "q98": float(disparity.disparity_max),
                "normalized_valid_min": float(disparity.values[disparity.valid_mask].min()),
                "normalized_valid_max": float(disparity.values[disparity.valid_mask].max()),
            },
            "teacher_source": "dense thermal image only; no depth/GT input",
        }
        if vae is not None:
            teacher = encode_seeded_condition_latent(
                vae,
                thermal.tensor,
                seed=args.seed + sample_index,
            )
            record["teacher_seed"] = args.seed + sample_index
            record["target_condition_latent"] = tensor_stats(teacher)
        records.append(record)

    summary = {
        "phase": "Adapter V2 Phase C fixed-8 audit",
        "training": False,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "sample_selection": "8 uniformly spaced rows from Train manifest",
        "sample_count": len(records),
        "uses_val": False,
        "uses_test": False,
        "lotus_model": None if args.data_only else args.lotus_model_path,
        "processing_res": args.processing_res,
        "depth_scale": args.depth_scale,
        "records": records,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
