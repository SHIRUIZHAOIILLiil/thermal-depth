"""VTD (VIS-TIR) dataset adapter for the RGB-teacher -> thermal-student line.

VTD gives *pixel-aligned* RGB (VIS) and thermal (TIR) via a beam-splitter rig,
with Velodyne-64 depth GT. That alignment is exactly what MS2 lacks (frozen
conclusion 15), so it is the legal venue for the ``rgb_vae`` teacher mode.

This module exposes the SAME per-sample interface as ``load_raw_sample`` in
``tools/train_ms2_rgb_teacher_thermal_student.py``:

    {rgb, thermal, thermal_features, depth, valid_mask, metadata}

so the training script only has to swap the loader, not the model/loss code.

HOW TO USE AFTER DOWNLOAD
-------------------------
1. Extract VTD somewhere, e.g. /mnt/e/dataset/vtd.
2. Look at the tree, then fill in the glob patterns / depth encoding in
   ``VTDConfig`` (every unknown is collected there, each marked TODO).
3. Run the probe — it loads ONE sample end-to-end (minus AnyThermal) and prints
   every shape/dtype/range plus a pass/fail checklist:

     python tools/vtd_dataset.py --probe --vtd-root /mnt/e/dataset/vtd

4. When the probe is green, train with:
     python tools/train_ms2_rgb_teacher_thermal_student.py \
       --dataset vtd --vtd-root /mnt/e/dataset/vtd --teacher-mode rgb_vae ...

NOTHING here is MS2-specific except that it reuses the two Lotus-input helpers;
all VTD format assumptions live in VTDConfig.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
for path in (ROOT, ROOT / "lotus", TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.anythermal_lotus_v2 import thermal_to_lotus_input  # noqa: E402
from train_ms2_rgb_teacher_thermal_student import rgb_to_lotus_input  # noqa: E402


# ============================================================================
#  FORMAT CONFIG — fill these in once you have looked at the extracted VTD tree.
#  Run `--probe` to have the loader tell you the real shapes/dtypes/ranges.
# ============================================================================
@dataclass
class VTDConfig:
    root: Path = Path("/mnt/e/dataset/vtd")

    # --- file discovery -----------------------------------------------------
    # Glob patterns (relative to root) for the LEFT camera of each modality.
    # They must each capture a frame key so the three modalities can be paired.
    # TODO: replace with the real VTD layout after inspecting the tree.
    vis_glob: str = "**/vis/left/*.png"        # RGB (visible)
    tir_glob: str = "**/tir/left/*.png"        # thermal (infrared)
    depth_glob: str = "**/depth/left/*.png"    # LiDAR-projected depth GT
    # How to extract the pairing key from a path (default: filename stem).
    frame_key_regex: str = r"(\d+)"            # first integer run in the filename

    # --- thermal ------------------------------------------------------------
    # thermal_to_lotus_input reads a single-channel high-bit array and maps it
    # to uint8. If VTD TIR is already 8-bit or pseudo-colour, set this False and
    # extend load_vtd_thermal accordingly. TODO: confirm via probe.
    thermal_single_channel: bool = True

    # --- depth GT -----------------------------------------------------------
    # encoding: "depth_uint16"  -> meters = raw / depth_scale
    #           "disparity_uint16" -> disparity = raw / depth_scale
    #           "depth_float"   -> raw already in meters (PFM/float TIFF)
    # TODO: confirm which one VTD uses and the scale via probe.
    depth_encoding: str = "depth_uint16"
    depth_scale: float = 256.0
    min_depth: float = 0.1
    max_depth: float = 80.0
    # If the depth map resolution differs from the thermal frame, resize it with
    # nearest-neighbour (keeps sparse LiDAR points crisp). Off by default so a
    # silent size mismatch surfaces loudly in the probe first.
    resize_depth_to_thermal: bool = False

    def signature(self) -> str:
        payload = json.dumps({k: str(v) for k, v in asdict(self).items()}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_args(cls, args) -> "VTDConfig":
        cfg = cls()
        if getattr(args, "vtd_root", None) is not None:
            cfg.root = Path(args.vtd_root)
        return cfg


# ---------------------------------------------------------------- discovery
def _frame_key(path: Path, regex: str) -> str:
    match = re.search(regex, path.stem)
    return match.group(1) if match else path.stem


def scan_vtd(cfg: VTDConfig) -> list[dict]:
    """Pair VIS/TIR/depth files by frame key into training rows."""
    root = Path(cfg.root)

    def collect(pattern: str) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for match in sorted(glob.glob(str(root / pattern), recursive=True)):
            p = Path(match)
            out[_frame_key(p, cfg.frame_key_regex)] = p
        return out

    vis = collect(cfg.vis_glob)
    tir = collect(cfg.tir_glob)
    depth = collect(cfg.depth_glob)
    keys = sorted(set(vis) & set(tir) & set(depth))
    if not keys:
        raise FileNotFoundError(
            "No paired VTD frames found. Check VTDConfig globs against the tree:\n"
            f"  vis  {cfg.vis_glob}  -> {len(vis)} files\n"
            f"  tir  {cfg.tir_glob}  -> {len(tir)} files\n"
            f"  depth {cfg.depth_glob} -> {len(depth)} files"
        )
    rows = []
    for index, key in enumerate(keys):
        rows.append(
            {
                "id": f"vtd_{key}",
                "manifest_index": index,
                "thermal_path": tir[key],
                "rgb_path": vis[key],
                "depth_path": depth[key],
            }
        )
    return rows


# ------------------------------------------------------------------- loaders
def load_vtd_thermal(path: Path, cfg: VTDConfig):
    """Return the Lotus thermal input tensor [1,3,H,W] in [-1,1] + diagnostics."""
    if cfg.thermal_single_channel:
        # Reuse the exact MS2 thermal path so AnyThermal sees the same statistics.
        out = thermal_to_lotus_input(path, processing_res=0)
        return out.tensor, out.diagnostics
    raise NotImplementedError(
        "VTD TIR is not single-channel; extend load_vtd_thermal for its format."
    )


def load_vtd_disparity(path: Path, cfg: VTDConfig, target_hw=None):
    """Return (disparity[1,H,W], valid_mask[1,H,W]) matching load_gt_disparity."""
    raw = np.asarray(Image.open(path)).astype(np.float64)
    if raw.ndim != 2:
        raise ValueError(f"Expected single-channel depth, got {raw.shape} from {path}.")

    if cfg.depth_encoding == "depth_uint16":
        depth = raw / cfg.depth_scale
        valid = (depth > cfg.min_depth) & (depth < cfg.max_depth) & np.isfinite(depth)
        disparity = np.zeros_like(depth)
        disparity[valid] = 1.0 / depth[valid]
    elif cfg.depth_encoding == "disparity_uint16":
        disparity = raw / cfg.depth_scale
        with np.errstate(divide="ignore"):
            depth = np.where(disparity > 0, 1.0 / np.maximum(disparity, 1e-9), 0.0)
        valid = (disparity > 0) & (depth > cfg.min_depth) & (depth < cfg.max_depth) & np.isfinite(depth)
        disparity = np.where(valid, disparity, 0.0)
    elif cfg.depth_encoding == "depth_float":
        depth = raw
        valid = (depth > cfg.min_depth) & (depth < cfg.max_depth) & np.isfinite(depth)
        disparity = np.zeros_like(depth)
        disparity[valid] = 1.0 / depth[valid]
    else:
        raise ValueError(f"Unknown depth_encoding {cfg.depth_encoding!r}.")

    disp_t = torch.from_numpy(disparity.astype(np.float32))[None]
    mask_t = torch.from_numpy(valid.astype(np.float32))[None]
    if target_hw is not None and tuple(disp_t.shape[-2:]) != tuple(target_hw):
        if not cfg.resize_depth_to_thermal:
            raise RuntimeError(
                f"Depth {tuple(disp_t.shape[-2:])} != thermal {tuple(target_hw)}; "
                "set VTDConfig.resize_depth_to_thermal=True or fix the pairing."
            )
        disp_t = F.interpolate(disp_t[None], size=tuple(target_hw), mode="nearest")[0]
        mask_t = F.interpolate(mask_t[None], size=tuple(target_hw), mode="nearest")[0]
    return disp_t, mask_t


def load_vtd_raw_sample(row, anythermal, args, device, cfg: VTDConfig):
    """Same interface as load_raw_sample: {rgb, thermal, thermal_features, depth, valid_mask, metadata}."""
    from models.anythermal_lotus_model import extract_anythermal_feature_pyramid

    thermal_tensor, thermal_diag = load_vtd_thermal(row["thermal_path"], cfg)
    thermal_tensor = thermal_tensor.to(device=device, dtype=torch.float32)
    target_hw = thermal_tensor.shape[-2:]

    rgb_tensor = rgb_to_lotus_input(row["rgb_path"], target_hw).to(device=device, dtype=torch.float32)

    features, _, anythermal_diag = extract_anythermal_feature_pyramid(
        anythermal, row["thermal_path"], enable_grad=False
    )
    features = [f.detach().float().to(device) for f in features]

    disparity, valid_mask = load_vtd_disparity(row["depth_path"], cfg, target_hw=target_hw)
    metadata = {
        "id": row["id"],
        "manifest_index": row["manifest_index"],
        "dataset": "vtd",
        "thermal_hw": list(map(int, thermal_tensor.shape[-2:])),
        "rgb_hw": list(map(int, rgb_tensor.shape[-2:])),
        "thermal_std": thermal_diag.get("converted_uint8_std"),
        "gt_valid_pixels": int(valid_mask.sum()),
    }
    return {
        "rgb": rgb_tensor,
        "thermal": thermal_tensor,
        "thermal_features": features,
        "depth": disparity.to(device),
        "valid_mask": valid_mask.to(device),
        "metadata": metadata,
    }


# --------------------------------------------------------------------- probe
def probe(cfg: VTDConfig) -> None:
    """Load ONE paired sample (file layer only, no AnyThermal) and report."""
    print(f"[probe] VTD root: {cfg.root}")
    rows = scan_vtd(cfg)
    print(f"[probe] paired frames found: {len(rows)}")
    row = rows[0]
    for tag in ("thermal_path", "rgb_path", "depth_path"):
        print(f"[probe]   {tag}: {row[tag]}")

    checks: list[tuple[str, bool, str]] = []

    thermal_tensor, thermal_diag = load_vtd_thermal(row["thermal_path"], cfg)
    th_hw = tuple(thermal_tensor.shape[-2:])
    checks.append(("thermal is [1,3,H,W] in [-1,1]",
                   thermal_tensor.ndim == 4 and thermal_tensor.shape[1] == 3,
                   f"shape={tuple(thermal_tensor.shape)} range=[{thermal_tensor.min():.2f},{thermal_tensor.max():.2f}]"))
    checks.append(("thermal not constant", float(thermal_tensor.std()) > 1e-4, f"std={float(thermal_tensor.std()):.4f}"))

    rgb_tensor = rgb_to_lotus_input(row["rgb_path"], th_hw)
    checks.append(("rgb resized to thermal frame",
                   tuple(rgb_tensor.shape[-2:]) == th_hw,
                   f"rgb={tuple(rgb_tensor.shape)} thermal={tuple(thermal_tensor.shape)}"))

    disparity, valid_mask = load_vtd_disparity(row["depth_path"], cfg, target_hw=th_hw)
    n_valid = int(valid_mask.sum())
    checks.append(("depth matches thermal frame", tuple(disparity.shape[-2:]) == th_hw,
                   f"depth={tuple(disparity.shape)}"))
    checks.append(("depth has >=100 valid pixels", n_valid >= 100, f"valid={n_valid}"))
    if n_valid:
        d = 1.0 / disparity[valid_mask > 0.5].clamp_min(1e-6)
        checks.append(("depth range plausible (0.1-80 m)", bool((d.min() >= 0.05) and (d.max() <= 200)),
                       f"depth[m]=[{float(d.min()):.2f},{float(d.max()):.2f}] median={float(d.median()):.2f}"))

    # latent-size agreement (the requirement-8 guarantee), predicted as H/8, W/8
    lh, lw = th_hw[0] // 8, th_hw[1] // 8
    checks.append(("latent grid divisible by 8", th_hw[0] % 8 == 0 and th_hw[1] % 8 == 0,
                   f"thermal {th_hw} -> latent ~({lh},{lw})"))

    print("\n=== VTD probe checklist ===")
    all_pass = True
    for name, ok, note in checks:
        all_pass &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({note})" if note else ""))
    print(f"=== {'READY — wire --dataset vtd' if all_pass else 'FIX VTDConfig above'} ===")
    if not all_pass:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="VTD dataset adapter / probe")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--vtd-root", type=Path, default=None)
    parser.add_argument("--depth-encoding", default=None,
                        choices=("depth_uint16", "disparity_uint16", "depth_float"))
    parser.add_argument("--depth-scale", type=float, default=None)
    args = parser.parse_args()
    cfg = VTDConfig()
    if args.vtd_root is not None:
        cfg.root = args.vtd_root
    if args.depth_encoding is not None:
        cfg.depth_encoding = args.depth_encoding
    if args.depth_scale is not None:
        cfg.depth_scale = args.depth_scale
    if args.probe:
        probe(cfg)
    else:
        rows = scan_vtd(cfg)
        print(f"scanned {len(rows)} VTD frames; run with --probe to validate one.")


if __name__ == "__main__":
    main()
