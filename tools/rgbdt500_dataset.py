"""RGBDT500 adapter/probe — the DENSE-GT venue for the caption-on-thermal test.

RGBDT500 (https://xuefeng-zhu5.github.io/RGBDT500/) is a multi-modal *tracking*
dataset: 500 videos, 203.7K spatially-aligned RGB / Depth / Thermal triplets,
GT = bounding boxes. We do NOT use it for tracking. We borrow only its
(thermal, dense depth) pairing to test one hypothesis the sparse MS2 LiDAR GT
could not settle:

    Does a caption help monocular THERMAL depth when the evaluator finally
    sees DENSE GT? (MS2 filtered GT is ~29% dense; the caption line's negative
    result there was confounded by the judge being blind on 71% of pixels.)

Role of each modality here (no leakage):
    thermal  -> model INPUT (predict depth)
    depth    -> GT only (borrowed from its tracking-input role)
    RGB      -> caption generation only (InternVL, same pipeline as MS2)

This file is the FILE/FORMAT layer + a `--probe` that loads one triplet and
measures everything the experiment depends on — most importantly the depth
DENSITY, which is the whole point. Every format unknown lives in RGBDT500Config
as a TODO; fill them in after looking at the extracted tree, then re-probe.

    python tools/rgbdt500_dataset.py --probe --root /mnt/e/dataset/rgbdt500
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "lotus", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


# ============================================================================
#  FORMAT CONFIG — fill in after inspecting the extracted RGBDT500 tree.
#  The webpage does not document the layout; --probe reports real shapes so you
#  can correct these, then re-probe until the checklist is green.
# ============================================================================
@dataclass
class RGBDT500Config:
    # VERIFIED 2026-07-18 against E:/dataset/RGBDT500/train/Train (400 seqs x 10
    # frames = 4000). Layout: <root>/<seq 001..400>/{color,infrared,depth}/<8-digit>.png
    # plus groundtruth.txt (tracking boxes, unused).
    root: Path = Path("/mnt/e/dataset/RGBDT500/train/Train")

    # --- file discovery (relative to root) ----------------------------------
    rgb_glob: str = "*/color/*.png"
    thermal_glob: str = "*/infrared/*.png"
    depth_glob: str = "*/depth/*.png"
    frame_key_regex: str = r"(\d+)"       # 8-digit frame index in the filename
    sequence_from_path_index: int = 0     # "001" .. "400"

    # --- thermal ------------------------------------------------------------
    # VERIFIED: 1920x1080 uint8 stored in PIL mode RGB. Channels are NOT bit-exact
    # (mean |ch0-ch1| ~0.2/255) but that is compression noise, not pseudo-colour —
    # it is a grayscale TIR frame, already 8-bit AGC-normalized by the authors
    # (unlike MS2's raw 16-bit which we min-max ourselves). Take one channel / L.
    thermal_single_channel: bool = False

    # --- depth GT (the crux) ------------------------------------------------
    # VERIFIED: uint16, raw range 213..19999 -> millimetres, i.e. scale 1000 and a
    # hard 20 m cap. Typical scene band is 7-19 m (compressed vs MS2's 0.1-80 m).
    depth_encoding: str = "depth_uint16"
    depth_scale: float = 1000.0
    min_depth: float = 0.1
    max_depth: float = 20.0   # sensor cap, NOT the 80 m driving cap
    # VERIFIED (mixed evidence): shift-search favours color (48% zero-shift vs 7%
    # for infrared); infrared frames carry warp borders, so the authors *did*
    # register it. Treat thermal<->depth alignment as UNCONFIRMED and never quote
    # absolute accuracy from this set — paired ablations only.
    depth_aligned_to: str = "color"

    # MEASURED: mean 75.8% valid (range 37.6-97.8%) across 40 sequences, i.e.
    # 2.6x denser than MS2 filtered GT (~29%). This is the premise of the whole
    # experiment and it holds. Holes concentrate at the top of frame (sky /
    # beyond-range), the signature of a real sensor rather than a model estimate.
    min_dense_fraction: float = 0.60

    def signature(self) -> str:
        payload = json.dumps({k: str(v) for k, v in asdict(self).items()}, sort_keys=True)
        import hashlib
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _frame_key(path: Path, regex: str) -> str:
    match = re.search(regex, path.stem)
    return match.group(1) if match else path.stem


def _sequence_key(path: Path, root: Path, index: int) -> str:
    rel = path.relative_to(root).parts
    return rel[index] if index < len(rel) else "seq"


def scan(cfg: RGBDT500Config) -> list[dict]:
    root = Path(cfg.root)

    def collect(pattern: str) -> dict[tuple[str, str], Path]:
        out: dict[tuple[str, str], Path] = {}
        for match in sorted(glob.glob(str(root / pattern), recursive=True)):
            p = Path(match)
            out[(_sequence_key(p, root, cfg.sequence_from_path_index),
                 _frame_key(p, cfg.frame_key_regex))] = p
        return out

    rgb, thermal, depth = collect(cfg.rgb_glob), collect(cfg.thermal_glob), collect(cfg.depth_glob)
    keys = sorted(set(rgb) & set(thermal) & set(depth))
    if not keys:
        raise FileNotFoundError(
            "No paired RGBDT500 frames. Fix the globs in RGBDT500Config:\n"
            f"  rgb     {cfg.rgb_glob}     -> {len(rgb)} files\n"
            f"  thermal {cfg.thermal_glob} -> {len(thermal)} files\n"
            f"  depth   {cfg.depth_glob}   -> {len(depth)} files"
        )
    return [
        {"id": f"{seq}_{frame}", "sequence": seq, "frame": frame,
         "rgb_path": rgb[(seq, frame)], "thermal_path": thermal[(seq, frame)],
         "depth_path": depth[(seq, frame)]}
        for seq, frame in keys
    ]


def decode_depth(path: Path, cfg: RGBDT500Config):
    raw = np.asarray(Image.open(path))
    if raw.ndim == 3:
        raw = raw[..., 0]
    raw = raw.astype(np.float64)
    if cfg.depth_encoding == "depth_uint16":
        depth = raw / cfg.depth_scale
        valid = (depth > cfg.min_depth) & (depth < cfg.max_depth) & np.isfinite(depth)
    elif cfg.depth_encoding == "depth_float":
        depth = raw
        valid = (depth > cfg.min_depth) & (depth < cfg.max_depth) & np.isfinite(depth)
    elif cfg.depth_encoding == "disparity_uint16":
        disparity = raw / cfg.depth_scale
        with np.errstate(divide="ignore"):
            depth = np.where(disparity > 0, 1.0 / np.maximum(disparity, 1e-9), 0.0)
        valid = (disparity > 0) & (depth > cfg.min_depth) & (depth < cfg.max_depth)
    else:
        raise ValueError(f"Unknown depth_encoding {cfg.depth_encoding!r}")
    return depth, valid


def probe(cfg: RGBDT500Config) -> None:
    print(f"[probe] RGBDT500 root: {cfg.root}   config sig: {cfg.signature()}")
    rows = scan(cfg)
    print(f"[probe] paired triplets found: {len(rows)}  across "
          f"{len({r['sequence'] for r in rows})} sequences")
    row = rows[len(rows) // 2]
    for tag in ("thermal_path", "rgb_path", "depth_path"):
        print(f"[probe]   {tag}: {row[tag]}")

    checks: list[tuple[str, bool, str]] = []

    thr = np.asarray(Image.open(row["thermal_path"]))
    checks.append(("thermal readable & non-constant",
                   float(thr.astype(float).std()) > 1.0,
                   f"shape={thr.shape} dtype={thr.dtype} range=[{thr.min()},{thr.max()}]"))

    rgb = np.asarray(Image.open(row["rgb_path"]).convert("RGB"))
    checks.append(("rgb readable (for caption gen)", rgb.ndim == 3,
                   f"shape={rgb.shape} dtype={rgb.dtype}"))

    depth, valid = decode_depth(row["depth_path"], cfg)
    dense = float(valid.mean())
    checks.append(("*** depth is DENSE (the whole point) ***",
                   dense >= cfg.min_dense_fraction,
                   f"valid={dense*100:.1f}%  (MS2 filtered GT ~29%; need >={cfg.min_dense_fraction*100:.0f}%)"))
    if valid.any():
        dv = depth[valid]
        checks.append(("depth range metric-plausible",
                       bool(dv.min() >= 0.05 and dv.max() <= 100),
                       f"depth[m]=[{dv.min():.2f},{dv.max():.2f}] median={np.median(dv):.2f}"))

    checks.append(("thermal & depth same resolution (alignment prereq)",
                   thr.shape[:2] == depth.shape[:2],
                   f"thermal={thr.shape[:2]} depth={depth.shape[:2]} (aligned_to={cfg.depth_aligned_to})"))
    h, w = thr.shape[:2]
    checks.append(("frame divisible by 8 (VAE latent grid)",
                   h % 8 == 0 and w % 8 == 0, f"{h}x{w}"))

    print("\n=== RGBDT500 probe checklist ===")
    all_pass = True
    for name, ok, note in checks:
        all_pass &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({note})" if note else ""))
    print("  [TODO] captions do NOT exist — must run the InternVL captioner on RGB frames")
    print(f"=== {'FORMAT READY — next: generate captions, then wire an eval runner' if all_pass else 'FIX RGBDT500Config above and re-probe'} ===")
    if not all_pass:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="RGBDT500 adapter / probe")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--depth-encoding", default=None,
                        choices=("depth_uint16", "depth_float", "disparity_uint16"))
    parser.add_argument("--depth-scale", type=float, default=None)
    parser.add_argument("--max-depth", type=float, default=None)
    args = parser.parse_args()
    cfg = RGBDT500Config()
    if args.root is not None: cfg.root = args.root
    if args.depth_encoding is not None: cfg.depth_encoding = args.depth_encoding
    if args.depth_scale is not None: cfg.depth_scale = args.depth_scale
    if args.max_depth is not None: cfg.max_depth = args.max_depth
    if args.probe:
        probe(cfg)
    else:
        rows = scan(cfg)
        print(f"scanned {len(rows)} RGBDT500 triplets; run with --probe to validate one.")


if __name__ == "__main__":
    main()
