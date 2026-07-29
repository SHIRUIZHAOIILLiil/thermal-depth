"""Probe: pixel-level spatial alignment between RGBDT500 modalities.

Motivation (2026-07-21): the RGB-teacher -> thermal-student line was closed on
MS2 by frozen conclusion 15 (RGB and thermal cameras do not share a viewpoint;
edge NCC 0.13-0.24, best-shift drift +-10-26 px).  RGBDT500 claims spatially
aligned RGB / Depth / Thermal, which would remove that blocker -- but our own
earlier shift search only compared *depth* against the other two and left
thermal<->depth UNCONFIRMED.  The pair that actually matters for a dense
pixel-wise teacher loss (experiment 5's bare L1) is **RGB <-> thermal**, which
has never been measured on its own.

Method (mirrors the MS2 probe so the numbers stay comparable): per frame, build
a Sobel gradient-magnitude edge map for each modality, standardise it to zero
mean / unit variance, then find the 2-D translation maximising normalised
cross-correlation over +-max_shift pixels.  Because both maps are standardised,
the FFT circular cross-correlation divided by the pixel count *is* the Pearson
correlation at each shift, which makes the search exact and cheap (numpy only --
this box has no cv2/scipy).  Report the peak shift and the NCC at the peak vs
at zero shift.

Three pairs are measured together so the target pair can be read against
controls measured with the identical estimator:

  rgb_thermal    the target pair (RGB teacher -> thermal student)
  rgb_depth      positive control (authors register depth to colour: expect ~0)
  thermal_depth  reproduces the earlier finding (only 7% zero-shift)

A shuffled-pair control (RGB of frame i vs thermal of frame j) establishes the
NCC floor, so a low-but-nonzero correlation is not over-read as alignment.

Pre-registered judgement for the experiment-5 recipe (set before running):
  median |shift| <= 2 px   bare L1 (experiment 5) is geometrically admissible
  2-8 px                   teacher usable for coverage only, not as a range anchor
  > 8 px                   line stays closed, same cause of death as MS2

CPU only; no GPU, no training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("E:/dataset/RGBDT500/clean_train/rgbdt500_train_manifest_train.jsonl"),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("E:/dataset/RGBDT500/clean_train"),
        help="Directory the manifest's relative paths resolve against.",
    )
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/lotus_line_v2/rgbdt500_modality_alignment.json"))
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--max-shift", type=int, default=24)
    parser.add_argument("--inset", type=int, default=16,
                        help="Extra margin dropped on every side before matching, "
                             "so registration borders cannot drive the peak.")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser.parse_args()


def load_rows(manifest: Path, samples: int, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in manifest.open(encoding="utf-8")]
    by_sequence: dict[str, list[dict]] = {}
    for row in rows:
        by_sequence.setdefault(str(row["sequence"]), []).append(row)
    rng = random.Random(seed)
    sequences = sorted(by_sequence)
    picked: list[dict] = []
    # Round-robin across sequences so the sample is not dominated by a few scenes.
    while len(picked) < min(samples, len(rows)):
        progressed = False
        for sequence in sequences:
            pool = by_sequence[sequence]
            if not pool:
                continue
            picked.append(pool.pop(rng.randrange(len(pool))))
            progressed = True
            if len(picked) >= min(samples, len(rows)):
                break
        if not progressed:
            break
    return picked


def sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray.astype(np.float64), 1, mode="edge")
    gx = (-padded[:-2, :-2] + padded[:-2, 2:]
          - 2.0 * padded[1:-1, :-2] + 2.0 * padded[1:-1, 2:]
          - padded[2:, :-2] + padded[2:, 2:])
    gy = (-padded[:-2, :-2] - 2.0 * padded[:-2, 1:-1] - padded[:-2, 2:]
          + padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:])
    return np.hypot(gx, gy)


def edge_map(image: np.ndarray) -> np.ndarray:
    magnitude = sobel_magnitude(image)
    # Per-image standardisation keeps NCC comparable across modalities whose
    # gradient magnitudes live on completely different scales, and is what makes
    # the FFT correlation below read directly as a correlation coefficient.
    std = float(magnitude.std())
    if std <= 1e-8:
        raise ValueError("Degenerate edge map (constant image)")
    return (magnitude - float(magnitude.mean())) / std


def load_modalities(row: dict, root: Path, args) -> dict[str, np.ndarray]:
    from PIL import Image

    rgb_gray = np.asarray(Image.open(root / row["rgb_path"]).convert("L"))
    thermal = np.asarray(Image.open(root / row["thermal_path"]).convert("L"))
    depth_raw = np.asarray(Image.open(root / row["depth_path"]))
    if depth_raw.ndim == 3:
        depth_raw = depth_raw[..., 0]

    depth = depth_raw.astype(np.float32) / args.depth_scale
    valid = np.isfinite(depth) & (depth > args.min_depth) & (depth < args.max_depth)
    # Depth edges are only meaningful on valid pixels; holes would otherwise
    # produce huge synthetic step edges at their borders.
    depth_filled = np.where(valid, depth, np.nan)
    median = float(np.nanmedian(depth_filled)) if valid.any() else 0.0
    depth_filled = np.where(valid, depth, median)

    return {
        "rgb": edge_map(rgb_gray),
        "thermal": edge_map(thermal),
        "depth": edge_map(depth_filled),
        "_depth_valid_fraction": float(valid.mean()),
    }


def _restandardise(patch: np.ndarray) -> np.ndarray:
    std = float(patch.std())
    if std <= 1e-8:
        raise ValueError("Degenerate patch after inset crop")
    return (patch - float(patch.mean())) / std


def best_shift(a: np.ndarray, b: np.ndarray, max_shift: int, inset: int):
    """Peak-correlation translation taking `a` onto `b`, searched by FFT.

    Both inputs are re-standardised over the inset-cropped region, so the
    circular cross-correlation divided by the pixel count is the Pearson
    correlation at each integer shift.
    """
    if inset:
        a = a[inset:-inset, inset:-inset]
        b = b[inset:-inset, inset:-inset]
    a = _restandardise(a)
    b = _restandardise(b)
    height, width = a.shape
    if max_shift * 2 >= min(height, width):
        raise ValueError("max_shift too large for image size")

    spectrum = np.conj(np.fft.rfft2(a)) * np.fft.rfft2(b)
    correlation = np.fft.irfft2(spectrum, s=a.shape) / float(a.size)

    offsets = np.arange(-max_shift, max_shift + 1)
    window = correlation[np.ix_(offsets % height, offsets % width)]
    peak = int(np.argmax(window))
    py, px = divmod(peak, window.shape[1])
    return (
        int(offsets[py]),
        int(offsets[px]),
        float(window[py, px]),
        float(correlation[0, 0]),
        window,
    )


def surface_peak(surface: np.ndarray, max_shift: int) -> dict:
    """Peak of a correlation surface averaged over many frames.

    Per-frame peaks are dominated by noise on a nearly flat cross-modal
    correlation surface; averaging the surfaces first raises the SNR by ~sqrt(n)
    and answers the question that actually matters -- is there a *systematic*
    offset -- rather than how far noise wanders on any single frame.
    """
    offsets = np.arange(-max_shift, max_shift + 1)
    peak = int(np.argmax(surface))
    py, px = divmod(peak, surface.shape[1])
    centre = surface[max_shift, max_shift]
    return {
        "dy": int(offsets[py]),
        "dx": int(offsets[px]),
        "ncc_at_peak": float(surface[py, px]),
        "ncc_at_zero": float(centre),
        "peak_minus_zero": float(surface[py, px] - centre),
    }


PAIRS = (("rgb_thermal", "rgb", "thermal"),
         ("rgb_depth", "rgb", "depth"),
         ("thermal_depth", "thermal", "depth"))


def summarise(records: list[dict], key: str) -> dict:
    shifts = np.array([[r[key]["dy"], r[key]["dx"]] for r in records], dtype=np.float64)
    magnitude = np.hypot(shifts[:, 0], shifts[:, 1])
    ncc_best = np.array([r[key]["ncc_best"] for r in records])
    ncc_zero = np.array([r[key]["ncc_zero"] for r in records])
    return {
        "n": int(len(records)),
        "median_shift_magnitude_px": float(np.median(magnitude)),
        "mean_shift_magnitude_px": float(magnitude.mean()),
        "p90_shift_magnitude_px": float(np.percentile(magnitude, 90)),
        "frac_within_1px": float((magnitude <= 1.0).mean()),
        "frac_within_2px": float((magnitude <= 2.0).mean()),
        "frac_within_8px": float((magnitude <= 8.0).mean()),
        "frac_exact_zero": float((magnitude == 0).mean()),
        "median_dy": float(np.median(shifts[:, 0])),
        "median_dx": float(np.median(shifts[:, 1])),
        "median_ncc_at_peak": float(np.median(ncc_best)),
        "median_ncc_at_zero_shift": float(np.median(ncc_zero)),
    }


def main() -> int:
    args = parse_args()
    rows = load_rows(args.manifest, args.samples, args.seed)
    if not rows:
        print("No rows sampled", file=sys.stderr)
        return 1

    records: list[dict] = []
    cache: list[dict[str, np.ndarray]] = []
    surfaces: dict[str, np.ndarray] = {}
    for index, row in enumerate(rows):
        try:
            modalities = load_modalities(row, args.root, args)
        except (FileNotFoundError, ValueError) as error:
            print(f"  skip {row['id']}: {error}")
            continue
        record = {"id": row["id"], "sequence": row["sequence"],
                  "depth_valid_fraction": modalities["_depth_valid_fraction"]}
        for name, left, right in PAIRS:
            dy, dx, ncc_best, ncc_zero, window = best_shift(
                modalities[left], modalities[right], args.max_shift, args.inset
            )
            record[name] = {"dy": dy, "dx": dx, "ncc_best": ncc_best, "ncc_zero": ncc_zero}
            surfaces[name] = window if name not in surfaces else surfaces[name] + window
        records.append(record)
        cache.append(modalities)
        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(rows)} frames")
    for name in surfaces:
        surfaces[name] = surfaces[name] / float(len(records))

    # Shuffled control: RGB of frame i against thermal of frame i+1, so the NCC
    # floor is measured with the same estimator on genuinely unrelated content.
    control = []
    control_surface = None
    for index in range(len(cache) - 1):
        dy, dx, ncc_best, ncc_zero, window = best_shift(
            cache[index]["rgb"], cache[index + 1]["thermal"], args.max_shift, args.inset
        )
        control.append({"rgb_thermal": {"dy": dy, "dx": dx,
                                        "ncc_best": ncc_best, "ncc_zero": ncc_zero}})
        control_surface = window if control_surface is None else control_surface + window
    if control_surface is not None:
        control_surface = control_surface / float(len(control))

    payload = {
        "manifest": str(args.manifest),
        "root": str(args.root),
        "samples": len(records),
        "max_shift": args.max_shift,
        "inset": args.inset,
        "judgement_thresholds_px": {"bare_l1_ok": 2, "coverage_only": 8},
        "pairs": {name: summarise(records, name) for name, _, _ in PAIRS},
        "averaged_surface_peak": {
            name: surface_peak(surfaces[name], args.max_shift) for name in surfaces
        },
        "averaged_surface_peak_shuffled_control": (
            surface_peak(control_surface, args.max_shift) if control_surface is not None else None
        ),
        "shuffled_control_rgb_thermal": summarise(control, "rgb_thermal") if control else None,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n=== RGBDT500 modality alignment ({len(records)} frames) ===")
    header = f"{'pair':<16}{'med|shift|':>11}{'p90':>7}{'<=1px':>8}{'<=2px':>8}{'ncc@peak':>10}{'ncc@0':>8}"
    print(header)
    for name, _, _ in PAIRS:
        s = payload["pairs"][name]
        print(f"{name:<16}{s['median_shift_magnitude_px']:>11.1f}"
              f"{s['p90_shift_magnitude_px']:>7.1f}{s['frac_within_1px']:>8.1%}"
              f"{s['frac_within_2px']:>8.1%}{s['median_ncc_at_peak']:>10.3f}"
              f"{s['median_ncc_at_zero_shift']:>8.3f}")
    if payload["shuffled_control_rgb_thermal"]:
        s = payload["shuffled_control_rgb_thermal"]
        print(f"{'(shuffled ctrl)':<16}{s['median_shift_magnitude_px']:>11.1f}"
              f"{s['p90_shift_magnitude_px']:>7.1f}{s['frac_within_1px']:>8.1%}"
              f"{s['frac_within_2px']:>8.1%}{s['median_ncc_at_peak']:>10.3f}"
              f"{s['median_ncc_at_zero_shift']:>8.3f}")
    print("\n--- peak of the frame-averaged correlation surface (systematic offset) ---")
    print(f"{'pair':<16}{'dy':>5}{'dx':>5}{'ncc@peak':>10}{'ncc@0':>8}{'peak-zero':>11}")
    for name in ("rgb_thermal", "rgb_depth", "thermal_depth"):
        p = payload["averaged_surface_peak"][name]
        print(f"{name:<16}{p['dy']:>5}{p['dx']:>5}{p['ncc_at_peak']:>10.4f}"
              f"{p['ncc_at_zero']:>8.4f}{p['peak_minus_zero']:>11.4f}")
    if payload["averaged_surface_peak_shuffled_control"]:
        p = payload["averaged_surface_peak_shuffled_control"]
        print(f"{'(shuffled ctrl)':<16}{p['dy']:>5}{p['dx']:>5}{p['ncc_at_peak']:>10.4f}"
              f"{p['ncc_at_zero']:>8.4f}{p['peak_minus_zero']:>11.4f}")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
