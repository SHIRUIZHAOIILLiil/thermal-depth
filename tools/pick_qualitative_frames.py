"""Pick candidate frames for the qualitative figure by measurement, not by eye.

The figure is the one place a reader looks at our predictions rather than our
numbers, and the frame it shows is chosen by us. A frame picked by eye is a
frame someone can ask about. So candidates come from three scores computed on
every frame of the official evaluation subset, and the figure is chosen from
the top of that list:

  structure  mean gradient magnitude of the stretched thermal frame -- how much
             is in the picture at all
  layering   entropy of the log-depth histogram over valid lidar returns -- a
             frame with near, middle and far content scores above an empty road
  near       share of lidar returns closer than 10 m -- vehicles, poles, people

A frame must also have lidar coverage at least 0.8 of its condition's median,
since a sparse ground-truth panel shows nothing. Score is the sum of the three
z-scores (near at half weight); candidates are taken greedily from the top with
at least 300 frames (30 s) between any two from the same sequence, so the list
is not eight views of one junction.

Writes a contact sheet per condition (thermal | ground truth, ranked) and a JSON
of every candidate with its scores.

    python tools/pick_qualitative_frames.py --ms2-root <root> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


D_MIN, D_MAX = 1e-3, 80.0
CONDITIONS = {"day": "test_day_list.txt", "night": "test_night_list.txt", "rainy": "test_rainy_list.txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-condition", type=int, default=8)
    parser.add_argument("--min-gap", type=int, default=300,
                        help="Minimum frame-index gap between two candidates of one sequence.")
    return parser.parse_args()


def stretch(raw: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(raw, (1.0, 99.0))
    return np.clip((raw - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(raw)


def scores_for(frame: dict) -> dict | None:
    thermal = stretch(np.asarray(Image.open(frame["image_path"]), np.float32))
    gx, gy = np.diff(thermal, axis=1), np.diff(thermal, axis=0)
    structure = float(np.abs(gx).mean() + np.abs(gy).mean())

    depth = np.asarray(Image.open(frame["depth_path"]), np.float32) / 256.0
    valid = np.isfinite(depth) & (depth > D_MIN) & (depth < D_MAX)
    coverage = float(valid.mean())
    if valid.sum() < 500:
        return None
    log_d = np.log(depth[valid])
    hist, _ = np.histogram(log_d, bins=20, range=(np.log(1.0), np.log(D_MAX)))
    p = hist[hist > 0] / hist.sum()
    layering = float(-(p * np.log(p)).sum())
    near = float((depth[valid] < 10.0).mean())
    return {"structure": structure, "layering": layering, "near": near, "coverage": coverage}


def render_depth(depth: np.ndarray, valid: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """The figure's scheme: depth space, magma, near dark and far bright, no-data
    grey -- black would read as 'near' under magma."""
    import matplotlib

    cmap = matplotlib.colormaps["magma"]
    t = np.clip((depth - lo) / (hi - lo + 1e-9), 0, 1)
    rgb = (cmap(t)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = (128, 128, 128)
    return rgb


def contact_sheet(picked: list[dict], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(picked), 2, figsize=(12.8, 2.75 * len(picked)), dpi=100)
    axes = np.atleast_2d(axes)
    for row, item in enumerate(picked):
        thermal = stretch(np.asarray(Image.open(item["image_path"]), np.float32))
        depth = np.asarray(Image.open(item["depth_path"]), np.float32) / 256.0
        valid = np.isfinite(depth) & (depth > D_MIN) & (depth < D_MAX)
        lo, hi = np.percentile(depth[valid], (2, 98))
        axes[row, 0].imshow(thermal, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(render_depth(depth, valid, lo, hi))
        axes[row, 0].set_title(
            f"#{row + 1}  {item['id']}   score {item['score']:+.2f}", fontsize=10, loc="left")
        axes[row, 1].set_title(
            f"structure {item['structure']:.3f}  layering {item['layering']:.2f}  "
            f"near {item['near']:.0%}  coverage {item['coverage']:.0%}", fontsize=9, loc="left")
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def official_frames(ms2_root: Path, list_file: str, step: int = 10) -> list[dict]:
    """`run_ms2_supdepth_baselines.official_frames` without its torch import, so
    this runs on a laptop: concatenate every frame of the listed sequences in
    list order, then take [0:-1:10]. Expected 2332 / 2292 / 2503 (our evaluated
    day results hold 2331: frame 15-46-56_003700 has no caption)."""
    sequences = [s.strip() for s in (ms2_root / list_file).read_text().splitlines() if s.strip()]
    frames = []
    for sequence in sequences:
        for path in sorted((ms2_root / "sync_data" / sequence / "thr" / "img_left").glob("*.png")):
            seq = sequence.lstrip("_")
            frames.append({
                "sequence": seq, "stem": path.stem, "id": f"{seq}_{path.stem}",
                "image_path": path,
                "depth_path": ms2_root / "proj_depth" / sequence / "thr" / "depth_filtered" / f"{path.stem}.png",
            })
    return frames[0:-1:step]


def main() -> None:
    args = parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, env in CONDITIONS.items():
        frames = official_frames(args.ms2_root, env)
        print(f"[{name}] {len(frames)} official frames", flush=True)
        rows = []
        for frame in frames:
            if not Path(frame["depth_path"]).is_file():
                continue
            s = scores_for(frame)
            if s is None:
                continue
            rows.append({"id": frame["id"], "sequence": frame["sequence"],
                         "index": int(frame["stem"]),
                         "image_path": str(frame["image_path"]),
                         "depth_path": str(frame["depth_path"]), **s})
        if not rows:
            print(f"[{name}] no ground truth found under {args.ms2_root} -- skipped", flush=True)
            continue
        floor = 0.8 * float(np.median([r["coverage"] for r in rows]))
        pool = [r for r in rows if r["coverage"] >= floor]

        def z(key: str) -> np.ndarray:
            v = np.array([r[key] for r in pool])
            return (v - v.mean()) / (v.std() + 1e-9)

        total = z("structure") + z("layering") + 0.5 * z("near")
        for r, s in zip(pool, total):
            r["score"] = float(s)
        pool.sort(key=lambda r: -r["score"])

        picked: list[dict] = []
        for r in pool:
            if any(p["sequence"] == r["sequence"] and abs(p["index"] - r["index"]) < args.min_gap
                   for p in picked):
                continue
            picked.append(r)
            if len(picked) == args.per_condition:
                break

        sheet = args.out_dir / f"candidates_{name}.png"
        contact_sheet(picked, sheet,
                      f"{name}: top {len(picked)} of {len(pool)} frames "
                      f"(coverage >= {floor:.0%}, {len(rows)} scored)")
        report[name] = {"scored": len(rows), "coverage_floor": floor, "candidates": picked}
        print(f"[{name}] {len(rows)} scored, {len(pool)} above coverage floor {floor:.0%}, "
              f"top {len(picked)} -> {sheet}", flush=True)
        for i, r in enumerate(picked, 1):
            print(f"   #{i}  {r['id']}  score {r['score']:+.2f}  structure {r['structure']:.3f}  "
                  f"layering {r['layering']:.2f}  near {r['near']:.0%}  coverage {r['coverage']:.0%}")

    (args.out_dir / "candidates.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[完成] {args.out_dir}")


if __name__ == "__main__":
    main()
