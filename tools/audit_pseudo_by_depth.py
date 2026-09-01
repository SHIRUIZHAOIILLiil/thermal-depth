"""Where the returns are, and how far the pseudo depth can be trusted there.

Two questions decide whether an image-space loss should be computed against the
completed map in metres:

  1. Does the far field have any ground truth at all? If almost no return lands
     beyond 30 m, the pseudo depth is the only signal there and using it is the
     only option available.
  2. Is the pseudo depth trustworthy where it is used? The per-frame affine that
     calibrates it is fitted over all returns, and those concentrate near, so its
     behaviour far away is extrapolation. A loss in metres puts most of its
     gradient exactly there, which is a good idea only if the target holds up.

Both are answered by binning on the measured depth: the density of returns per
bin, and the disagreement between the return and the pseudo depth it replaces.
No network and no GPU -- this reads the depth PNGs and the calibrated .npy files
and does arithmetic.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

BIN_EDGES = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0, np.inf]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--pseudo-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--limit", type=int, default=300)
    return parser.parse_args()


def rows_from(manifest: Path, limit: int):
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    if limit and limit < len(rows):
        step = len(rows) / limit
        rows = [rows[int(i * step)] for i in range(limit)]
    return rows


def resolve(root: Path, row: dict) -> Path:
    for key in ("depth_path", "gt_path"):
        if key in row:
            path = Path(row[key])
            return path if path.is_absolute() else root / path
    raise KeyError(f"no depth path in {sorted(row)[:6]}")


def main() -> int:
    args = parse_args()
    rows = rows_from(args.manifest, args.limit)

    bins = len(BIN_EDGES) - 1
    counts = np.zeros(bins, dtype=np.int64)
    rel_sum = np.zeros(bins)
    frames = 0
    total_pixels = 0
    # The teacher's own score. Every pixel of the latent target that was not
    # measured is this predictor's opinion, so whatever error it carries is the
    # floor a model trained to imitate it can reach. Per frame, like every other
    # number in this project, then averaged.
    teacher = {"abs_rel": [], "rmse": [], "delta1": []}

    for row in rows:
        pseudo_path = args.pseudo_dir / f"{row['id']}.npy"
        if not pseudo_path.is_file():
            continue
        gt = np.asarray(Image.open(resolve(args.ms2_root, row)), dtype=np.float64) / args.depth_scale
        pseudo = np.load(pseudo_path, allow_pickle=False).astype(np.float64)
        if pseudo.shape != gt.shape:
            continue
        valid = np.isfinite(gt) & (gt > args.min_depth) & (gt < args.max_depth)
        if valid.sum() < 100:
            continue
        frames += 1
        total_pixels += gt.size

        p = np.clip(pseudo[valid], args.min_depth, args.max_depth)
        g = gt[valid]
        teacher["abs_rel"].append(float(np.mean(np.abs(p - g) / g)))
        teacher["rmse"].append(float(np.sqrt(np.mean((p - g) ** 2))))
        ratio = np.maximum(p / g, g / p)
        teacher["delta1"].append(float(np.mean(ratio < 1.25)))

        index = np.digitize(gt[valid], BIN_EDGES) - 1
        rel = np.abs(gt[valid] - pseudo[valid]) / np.maximum(gt[valid], 1e-6)
        for b in range(bins):
            take = index == b
            n = int(take.sum())
            if n:
                counts[b] += n
                rel_sum[b] += float(rel[take].sum())

    if not frames:
        raise SystemExit("no frame produced a measurement")

    print(f"[frames] {frames}")
    print(f"[density] {counts.sum() / max(1, total_pixels):.1%} of all pixels carry a return")
    print(f"\n[teacher] the calibrated pseudo depth, scored against the returns it did not "
          f"replace being irrelevant -- this is what the latent objective imitates:")
    print(f"          AbsRel {100 * np.mean(teacher['abs_rel']):.2f}   "
          f"RMSE {np.mean(teacher['rmse']):.3f} m   "
          f"delta1 {100 * np.mean(teacher['delta1']):.2f}")
    print("          A model trained to reproduce this map cannot score better than it does,")
    print("          however well the U-Net learns. Compare against the model's own row.\n")
    header = f"{'depth bin (m)':>16s}{'returns':>12s}{'share':>9s}{'per frame':>12s}{'|gt-pseudo|/gt':>17s}"
    print(header)
    print("-" * len(header))
    share = counts / max(1, counts.sum())
    for b in range(bins):
        lo, hi = BIN_EDGES[b], BIN_EDGES[b + 1]
        label = f"{lo:.0f}-{hi:.0f}" if np.isfinite(hi) else f">{lo:.0f}"
        mean_rel = rel_sum[b] / counts[b] if counts[b] else float("nan")
        print(f"{label:>16s}{counts[b]:>12d}{share[b]:>9.1%}"
              f"{counts[b] / frames:>12.0f}{mean_rel:>17.1%}")

    print("\nReading it. The share column answers whether the far field has ground")
    print("truth: if the bins beyond 30 m hold a percent or two of the returns, the")
    print("pseudo depth is effectively the only target there. The last column answers")
    print("whether that target can be trusted: the affine is fitted over all returns,")
    print("which are concentrated near, so a disagreement that grows with depth means")
    print("the far field is extrapolation. A loss in metres would put most of its")
    print("gradient into whichever bins are worst on that column.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "frames": frames,
            "bin_edges": [None if np.isinf(e) else e for e in BIN_EDGES],
            "returns": counts.tolist(),
            "share": share.tolist(),
            "mean_rel_error": [rel_sum[b] / counts[b] if counts[b] else None for b in range(bins)],
        }, indent=2), encoding="utf-8")
        print(f"\n-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
