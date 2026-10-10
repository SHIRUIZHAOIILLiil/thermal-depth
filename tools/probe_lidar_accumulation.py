"""Feasibility probe: real LiDAR depth where the current frame's LiDAR does not reach.

The question it serves: is our model better than the baselines *outside* the LiDAR
footprint? Scoring there needs an independent reference, and the completed target is
not one (it is AnyThermal there). Real LiDAR from neighbouring frames, carried into
the current thermal view with MS2's own odometry, is.

Two steps, the second only meaningful if the first passes:

  check       project the current frame's own LiDAR into its thermal view and compare
              with MS2's projected GT (proj_depth/<seq>/thr/depth*). Tries both readings
              of the calibration's direction convention; the one that reproduces the
              GT is the one to use.
  accumulate  carry frames i-K..i+K into frame i's thermal view. Per pixel: the nearest
              point over all frames (static-scene visibility), kept only if at least
              `--min-support` frames place a point within `--agree` of it (drops moving
              objects and stray occluded points). Reports how many pixels this adds
              where frame i's own GT has nothing, and draws them over the thermal image.

Data: sync_data/<seq>/{lidar/{left,right}/*.mat, calib.npy, thr/img_left},
      odom/<seq>/thr/*.txt (4x4 thermal-camera pose relative to its first frame).

    python tools/probe_lidar_accumulation.py --ms2-root /mnt/e/dataset/ms2 \
        --seq 2021-08-06-11-23-45 --frames 300 2000 4000 --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

H, W = 256, 640


def load_pose(odom: Path, idx: int) -> np.ndarray:
    return np.loadtxt(odom / "thr" / f"{idx:06d}.txt").reshape(4, 4)


def load_lidar(seq_dir: Path, side: str, idx: int) -> np.ndarray:
    import scipy.io as sio
    return sio.loadmat(seq_dir / "lidar" / side / f"{idx:06d}.mat")["data"][:, :3].astype(np.float64)


def rt(R, T_mm) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = np.asarray(R, np.float64)
    M[:3, 3] = np.asarray(T_mm, np.float64).reshape(3) / 1000.0
    return M


def lidar_to_thr(calib: dict, side: str, convention: str) -> np.ndarray:
    """4x4 taking LiDAR(side) coordinates to the rectified left thermal camera."""
    tag = "L" if side == "left" else "R"
    nir2lidar = rt(calib[f"R_nir2lidar{tag}"], calib[f"T_nir2lidar{tag}"])
    nir2thr = rt(calib["R_nir2thr"], calib["T_nir2thr"])
    if convention == "A":   # X = R * nir + T, i.e. the matrices map NIR into X
        return nir2thr @ np.linalg.inv(nir2lidar)
    return np.linalg.inv(nir2thr) @ nir2lidar  # "B": the matrices map X into NIR


def project(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Z-buffered depth image (metres, 0 = empty) of camera-frame points."""
    z = points_cam[:, 2]
    keep = z > 0.5
    p = points_cam[keep]
    u = np.round(K[0, 0] * p[:, 0] / p[:, 2] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * p[:, 1] / p[:, 2] + K[1, 2]).astype(int)
    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, d = u[inside], v[inside], p[inside, 2]
    depth = np.full(H * W, np.inf)
    np.minimum.at(depth, v * W + u, d)
    depth[~np.isfinite(depth)] = 0.0
    return depth.reshape(H, W)


def frame_points_in(seq_dir, odom, calib, conv, j, i) -> np.ndarray:
    """LiDAR points of frame j (both sensors) in thermal camera i coordinates."""
    rel = np.linalg.inv(load_pose(odom, i)) @ load_pose(odom, j)
    out = []
    for side in ("left", "right"):
        pts = load_lidar(seq_dir, side, j)
        homo = np.c_[pts, np.ones(len(pts))]
        out.append((rel @ lidar_to_thr(calib, side, conv) @ homo.T).T[:, :3])
    return np.vstack(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ms2-root", type=Path, required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--frames", type=int, nargs="+", required=True)
    ap.add_argument("--k", type=int, default=10, help="Neighbours on each side (10 = +-1 s).")
    ap.add_argument("--agree", type=float, default=0.03)
    ap.add_argument("--min-support", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seq_dir = args.ms2_root / "sync_data" / f"_{args.seq}"
    odom = args.ms2_root / "odom" / f"_{args.seq}"
    calib = np.load(seq_dir / "calib.npy", allow_pickle=True).item()
    K = np.asarray(calib["K_thrL"], np.float64)
    n_frames = len(list((odom / "thr").glob("*.txt")))

    # ---- check: which convention reproduces MS2's own projection ----
    best = None
    for conv in ("A", "B"):
        stats = []
        for i in args.frames:
            mine = project(frame_points_in(seq_dir, odom, calib, conv, i, i), K)
            for variant in ("depth", "depth_filtered"):
                gt = np.asarray(Image.open(args.ms2_root / "proj_depth" / f"_{args.seq}" / "thr" / variant / f"{i:06d}.png"),
                                np.float64) / 256.0
                both = (gt > 0) & (mine > 0)
                if both.sum() == 0:
                    stats.append((variant, 0.0, np.nan, int((gt > 0).sum())))
                    continue
                rel = np.abs(mine[both] - gt[both]) / gt[both]
                stats.append((variant, float((rel < 0.02).mean()), float(np.median(rel)), int((gt > 0).sum())))
        print(f"[check] convention {conv}:")
        for variant in ("depth", "depth_filtered"):
            rows = [s for s in stats if s[0] == variant]
            share = np.mean([s[1] for s in rows])
            print(f"   vs {variant:15s} pixels agreeing within 2%: {share:.1%}   "
                  f"median rel err {np.nanmean([s[2] for s in rows]):.4f}")
            if variant == "depth" and (best is None or share > best[1]):
                best = (conv, share)
    conv = best[0]
    print(f"[check] using convention {conv} ({best[1]:.1%} of GT pixels reproduced within 2%)")
    if best[1] < 0.8:
        print("!! neither convention reproduces MS2's projection; accumulation would be meaningless. Stopping.")
        return

    # ---- accumulate ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for i in args.frames:
        gt = np.asarray(Image.open(args.ms2_root / "proj_depth" / f"_{args.seq}" / "thr" / "depth_filtered" / f"{i:06d}.png"),
                        np.float64) / 256.0
        own = (gt > 1e-3) & (gt < 80)
        stack = [project(frame_points_in(seq_dir, odom, calib, conv, j, i), K)
                 for j in range(max(0, i - args.k), min(n_frames, i + args.k + 1)) if j != i]
        stack = np.stack(stack)                                   # (F, H, W), 0 = empty
        nearest = np.where(stack > 0, stack, np.inf).min(0)
        support = ((stack > 0) & (np.abs(stack - nearest) <= args.agree * nearest)).sum(0)
        acc = np.where(np.isfinite(nearest) & (support >= args.min_support) & (nearest < 80), nearest, 0.0)
        new = (acc > 0) & ~own
        # where both exist, how well does accumulation agree with the frame's own GT?
        both = (acc > 0) & own
        rel = np.abs(acc[both] - gt[both]) / gt[both]
        print(f"[frame {i}] own GT {own.mean():.1%} of frame; accumulation adds {new.mean():.1%} new "
              f"({new.sum()} px) -> {(own | new).mean():.1%} total; "
              f"on overlap with own GT: median rel err {np.median(rel):.4f}, within 5%: {(rel < 0.05).mean():.1%}")
        thermal = np.asarray(Image.open(seq_dir / "thr" / "img_left" / f"{i:06d}.png"), np.float32)
        lo, hi = np.percentile(thermal, (1, 99))
        thermal = np.clip((thermal - lo) / (hi - lo), 0, 1)
        fig, ax = plt.subplots(3, 1, figsize=(9, 10), dpi=100)
        for a, (title, d) in zip(ax, (("frame's own GT", np.where(own, gt, np.nan)),
                                      ("added by accumulation (new pixels only)", np.where(new, acc, np.nan)),
                                      ("own + added", np.where(own, gt, np.where(new, acc, np.nan))))):
            a.imshow(thermal, cmap="gray", vmin=0, vmax=1)
            a.imshow(np.log(d), cmap="magma", vmin=np.log(2), vmax=np.log(80), alpha=0.85)
            a.set_title(title, fontsize=10)
            a.axis("off")
        fig.tight_layout()
        fig.savefig(args.out / f"accum_{args.seq}_{i:06d}.png")
        plt.close(fig)
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
