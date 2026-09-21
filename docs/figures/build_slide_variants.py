# -*- coding: utf-8 -*-
"""Slide-shaped versions of the two report figures.

The document figures are three rows by four columns, which is right for a page
and wrong for 16:9 -- placed at full body width they stand 6.6 inches tall on a
7.5 inch slide and run off the bottom. Shrinking them to fit makes the panel
labels unreadable, which defeats the point of showing a figure at all.

So each becomes a single row here, carrying one example instead of three. A
slide is read once, from a distance, while someone is talking over it; the
document keeps the version that rewards a second look.
"""
import glob
import json
import os

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for name in ("Microsoft YaHei", "SimHei", "DengXian"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [name]
        break
plt.rcParams["axes.unicode_minus"] = False

D_MIN, D_MAX = 1e-3, 80.0


def stretched(path):
    raw = np.asarray(Image.open(path), dtype=np.float32)
    low, high = (float(v) for v in np.percentile(raw, (1.0, 99.0)))
    return (np.clip((raw - low) / (high - low), 0.0, 1.0)
            if high > low else np.zeros_like(raw))


def metric_chain_slide():
    """Thermal, the unitless output, the fitted metres, the sparse lidar."""
    # The middle frame: 25.4% coverage, closest to the split's median, so the
    # example is typical rather than flattering.
    d = sorted(g for g in glob.glob("figs_report/f*") if os.path.isdir(g))[1]
    meta = json.load(open(f"{d}/meta.json", encoding="utf-8"))
    thermal = stretched(f"{d}/thermal.png")
    lidar = np.asarray(Image.open(f"{d}/lidar.png"), dtype=np.float32) / 256.0
    y = np.load(f"{d}/pred.npy").astype(np.float32)
    real = np.isfinite(lidar) & (lidar > D_MIN) & (lidar < D_MAX)

    a, b = np.polyfit(y[real].astype(np.float64),
                      np.log(lidar[real].astype(np.float64)), 1)
    metres = np.exp(np.clip(a * y + b, -9.0, 9.0))
    lo, hi = np.percentile(metres, [2, 98])
    depth_kw = dict(cmap="turbo_r", vmin=lo, vmax=hi, interpolation="nearest")

    fig = plt.figure(figsize=(17.0, 3.15), dpi=150)
    gs = fig.add_gridspec(1, 4, wspace=0.05, left=0.004, right=0.996,
                          top=0.80, bottom=0.10)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(thermal, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("热像输入", fontsize=15, pad=7)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(y, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax2.set_title("网络直接输出", fontsize=15, pad=7)
    ax2.set_xlabel("稠密，0 到 1，没有单位", fontsize=12, color="#C00000", labelpad=4)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(metres, **depth_kw)
    ax3.set_title("拟合两个参数之后", fontsize=15, pad=7)
    ax3.set_xlabel(f"$\\log D = {a:.2f}\\,y {b:+.2f}$　→　米",
                   fontsize=12, color="#C00000", labelpad=4)

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(np.zeros_like(lidar), cmap="gray", vmin=0, vmax=1,
               interpolation="nearest")
    ax4.imshow(np.where(real, lidar, np.nan), **depth_kw)
    ax4.set_title("官方激光 GT", fontsize=15, pad=7)
    ax4.set_xlabel(f"有单位，但只覆盖 {real.mean():.1%}", fontsize=12,
                   color="#C00000", labelpad=4)

    for ax in (ax, ax2, ax3, ax4):
        ax.set_xticks([]); ax.set_yticks([])
    out = "slide_metric_chain.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"写入 {out}   帧 {meta['id']}  a={a:.3f} b={b:+.3f}")


def gt_density_slide():
    """The published GT, the training target, and a pixel-level zoom."""
    d = "figs_report/train0"
    lidar = np.asarray(Image.open(f"{d}/lidar.png"), dtype=np.float32) / 256.0
    pseudo = np.load(f"{d}/pseudo.npy").astype(np.float32)
    real = np.isfinite(lidar) & (lidar > D_MIN) & (lidar < D_MAX)
    completed = np.clip(np.where(real, lidar, pseudo), D_MIN, D_MAX)
    coverage = real.mean()

    lo, hi = np.percentile(completed, [2, 98])
    depth_kw = dict(cmap="turbo_r", vmin=lo, vmax=hi, interpolation="nearest")

    size = 64
    counts = np.add.reduceat(
        np.add.reduceat(real.astype(np.int32),
                        np.arange(0, real.shape[0], size), axis=0),
        np.arange(0, real.shape[1], size), axis=1)
    by, bx = np.unravel_index(counts.argmax(), counts.shape)
    window = (slice(by * size, by * size + size), slice(bx * size, bx * size + size))

    fig = plt.figure(figsize=(16.0, 3.55), dpi=150)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.6, 1], wspace=0.06,
                          left=0.004, right=0.996, top=0.79, bottom=0.11)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(np.zeros_like(lidar), cmap="gray", vmin=0, vmax=1,
              interpolation="nearest")
    ax.imshow(np.where(real, lidar, np.nan), **depth_kw)
    ax.set_title(f"官方激光 GT　覆盖 {coverage:.1%}", fontsize=15, pad=7)
    ax.set_xlabel("黑色 = 没有测量值，未做膨胀或插值", fontsize=12,
                  color="#C00000", labelpad=4)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(completed, **depth_kw)
    ax2.set_title("训练用的目标", fontsize=15, pad=7)
    ax2.set_xlabel("伪深度铺满 + 真实激光盖上去", fontsize=12,
                   color="#C00000", labelpad=4)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(np.zeros((size, size)), cmap="gray", vmin=0, vmax=1,
               interpolation="nearest")
    ax3.imshow(np.where(real[window], lidar[window], np.nan), **depth_kw)
    ax3.set_title(f"放大：最密的一块 {real[window].mean():.0%}", fontsize=15, pad=7)
    ax3.set_xlabel("一个方块 = 一个像素", fontsize=12, color="#C00000", labelpad=4)

    for axis in (ax, ax2, ax3):
        axis.set_xticks([]); axis.set_yticks([])
    out = "slide_gt_density.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"写入 {out}   整帧 {coverage:.1%}  最密块 {real[window].mean():.1%}")


if __name__ == "__main__":
    metric_chain_slide()
    gt_density_slide()
