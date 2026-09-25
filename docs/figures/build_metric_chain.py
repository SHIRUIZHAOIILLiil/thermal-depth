# -*- coding: utf-8 -*-
"""What the network emits, and where the metres come from.

The recurring question is where the metric information goes. This answers it on
three real test frames: the network's own output is dense and has no unit, the
lidar has units and is sparse, and the metres in every number this project
reports arrive from a two-parameter fit between them -- solved per frame, in log
space, on the measured pixels only.

The fit is the same one the official evaluator performs, reimplemented here so
the figure shows the actual arithmetic rather than a picture of it. The middle
panel is deliberately given its own colour bar reading 0 to 1: it is not depth,
and giving it a metre scale would assert the opposite of the point.

Test frames have no pseudo depth by design -- producing it needs a fit against
ground truth, and doing that on the test split would contaminate it.
"""
import glob
import json
import os

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import font_manager

for name in ("Microsoft YaHei", "SimHei", "DengXian"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [name]
        break
plt.rcParams["axes.unicode_minus"] = False

D_MIN, D_MAX = 1e-3, 80.0
dirs = sorted(d for d in glob.glob("figs_report/f*") if os.path.isdir(d))

fig = plt.figure(figsize=(17.6, 9.4), dpi=130)
gs = fig.add_gridspec(len(dirs), 4, hspace=0.30, wspace=0.06,
                      left=0.075, right=0.975, top=0.885, bottom=0.035)

for row, d in enumerate(dirs):
    meta = json.load(open(f"{d}/meta.json", encoding="utf-8"))
    # The same 1%/99% stretch the model is fed. Showing the raw 16-bit frame
    # under a min-max scale instead makes a night frame read as almost black,
    # which is not what the network sees and would misrepresent the input.
    raw = np.asarray(Image.open(f"{d}/thermal.png"), dtype=np.float32)
    low, high = (float(v) for v in np.percentile(raw, (1.0, 99.0)))
    thermal = (np.clip((raw - low) / (high - low), 0.0, 1.0)
               if high > low else np.zeros_like(raw))
    lidar = np.asarray(Image.open(f"{d}/lidar.png"), dtype=np.float32) / 256.0
    y = np.load(f"{d}/pred.npy").astype(np.float32)

    real = np.isfinite(lidar) & (lidar > D_MIN) & (lidar < D_MAX)

    # The evaluator's ssi_log, on this frame: least squares of a*y + b against
    # log(GT) over the measured pixels, then exponentiate. Two numbers, fitted
    # per frame. They are the entire metric content of the result.
    a, b = np.polyfit(y[real].astype(np.float64),
                      np.log(lidar[real].astype(np.float64)), 1)
    metres = np.exp(np.clip(a * y + b, -9.0, 9.0))
    aligned_error = float(np.mean(np.abs(metres[real] - lidar[real]) / lidar[real]))

    lo, hi = (float(v) for v in np.percentile(metres, [2, 98]))
    depth_norm = colors.FuncNorm(
        (lambda d: -1.0 / np.clip(d, 0.5, None),
         lambda v: -1.0 / np.clip(v, None, -1e-6)),
        vmin=lo, vmax=hi)
    depth_kw = dict(cmap="magma_r", norm=depth_norm, interpolation="nearest")

    def metre_ticks(bar):
        """刻度选在米制的整数上，位置由 norm 决定，所以间距会不均匀。"""
        nice = [3, 5, 8, 12, 20, 30, 50, 80]
        bar.set_ticks([t for t in nice if lo <= t <= hi])
        bar.ax.tick_params(labelsize=8)

    def cell(column, image, title, **show):
        ax = fig.add_subplot(gs[row, column])
        handle = ax.imshow(image, **(show or depth_kw))
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(title, fontsize=13, pad=9)
        return ax, handle

    ax, _ = cell(0, thermal, "热像输入（1%/99% 拉伸，与训练一致）",
                 cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_ylabel(meta["id"] + "\n" + f"覆盖 {meta['coverage']:.1%}",
                  fontsize=9.5, labelpad=8, linespacing=1.6)

    # 量程是它自己的 0–1，因为这不是深度 —— 但方向必须和左右两格一致：越亮越近。
    # 曾经这一格用 viridis、邻格用别的，同一个量在相邻两张图里跑反了方向。
    ax, handle = cell(1, y, "网络直接输出　$y$",
                      cmap="magma_r", vmin=0, vmax=1, interpolation="nearest")
    bar = fig.colorbar(handle, ax=ax, fraction=0.030, pad=0.010)
    bar.ax.tick_params(labelsize=8)
    if row == len(dirs) - 1:
        ax.set_xlabel("稠密，但没有单位", fontsize=10, color="#555", labelpad=4)

    ax, handle = cell(2, metres, "拟合两个参数之后（米）")
    metre_ticks(fig.colorbar(handle, ax=ax, fraction=0.030, pad=0.010))
    ax.set_xlabel(f"$\\log D = {a:.3f}\\,y {b:+.3f}$", fontsize=10,
                  color="#C00000", labelpad=4)

    ax = fig.add_subplot(gs[row, 3])
    ax.imshow(np.zeros_like(lidar), cmap="gray", vmin=0, vmax=1,
              interpolation="nearest")
    ax.imshow(np.where(real, lidar, np.nan), **depth_kw)
    ax.set_xticks([]); ax.set_yticks([])
    if row == 0:
        ax.set_title("官方激光 GT（米）", fontsize=13, pad=9)
    if row == len(dirs) - 1:
        ax.set_xlabel("有单位，但稀疏。那两个参数就是从这些像素上拟合的",
                      fontsize=10, color="#555", labelpad=4)

    print(f"{meta['id']}  覆盖 {meta['coverage']:.1%}  "
          f"a={a:.4f} b={b:+.4f}  对齐后 AbsRel {aligned_error:.4f}  "
          f"深度范围 {metres.min():.1f}-{metres.max():.1f} m")

fig.suptitle(
    "网络出的是没有单位的相对量，米来自逐帧拟合的两个参数　·　"
    "官方 test 三帧，按激光覆盖率分位数选取",
    fontsize=15, y=0.955)
out = "metric_chain.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"\n写入 {out}")
