# -*- coding: utf-8 -*-
"""Two crops of the lidar mask, small enough to count by eye.

Coverage figures kept losing this argument. A colour map at 3x looks solid, a
per-row profile is a number about a picture rather than the picture, and the
same frame at three display widths came out indistinguishable. None of them let
anyone check.

So: two 32x32 blocks, no colour, a grid every fourth pixel, and the count in the
title. The right-hand crop is the useful one -- the bottom of the frame reads as
covered in points and holds 51 measurements out of 1024, because regularly
spaced dots on black are read as a surface. The left is the densest band, and
even there it is about half.

The frame's 25.4% comes straight from the file: 122,113 of 163,840 pixels are
exactly zero before any mask, threshold or rendering of ours touches them.
"""
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

FRAME = "figs_report/f1_2021-08-13-16-31-10_004184/lidar.png"
SIZE = 32
CROPS = [(150, 300, "中间亮带（看着最实的地方）"),
         (215, 300, "最下方（看着有很多点）")]

raw = np.asarray(Image.open(FRAME))
depth = raw.astype(np.float32) / 256.0
real = (depth > 1e-3) & (depth < 80.0)
print(f"整帧 {real.mean():.2%}   文件里 =0 的像素 {int((raw == 0).sum()):,}"
      f" / {raw.size:,}")

fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.2), dpi=140)
for ax, (y0, x0, label) in zip(axes, CROPS):
    block = real[y0:y0 + SIZE, x0:x0 + SIZE]
    ax.imshow(block, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(np.arange(-.5, SIZE, 4)); ax.set_yticks(np.arange(-.5, SIZE, 4))
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.grid(color="#4472C4", lw=0.7, alpha=0.55)
    ax.set_title(f"{label}\n{SIZE}×{SIZE} = {SIZE * SIZE} 个像素，"
                 f"其中 {int(block.sum())} 个有测量值", fontsize=14, pad=10)
    ax.set_xlabel(f"覆盖 {block.mean():.1%}　白 = 有，黑 = 没有", fontsize=13,
                  color="#C00000", labelpad=8)
    print(f"  {label}: {int(block.sum())}/{SIZE * SIZE} = {block.mean():.1%}")

fig.suptitle("同一帧的两个 32×32 小块，格子线每 4 像素一道 —— 可以直接数",
             fontsize=15.5, y=0.965)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("lidar_countable.png", bbox_inches="tight", facecolor="white")
print("写入 lidar_countable.png")
