# -*- coding: utf-8 -*-
"""Why the published GT reads as dense when it covers 26% of the frame.

The guess going in was rendering: a sparse map drawn into a 300-pixel panel gets
interpolated and the gaps close. That is real, so every panel here uses
interpolation='nearest' and one zoom goes down to where a pixel is a visible
square. But it was the smaller half of the answer.

The larger half is that the 26% is not spread thin. It sits on the road surface
and stops at the horizon: the densest 64x64 block of this frame is 77% covered
while the frame is 27%, and the per-row profile runs from nothing at the top to
about 70% near the bottom. A picture whose lower half is nearly filled reads as
a dense ground truth, and in the part of the image that has structure, it is one.

The dense map beside it is a different object -- the training target, which is
AnyThermal pseudo depth everywhere with the real returns written over it.
"""
import json
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

D = "figs_report/train0"
D_MIN, D_MAX = 1e-3, 80.0

meta = json.load(open(f"{D}/meta.json", encoding="utf-8"))
thermal = np.asarray(Image.open(f"{D}/thermal.png"))
lidar = np.asarray(Image.open(f"{D}/lidar.png"), dtype=np.float32) / 256.0
pseudo = np.load(f"{D}/pseudo.npy").astype(np.float32)

real = np.isfinite(lidar) & (lidar > D_MIN) & (lidar < D_MAX)
completed = np.clip(np.where(real, lidar, pseudo), D_MIN, D_MAX)
coverage = real.mean()

lo, hi = np.percentile(completed, [2, 98])
depth_kw = dict(cmap="turbo_r", vmin=lo, vmax=hi, interpolation="nearest")

fig = plt.figure(figsize=(16.8, 7.6), dpi=130)
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.26, wspace=0.10,
                      left=0.015, right=0.985, top=0.885, bottom=0.055)


def panel(position, image, title, sub=None, **show):
    ax = fig.add_subplot(position)
    ax.imshow(image, **(show or depth_kw))
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    if sub:
        ax.set_xlabel(sub, fontsize=10, color="#555", labelpad=5)
    return ax


panel(gs[0, 0], thermal, "热像输入", sub="模型看到的",
      cmap="gray", interpolation="nearest")

# Sparse on black, never dilated: an unmeasured pixel stays unmeasured.
ax = fig.add_subplot(gs[0, 1])
ax.imshow(np.zeros_like(lidar), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax.imshow(np.where(real, lidar, np.nan), **depth_kw)
ax.set_title(f"官方激光 GT　覆盖 {coverage:.1%}", fontsize=13, pad=8)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("黑色 = 没有测量值。未做膨胀，未做插值", fontsize=10,
              color="#555", labelpad=5)

panel(gs[0, 2], completed, "补全伪深度（训练目标）",
      sub="伪深度填满 + 激光覆写，处处有值")

# A window where returns are individually visible, so nothing can be hiding in
# the rendering. The densest 64x64 block, which turns out to be the more useful
# choice: it is 77% covered against 27% for the frame, and that gap is the
# actual answer to why this GT reads as dense. Coverage is not thin everywhere
# -- it is concentrated on the road surface and absent above the horizon.
size = 64
counts = np.add.reduceat(np.add.reduceat(real.astype(np.int32),
                                         np.arange(0, real.shape[0], size), axis=0),
                         np.arange(0, real.shape[1], size), axis=1)
by, bx = np.unravel_index(counts.argmax(), counts.shape)
y0, x0 = by * size, bx * size
window = (slice(y0, y0 + size), slice(x0, x0 + size))
patch_coverage = real[window].mean()

ax = fig.add_subplot(gs[1, 0])
ax.imshow(np.zeros((size, size)), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
ax.imshow(np.where(real[window], lidar[window], np.nan), **depth_kw)
ax.set_title(f"放大 · 激光最密的 64×64 块　{patch_coverage:.1%}", fontsize=12.5, pad=7)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("一个方块 = 一个像素", fontsize=10, color="#555", labelpad=5)

panel(gs[1, 1], completed[window], "放大 · 同一块的训练目标",
      sub="每个像素都有值，来自伪深度")

# Coverage per image row, which says in one line what the two panels above show
# by example: the 27% is not spread thin, it sits on the road and stops at the
# horizon. A frame whose lower half is nearly filled reads as a dense map even
# though most of the picture has no measurement at all.
ax = fig.add_subplot(gs[1, 2])
per_row = real.mean(axis=1) * 100
rows = np.arange(real.shape[0])
ax.plot(per_row, rows, color="#C00000", lw=1.6)
ax.fill_betweenx(rows, 0, per_row, color="#C00000", alpha=0.15)
ax.axvline(coverage * 100, color="#4472C4", lw=1.3, ls="--",
           label=f"整帧平均 {coverage:.1%}")
ax.set_ylim(real.shape[0] - 1, 0)
ax.set_xlim(0, 100)
ax.set_xlabel("该行有激光的像素占比（%）", fontsize=10)
ax.set_ylabel("图像行（上→下）", fontsize=10)
ax.set_title("覆盖率沿高度的分布", fontsize=12.5, pad=7)
ax.legend(fontsize=9.5, loc="upper right", framealpha=0.95)
ax.tick_params(labelsize=9)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

fig.suptitle(
    f"官方 GT 覆盖 {coverage:.1%}，但它集中在路面上　·　"
    f"稠密的那张是补出来的　·　帧 {meta['id']}",
    fontsize=15, y=0.955)
out = "gt_density_compare.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"写入 {out}")
print(f"  覆盖 {coverage:.2%}  最密 64×64 块 {patch_coverage:.2%}  "
      f"位置 ({y0}, {x0})")
print(f"  伪深度 {pseudo.min():.2f}–{pseudo.max():.2f} m  "
      f"覆写像素 {int(real.sum()):,}")
top = real[: real.shape[0] // 2].mean()
bottom = real[real.shape[0] // 2 :].mean()
print(f"  上半幅覆盖 {top:.1%}   下半幅覆盖 {bottom:.1%}")
