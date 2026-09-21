# -*- coding: utf-8 -*-
"""A 25.4% ground truth that looks solid: which number is wrong, and why neither is.

Kept as a standalone figure to bring out if the question is asked, because the
question is a fair one -- the lidar panel on the slide does not look like a
quarter of the pixels.

Both readings are correct, and there are two reasons, in this order.

The larger one is that the 25.4% is not spread evenly. Returns land where the
scene has structure: the densest 96x96 block of this frame is 69% covered and
the per-row profile peaks near 75%, while the sky contributes almost nothing.
So in the part of the picture anyone actually looks at, the map really is close
to dense, and reading it that way is not a mistake.

The smaller one is display scale. Shrinking a 640-pixel-wide map into a
400-pixel-wide panel makes one output pixel stand for several input ones, and
it lights up if any of them has a return: measured on the rendering itself, the
visible fraction goes from 25.4% at 1:1 to 37% at 600 and 44% at 300.
interpolation='nearest' does not prevent this -- it stops the renderer blurring
between pixels, and does nothing about several pixels competing for one on the
way down.

The top row measures the second effect; the bottom row magnifies instead of
shrinking, where a return is a visible square and the first can be counted.
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

D = "figs_report/f1_2021-08-13-16-31-10_004184"
RED = "#C00000"

lidar = np.asarray(Image.open(f"{D}/lidar.png"), np.float32) / 256.0
real = np.isfinite(lidar) & (lidar > 1e-3) & (lidar < 80.0)
H, W = real.shape
lo, hi = np.percentile(lidar[real], [2, 98])


def render(mask, values):
    """The panel as pixels: black where unmeasured, colour where measured."""
    rgba = plt.get_cmap("turbo_r")((np.clip(values, lo, hi) - lo) / (hi - lo))
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    rgb[~mask] = 0
    return rgb


full = render(real, lidar)


def at_width(target):
    """Downsample the rendering the way a viewer does, then measure what lit up."""
    small = Image.fromarray(full).resize((target, round(target * H / W)), Image.LANCZOS)
    array = np.asarray(small)
    visible = float((array.sum(axis=2) > 20).mean())
    # Back up to a common size with nearest, so the figure shows what the
    # downsample did rather than hiding it under another resample.
    shown = np.asarray(small.resize((W, H), Image.NEAREST))
    return shown, visible


fig = plt.figure(figsize=(16.4, 7.0), dpi=140)
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.35], hspace=0.30, wspace=0.07,
                      left=0.012, right=0.988, top=0.855, bottom=0.05)

for column, width in enumerate((300, 600, W)):
    ax = fig.add_subplot(gs[0, column])
    if width == W:
        shown, visible = full, float(real.mean())
        title = f"原始分辨率 {W} 像素宽"
    else:
        shown, visible = at_width(width)
        title = f"缩到 {width} 像素宽"
    ax.imshow(shown, interpolation="nearest")
    ax.set_title(title, fontsize=13.5, pad=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"看上去有 {visible:.1%} 的像素是亮的", fontsize=12,
                  color=RED if width != W else "#1E6B52", labelpad=5)

# A window big enough to read a scene from, small enough that one array pixel
# gets several screen pixels. Placed where the returns are densest, so the
# count is the friendliest one available to the claim being questioned.
size = 96
counts = np.add.reduceat(np.add.reduceat(real.astype(np.int32),
                                         np.arange(0, H, size), axis=0),
                         np.arange(0, W, size), axis=1)
by, bx = np.unravel_index(counts.argmax(), counts.shape)
y0, x0 = by * size, bx * size
window = (slice(y0, min(y0 + size, H)), slice(x0, min(x0 + size, W)))
patch = real[window]

ax = fig.add_subplot(gs[1, 0])
ax.imshow(full[window], interpolation="nearest")
ax.set_title(f"放大 · 回波最密的 {patch.shape[0]}×{patch.shape[1]} 块",
             fontsize=13, pad=7)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel(f"这一块 {patch.mean():.1%}，一个方块就是一个像素", fontsize=12,
              color=RED, labelpad=5)

ax = fig.add_subplot(gs[1, 1])
ax.imshow(patch, cmap="gray", interpolation="nearest")
ax.set_title("同一块，只看有没有回波", fontsize=13, pad=7)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("白 = 有测量值，黑 = 没有", fontsize=12, color="#555", labelpad=5)

ax = fig.add_subplot(gs[1, 2])
per_row = real.mean(axis=1) * 100
rows = np.arange(H)
ax.plot(per_row, rows, color=RED, lw=1.7)
ax.fill_betweenx(rows, 0, per_row, color=RED, alpha=0.14)
ax.axvline(real.mean() * 100, color="#4472C4", lw=1.4, ls="--",
           label=f"整帧 {real.mean():.1%}")
ax.set_ylim(H - 1, 0); ax.set_xlim(0, 100)
ax.set_xlabel("该行有回波的像素占比（%）", fontsize=11)
ax.set_ylabel("图像行（上→下）", fontsize=11)
ax.set_title("覆盖沿高度的分布", fontsize=13, pad=7)
ax.legend(fontsize=10, loc="lower right", framealpha=0.95)
ax.tick_params(labelsize=9)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

fig.suptitle(
    "整帧覆盖 25.4%，但回波集中在有结构的地方：最密的一块 69%，逐行峰值约 75%"
    "　·　缩小显示会再虚增一截",
    fontsize=15, y=0.945)
out = "lidar_density_answer.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print(f"写入 {out}")
print(f"  整帧 {real.mean():.2%}   最密 {size}×{size} 块 {patch.mean():.1%} "
      f"（位置 {y0},{x0}）")
for width in (300, 600):
    print(f"  缩到 {width} 宽后看上去 {at_width(width)[1]:.1%}")
