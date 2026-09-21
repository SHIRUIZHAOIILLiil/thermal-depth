# -*- coding: utf-8 -*-
"""What trains, what does not, and what the loss is made of.

Two diagrams. The first is the backbone: one U-Net takes gradients and
everything around it is frozen, which is the fact the picture exists to make
obvious. The second is the metric head, a separate and much later stage that
trains on the first one's frozen output.

Colour says nothing here; the badge does. A flame means the block takes
gradients, a snowflake means it does not, and both are printed on the block
rather than collected in a legend, so no one has to hold a key in their head
while reading the path.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib import font_manager

for name in ("Microsoft YaHei", "SimHei", "DengXian"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [name]
        break
plt.rcParams["axes.unicode_minus"] = False

# The CJK font has no flame or snowflake, and matplotlib will not mix fonts
# inside one string -- it substitutes a blank box and warns, which is how a
# diagram ends up shipping with two empty squares where its legend was. So the
# badge is drawn as its own text object in a font that has the glyphs.
_installed = {f.name for f in font_manager.fontManager.ttflist}
BADGE_FONT = next((n for n in ("Segoe UI Emoji", "Segoe UI Symbol",
                               "Noto Color Emoji", "Symbola") if n in _installed), None)

INK = "#1F2A44"
TRAIN_FILL, TRAIN_LINE = "#FDECE4", "#C0392B"     # takes gradients
FROZEN_FILL, FROZEN_LINE = "#EDF2FA", "#7B8CA6"   # does not
DATA_FILL, DATA_LINE = "#F5F7FA", "#B9C2D0"
RED = "#C00000"
MUTED = "#5A667A"


def box(ax, x, y, w, h, *, state="frozen", radius=0.10):
    fill, line = {"train": (TRAIN_FILL, TRAIN_LINE),
                  "frozen": (FROZEN_FILL, FROZEN_LINE),
                  "data": (DATA_FILL, DATA_LINE)}[state]
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0,rounding_size={radius}",
                                facecolor=fill, edgecolor=line, linewidth=1.6))


def block(ax, x, y, w, h, title, subtitle=None, *, state="frozen", radius=0.10,
          fontsize=11.5):
    """state: train (flame), frozen (snowflake), or data (neither)."""
    box(ax, x, y, w, h, state=state, radius=radius)
    badge = {"train": "\U0001F525", "frozen": "❄", "data": ""}[state]
    # The title sits above centre only when there is a subtitle to sit below
    # it. At 11.5pt over 9.5pt they need about a quarter inch between
    # baselines; the first version gave them half that and printed one line
    # through the other.
    title_y = y + h / 2 + (0.14 if subtitle else 0.0)
    if badge and BADGE_FONT:
        # Badge to the left in its own font, title nudged right so it stays
        # centred on what is left of the block.
        ax.text(x + 0.17, title_y, badge, ha="left", va="center", fontsize=13,
                fontname=BADGE_FONT)
        ax.text(x + w / 2 + 0.14, title_y, title, ha="center", va="center",
                fontsize=fontsize, color=INK, weight="bold")
    else:
        ax.text(x + w / 2, title_y, title, ha="center", va="center",
                fontsize=fontsize, color=INK, weight="bold")
    if subtitle:
        ax.text(x + w / 2, title_y - 0.28, subtitle, ha="center", va="center",
                fontsize=9.5, color=MUTED)


def arrow(ax, x0, y0, x1, y1, *, style="-|>", colour=INK, dashed=False, width=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=13, color=colour, lw=width,
                                 linestyle="--" if dashed else "-",
                                 shrinkA=1, shrinkB=1))


def backbone():
    fig, ax = plt.subplots(figsize=(15.6, 7.6), dpi=140)
    ax.set_xlim(0, 15.6); ax.set_ylim(0, 7.6); ax.axis("off")

    ax.text(0.25, 7.25, "第一阶段：训练主干", fontsize=17, weight="bold", color=INK)
    ax.text(0.25, 6.92, "865 M 的 U-Net 是唯一吃梯度的部分；VAE 与文本编码器全程冻结",
            fontsize=12, color=MUTED)

    # A key, so the slide underneath does not have to spend a line saying what
    # the two badges mean -- and so the figure still reads on its own if it is
    # ever pulled out of the deck.
    if BADGE_FONT:
        ax.text(11.55, 7.18, "🔥", fontsize=13, fontname=BADGE_FONT,
                va="center")
        ax.text(11.85, 7.18, "= 训练（吃梯度）", fontsize=11.5, color=INK,
                va="center")
        ax.text(13.70, 7.18, "❄", fontsize=13, fontname=BADGE_FONT,
                va="center")
        ax.text(13.98, 7.18, "= 冻结", fontsize=11.5, color=INK, va="center")

    # Three inputs down the left, each through a frozen encoder.
    block(ax, 0.25, 4.70, 2.15, 0.90, "热像", "256×640，16 位", state="data")
    block(ax, 2.85, 4.70, 2.45, 0.90, "VAE 编码器", "冻结", state="frozen")
    arrow(ax, 2.40, 5.15, 2.85, 5.15)

    block(ax, 0.25, 3.05, 2.15, 0.90, "caption", "InternVL3 生成", state="data")
    block(ax, 2.85, 3.05, 2.45, 0.90, "CLIP 文本编码器", "冻结", state="frozen")
    arrow(ax, 2.40, 3.50, 2.85, 3.50)

    block(ax, 0.25, 1.40, 2.15, 0.90, "训练目标", "伪深度 + 激光覆写", state="data")
    block(ax, 2.85, 1.40, 2.45, 0.90, "VAE 编码器", "冻结（同一份）", state="frozen")
    arrow(ax, 2.40, 1.85, 2.85, 1.85)
    ax.text(1.32, 1.14, "先归一化：log 深度的 2%/98% 分位 → [-1, 1]",
            ha="center", fontsize=9.5, color=RED)

    # The one trainable block.
    block(ax, 6.05, 3.05, 2.95, 2.55, "U-Net", "865 M · 唯一训练的部分",
          state="train", radius=0.14)
    arrow(ax, 5.30, 5.15, 6.05, 4.75)          # image latent
    arrow(ax, 5.30, 3.50, 6.05, 3.85)          # text, into cross-attention
    ax.text(5.68, 3.28, "跨注意力", ha="center", fontsize=9, color=MUTED)

    # Target latent joins at the loss, not at the network.
    arrow(ax, 5.30, 1.85, 10.15, 1.85)
    block(ax, 10.15, 1.42, 2.35, 0.86, "目标 latent", state="data")

    block(ax, 9.55, 3.70, 2.55, 1.25, "预测 latent", "t=999，单步", state="data")
    arrow(ax, 9.00, 4.33, 9.55, 4.33)

    # The loss panel is laid out by hand: `block` centres its title, and this
    # box carries five lines under the title, so a centred title lands on top
    # of them.
    lx, ly, lw, lh = 12.85, 2.50, 2.55, 2.30
    box(ax, lx, ly, lw, lh, state="data", radius=0.12)
    cx = lx + lw / 2
    ax.text(cx, ly + lh - 0.26, "损失", ha="center", fontsize=12,
            weight="bold", color=INK)
    ax.text(cx, ly + 1.62, "SL_A", ha="center", fontsize=11.5, weight="bold",
            color=RED)
    ax.text(cx, ly + 1.36, "深度分支", ha="center", fontsize=9.5, color=MUTED)
    ax.text(cx, ly + 0.92, "SL_R", ha="center", fontsize=11.5, weight="bold",
            color=RED)
    ax.text(cx, ly + 0.66, "热像重建分支", ha="center", fontsize=9.5, color=MUTED)
    ax.text(cx, ly + 0.24, "两项都是 latent 上的 MSE", ha="center", fontsize=8.5,
            color=MUTED)

    arrow(ax, 12.10, 4.33, lx + 0.72, ly + lh, dashed=True, colour=RED)
    arrow(ax, 12.50, 1.85, lx + 0.30, ly, dashed=True, colour=RED)

    # Gradient path: back along the bottom, up into the U-Net, and nowhere else.
    arrow(ax, lx - 0.10, 2.90, 7.52, 2.90, colour=RED, dashed=True, width=2.0)
    ax.text(10.2, 2.66, "梯度只回到 U-Net", ha="center", fontsize=10.5,
            color=RED, weight="bold")
    arrow(ax, 7.52, 2.90, 7.52, 3.05, colour=RED, dashed=True, width=2.0)

    ax.text(0.25, 0.46,
            "损失  L = 1.0 × SL_A + 1.0 × SL_R"
            "　　（两个权重在基础配方里被固定为 1，改动需显式开启 metric 适配）",
            fontsize=11.5, color=INK)
    ax.text(0.25, 0.14,
            "注意：绝对尺度在“造训练目标”那一步就被逐帧分位数除掉了，"
            "所以网络输出必然没有单位",
            fontsize=10.5, color=RED)

    fig.savefig("arch_backbone.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("写入 arch_backbone.png")


def metric_head():
    fig, ax = plt.subplots(figsize=(15.6, 5.8), dpi=140)
    ax.set_xlim(0, 15.6); ax.set_ylim(0, 5.8); ax.axis("off")

    ax.text(0.25, 5.45, "第二阶段：米制重标定头（可选，仍在实验）",
            fontsize=17, weight="bold", color=INK)
    ax.text(0.25, 5.12,
            "第一阶段整个冻结；只训一个 0.39 M 的小网络，把没有单位的输出换算成米",
            fontsize=12, color=MUTED)

    block(ax, 0.25, 3.25, 2.55, 1.00, "第一阶段模型", "整个冻结", state="frozen")
    block(ax, 0.25, 1.95, 2.55, 1.00, "热像", state="data")
    block(ax, 0.25, 0.65, 2.55, 1.00, "caption", state="data")

    block(ax, 3.45, 0.65, 2.45, 1.00, "CLIP 文本编码器", "冻结", state="frozen")
    arrow(ax, 2.80, 1.15, 3.45, 1.15)

    block(ax, 6.45, 1.05, 3.05, 3.15, "重标定头", "0.39 M · 唯一训练的部分",
          state="train", radius=0.14)
    arrow(ax, 2.80, 3.75, 6.45, 3.55)
    ax.text(4.60, 3.88, "相对深度 y（0 到 1）", ha="center", fontsize=9.5,
            color=MUTED)
    arrow(ax, 2.80, 2.45, 6.45, 2.62)
    arrow(ax, 5.90, 1.15, 6.45, 1.70)
    ax.text(4.65, 1.82, "跨注意力", ha="center", fontsize=9, color=MUTED)

    block(ax, 10.15, 2.05, 2.55, 1.10, "逐像素 A、B", "两张与图同大的图",
          state="data")
    arrow(ax, 9.50, 2.60, 10.15, 2.60)

    block(ax, 13.05, 2.05, 2.30, 1.10, "米制深度", state="data")
    arrow(ax, 12.70, 2.60, 13.05, 2.60)
    ax.text(14.20, 1.78, "$D = \\exp(A\\odot y + B)$", ha="center", fontsize=11,
            color=RED)

    ax.text(0.25, 0.22,
            "损失：真实激光上的米制 L1（约 26% 的像素）　"
            "注意：监督必须用真实激光，不能用补全伪深度 —— 后者本身是拟合出来的，"
            "拿它监督米制是循环论证",
            fontsize=10.5, color=RED)

    fig.savefig("arch_metric_head.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("写入 arch_metric_head.png")


if __name__ == "__main__":
    backbone()
    metric_head()
