"""汇报：log 目标线、双种子、caption 三层、米制映射、架构（2026-09-21）。

十页，一页一件事。表只放该放的数：三个指标一起，原始值并排，不放胜率、不放
置信区间、不放相对差值 —— 让读者自己看两行的差，而不是替他算好。

精度按评估器报的位数写。此前两次缩位都出过事：两位小数把 3.605 / 3.598 印成
一样，四位把 0.07996 印成 0.0800。下面的断言挡第一种。

    python tools/build_report_20260921.py --output docs/data/汇报_20260921.pptx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from PIL import Image
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_report8_slides import (  # noqa: E402
    BAND, GOOD, GREY, INK, RULE, WHITE,
    BODY_W, MARGIN, W, H,
    conclusion, footnote, header, table, textbox, write,
)

DATE = "2026-09-21"
FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures"

METRICS = ["AbsRel ↓", "RMSE (m) ↓", "δ1 ↑"]
NUMCOLS = [PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 3


def new_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def fit_height(slide, picture: Path, top: float, height: float):
    """Place a figure by height, centred, reading its aspect off the file.

    The aspect used to be written into this script as a pair of pixel counts,
    which meant every edit to a figure silently rescaled it here -- the numbers
    had already gone stale once by the time anyone looked.
    """
    width = height * Image.open(picture).size[0] / Image.open(picture).size[1]
    slide.shapes.add_picture(str(picture), Inches(MARGIN + (BODY_W - width) / 2),
                             Inches(top), height=Inches(height))


def unbold(tbl, *, rows_from=1):
    """Drop the bold that `table`'s highlight applies, keeping the shading."""
    for row in range(rows_from, len(tbl.rows)):
        for column in range(len(tbl.columns)):
            for para in tbl.cell(row, column).text_frame.paragraphs:
                for run in para.runs:
                    run.font.bold = False


def check_paired_rows_differ(rows: list[list[str]]) -> None:
    """Consecutive rows are the two arms of one comparison; they must print apart.

    The reader is being asked to see a difference between two lines. If rounding
    prints them identically, the slide asserts a tie the data does not contain --
    which has happened here twice, once at two decimals on an RMSE pair and once
    at four on an AbsRel. This refuses to build rather than let it through a
    third time.
    """
    body = rows[1:]
    assert len(body) % 2 == 0, "成对的表必须有偶数行"
    for index in range(0, len(body), 2):
        first, second = body[index], body[index + 1]
        same = [column for column in range(1, len(first))
                if first[column] == second[column]]
        if same:
            raise SystemExit(
                f"⛔ 「{first[0].strip()}」与「{second[0].strip()}」在第 {same} 列"
                f"印出来一模一样（{[first[c] for c in same]}）。"
                "两行并排就是让人看差别的，印成一样等于在说它们相等。")


# ── 1. 封面 ────────────────────────────────────────────────────────────────
def cover(prs):
    slide = new_slide(prs)
    bar = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(W), Inches(2.55))
    bar.fill.solid(); bar.fill.fore_color.rgb = INK
    bar.line.fill.background(); bar.shadow.inherit = False
    write(textbox(slide, MARGIN, 0.72, 12.0, 0.36), ["热像深度估计 · 进展汇报"],
          size=13, colour=RULE, bold=True)
    write(textbox(slide, MARGIN, 1.12, 12.0, 1.10),
          ["caption 起作用，且在两个随机种子上复现"], size=30, colour=WHITE, bold=True)
    write(textbox(slide, MARGIN, 2.95, BODY_W, 2.6), [
        ("本次汇报三件事", {"size": 15, "bold": True}),
        "一、换 log 深度作训练目标，三个指标同时变好，双种子验证",
        "二、caption 的作用拆成两层：文本在不在，文本说了什么",
        "三、模型直接输出的是什么，米是怎么加上去的",
    ], size=14, space_after=9)
    write(textbox(slide, MARGIN, 6.70, BODY_W, 0.3), [DATE], size=11, colour=GREY)


# ── 2. 换 log 目标 ─────────────────────────────────────────────────────────
def objective(prs):
    slide = new_slide(prs)
    header(slide, "训练目标", "把目标从视差空间换到 log 深度空间")
    rows = [
        ["场景 / 训练目标"] + METRICS,
        ["白天　视差空间", "0.07935", "3.821", "0.9218"],
        ["白天　log 深度", "0.07333", "2.914", "0.9459"],
        ["夜间　视差空间", "0.08179", "3.515", "0.9298"],
        ["夜间　log 深度", "0.07601", "2.626", "0.9490"],
        ["雨天　视差空间", "0.10584", "4.600", "0.8793"],
        ["雨天　log 深度", "0.09654", "3.542", "0.9098"],
    ]
    check_paired_rows_differ(rows)
    table(slide, rows, MARGIN, 1.62, BODY_W, 3.30, [2.3, 1, 1, 1],
          size=14, align=NUMCOLS, highlight=(2, 4, 6))
    footnote(slide, "同起点、同数据、同步数，只换训练目标的归一化空间。"
                    "带 caption 训练，推理给本帧 caption。", top=5.10)
    write(textbox(slide, MARGIN, 5.48, BODY_W, 0.9), [
        ("为什么这次没有“买一个赔一个”", {"bold": True, "size": 13}),
        "视差空间把分辨率花在近处、深度空间花在远处，所以之前换深度空间时 RMSE 降了但 AbsRel 变差；"
        "log 空间对每个距离是同样的相对精度，两者不再互换。",
    ], size=12.5, space_after=5)
    conclusion(slide, "三个场景、三个指标同时变好，RMSE 降约 0.9 米")


# ── 3. 双种子 ──────────────────────────────────────────────────────────────
def seeds(prs):
    slide = new_slide(prs)
    header(slide, "复现", "换一个随机种子重训，结论不变")
    rows = [
        ["场景 / 种子"] + METRICS,
        ["白天　种子 42", "0.07594", "2.979", "0.94308"],
        ["白天　种子 43", "0.07333", "2.914", "0.94588"],
        ["夜间　种子 42", "0.07648", "2.657", "0.94827"],
        ["夜间　种子 43", "0.07601", "2.626", "0.94896"],
        ["雨天　种子 42", "0.09681", "3.564", "0.90979"],
        ["雨天　种子 43", "0.09654", "3.542", "0.90985"],
    ]
    # δ1 carries five decimals here and four on the previous slide. That is not
    # an inconsistency to tidy up: at four, the two rainy seeds both print
    # 0.9098 while the values are 0.90979 and 0.90985, and a table whose whole
    # job is to put two runs side by side would be claiming they are identical.
    check_paired_rows_differ(rows)
    # Shaded by pair, not by winner. The previous slide marks the better row of
    # each pair because there is a real effect to point at; here the two seeds
    # differ by less than a rerun would, so marking one would present run-to-run
    # noise as a result. The banding groups each pair instead, which is what the
    # slide asks the reader to look at.
    tbl = table(slide, rows, MARGIN, 1.62, BODY_W, 3.30, [2.3, 1, 1, 1],
                size=14, align=NUMCOLS, highlight=(1, 2, 5, 6))
    # `highlight` shades and bolds together. Bold is emphasis and shading is
    # grouping, and here only grouping is meant: left as it came, four rows
    # would read as important and the night pair as an aside.
    unbold(tbl, rows_from=1)
    footnote(slide, "两条臂各自由验证集十个候选点自动选出最好的一个，"
                    "全程不看测试集。", top=5.10)
    write(textbox(slide, MARGIN, 5.48, BODY_W, 0.9), [
        "两个种子的六组结果彼此接近，并且都明显优于上一页的基线那一栏。",
    ], size=13, space_after=5)
    conclusion(slide, "不是单次运行的运气")


# ── 4. caption 分两层 ──────────────────────────────────────────────────────
def caption_levels(prs):
    slide = new_slide(prs)
    header(slide, "caption", "同一份权重，只换推理时给的文本")
    rows = [
        ["推理时喂进去的文本"] + METRICS,
        ["不给文本（空）", "0.07556", "2.996", "0.9426"],
        ["给本帧自己的 caption", "0.07333", "2.914", "0.9459"],
        ["给随机另一张图的 caption", "0.07355", "2.923", "0.9454"],
    ]
    table(slide, rows, MARGIN, 1.66, BODY_W, 1.95, [2.6, 1, 1, 1],
          size=14.5, align=NUMCOLS, highlight=(2,))
    write(textbox(slide, MARGIN, 3.92, BODY_W, 2.3), [
        ("三行怎么读", {"bold": True, "size": 14}),
        "第一行 → 第二行：给文本，三个指标都变好 —— 这是主要的那部分",
        "第三行 → 第二行：文本换成对的那一句，还能再好一点 —— 这部分小得多",
        ("所以：作用主要来自“有文本”，一部分来自“文本内容对得上”。",
         {"bold": True, "colour": GOOD, "space_after": 8}),
        "这个方向在两个种子 × 白天/夜间/雨天 × 三个指标上完全一致。",
    ], size=13, space_after=7)
    footnote(slide, "白天，种子 43。权重完全相同，差别只在送进网络的那句话。", top=6.22)
    conclusion(slide, "文本确实在起作用，而且能分清是哪一部分在起作用")


# ── 对比：同一数据上训练的热像模型 ────────────────────────────────────────
def against_anythermal(prs):
    slide = new_slide(prs)
    header(slide, "对比", "和一个在同一批数据上训练过的热像模型")
    rows = [
        ["场景 / 模型"] + METRICS,
        ["白天　AnyThermal", "0.0810", "2.614", "0.9424"],
        ["白天　我们（带 caption）", "0.0737", "2.924", "0.9453"],
        ["夜间　AnyThermal", "0.0872", "2.514", "0.9380"],
        ["夜间　我们（带 caption）", "0.0763", "2.639", "0.9482"],
        ["雨天　AnyThermal", "0.1020", "3.159", "0.9048"],
        ["雨天　我们（带 caption）", "0.0965", "3.538", "0.9099"],
    ]
    check_paired_rows_differ(rows)
    # Shaded by pair, and no row is marked better, because none is: two of the
    # three metrics go our way and the third does not. Marking a winner row
    # here would assert something the numbers refuse.
    tbl = table(slide, rows, MARGIN, 1.62, BODY_W, 3.30, [2.3, 1, 1, 1],
                size=14, align=NUMCOLS, highlight=(1, 2, 5, 6))
    unbold(tbl, rows_from=1)
    footnote(slide, "官方帧集，每 10 帧抽 1（白天 2331 / 夜间 2292 / 雨天 2503）——"
                    "外部模型的数只在这个帧集上存在。第 2–4 页按全帧计，同一条臂，两种口径差约 0.0005。",
             top=5.06)
    write(textbox(slide, MARGIN, 5.44, BODY_W, 1.0), [
        "我们这三行就是第 2–4 页那条臂：带 caption 训练，推理给本帧 caption。"
        "AnyThermal 的深度监督也是 MS2 训练集 + 稀疏激光，与我们同源。",
        ("我们的训练目标由它的预测标定而来，所以赢的那两项说明的是"
         "「激光覆写和训练补进了它没有的东西」，不是架构更好。", {"colour": GREY, "size": 12}),
    ], size=12.5, space_after=5)
    conclusion(slide, "AbsRel 与 δ1 三个场景都更好，RMSE 三个场景都更差")


# ── 5. 架构：出来的是什么 ──────────────────────────────────────────────────
def chain(prs):
    slide = new_slide(prs)
    header(slide, "架构", "网络直接出的没有单位，米是后面加上去的")
    # The slide variant: one frame across, not three down. The document figure
    # at full body width stands 6.6 inches tall on a 7.5 inch slide.
    picture = FIGURES / "slide_metric_chain.png"
    if picture.is_file():
        slide.shapes.add_picture(str(picture), Inches(MARGIN), Inches(1.72),
                                 width=Inches(BODY_W))
    write(textbox(slide, MARGIN, 3.80, BODY_W, 2.4), [
        "网络吐出的是一张 0 到 1 的图，稠密但没有单位；官方激光有单位但只覆盖约四分之一的像素。"
        "每一帧在有激光的那些像素上拟合两个数，就把整张图换算成米 —— 图中红色那行公式即是。",
        ("这两个数是逐帧拟合的，绝对尺度在造训练目标时就被除掉了，所以模型输出没有单位是必然的。",
         {"colour": GREY, "size": 12}),
    ], size=13, space_after=7)
    conclusion(slide, "稠密的那张没单位，有单位的那张不稠密，米来自两者之间的拟合")


# ── 架构三页：谁训练、谁冻结、损失是什么 ───────────────────────────────────
def code(slide, left, top, width, height, lines, *, size=10):
    """A fixed-width excerpt with a tinted backing, so it reads as quoted code.

    Lines are quoted from the source with the long comments dropped, since a
    slide has room for the statement or for the reasoning behind it but not
    both. `word_wrap` is off: a wrapped line of code reads as two statements.
    """
    from pptx.enum.shapes import MSO_SHAPE
    plate = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left),
                                   Inches(top), Inches(width), Inches(height))
    plate.fill.solid(); plate.fill.fore_color.rgb = RGBColor(0xF4, 0xF6, 0xFA)
    plate.line.color.rgb = RGBColor(0xD5, 0xDD, 0xE8)
    plate.shadow.inherit = False
    frame = plate.text_frame
    frame.word_wrap = False
    frame.margin_left = frame.margin_right = Inches(0.14)
    frame.margin_top = frame.margin_bottom = Inches(0.09)
    for index, line in enumerate(lines):
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        para.space_after = Pt(1.5)
        run = para.add_run()
        run.text = line
        run.font.name = "Consolas"
        run.font.size = Pt(size)
        run.font.color.rgb = RGBColor(0x24, 0x2E, 0x44)
    return plate


def arch_backbone(prs):
    slide = new_slide(prs)
    header(slide, "架构", "只有 U-Net 在训练，其余全部冻结")
    picture = FIGURES / "arch_backbone.png"
    if picture.is_file():
        # 按高度摆：满版心宽时它会一路压过结论条。
        fit_height(slide, picture, top=1.44, height=5.02)
    conclusion(slide, "一个网络在学，其余都是固定的编码器")


def arch_loss(prs):
    slide = new_slide(prs)
    header(slide, "损失", "两项 latent MSE，权重都是 1")
    write(textbox(slide, MARGIN, 1.52, BODY_W, 0.42), [
        "L = 1.0 × SL_A（深度分支） + 1.0 × SL_R（热像重建分支）",
    ], size=16, bold=True, space_after=4)
    code(slide, MARGIN, 2.02, BODY_W, 1.28, [
        "anno_loss = F.mse_loss(model_pred[:bsz][mask_anno], target[:bsz][mask_anno])",
        "rgb_loss  = F.mse_loss(model_pred[bsz:][mask_rgb],  target[bsz:][mask_rgb])",
        "loss = args.lambda_dense * anno_loss + args.lambda_recon * rgb_loss",
    ], size=10.5)
    write(textbox(slide, MARGIN, 3.44, BODY_W, 0.36), [
        "训练目标怎么算出来的（本次汇报换掉的就是这一段）",
    ], size=13.5, bold=True, space_after=3)
    code(slide, MARGIN, 3.86, BODY_W, 1.30, [
        "log_depth = torch.log(depth.clamp(min=1e-6))",
        "dmin = torch.quantile(log_depth[valid], 0.02)   # 每一帧自己的分位数",
        "dmax = torch.quantile(log_depth[valid], 0.98)",
        "depth_norm = ((log_depth - dmin) / (dmax - dmin) - 0.5) * 2.0",
    ], size=10.5)
    write(textbox(slide, MARGIN, 5.30, BODY_W, 1.0), [
        "两个权重在基础配方里固定为 1，本次汇报的对照没有动过它们；两条臂的差别只有上面第二段里"
        "取 log 这一下。绝对尺度就是在这一步被逐帧分位数除掉的 —— 所以模型输出没有单位是设计使然，不是缺陷。",
    ], size=12.5, space_after=6)
    conclusion(slide, "损失没变，变的是送进损失的那个目标处在哪个空间")


def arch_inference(prs):
    slide = new_slide(prs)
    header(slide, "推理", "网络出的是 [-1, 1]，米是评估器逐帧贴上去的")
    picture = FIGURES / "arch_inference.png"
    if picture.is_file():
        fit_height(slide, picture, top=1.46, height=4.62)
    write(textbox(slide, MARGIN, 6.20, BODY_W, 0.40), [
        "推理时整个网络都冻结，一次前向，没有去噪循环。虚线以下不属于模型。",
    ], size=12.5, space_after=4)
    conclusion(slide, "模型不产出米；那两个数每一帧重新解")


# ── 6. 训练目标为什么是稠密的 ──────────────────────────────────────────────
def target(prs):
    slide = new_slide(prs)
    header(slide, "训练目标", "官方 GT 覆盖约四分之一，训练用的那张是补出来的")
    picture = FIGURES / "slide_gt_density.png"
    if picture.is_file():
        slide.shapes.add_picture(str(picture), Inches(MARGIN), Inches(1.60),
                                 width=Inches(BODY_W))
    write(textbox(slide, MARGIN, 4.32, BODY_W, 1.9), [
        "整帧覆盖 26.9%，但激光集中在路面：最密的一块有 76.6%，天空一个点都没有。"
        "所以官方 GT 看着像稠密的，在有结构的地方它确实接近稠密。",
        ("训练用的是右上那张：伪深度铺满，再把真实激光盖上去。两者是不同的东西。",
         {"colour": GREY, "size": 12}),
    ], size=13, space_after=7)
    conclusion(slide, "放大到像素级，稀疏与补全的差别一眼可见")


def build(path: Path) -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    for page in (cover, objective, seeds, caption_levels, against_anythermal,
                 chain, target,
                 arch_backbone, arch_loss, arch_inference):
        page(prs)
    path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(path)
    print(f"{len(prs.slides.__iter__.__self__._sldIdLst)} 页 -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("docs/data/汇报_20260921.pptx"))
    build(parser.parse_args().output)


if __name__ == "__main__":
    main()
