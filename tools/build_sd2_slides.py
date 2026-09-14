"""Build the SD2-start deck. Every number is transcribed from the eval JSONs.

Layout vocabulary is imported from build_report8_slides so this reads as the
next deck in the same series rather than a new house style.

The spine, in the order a supervisor needs it:
    what we had been doing wrong -> what the corrected run shows ->
    what it does not show -> why the objective makes both of those expected.

Three things are deliberately absent. Per-frame win rates, because a rate over
frames says nothing about whether the effect survives another training run.
Difference columns, because the checkpoint-selection protocol cannot support
them and a printed difference gets read anyway. Stratified breakdowns, because
the overall three metrics have not improved and a split would be shopping.

    python tools/build_sd2_slides.py --output docs/data/SD2_report.pptx
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from build_report8_slides import (
    BAD, BAND, FONT, GOOD, GREY, ICE, INK, RULE, WHITE, cell_text,
    BODY_W, MARGIN, W, H,
    conclusion, footnote, header, placeholder, table, textbox, write,
)

RED = RGBColor(0xC0, 0x00, 0x00)
DATE = "2026-09-15"

COLS = [1.6, 1, 1, 1, 1, 1, 1]
ALIGN = [PP_ALIGN.LEFT] + [PP_ALIGN.CENTER] * 6


def new(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def box(slide, left, top, width, height, text, *, fill=BAND, line=RULE,
        size=11, colour=INK, bold=False, align=PP_ALIGN.CENTER, anchor=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left),
                                   Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(1.1)
    shape.shadow.inherit = False
    frame = shape.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor or MSO_ANCHOR.MIDDLE
    frame.margin_left = frame.margin_right = Inches(0.16)
    frame.margin_top = frame.margin_bottom = Inches(0.10)
    lines = text if isinstance(text, list) else [text]
    for index, item in enumerate(lines):
        body, over = item if isinstance(item, tuple) else (item, {})
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        para.alignment = over.get("align", align)
        para.space_after = Pt(over.get("space_after", 3))
        run = para.add_run()
        run.text = body
        run.font.name = FONT
        run.font.size = Pt(over.get("size", size))
        run.font.bold = over.get("bold", bold)
        run.font.color.rgb = over.get("colour", colour)
    return shape


def arrow(slide, x0, y, x1):
    shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x0), Inches(y - 0.07),
                                   Inches(x1 - x0), Inches(0.14))
    shape.fill.solid()
    shape.fill.fore_color.rgb = RGBColor(0x9A, 0xA6, 0xB8)
    shape.line.fill.background()
    shape.shadow.inherit = False


def result_table(slide, left_label, right_label, rows, left, top, width, height,
                 *, size=12.5):
    """A six-column metric table under two merged condition headers.

    The two headers name the two factors separately -- whether the arm was
    trained with captions, and what text it is fed at inference -- because the
    whole design turns on those being independent, and a single run-on label
    ("caption-trained, fed the real caption") hides that.

    Every column keeps the precision the evaluator reports. Shortening them was
    tried and reverted twice: two decimals print the RMSE pair 3.605 / 3.598
    identically, so one of two equal-looking cells carried the red mark, and
    four decimals turn 0.07996 into 0.0800, which reads as eight point zero when
    the value is below it. The assertion below catches the first kind; the
    second has no automatic guard, which is the reason for this paragraph.
    """
    n = len(rows) + 2
    shape = slide.shapes.add_table(n, 7, Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    tbl = shape.table
    tbl.first_row = False
    tbl.horz_banding = False
    for index, fraction in enumerate(COLS):
        tbl.columns[index].width = Inches(width * fraction / sum(COLS))

    tbl.cell(0, 0).merge(tbl.cell(1, 0))
    tbl.cell(0, 1).merge(tbl.cell(0, 3))
    tbl.cell(0, 4).merge(tbl.cell(0, 6))
    for cell, text in ((tbl.cell(0, 0), "场景"),
                       (tbl.cell(0, 1), left_label),
                       (tbl.cell(0, 4), right_label)):
        cell.fill.solid()
        cell.fill.fore_color.rgb = INK
        cell_text(cell, text, size=size, bold=True, colour=WHITE,
                  align=PP_ALIGN.CENTER)
    for col, name in enumerate(["AbsRel ↓", "RMSE ↓", "δ1 ↑"] * 2, start=1):
        cell = tbl.cell(1, col)
        cell.fill.solid()
        cell.fill.fore_color.rgb = BAND
        cell_text(cell, name, size=size - 1.5, bold=True, align=PP_ALIGN.CENTER)

    for r, (label, group, lv, rv) in enumerate(rows, start=2):
        band = BAND if group else WHITE
        cell = tbl.cell(r, 0)
        cell.fill.solid(); cell.fill.fore_color.rgb = band
        cell_text(cell, label, size=size, bold=False, align=PP_ALIGN.LEFT)
        col = 1
        for side, other in ((lv, rv), (rv, lv)):
            for i, fmt in enumerate(("{:.5f}", "{:.3f}", "{:.4f}")):
                better = side[i] > other[i] if i == 2 else side[i] < other[i]
                assert fmt.format(side[i]) != fmt.format(other[i]), (
                    f"{label}: {side[i]} and {other[i]} both print as "
                    f"{fmt.format(side[i])}, so the mark would be unreadable")
                cell = tbl.cell(r, col)
                cell.fill.solid(); cell.fill.fore_color.rgb = band
                cell_text(cell, fmt.format(side[i]), size=size,
                          bold=better, colour=RED if better else INK,
                          align=PP_ALIGN.CENTER)
                col += 1
    return tbl


INJ = [("种子 42 · 白天", False, (0.07996, 3.876, 0.9207), (0.08201, 3.941, 0.9158)),
       ("种子 42 · 夜间", False, (0.08240, 3.533, 0.9265), (0.08408, 3.542, 0.9228)),
       ("种子 42 · 雨天", False, (0.10347, 4.533, 0.8796), (0.10697, 4.621, 0.8719)),
       ("种子 43 · 白天", True, (0.08265, 3.844, 0.9183), (0.08538, 3.946, 0.9115)),
       ("种子 43 · 夜间", True, (0.08578, 3.605, 0.9199), (0.08813, 3.679, 0.9135)),
       ("种子 43 · 雨天", True, (0.10850, 4.638, 0.8705), (0.11080, 4.684, 0.8646))]

CONTENT = [("种子 42 · 白天", False, (0.07996, 3.876, 0.9207), (0.08028, 3.892, 0.9199)),
           ("种子 42 · 夜间", False, (0.08240, 3.533, 0.9265), (0.08255, 3.524, 0.9262)),
           ("种子 42 · 雨天", False, (0.10347, 4.533, 0.8796), (0.10402, 4.550, 0.8784)),
           ("种子 43 · 白天", True, (0.08265, 3.844, 0.9183), (0.08337, 3.870, 0.9166)),
           ("种子 43 · 夜间", True, (0.08578, 3.605, 0.9199), (0.08606, 3.616, 0.9191)),
           ("种子 43 · 雨天", True, (0.10850, 4.638, 0.8705), (0.10936, 4.685, 0.8688))]

ARMS8 = [("白天", False, (0.08265, 3.844, 0.9183), (0.08398, 3.952, 0.9143)),
         ("夜间", False, (0.08578, 3.605, 0.9199), (0.08600, 3.598, 0.9191)),
         ("雨天", False, (0.10850, 4.638, 0.8705), (0.11335, 4.778, 0.8580))]

ARMS20 = [("白天", False, (0.08387, 3.950, 0.9208), (0.08259, 3.865, 0.9212)),
          ("夜间", False, (0.09071, 3.968, 0.9179), (0.08840, 3.798, 0.9205)),
          ("雨天", False, (0.10652, 4.666, 0.8793), (0.10739, 4.632, 0.8760))]


def cover(prs):
    slide = new(prs)
    back = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(W), Inches(H))
    back.fill.solid()
    back.fill.fore_color.rgb = INK
    back.line.fill.background()
    back.shadow.inherit = False
    write(textbox(slide, MARGIN, 2.20, 11.6, 0.4), [f"热像深度估计 · 进展汇报　{DATE}"],
          size=13, colour=ICE, bold=True)
    write(textbox(slide, MARGIN, 2.70, 11.8, 1.6),
          ["改从 SD2 起训之后，", "caption 第一次跨种子稳定生效"],
          size=34, colour=WHITE, bold=True, space_after=8)
    write(textbox(slide, MARGIN, 4.60, 11.8, 1.4),
          ["· 同一模型喂真实 caption 优于喂空 caption：两个种子 × 三个场景 × 三个指标，十八格同向",
           "· 语义内容只占其中约两成，其余来自「有文本」这件事本身",
           "· 绝对精度未提升，横向仍不及监督基线"],
          size=13.5, colour=ICE, space_after=8)


def lineage(prs):
    slide = new(prs)
    header(slide, "问题定位", "我们一直接在 Lotus 的权重上做，而 Iris 是从 SD2 重训")
    for top, mid_text, mid_fill, mid_line, tail in (
            (2.05, ["Lotus 训练：合成数据", "全程空文本"], BAND, RULE, "我们（之前）"),
            (3.25, ["Iris 训练：同一配方", "＋ caption"], RGBColor(0xDF, 0xEC, 0xE4), GOOD, None)):
        box(slide, MARGIN, top, 1.95, 0.62, "SD 2 基础模型", fill=WHITE, bold=True)
        arrow(slide, MARGIN + 2.05, top + 0.31, MARGIN + 2.75)
        box(slide, MARGIN + 2.80, top, 3.40, 0.62, mid_text, size=10.5,
            fill=mid_fill, line=mid_line)
        arrow(slide, MARGIN + 6.30, top + 0.31, MARGIN + 7.00)
        box(slide, MARGIN + 7.05, top, 2.45, 0.62,
            "Lotus-G 权重" if tail else "Iris 的模型", fill=WHITE, bold=True)
        if tail:
            arrow(slide, MARGIN + 9.60, top + 0.31, MARGIN + 10.30)
            box(slide, MARGIN + 10.35, top, 1.75, 0.62, tail, size=10.5,
                fill=RGBColor(0xF6, 0xE0, 0xDC), line=BAD)
        else:
            write(textbox(slide, MARGIN + 9.65, top + 0.08, 2.4, 0.6),
                  ["Iris 论文的做法", "也是我们这次的做法"], size=10.5,
                  colour=GOOD, bold=True)
    write(textbox(slide, MARGIN, 4.40, BODY_W, 1.6),
          ["Iris 的脚本里模型路径指向 SD2，输出目录却叫 train-lotus-g —— 那是「用 Lotus-G 这个配方」的意思，"
           "不是「用 Lotus-G 这份权重」。同名造成了之前的误接。",
           "他们必须从 SD2 重训：否则实验臂比对照臂多训了一整轮，增益里有多少来自文本就说不清。"],
          size=12.5, colour=GREY, space_after=9)
    conclusion(slide, "之前的 caption 实验都建立在一个多训过一轮的起点上；这一轮把起点改回 SD2，其余配方逐项不变。")


def setup(prs):
    slide = new(prs)
    header(slide, "实验设置", "只改起点，其余逐项不动")
    table(slide, [["项目", "本轮设置", "与 Iris 的关系"],
                  ["起点权重", "SD 2.1-base（社区镜像，哈希已验）", "Iris 用 2.0-base ← 唯一偏差"],
                  ["训练配方", "conv_in 4→8、任务嵌入、双分支、t=999 单步", "逐项一致"],
                  ["目标表示", "trunc_disparity（逐帧分位数归一化）", "一致"],
                  ["优化器 / 学习率", "8-bit Adam，3e-05 常数，20000 步", "一致"],
                  ["有效 batch", "12", "Iris 为 36"],
                  ["训练数据", "MS2 热像 75,688 帧 ＋ 校准伪深度（激光覆写）", "Iris 为合成 RGB"],
                  ["种子", "42 与 43", "复现用"]],
          MARGIN, 1.95, BODY_W, 3.5, [2.2, 5.0, 3.4], size=12,
          align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT])
    footnote(slide, "评估：官方划分的测试段、全帧、逐帧两参数对齐；checkpoint 由验证集机械选点（AbsRel 最小）。")
    conclusion(slide, "两条臂只差一件事：训练时是否喂 caption。起点、数据、超参、随机种子全部相同。")


def injection(prs):
    slide = new(prs)
    header(slide, "结果 · 主证据", "同一份权重，只换推理时喂进去的文本")
    result_table(slide, "训练：带 caption　·　推理：真实 caption",
                 "训练：带 caption　·　推理：空 caption",
                 INJ, MARGIN, 1.92, BODY_W, 3.45)
    write(textbox(slide, MARGIN, 5.55, BODY_W, 0.5),
          ["十八格全部偏真实 caption。此比较不涉及第二次训练，因此不含训练随机性。"],
          size=13.5, colour=INK, bold=True)
    footnote(slide, "此前在 Lotus-G 起点上的六条实验线，这一比较全部是相反方向。每行较好的值标红。")
    conclusion(slide, "文本条件在 SD2 起点上由有害转为有益，且在两个种子上一致。")


def content(prs):
    slide = new(prs)
    header(slide, "结果 · 控制实验", "把 caption 与图像的配对打乱，还剩多少？")
    result_table(slide, "训练：带 caption　·　推理：真实 caption",
                 "训练：带 caption　·　推理：打乱 caption",
                 CONTENT, MARGIN, 1.92, BODY_W, 3.45)
    write(textbox(slide, MARGIN, 5.55, BODY_W, 0.5),
          ["方向仍然一致（十八格中十七格），但打乱后仍保留约八成收益 —— 语义内容约占两成。"],
          size=13.5, colour=INK, bold=True)
    footnote(slide, "打乱 ＝ 把每帧的描述换成另一帧的描述；文本长度与分布不变，只破坏与图像的对应关系。")
    conclusion(slide, "模型主要响应「有一段文本在场」，而不是文本说了什么。")


def arms(prs):
    slide = new(prs)
    header(slide, "结果 · 未能确立的一项", "带 caption 训练 vs 不带：读不出方向")
    for left, label, rows in ((MARGIN, "两臂都取 8000 步", ARMS8),
                              (MARGIN + 6.25, "两臂都取 20000 步", ARMS20)):
        write(textbox(slide, left, 1.78, 5.85, 0.26), [label], size=12.5, colour=INK, bold=True)
        result_table(slide, "训练带 caption　推理真实",
                     "训练无 caption　推理空",
                     rows, left, 2.08, 5.85, 2.10, size=10.5)
    write(textbox(slide, MARGIN, 4.35, BODY_W, 1.7),
          ["同一对训练，只换取用的 checkpoint，结论完全相反。",
           "单条臂自己在不同 checkpoint 之间的跳动最大 0.0060，而两臂之间的差只有 0.001–0.003 ——"
           "噪声大于信号，任何单一步数上的比较都不可靠。"],
          size=12.5, colour=GREY, space_after=9)
    conclusion(slide, "这一项不作为主张。注入对照之所以站得住，正因为它不跨训练运行。")


def objective(prs):
    slide = new(prs)
    header(slide, "方法 · 目标函数", "两个分支各自在监督什么")
    top = 1.84
    xs = [MARGIN, MARGIN + 2.05, MARGIN + 3.95, MARGIN + 6.80]
    widths = [1.80, 1.65, 2.60, 2.00]
    labels = [["深度图（米）"], ["取倒数 → 视差"],
              ["逐帧 2%/98% 分位数", "归一化到 [-1, 1]"], ["VAE 编码", "→ 目标 latent"]]
    for x, width, label in zip(xs, widths, labels):
        box(slide, x, top, width, 0.66, label, size=11, fill=WHITE)
    for index in range(3):
        arrow(slide, xs[index] + widths[index] + 0.06, top + 0.33, xs[index + 1] - 0.06)
    write(textbox(slide, MARGIN + 3.95, top + 0.70, 2.60, 0.28),
          ["↑ 绝对尺度在这一步丢失"], size=10.5, colour=BAD, bold=True)
    write(textbox(slide, MARGIN + 9.05, top + 0.12, 3.1, 0.5),
          ["深度分支的目标", "（重建分支的目标是热像自身）"], size=10.5, colour=GREY)

    body = 2.92
    box(slide, MARGIN, body, 5.85, 2.55,
        [("L_dense　深度分支", {"size": 14, "bold": True, "colour": INK, "space_after": 7}),
         ("输入　[热像 latent ‖ 加噪 latent]　任务开关 [1,0]　文本＝caption", {"size": 11.5}),
         ("目标　该帧归一化视差图的 VAE latent", {"size": 11.5}),
         ("损失　两者在 latent 空间的 MSE", {"size": 11.5, "space_after": 7}),
         ("教模型：从热像推断相对深度。这是唯一在学正事的项。",
          {"size": 11.5, "bold": True, "colour": GOOD, "space_after": 7}),
         ("掩码取伪深度有效区 ∪ 天空，8×8 池化到 latent 分辨率；实测覆盖 100%，"
          "因此等价于全 1。", {"size": 10.5, "colour": GREY})],
        fill=WHITE, line=GOOD, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)

    box(slide, MARGIN + 6.25, body, 5.85, 2.55,
        [("L_recon　重建分支", {"size": 14, "bold": True, "colour": INK, "space_after": 7}),
         ("输入　同一张热像 latent　任务开关 [0,1]　文本＝空串", {"size": 11.5}),
         ("目标　热像自身的 VAE latent", {"size": 11.5}),
         ("损失　同样是 latent 空间的 MSE，掩码恒为全 1", {"size": 11.5, "space_after": 7}),
         ("教模型：学深度时不要毁掉输入的细结构（Lotus 的「细节保持器」）。",
          {"size": 11.5, "bold": True, "colour": GOOD, "space_after": 7}),
         ("但输入的前四个通道已经就是热像 latent，这接近恒等映射：损失从第 20 步的 0.53 "
          "掉到第 1300 步的 0.001，约为深度分支的百分之一，此后几乎不提供梯度。",
          {"size": 10.5, "colour": GREY})],
        fill=WHITE, line=RULE, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP)

    write(textbox(slide, MARGIN, 5.62, BODY_W, 0.5),
          ["两项都只比较 latent：目标函数里没有任何一个量的单位是米，也没有任何一项引用 caption。"],
          size=13, colour=BAD, bold=True)
    conclusion(slide, "文本只作为条件输入进入 cross-attention，没有梯度要求模型去读它 —— "
                      "Iris 的目标函数与此逐字相同。")



def limits(prs):
    slide = new(prs)
    header(slide, "限制与下一步", "能说什么、不能说什么")
    write(textbox(slide, MARGIN, 1.85, 5.85, 3.3),
          [("可以主张", {"size": 15, "bold": True, "colour": GOOD}),
           "· 文本条件在 SD2 起点上有益：两个种子、三个场景、三个指标一致",
           "· 该比较使用同一份权重，不受训练随机性影响",
           "· 此前六条实验线在这一比较上全部相反，方向性改变是新的",
           "· 语义内容约贡献两成，可定量陈述"],
          size=12.5, colour=INK, space_after=10)
    write(textbox(slide, MARGIN + 6.25, 1.85, 5.85, 3.3),
          [("不能主张", {"size": 15, "bold": True, "colour": BAD}),
           "· 「模型变好了」—— 绝对精度比原路线退约 0.006，横向不及 NeWCRF、BTS",
           "· 「语言的几何信息在起作用」—— 打乱后仍保留八成",
           "· 「带 caption 训练的模型更好」—— 两臂之差小于 checkpoint 噪声",
           "· 米制 RMSE 的竞争力 —— 目标函数不含米，这是路线属性而非缺陷"],
          size=12.5, colour=INK, space_after=10)
    placeholder(slide, MARGIN, 5.25, BODY_W, 1.0,
                [("需要定的方向", {"size": 13, "bold": True, "colour": INK}),
                 "「让 caption 起效」与「把绝对指标做上去」目前落在两个不同的配置里，且有迹象表明二者互相牵制。"
                 "是继续投入追绝对指标，还是就现有结果整理成以机制为主的文章？"])


def build(path: Path) -> None:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)
    for page in (cover, lineage, setup, injection, content, arms, objective, limits):
        page(prs)
    path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(path)
    print(f"{len(prs.slides._sldIdLst)} 页 -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/data/SD2_report.pptx"))
    build(parser.parse_args().output)


if __name__ == "__main__":
    main()
