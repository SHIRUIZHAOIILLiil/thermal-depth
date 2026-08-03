"""Build the report-8 deck from docs/AIRE_RESULTS_20260802.md.

Every number here is transcribed from that document, which is the single source
of truth for the Aire run; nothing is recomputed and nothing is rounded further.
The deck exists as a script rather than as a hand-edited pptx because report 6's
build script was not kept, and the numbers had to be re-entered by hand for
report 7.

Storyline (the supervisor-facing spine, in this order):
    six-route baseline -> caption's two opposite effects -> d2's far-field
    advantage under a controlled attribution -> the external reference point
    (original AnyThermal) -> limits.
Rain and gradient matching are supplements, not spine: the first is a condition
boundary on the caption finding, the second is a clean negative result.

Two slides carry placeholders on purpose -- the zero-shot direct run and the
qualitative comparison are cluster jobs that had not returned when the deck was
built (docs §11).  They are marked in the deck itself, not silently omitted, so
an unfilled box is visible rather than forgotten.

    python tools/build_report8_slides.py --output "E:/.../8-thermal_depth_report_8.pptx"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

INK = RGBColor(0x1F, 0x2A, 0x44)      # deep charcoal blue -- titles, dark bars
ICE = RGBColor(0xCA, 0xDC, 0xFC)      # ice blue -- kickers on dark, accents
GREY = RGBColor(0x5A, 0x66, 0x7A)     # muted -- kickers on light, footnotes
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GOOD = RGBColor(0x1E, 0x6B, 0x52)     # an improvement
BAD = RGBColor(0xA6, 0x39, 0x2E)      # a regression
RULE = RGBColor(0xD5, 0xDB, 0xE4)
BAND = RGBColor(0xF2, 0xF5, 0xF9)     # table header / zebra fill

FONT = "微软雅黑"
DATE = "2026-08-03"

W, H = 13.333, 7.5
MARGIN, BODY_W = 0.60, 12.13
BAR_TOP, BAR_H = 6.62, 0.66


# ── primitives ──────────────────────────────────────────────────────────────

def textbox(slide, left, top, width, height):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    return frame


def write(frame, lines, *, size, colour=INK, bold=False, space_after=4, line_spacing=None):
    """`lines` is a list of str, or of (text, {overrides}) pairs."""
    for index, line in enumerate(lines):
        text, over = line if isinstance(line, tuple) else (line, {})
        para = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        para.space_after = Pt(over.get("space_after", space_after))
        if line_spacing:
            para.line_spacing = line_spacing
        run = para.add_run()
        run.text = text
        run.font.name = FONT
        run.font.size = Pt(over.get("size", size))
        run.font.bold = over.get("bold", bold)
        run.font.color.rgb = over.get("colour", colour)
    return frame


def header(slide, kicker, title):
    write(textbox(slide, MARGIN, 0.38, 12.00, 0.30), [kicker], size=11, colour=GREY, bold=True)
    write(textbox(slide, MARGIN, 0.68, 12.10, 0.72), [title], size=26, colour=INK, bold=True)


def conclusion(slide, text):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(MARGIN), Inches(BAR_TOP),
                                 Inches(BODY_W), Inches(BAR_H))
    bar.fill.solid()
    bar.fill.fore_color.rgb = INK
    bar.line.fill.background()
    bar.shadow.inherit = False
    frame = bar.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame.margin_left = Inches(0.18)
    frame.margin_right = Inches(0.18)
    para = frame.paragraphs[0]
    prefix = para.add_run()
    prefix.text = "结论   "
    prefix.font.name, prefix.font.size, prefix.font.bold = FONT, Pt(12), True
    prefix.font.color.rgb = ICE
    body = para.add_run()
    body.text = text
    body.font.name, body.font.size, body.font.bold = FONT, Pt(12), False
    body.font.color.rgb = WHITE


def footnote(slide, text, top=6.24):
    write(textbox(slide, MARGIN, top, BODY_W, 0.32), [text], size=10.5, colour=GREY)


def cell_text(cell, text, *, size, bold=False, colour=INK, align=PP_ALIGN.LEFT):
    frame = cell.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame.margin_left = frame.margin_right = Inches(0.06)
    frame.margin_top = frame.margin_bottom = Inches(0.02)
    para = frame.paragraphs[0]
    para.alignment = align
    run = para.add_run()
    run.text = text
    run.font.name, run.font.size, run.font.bold = FONT, Pt(size), bold
    run.font.color.rgb = colour


def table(slide, rows, left, top, width, height, col_widths, *, size=11.5,
          align=None, highlight=()):
    """`rows[0]` is the header. A cell may be a str or (str, {overrides})."""
    shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    tbl = shape.table
    tbl.first_row = True
    tbl.horz_banding = False
    for index, fraction in enumerate(col_widths):
        tbl.columns[index].width = Inches(width * fraction / sum(col_widths))
    for r, row in enumerate(rows):
        for c, raw in enumerate(row):
            text, over = raw if isinstance(raw, tuple) else (raw, {})
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = BAND if (r == 0 or r in highlight) else WHITE
            cell_text(cell, text,
                      size=over.get("size", size),
                      bold=over.get("bold", r == 0 or r in highlight),
                      colour=over.get("colour", INK if r == 0 else INK),
                      align=over.get("align",
                                     (align[c] if align else PP_ALIGN.LEFT) if c else PP_ALIGN.LEFT))
    return tbl


def placeholder(slide, left, top, width, height, lines):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
                                 Inches(width), Inches(height))
    box.fill.solid()
    box.fill.fore_color.rgb = BAND
    box.line.color.rgb = RULE
    box.line.width = Pt(1.25)
    box.shadow.inherit = False
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(0.24)
    frame.margin_top = Inches(0.20)
    write(frame, lines, size=12, colour=GREY)


def picture_or_placeholder(slide, image: Path | None, left, top, width, height, lines):
    if image and image.is_file():
        slide.shapes.add_picture(str(image), Inches(left), Inches(top), width=Inches(width))
    else:
        placeholder(slide, left, top, width, height, lines)


# ── slides ──────────────────────────────────────────────────────────────────

def cover(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    back = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    back.fill.solid()
    back.fill.fore_color.rgb = INK
    back.line.fill.background()
    back.shadow.inherit = False
    write(textbox(slide, 1.00, 2.25, 11.30, 0.40),
          ["THERMAL DEPTH · PROGRESS REPORT 8"], size=13, colour=ICE, bold=True)
    write(textbox(slide, 1.00, 2.75, 11.30, 1.20),
          ["caption 的两个相反效应，与远端误差的归因"], size=40, colour=WHITE, bold=True)
    write(textbox(slide, 1.00, 4.05, 11.30, 1.50), [
        "六条路线 × 20 epoch，全部在利兹 Aire 超算完成并在独立 test 集出数",
        "MS2 官方 BMSD 协议 · test = 16-08-46，2,543 帧，训练与调参从未接触",
    ], size=15, colour=ICE, space_after=6)
    write(textbox(slide, 1.00, 6.30, 11.30, 0.40), [DATE], size=13, colour=GREY)


def overview(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, "WHAT THIS ROUND ESTABLISHED", "本轮拿到的三件事")
    rows = [
        ["", "发现", "证据强度"],
        ["1",
         "caption 的作用是两个方向相反的效应：训练时用它有益，推理时喂它有害。"
         "全项目最好的热像成绩来自「用 caption 训练、推理给空 prompt」＝ 0.0869。",
         "注入效应：同权重同帧，仅 prompt 不同，无懈可击\n权重效应：单 seed"],
        ["2",
         "远端误差可以归因到 condition 的来源，而不是 adapter 架构：\n"
         "c2 与 d2 结构完全相同，只换 condition，远端 0.1817 → 0.1524（−16%）。",
         "受控对照 + donor-swap 证伪（误差翻 3.3 倍）"],
        ["3",
         "远端退化不是热像深度的通病：原版 AnyThermal 用同一份稀疏 LiDAR 监督，"
         "远端几乎不退化（×1.13 vs 我方 ×1.90）。差别在架构，不在监督。",
         "外部模型，同一考卷，同一分层定义"],
    ]
    table(slide, rows, MARGIN, 1.70, BODY_W, 3.90, [0.4, 7.2, 3.6], size=12)
    write(textbox(slide, MARGIN, 5.72, BODY_W, 0.80), [
        "以往把 caption 当成一个整体来评，结论总是「不显著」——"
        "两个方向相反的效应叠在一起，正好互相抵消。这解释了此前为什么总测不出效果。",
    ], size=12.5, colour=GREY)
    conclusion(slide, "六条线全部完成 20 epoch 并在独立 test 集出数；三件事各有可检验的证据，强度不同，逐条标注。")


def protocol(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, "PROTOCOL & GOVERNANCE", "口径：所有数字共用的那一套")
    rows = [
        ["项", "值"],
        ["val（选 epoch 用）", "11-23-45，5,810 帧"],
        ["test（对外数字）", "16-08-46，2,543 帧，官方 val split、白天，训练与调参从未接触"],
        ["协议", "official BMSD ssi_disparity，min_depth 1e-3，max_depth 80"],
        ["对齐", "六线一律 ssi_disparity；原版 AnyThermal 用 ssi（官方表）"],
        ["训练 manifest", "ms2_train_day2seq_clip75_20260728.jsonl，19,949 帧"],
        ["环境", "torch 2.3.1+cu121 / L40S 48GB"],
    ]
    table(slide, rows, MARGIN, 1.66, BODY_W, 2.55, [2.6, 9.5], size=12)

    write(textbox(slide, MARGIN, 4.42, 5.90, 1.70), [
        ("⚠️ 两族模型的对齐方式不可混用", {"bold": True, "colour": BAD}),
        "把六线的预测按 ssi 对齐会得到 0.37（正确值 0.0884）；",
        "把 AnyThermal 按 ssi_disparity 对齐会得到 0.2724（正确值 0.0821）。",
        "所以不同对齐的模型必须分开跑、并排读。",
    ], size=12, colour=INK, space_after=3)

    write(textbox(slide, 6.85, 4.42, 5.88, 1.70), [
        ("✅ 跨机器标定已通过", {"bold": True, "colour": GOOD}),
        "集群 b(e05) 全量 val 0.07749105 vs 本地 0.0775，",
        "差 0.000009（阈值 0.0005）。",
        "本地与集群结果可以并进同一张表。",
    ], size=12, colour=INK, space_after=3)

    conclusion(slide, "治理约束：11-23-45 只用来选 epoch，16-08-46 只在定稿时评一次；本报告所有对外数字都带 test 列。")


def main_table(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, "BASELINE · SIX ROUTES ON THE HELD-OUT TEST SET", "主表：六条路线的 val 与 test")
    rows = [
        ["线", "输入", "Condition", "可训练参数", "val", "test", "rmse", "a1"],
        ["a  RGB+U-Net", "RGB", "冻结 VAE latent", "867.57 M", "0.0791", ("0.0844", {"bold": True}), "3.827", "0.9269"],
        ["b  Thermal+U-Net", "Thermal", "冻结 VAE latent", "867.57 M", ("0.0775", {"bold": True}), "0.0884", "3.956", "0.9071"],
        ["c1  VAE-adapter", "Thermal", "VAE latent + Adapter", "7.11 M", "0.1127", "0.1217", "5.465", "0.8365"],
        ["c2  VAE-adapter+U-Net", "Thermal", "VAE latent + Adapter", "874.67 M", "0.0932", "0.1013", "4.418", "0.8826"],
        ["d1  AnyTh-adapter", "Thermal", "AnyThermal 特征 + Adapter", "9.41 M", "0.0860", "0.0973", "4.432", "0.8883"],
        ["d2  AnyTh-adapter+U-Net", "Thermal", "AnyThermal 特征 + Adapter", "876.98 M", "0.0815", "0.0903", "3.855", "0.9086"],
        [("零训练直推", {"colour": GREY}), ("Thermal", {"colour": GREY}),
         ("冻结 VAE latent（U-Net 也不训）", {"colour": GREY}), ("0", {"colour": GREY}),
         ("0.1291", {"colour": GREY}), ("待填", {"colour": BAD, "bold": True}),
         ("待填", {"colour": GREY}), ("待填", {"colour": GREY})],
    ]
    align = [PP_ALIGN.LEFT] + [PP_ALIGN.LEFT] * 2 + [PP_ALIGN.RIGHT] * 5
    table(slide, rows, MARGIN, 1.62, BODY_W, 3.10, [2.5, 1.0, 2.9, 1.5, 1.0, 1.0, 1.0, 1.0],
          size=11.5, align=align)

    write(textbox(slide, MARGIN, 4.92, BODY_W, 1.30), [
        "· 热像几乎追平 RGB：a 0.0844 vs b 0.0884，仅差 4.5%。但 b 的跨序列泛化更差 —— "
        "val→test 退化 +0.0109，a 只有 +0.0053。",
        "· d 系全面压过 c 系（d1 0.0973 vs c1 0.1217；d2 0.0903 vs c2 0.1013）："
        "AnyThermal 特征作为 condition 优于 VAE latent，在两个参数量级上都成立。",
        "· d1 用 9.41 M（1%）参数达到 0.0973，逼近 867 M 的 b。",
    ], size=12, colour=INK, space_after=4)
    footnote(slide, "零训练直推 = b 线但 U-Net 也不训，是「训 U-Net 买来了什么」的受控对照；"
                    "val 0.1291 为历史值，test 待集群作业回填。")
    conclusion(slide, "六线的排序稳定：condition 用什么，比在 condition 上挂多少参数更要紧。")


def caption_effects(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, "★ CAPTION · TWO OPPOSITE EFFECTS", "caption 拆成两个效应后，两个都显著")
    write(textbox(slide, MARGIN, 1.48, BODY_W, 0.34), [
        "同一条线的三个考法（test 2,543 帧，AbsRel，越小越好）：",
    ], size=12, colour=GREY)
    rows = [
        ["线", "无 cap 训练", "cap 训练 / 空 prompt", "cap 训练 / 真 caption"],
        ["b  Thermal+U-Net", "0.0884", ("0.0869", {"bold": True, "colour": GOOD}), "0.0882"],
        ["c2  VAE-adapter+U-Net", "0.1013", "0.1030", "0.1022"],
        ["d2  AnyTh-adapter+U-Net", "0.0903", ("0.0876", {"bold": True, "colour": GOOD}), "0.0881"],
    ]
    align = [PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT] * 3
    table(slide, rows, MARGIN, 1.86, BODY_W, 1.20, [3.2, 2.6, 3.2, 3.2], size=12, align=align)

    write(textbox(slide, MARGIN, 3.30, BODY_W, 0.34), [
        "拆成两个效应（负 = 更好；n=2,543 配对，bootstrap 3000 次）：",
    ], size=12, colour=GREY)
    rows = [
        ["线", "① 权重效应（训练时用 caption）", "② 注入效应（推理时喂 caption）"],
        ["b", ("−0.00153  [−0.00188, −0.00118]  显著", {"colour": GOOD}),
         ("+0.00130  [+0.00110, +0.00152]  显著", {"colour": BAD})],
        ["c2", ("+0.00169  [+0.00146, +0.00194]  显著", {"colour": BAD}),
         ("−0.00082  [−0.00104, −0.00062]  显著", {"colour": GOOD})],
        ["d2", ("−0.00270  [−0.00332, −0.00206]  显著", {"colour": GOOD}),
         ("+0.00046  [+0.00037, +0.00056]  显著", {"colour": BAD})],
    ]
    table(slide, rows, MARGIN, 3.68, BODY_W, 1.20, [1.6, 5.3, 5.3], size=12,
          align=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT])

    write(textbox(slide, MARGIN, 5.12, BODY_W, 1.10), [
        "· 全项目最好的热像成绩 0.0869 —— b 用 caption 训练、推理时给空 prompt。"
        "caption 的价值在训练期，不在推理期。",
        "· 注入税的大小由 condition 架构决定：b +0.00130，d2 只有 +0.00046，小 3 倍。"
        "语义层级的 condition 不能让注入变得有益，但能把代价降到三分之一。",
        "· c2 两个方向都反着来，与它在所有其他指标上的劣势一致 —— VAE latent 这条路整体是错的。",
    ], size=12, colour=INK, space_after=3)
    conclusion(slide, "以往「caption 不显著」是两个相反效应互相抵消的假象；拆开后两个都显著。")


def caption_strength(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, "★ CAPTION · HOW STRONG IS EACH CLAIM", "两个效应的证据强度不同，必须分开说")
    rows = [
        ["", "② 注入效应", "① 权重效应"],
        ["比的是什么", "同一份权重、同一批帧，只有 prompt 不同",
         "两次独立训练的 checkpoint"],
        ["混淆变量", "没有 —— 除 prompt 外一切固定", "训练随机性（单 seed）"],
        ["能说到多硬", "无懈可击，可以直接下结论",
         "有力但不等于定论，汇报时标注「单 seed」"],
        ["旁证", "—",
         "三条线方向不一致（c2 反向）。若只是「重训一次就变好」的假象，三条应同向"],
    ]
    table(slide, rows, MARGIN, 1.70, BODY_W, 2.60, [1.9, 4.6, 5.6], size=12)

    write(textbox(slide, MARGIN, 4.55, BODY_W, 1.55), [
        ("为什么这一条值得单独一页", {"bold": True, "size": 13}),
        "① 和 ② 是同一张表里读出来的，但它们的可信度差一个量级。把它们并列成"
        "「caption 有两个效应」而不区分强度，是这轮最容易被问倒的地方。",
        "补齐 ① 的办法很明确：d2+caption 跑第二个 seed（约 27 小时），"
        "把单次观测升级成可复现观测。这是下一步的第一优先项。",
    ], size=12, colour=INK, space_after=5)
    conclusion(slide, "② 可以当结论讲；① 只能当「单 seed 下的强观测」讲，第二 seed 是明确的补救路径。")


def far_field(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, "FAR FIELD · A CONTROLLED ATTRIBUTION", "d2 的远端优势，以及它能归因到什么")
    rows = [
        ["分层", "b", "b+cap", "d1", "c2", "d2"],
        ["all", "0.0884", "0.0882", "0.0973", "0.1013", "0.0903"],
        ["depth/far >30m", "0.1680", "0.1679", "0.1726", ("0.1817", {"colour": BAD}),
         ("0.1524", {"bold": True, "colour": GOOD})],
        ["row/top", "0.1572", "0.1598", "0.1767", "0.1858", "0.1705"],
        ["structure/boundary", "0.1075", "0.1089", "0.1184", "0.1242", "0.1112"],
        ["structure/interior", "0.0862", "0.0857", "0.0946", "0.0989", "0.0878"],
        ["锐利度 b/i", "1.25", "1.27", "1.25", "1.26", "1.27"],
        ["远端退化 far/all", "×1.90", "×1.90", "×1.77", "×1.79",
         ("×1.69", {"bold": True, "colour": GOOD})],
    ]
    align = [PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT] * 5
    table(slide, rows, MARGIN, 1.62, 7.55, 3.55, [2.3, 1.0, 1.0, 1.0, 1.0, 1.0],
          size=11.5, align=align)

    write(textbox(slide, 8.40, 1.62, 4.33, 3.60), [
        ("受控归因", {"bold": True, "size": 13}),
        "c2 与 d2 结构完全相同，可训练参数几乎一样，"
        "唯一差别是 condition 来自哪里。",
        ("far  0.1817 → 0.1524，−16%", {"bold": True, "colour": GOOD, "size": 13}),
        "所以远端改善可以归因到 AnyThermal 特征本身，不是 adapter 架构。",
        "",
        "d2 是唯一在远端改善的线（vs b 的 0.1680，−9.3%），退化倍数也最小。"
        "它用一点整体精度换来了远端的实质改善 —— 正是任务关心的区域。",
        "",
        ("任务 1b 的答案：没修好。", {"bold": True, "colour": BAD}),
        "c2（先训 adapter 再训 U-Net）远端 0.1817、row/top 0.1858，都是所有线里最差。",
    ], size=11.5, colour=INK, space_after=4)

    footnote(slide,
             "⚠️ 分层只统计有 GT 的像素。MS2 每帧 valid 平均 26.5%，天空没有 LiDAR 回波 —— "
             "row/top 是「上三分之一里有回波的像素」（楼顶、电线杆），不是天空。措辞不能混。",
             top=5.42)
    conclusion(slide, "远端改善来自 condition 的语义层级，不是参数量、不是 adapter 结构 —— 这是一个受控结论。")


def donor_swap(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, "FALSIFICATION · DOES THE FEATURE BRANCH DO ANYTHING",
           "证伪对照：d2 真的在用 AnyThermal 特征吗")
    write(textbox(slide, MARGIN, 1.48, BODY_W, 0.70), [
        "担心的是：adapter 可能学会忽略 AnyThermal 特征，d2 退化成一个带额外参数的 b。"
        "损失曲线看不出这件事。做法是推理时把 condition 换成随机 donor 帧的（uniform，自配对已换走）。",
    ], size=12.5, colour=INK)
    rows = [
        ["", "abs_rel", "rmse", "a1"],
        ["d2 原始", "0.0903", "3.855", "0.9086"],
        ["d2 只打乱 AnyThermal 特征", ("0.2971", {"bold": True, "colour": BAD}), "10.005",
         ("0.5186", {"bold": True, "colour": BAD})],
        ["d2 打乱整个 condition", "0.2981", "10.034", "0.5171"],
        ["b 打乱整个 condition（标尺）", "0.2945", "9.952", "0.5206"],
    ]
    align = [PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT] * 3
    table(slide, rows, MARGIN, 2.36, 7.30, 1.75, [3.4, 1.4, 1.4, 1.4], size=12, align=align)

    write(textbox(slide, 8.20, 2.36, 4.53, 3.30), [
        ("读法", {"bold": True, "size": 13}),
        "只换特征金字塔，误差翻 3.3 倍、a1 从 0.91 塌到 0.52。",
        "「只打乱特征」与「打乱全部」几乎相同（0.2971 vs 0.2981）——"
        "AnyThermal 特征承载了几乎全部 condition 信号，"
        "adapter 的 thermal 张量输入不独立贡献。",
        "崩塌幅度与 b 打乱 VAE latent 相当（0.2945）。",
    ], size=11.5, colour=INK, space_after=5)

    write(textbox(slide, MARGIN, 4.42, 7.30, 0.60), [
        "anythermal 模式只换特征、保留正确的热像张量，"
        "所以零结果不能推给「adapter 丢了图像」—— 这是该检验的设计关键。",
    ], size=11.5, colour=GREY)

    write(textbox(slide, MARGIN, 5.16, BODY_W, 1.00), [
        ("反过来的一半：adapter 也是必需的", {"bold": True, "size": 12.5}),
        "把 AnyThermal 特征经零参数 bridge 直接塞给冻结的 Lotus-G（不经 adapter、不训练），"
        "test AbsRel = 0.4697 —— 比上表「打乱 condition」的 0.2971 还差。"
        "特征金字塔的格式是冻结 U-Net 从没见过的，「乱但同分布」好过「不同分布」。",
    ], size=11.5, colour=INK, space_after=3)
    conclusion(slide, "两个方向都验了：adapter 在用 AnyThermal 特征（打乱就塌），而少了 adapter 这些特征也用不了。")


def external_reference(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, "TASK 4 · THE EXTERNAL REFERENCE POINT", "原版 AnyThermal：它没有远端问题")
    rows = [
        ["", "all", "far >30m", "row/top", "boundary", "interior", "b/i", "far/all"],
        ["b（我方最佳 baseline）", "0.0884", "0.1680", "0.1572", "0.1075", "0.0862", "1.25",
         ("×1.90", {"colour": BAD, "bold": True})],
        ["原版 AnyThermal", ("0.0821", {"bold": True}), ("0.0930", {"bold": True, "colour": GOOD}),
         "0.1257", "0.0973", "0.0808", "1.20",
         ("×1.13", {"colour": GOOD, "bold": True})],
    ]
    align = [PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT] * 7
    table(slide, rows, MARGIN, 1.62, BODY_W, 1.05, [2.6, 1.2, 1.4, 1.2, 1.3, 1.2, 0.9, 1.2],
          size=11.5, align=align)
    footnote(slide, "AnyThermal 用 ssi 对齐（独立作业），两表分层定义共享，可并排读。"
                    "复现验证：论文 Table IV 报 AbsRel 0.0883（day/night/rainy 平均），我方实测 0.0821（仅白天 test），量级吻合。",
             top=2.76)

    write(textbox(slide, MARGIN, 3.34, BODY_W, 0.34),
          ["它用了什么监督（论文 §III-B / §III-F）："], size=12.5, colour=INK, bold=True)
    rows = [
        ["", "原版 AnyThermal", "我方六线"],
        ["深度监督", "MS2 自己的 train split + 稀疏 LiDAR GT", "同左"],
        ["骨干", "DINOv2 ViT-B/14，CLS token 对比损失蒸馏（自监督、无标注）", "Stable Diffusion U-Net"],
        ["深度头", "MiDaS 架构，「其余部分不变」", "扩散单步 x0 预测"],
        ["训练时骨干", ("冻结，只训头", {"bold": True}), ("整个 U-Net 解冻微调", {"bold": True})],
    ]
    table(slide, rows, MARGIN, 3.72, BODY_W, 1.65, [1.7, 6.6, 3.8], size=11.5)

    write(textbox(slide, MARGIN, 5.55, BODY_W, 0.90), [
        ("机制假设：", {"bold": True}),
        "稀疏 LiDAR 在远端/天空无信号；全解冻的 U-Net 在无监督区自由漂移，"
        "而冻结骨干保住了 DINOv2 的语义先验。d2 部分借到了这个好处（用 AnyThermal 特征），"
        "但下游 U-Net 仍全解冻，所以只到 ×1.69。",
    ], size=11.5, colour=INK, space_after=2)
    conclusion(slide, "同一份稀疏 LiDAR 监督，差别在架构 —— 远端退化是我方方法特有的，不是热像深度的通病。")


def qualitative(prs, figure: Path | None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, "QUALITATIVE · SAME FRAMES, THREE REFERENCE POINTS", "对比可视化：零训练 / 我方 / 原版")
    picture_or_placeholder(
        slide, figure, MARGIN, 1.55, BODY_W, 3.55,
        ["【待填：集群 vis 作业的输出 PNG】",
         "",
         "tools/build_comparison_figure.py（新增），经 slurm/vis.sbatch 在集群上出图。",
         "挑帧按分层结果驱动，不随机：--pick gap --gap d2:b --stratum 'depth/far >30m' "
         "挑 d2 在远端领先 b 最多的帧；--pick flip 挑逐帧排序与全集排序相反的帧。",
         "每列下方标注它自己用的对齐空间 —— 原版 AnyThermal 是 ssi，六线是 ssi_disparity，"
         "混用会得到几倍大的数（见口径页）。"])
    write(textbox(slide, MARGIN, 5.30, BODY_W, 0.90), [
        "两种上色都出：官方显示函数（每格用自己的量程归一化）与共用色标（全行按该帧真值量程）。"
        "前者是仓库的显示助手，不是 BMSD 协议的一部分；它会把预测量程更宽的模型压缩得更狠，"
        "所以两种并排给出，读者不必凭信任接受其中一种。",
    ], size=11.5, colour=GREY)
    conclusion(slide, "图上的每一格都由 raw 预测按各自对齐空间现算，AbsRel 与分层数字同源。")


def rain(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, "SUPPLEMENT · WHEN DOES INJECTION HELP", "补充一：雨天 —— 注入效应的条件边界")
    write(textbox(slide, MARGIN, 1.46, BODY_W, 0.34), [
        "三个考卷，同一份 b+caption 权重，只有 prompt 内容不同（②注入效应口径）。",
    ], size=12, colour=GREY)
    rows = [
        ["考卷", "帧数", "b 的 abs_rel", "注入效应", "胜率", "caption 词表(等n)", "唯一率"],
        ["晴  16-08-46", "2,543", "0.0884", ("+0.00130  显著（有害）", {"colour": BAD}), "—", "810", "92.7%"],
        ["轻雨  16-19-00", "2,374", "0.0930",
         ("−0.00169  显著（有帮助）", {"colour": GOOD, "bold": True}), "60.7%",
         ("831", {"bold": True}), "90.9%"],
        ["重雨  16-59-13", "2,161", "0.1170", ("+0.00049  显著（有害）", {"colour": BAD}), "45.0%",
         ("709", {"bold": True}), "87.2%"],
    ]
    align = [PP_ALIGN.LEFT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.LEFT,
             PP_ALIGN.RIGHT, PP_ALIGN.RIGHT, PP_ALIGN.RIGHT]
    table(slide, rows, MARGIN, 1.84, BODY_W, 1.35, [1.9, 0.9, 1.4, 3.3, 0.9, 1.9, 1.0],
          size=11.5, align=align)

    rows = [
        ["考卷", "视觉通路", "caption 信息量", "结果"],
        ["晴", "强", "高（810）", "文本冗余 → 注入税"],
        ["轻雨", ("被削弱", {"bold": True}), ("最高（831）", {"bold": True}),
         ("补位成功", {"bold": True, "colour": GOOD})],
        ["重雨", "严重削弱", "最低（709）", "文本自己也失效 → 注入税"],
    ]
    table(slide, rows, MARGIN, 3.44, 7.10, 1.35, [1.2, 1.6, 1.8, 3.0], size=11.5)

    write(textbox(slide, 8.00, 3.44, 4.73, 2.60), [
        ("两个条件必须同时满足", {"bold": True, "size": 12.5}),
        "视觉通路被削弱 且 文本仍携带有效信息。",
        "这修正了原先的「文本价值 = 视觉通路有多弱」—— "
        "那个说法在 caption 源与模型输入同模态时看不出区别；"
        "本设置 caption 来自 RGB、模型吃热像，两个变量解耦后才露出真正的驱动因素。",
    ], size=11.5, colour=INK, space_after=4)

    write(textbox(slide, MARGIN, 5.02, 7.10, 1.15), [
        ("未复现的部分：", {"bold": True, "colour": BAD}),
        "d2 上轻雨与重雨的注入效应几乎相同（−0.00059 / −0.00062），尽管词表差 15%。"
        "「效应跟着 caption 信息量走」在 b 上成立、在 d2 上不成立。",
    ], size=11, colour=INK, space_after=2)
    conclusion(slide, "可证伪的下一步：夜间 21-58-13（RGB 近全黑）。预测词表低于重雨、注入税大于 +0.0005。")


def gradient_matching(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, "SUPPLEMENT · A CLEAN NEGATIVE RESULT",
           "补充二：MiDaS 多尺度梯度匹配损失 —— 试了，失败了")
    write(textbox(slide, MARGIN, 1.46, BODY_W, 0.80), [
        "动机：逐点 L1 让「边界抹平」比「锐利但错位」便宜（合成阶跃上 0.0385 vs 0.0729），"
        "而 AbsRel/RMSE 同为逐点平均，奖励同一种投机 —— 这解释了「数值好看但可视化模糊」。"
        "MiDaS 的损失有第二项（梯度匹配），我方只有数据项。",
    ], size=12, colour=INK)
    rows = [
        ["臂", "best epoch", "val abs_rel", "rmse", "a1"],
        ["w=0", "2", ("0.07974", {"bold": True, "colour": GOOD}), "3.464", "0.9256"],
        ["w=5", "3", "0.08539", "3.760", "0.9136"],
        ["w=20", "2", ("0.09284", {"colour": BAD}), "4.277", "0.9011"],
    ]
    align = [PP_ALIGN.LEFT] + [PP_ALIGN.RIGHT] * 4
    table(slide, rows, MARGIN, 2.46, 6.10, 1.35, [1.2, 1.4, 1.6, 1.2, 1.2], size=12, align=align)
    footnote(slide, "三条臂各 3 epoch，单调恶化。", top=3.92)

    write(textbox(slide, 7.05, 2.46, 5.68, 3.40), [
        ("失败机制（有剂量-反应关系、有机制，可写）", {"bold": True, "size": 12.5}),
        "alignment_scale 从 0.26 涨到 3.3 —— 输出动态范围被压缩 12 倍；"
        "clamped_above 从 5 涨到 80。",
        "scale 虽 detach，但出现在损失对预测的梯度里，形成正反馈 —— "
        "w=20 的梯度范数常态触顶 --max-grad-norm 1.0。",
        "",
        ("实现本身是对的：", {"bold": True}),
        "合成阶跃上正确锐利预测两项均为 0，模糊预测梯度项 0.0262；"
        "26.5% 稀疏 mask 下 0.026164 vs 稠密 0.026163 —— "
        "多尺度掩码在 MS2 的 GT 密度下成立。",
    ], size=11.5, colour=INK, space_after=4)

    write(textbox(slide, MARGIN, 4.35, 6.10, 1.70), [
        ("学到的方法论", {"bold": True, "size": 12.5}),
        "权重标定按损失值之比定是错的：合成数据比值 0.68，"
        "真实数据实测 gmatch/gt_ssi_l1 = 0.087，差 8 倍。",
        "应该按梯度量级定权重，不是按损失值。",
    ], size=11.5, colour=INK, space_after=4)
    conclusion(slide, "负面结果，但不是空手：机制清楚、有剂量-反应关系，且给出了一条可复用的方法论。")


def limits(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, "LIMITS", "局限（这些必须写进结论）")
    rows = [
        ["", "局限", "影响到哪条结论"],
        ["1", "GT 覆盖率 26.5%，天空无 LiDAR 回波 —— AbsRel/RMSE/所有分层都测不到天空。"
              "RMSE 逃不出这个问题（同一个 valid mask），它只是单位可读。",
         "所有「天空」措辞。分层里的 row/top 是「上三分之一有回波的像素」，不是天空"],
        ["2", "权重效应（①）单 seed；注入效应（②）无此问题。", "caption 结论的前半句"],
        ["3", "a + caption 未跑（第一周任务 4 缺口，67 小时）。", "RGB 线上的 caption 效应无数据"],
        ["4", "零训练直推、对比可视化尚未回来（第二周任务 3）。", "主表最后一行、可视化页"],
        ["5", "雨天两因素模型在 d2 上未复现。", "「效应跟着 caption 信息量走」只在 b 上成立"],
        ["6", "现有工具无法测量真正的天空区域；需要不依赖 GT 的判据"
              "（如「无回波区域被预测为近处的比例」），该工具尚未实现。",
         "「它没有天空问题」这类表述只能说到「有回波的上三分之一」"],
    ]
    table(slide, rows, MARGIN, 1.66, BODY_W, 4.40, [0.4, 7.0, 4.0], size=11)
    conclusion(slide, "最大的一条是 GT 覆盖率：本轮所有关于天空的说法，严格讲都只覆盖「有回波的上三分之一」。")


def next_steps(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    header(slide, "NEXT", "下一步，按优先级")
    rows = [
        ["", "做什么", "为什么", "成本"],
        ["1", "d2 + caption 第二个 seed",
         "把权重效应 ① 从单次观测升级为可复现观测 —— 这是本轮最软的一环", "约 27 小时"],
        ["2", "零训练直推 + 对比可视化",
         "补上主表的参照点与任务 3 的图（作业与工具已就绪）", "约 15 分钟 + CPU 作业"],
        ["3", "夜间 21-58-13 第四个数据点",
         "雨天两因素模型的可证伪预测：词表低于重雨、注入税大于 +0.0005", "一个评估作业"],
        ["4", "a + caption",
         "补齐第一周任务 4 的缺口，看 caption 双效应在 RGB 线上是否同构", "约 67 小时"],
        ["5", "不依赖 GT 的天空判据工具",
         "堵上局限 1 那个窟窿 —— 现在没有任何指标能测天空", "约 60 行"],
    ]
    table(slide, rows, MARGIN, 1.70, BODY_W, 3.60, [0.4, 3.3, 6.4, 1.7], size=11.5)
    write(textbox(slide, MARGIN, 5.50, BODY_W, 0.80), [
        "1 和 2 决定这一轮结论能讲多硬，优先做；3–5 是扩展。"
        "集群上的产物路径、自查工具与全部原始数字见 docs/AIRE_RESULTS_20260802.md。",
    ], size=12, colour=GREY)
    conclusion(slide, "先补 ① 的第二 seed 和零训练参照点，再谈扩展。")


def build(output: Path, figure: Path | None) -> None:
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    cover(prs)
    overview(prs)
    protocol(prs)
    main_table(prs)
    caption_effects(prs)
    caption_strength(prs)
    far_field(prs)
    donor_swap(prs)
    external_reference(prs)
    qualitative(prs, figure)
    rain(prs)
    gradient_matching(prs)
    limits(prs)
    next_steps(prs)
    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(output))
    print(f"{output}  ({len(prs.slides)} slides)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, default=None,
                        help="Comparison PNG from build_comparison_figure.py. "
                             "Omitted -> the qualitative slide keeps its marked placeholder.")
    args = parser.parse_args()
    build(args.output, args.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
