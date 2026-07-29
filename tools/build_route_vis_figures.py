"""Qualitative comparison figures for task1 / task2 / task4, in two colour styles.

Both styles render the SAME quantity through the SAME pipeline as the official
evaluator (least-squares fit in disparity space -> invert -> clip to
[min_depth, max_depth]) and use the SAME Spectral colour map.  They differ in
one line only:

  official  each panel is normalised by its own min/max -- exactly
            lotus/utils/image_utils.py::colorize_depth_map.  For the model
            columns the untouched vis/*.png export is pasted in directly.
  shared    every panel of a frame is normalised by that frame's GROUND TRUTH
            depth range, so one colour means one depth across the whole row.

The official style is the repo's display helper -- it is not part of the BMSD
protocol, which specifies alignment, masking, metric formulas and aggregation
only.  Per-image normalisation is fine when every model is affine-invariant; it
misrepresents a model that predicts a genuinely wider depth range, because the
widest-range panel gets compressed the most.  Every cell is therefore annotated
with its own aligned depth range, so the two styles can be read against each
other rather than taken on trust.

Outputs (outputs/lotus_line_v2/route_vis_figures/)
  official_<set>.png / shared_<set>.png     the three sets, both styles
  ranking_flip.png                          both styles on one frame

Usage
  python tools/build_route_vis_figures.py
  python tools/build_route_vis_figures.py --frames 8 --which task4_five_experiments
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ms2_eval.io import load_ms2_gt                                          # noqa: E402
from ms2_eval.official_protocol import fit_scale_shift, official_valid_mask  # noqa: E402
from ms2_eval.resize import resize_dense_prediction                          # noqa: E402

BASE = ROOT / "outputs" / "lotus_line_v2"
OUT = BASE / "route_vis_figures"
MS2 = Path("E:/dataset/ms2")
SEQ = "2021-08-06-11-23-45"
N_FRAMES_IN_SEQ = 5810

CMAP = matplotlib.colormaps["Spectral"]   # the evaluator's own colour map
MIN_D, MAX_D = 0.1, 80.0                  # dataset.min_depth / max_depth

BG, INK, DIM = (255, 255, 255), (20, 24, 34), (110, 120, 138)
EDGE, HOLE = (208, 213, 222), (58, 62, 70)
TILE_W, GAP, HEAD, CAP, ROWLAB, FOOT = 470, 8, 128, 44, 96, 46

CJK = ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf",
       "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc")


def font(size):
    for p in CJK:
        if Path(p).is_file():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


FIGURES = {
    "task1_six_routes": (
        "任务1 · 六条路线",
        [("a  冻结 RGB 直推", "route_a_rgb_frozen_val_full", "rgb", "disp"),
         ("b  RGB + 训 U-Net", "route_b_rgb_unet_val_full", "rgb", "disp"),
         ("c  冻结热像直推", "route_c_thermal_frozen_val_full", "thr", "disp"),
         ("d  热像 + 训 U-Net", "route_d_thermal_unet_val_full", "thr", "disp"),
         ("e  VAE latent + Adapter", "route_e_vae_adapter_val_full", "thr", "disp"),
         ("f  AnyThermal + Adapter", "route_f_adapter_only_val_full", "thr", "disp")]),
    "task2_six_captions": (
        "任务2 · 六条路线注入 caption",
        [("a  冻结 RGB + caption", "route_a_rgb_frozen_val_full_caption", "rgb", "disp"),
         ("b  RGB U-Net + caption", "route_b_rgb_unet_val_full_caption", "rgb", "disp"),
         ("c  冻结热像 + caption", "route_c_thermal_frozen_val_full_caption", "thr", "disp"),
         ("d  热像 U-Net + caption", "route_d_fp32dec_val_full_caption", "thr", "disp"),
         ("e  VAE Adapter + caption", "route_e_vae_adapter_val_full_caption", "thr", "disp"),
         ("f  AnyThermal Adapter + caption", "route_f_adapter_only_val_full_caption", "thr", "disp")]),
    "task4_five_experiments": (
        "任务4 · 五个实验",
        [("实验1  约束 + GT", "route_d_fp32dec_val_full_empty", "thr", "disp"),
         ("实验2  纯 GT", "route_gt_only_val_full", "thr", "disp"),
         ("实验3  约束 + GT + teacher(ssi)", "route_amteacher_val_full", "thr", "disp"),
         ("实验4  去约束 + teacher(ssi)", "route_cprime_val_full", "thr", "disp"),
         ("实验5  去约束 + teacher(l1)", "route_cprime_l1_val_full", "thr", "disp"),
         ("AnyThermal 全监督老师", "anythermal_midas_val_full", "thr", "depth")]),
}

SUBTITLE = {
    "official": ("官方显示函数：每一格用自己的深度量程归一化（colorize_depth_map，逐图 min-max + Spectral）",
                 "同一颜色在不同格中不代表同一深度；每格下方标注的是它自己的量程"),
    "shared": ("同帧共用色标：全部按该帧真值的深度量程归一化（colormap 与渲染流程与官方完全一致）",
               "同一颜色在整行中代表同一深度，可以直接横向比较"),
}


def cam_dir(cam):
    return "rgb" if cam == "rgb" else "thr"


def gt_path(cam, fr):
    return MS2 / "proj_depth" / f"_{SEQ}" / cam_dir(cam) / "depth_filtered" / f"{fr:06d}.png"


def input_path(cam, fr):
    return MS2 / "sync_data" / f"_{SEQ}" / cam_dir(cam) / "img_left" / f"{fr:06d}.png"


def vis_path(route, cam, fr):
    return BASE / route / "vis" / f"sync_data__{SEQ}_{cam_dir(cam)}_img_left_pred_{fr:06d}.png"


def colour(depth, lo, hi, mask=None):
    """The evaluator's colouring, with the bounds made explicit."""
    n = np.clip((depth - lo) / max(hi - lo, 1e-9), 0, 1)
    rgb = (CMAP(n)[..., :3] * 255).astype(np.uint8)
    if mask is not None:
        rgb[~mask] = HOLE
    return rgb


def input_tile(cam, fr):
    src = Image.open(input_path(cam, fr))
    if src.mode in ("I;16", "I"):
        a = np.asarray(src).astype(np.float64)
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        src = Image.fromarray(((((a - lo) / max(hi - lo, 1e-9)).clip(0, 1)) * 255).astype("uint8"))
    return src.convert("RGB")


class FrameData:
    def __init__(self, cam, fr):
        self.cam, self.fr = cam, fr
        _, self.gt = load_ms2_gt(gt_path(cam, fr), 256.0)
        self.valid = official_valid_mask(self.gt, 0.001, MAX_D)
        self.disp = np.zeros_like(self.gt, np.float64)
        self.disp[self.valid] = 1.0 / self.gt[self.valid].astype(np.float64)
        g = np.clip(self.gt[self.valid], MIN_D, MAX_D)
        self.gt_lo, self.gt_hi = float(g.min()), float(g.max())

    def aligned_depth(self, route, space="disp"):
        """Exactly the evaluator's path: LS fit in disparity, invert, clip."""
        raw = np.load(BASE / route / "raw_predictions" / f"{SEQ}_{self.fr:06d}.npy").astype(np.float32)
        if raw.ndim == 3:
            raw = raw.mean(axis=0) if raw.shape[0] <= 4 else raw.mean(axis=2)
        raw = resize_dense_prediction(raw, self.gt.shape)
        if space == "depth":            # MiDaS-style affine depth (AnyThermal)
            sc, sh = fit_scale_shift(raw, self.gt.astype(np.float32), self.valid)
            d = np.clip(raw.astype(np.float64) * sc + sh, MIN_D, MAX_D)
        else:
            sc, sh = fit_scale_shift(raw, self.disp.astype(np.float32), self.valid)
            d = np.clip(1.0 / np.clip(raw.astype(np.float64) * sc + sh, 1e-3, None), MIN_D, MAX_D)
        rel = np.abs(d[self.valid] - self.gt[self.valid]) / self.gt[self.valid]
        return d, float(rel.mean())


def build(key, style, frames):
    title, cols = FIGURES[key]
    cams = []
    for _, _, c, _sp in cols:
        if c not in cams:
            cams.append(c)
    cells = ([("RGB 输入" if c == "rgb" else "热像输入", None, c, "disp") for c in cams]
             + [("真值", "__gt__", cams[0], "disp")] + cols)

    heights = []
    for _, d, cam, _sp in cells:
        probe = input_path(cam, frames[0]) if d is None else gt_path(cam, frames[0])
        w, h = Image.open(probe).size
        heights.append(round(TILE_W * h / w))
    row_h = max(heights)

    W = ROWLAB + len(cells) * (TILE_W + GAP)
    H = HEAD + len(frames) * (row_h + CAP) + FOOT
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((ROWLAB, 18), f"{title}　·　{'官方上色' if style == 'official' else '共用色标'}",
              font=font(31), fill=INK)
    draw.text((ROWLAB, 60), SUBTITLE[style][0], font=font(17), fill=DIM)
    draw.text((ROWLAB, 84), SUBTITLE[style][1], font=font(16), fill=DIM)

    for ci, (label, *_r) in enumerate(cells):
        draw.text((ROWLAB + ci * (TILE_W + GAP), HEAD - 26), label, font=font(18), fill=INK)

    fdata = {c: {fr: FrameData(c, fr) for fr in frames} for c in cams}

    for ri, fr in enumerate(frames):
        y = HEAD + ri * (row_h + CAP)
        ref = fdata[cams[0]][fr]
        draw.text((10, y + row_h // 2 - 26), f"帧\n{fr:06d}", font=font(17), fill=INK)
        if style == "shared":
            draw.text((10, y + row_h // 2 + 18), f"色标\n{ref.gt_lo:.1f}–{ref.gt_hi:.0f}m",
                      font=font(14), fill=DIM)
        for ci, (label, d, cam, space) in enumerate(cells):
            x = ROWLAB + ci * (TILE_W + GAP)
            fd = fdata[cam][fr]
            note = ""
            if d is None:
                img = input_tile(cam, fr)
            elif d == "__gt__":
                img = Image.fromarray(colour(np.clip(fd.gt, MIN_D, MAX_D),
                                             fd.gt_lo, fd.gt_hi, fd.valid))
                note = f"{fd.gt_lo:.1f}–{fd.gt_hi:.0f} m　有效 {fd.valid.mean()*100:.0f}%"
            else:
                dep, absrel = fd.aligned_depth(d, space)
                if style == "official":
                    vp = vis_path(d, cam, fr)
                    img = (Image.open(vp).convert("RGB") if vp.is_file()
                           else Image.fromarray(colour(dep, dep.min(), dep.max())))
                    mark = "" if vp.is_file() else "＊"
                    note = f"AbsRel {absrel:.4f}　本格量程 {dep.min():.1f}–{dep.max():.0f} m{mark}"
                else:
                    img = Image.fromarray(colour(dep, fd.gt_lo, fd.gt_hi))
                    note = f"AbsRel {absrel:.4f}"
            w, h = img.size
            th = round(TILE_W * h / w)
            ty = y + (row_h - th) // 2
            canvas.paste(img.resize((TILE_W, th), Image.LANCZOS), (x, ty))
            draw.rectangle([x - 1, ty - 1, x + TILE_W, ty + th], outline=EDGE)
            if note:
                draw.text((x, y + row_h + 6), note, font=font(15), fill=DIM)

    foot = ("模型列为官方评估流程导出的 vis/*.png 原图；标＊者当时未存 vis，由同一函数现渲染（逐格 min-max）；真值列同法，空洞为灰。"
            if style == "official" else
            "全部由原始输出重算：官方视差空间最小二乘对齐 → 转深度 → clip 到 0.1–80m → "
            "按该帧真值量程归一化 → Spectral。与官方唯一的差别是归一化上下界改为共用。")
    draw.text((ROWLAB, H - 32), foot, font=font(15), fill=DIM)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{style}_{key}.png"
    canvas.save(p)
    print(f"  {p.relative_to(ROOT)}   {W}x{H}")


def ranking_flip(fr):
    """One frame, the task-4 arms, both styles stacked: the orderings disagree."""
    _, cols = FIGURES["task4_five_experiments"]
    fd = FrameData("thr", fr)
    tw = 560
    w0, h0 = Image.open(gt_path("thr", fr)).size
    th = round(tw * h0 / w0)
    W = ROWLAB + (len(cols) + 1) * (tw + GAP)
    H = 156 + 2 * (th + 64) + 40
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((ROWLAB, 18), f"同一帧、同一批预测，两种上色给出相反的观感　·　帧 {fr:06d}",
              font=font(31), fill=INK)
    draw.text((ROWLAB, 64),
              "全量 AbsRel 排名：实验5 最好 0.0862 → 实验1 0.1275 → 实验3 0.1276 → "
              "实验2 0.1184 → 实验4 最差 0.1688。", font=font(18), fill=DIM)
    draw.text((ROWLAB, 92),
              "官方显示函数逐图 min-max，量程越宽的模型被压缩得越狠——"
              "而只有实验5 把远景推到了 80m。", font=font(18), fill=DIM)

    for row, style in enumerate(("official", "shared")):
        y = 156 + row * (th + 64)
        draw.text((ROWLAB, y - 26),
                  "① 官方上色（每格自己的量程）" if style == "official"
                  else "② 共用色标（全行同一量程，取自真值）", font=font(20), fill=INK)
        panels = [("真值", Image.fromarray(colour(np.clip(fd.gt, MIN_D, MAX_D),
                                                 fd.gt_lo, fd.gt_hi, fd.valid)),
                   f"{fd.gt_lo:.1f}–{fd.gt_hi:.0f} m")]
        for label, d, cam, space in cols:
            dep, absrel = fd.aligned_depth(d, space)
            vp = vis_path(d, cam, fr)
            if style == "official":
                img = (Image.open(vp).convert("RGB") if vp.is_file()
                       else Image.fromarray(colour(dep, dep.min(), dep.max())))
            else:
                img = Image.fromarray(colour(dep, fd.gt_lo, fd.gt_hi))
            note = (f"AbsRel {absrel:.4f}　量程 {dep.min():.1f}–{dep.max():.0f} m"
                    if style == "official" else f"AbsRel {absrel:.4f}")
            panels.append((label.split("  ")[0], img, note))
        for ci, (name, img, note) in enumerate(panels):
            x = ROWLAB + ci * (tw + GAP)
            canvas.paste(img.resize((tw, th), Image.LANCZOS), (x, y))
            draw.rectangle([x - 1, y - 1, x + tw, y + th], outline=EDGE)
            draw.text((x, y + th + 5), name, font=font(17), fill=INK)
            draw.text((x, y + th + 27), note, font=font(14), fill=DIM)

    draw.text((ROWLAB, H - 28),
              "两行的像素来自同一批 raw_predictions，唯一差别是归一化的上下界。",
              font=font(15), fill=DIM)
    p = OUT / "ranking_flip.png"
    canvas.save(p)
    print(f"  {p.relative_to(ROOT)}   {W}x{H}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--which", default="all", choices=("all",) + tuple(FIGURES))
    ap.add_argument("--style", default="both", choices=("both", "official", "shared"))
    args = ap.parse_args()

    frames = [int(i) for i in np.linspace(0, N_FRAMES_IN_SEQ - 1, args.frames)]
    print(f"等距抽样 {args.frames} 帧: {frames}")

    styles = ("official", "shared") if args.style == "both" else (args.style,)
    for key in (FIGURES if args.which == "all" else [args.which]):
        missing = [d for _, d, cam, _sp in FIGURES[key][1]
                   if not (BASE / d / "raw_predictions").is_dir()]
        if missing:
            print(f"{key}: 缺少 vis 导出 {missing}")
            continue
        print(f"{key}:")
        for st in styles:
            build(key, st, frames)
    print("ranking_flip:")
    ranking_flip(frames[len(frames) // 2])


if __name__ == "__main__":
    main()
