"""Side-by-side qualitative comparison for an arbitrary set of saved predictions.

`build_route_vis_figures.py` did this for the old a-f route names on a hardcoded
`E:/dataset/ms2` and a hardcoded val sequence, and it picked frames by even
spacing.  None of the three survives contact with the Aire results: the routes
are now a/b/c1/c2/d1/d2 plus two external models, the data lives at
`$SCRATCH/data/ms2`, and evenly spaced frames say nothing about the claim the
figures exist to support.

This tool keeps that file's rendering (same Spectral map, same two colour
styles, same "align in the official space, invert, clip" pipeline) and replaces
the three hardcoded parts:

  models     `--predictions DIR[:LABEL][@ALIGN]`, repeatable.  The `@ALIGN`
             suffix is why this is not just a flag on the sibling tool:
             `analyze_prediction_regions.py --align` is global, so the original
             AnyThermal (`ssi`) and our routes (`ssi_disparity`) cannot appear
             in one of its runs.  In a figure they must.  Aligning either family
             in the other's space multiplies its error several-fold, so a
             mislabelled column is not a cosmetic error -- the alignment used is
             printed under every column.
  data       `--manifest` + `--data-root`, same contract as the evaluator.
  frames     `--pick`, driven by a stratified scan rather than by spacing:

               far      worst error in one stratum (default `depth/far >30m`)
               gap      where model A beats model B most in that stratum
               flip     where the per-frame ranking of the models disagrees
                        most with their ranking on the full set
               uniform  the old behaviour, kept for a sanity strip
               manual   `--frame-ids`

The scan reuses `strata_for` and the alignment mirror from the region tools, so
"the frame where d2's far field wins most" means the same thing here as in the
tables.  It is cached to `scan.json`; re-rendering is then instant.

    python tools/build_comparison_figure.py \\
        --manifest $SCRATCH/manifests/.../ms2_test_16-08-46_....jsonl \\
        --data-root $SCRATCH/data/ms2 \\
        --predictions $SCRATCH/runs/eval/b_e05_raw/raw_predictions:b_thermal \\
        --predictions $SCRATCH/runs/eval/d2_raw/raw_predictions:d2_anythermal \\
        --predictions $SCRATCH/runs/anythermal/Midas_anythermal/raw_predictions:anythermal_orig@ssi \\
        --pick gap --gap d2_anythermal:b_thermal --frames 5 \\
        --output-dir $SCRATCH/runs/vis/far_gap
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "lotus", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ms2_eval.io import load_ms2_gt                                          # noqa: E402
from ms2_eval.official_protocol import (                                     # noqa: E402
    ALIGN_MODES,
    collapse_channels,
    evaluate_sample,
    official_valid_mask,
)
from ms2_eval.resize import resize_dense_prediction                          # noqa: E402
from ms2_eval.stratify import align_prediction, strata_for                   # noqa: E402
from run_official_ms2_evaluation import prediction_path                      # noqa: E402

CMAP = matplotlib.colormaps["Spectral"]   # the evaluator's own colour map
MIN_D, MAX_D = 0.1, 80.0                  # display clip; scoring uses --min-depth

FAR = "depth/far >30m"

BG, INK, DIM = (255, 255, 255), (20, 24, 34), (110, 120, 138)
EDGE, HOLE = (208, 213, 222), (58, 62, 70)
TILE_W, GAP, HEAD, CAP, ROWLAB, FOOT = 470, 8, 186, 62, 104, 46

FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
)

TEXT = {
    "zh": {
        "input_thr": "热像输入", "input_rgb": "RGB 输入", "gt": "真值",
        "frame": "帧", "scale": "色标", "valid": "有效",
        "range": "量程", "align": "对齐",
        "sub_official": ("逐格上色：每一格用自己的深度量程归一化"
                         "（colorize_depth_map，逐图 min-max + Spectral）"),
        "sub_official2": "同一颜色在不同格中不代表同一深度；每格下方标注它自己的量程",
        "sub_shared": "同帧共用色标：全部按该帧真值的深度量程归一化（colormap 与官方一致）",
        "sub_shared2": "同一颜色在整行中代表同一深度，可以直接横向比较",
        "foot": ("全部由 raw_predictions 重算：按各列标注的空间做最小二乘对齐 → 转深度 → "
                 "clip 到 {lo}–{hi} m → 上色。真值列空洞为灰。"),
    },
    "en": {
        "input_thr": "thermal input", "input_rgb": "RGB input", "gt": "ground truth",
        "frame": "frame", "scale": "scale", "valid": "valid",
        "range": "range", "align": "align",
        "sub_official": "per-panel colouring: each panel normalised by its own depth range "
                        "(colorize_depth_map, per-image min-max + Spectral)",
        "sub_official2": "one colour does NOT mean one depth across panels; each panel is "
                         "annotated with its own range",
        "sub_shared": "shared scale: every panel normalised by this frame's ground-truth "
                      "depth range (same colormap as the official helper)",
        "sub_shared2": "one colour means one depth across the whole row; read it sideways",
        "foot": ("all recomputed from raw_predictions: least-squares alignment in the space "
                 "printed under each column -> depth -> clip to {lo}-{hi} m -> colour. "
                 "Holes in the GT column are grey."),
    },
}


def resolve_font(explicit: str | None, lang: str) -> tuple[Path | None, str]:
    """Return (font path or None, effective language).

    The cluster's conda env has no CJK font, and PIL's default bitmap font draws
    every Chinese glyph as a box -- a figure that looks broken rather than one
    that looks translated.  So a missing font downgrades the language instead of
    the legibility; `--font` overrides.
    """
    candidates = [explicit] if explicit else list(FONT_CANDIDATES)
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate), lang
    if explicit:
        raise SystemExit(f"--font not found: {explicit}")
    if lang == "zh":
        print("[font] 找不到中文字体，图上文字降级为英文；"
              "要中文就 --font 指到一个 CJK ttf（例如把 msyh.ttc 传到 $SCRATCH/fonts/）",
              flush=True)
    return None, "en"


class Fonts:
    def __init__(self, path: Path | None):
        self.path, self._cache = path, {}

    def __call__(self, size: int):
        if size not in self._cache:
            if self.path is None:
                self._cache[size] = ImageFont.load_default()
            else:
                self._cache[size] = ImageFont.truetype(str(self.path), size)
        return self._cache[size]


def parse_sources(values: list[str], default_align: str) -> list[dict]:
    """`DIR[:LABEL][@ALIGN]` -> [{label, dir, align}]."""
    sources = []
    for value in values:
        body, _, align = value.partition("@")
        align = align or default_align
        if align not in ALIGN_MODES:
            raise SystemExit(f"{value!r}: unknown alignment {align!r}, pick from {ALIGN_MODES}")
        # a Windows drive letter is not a label separator (same rule as the region tool)
        if ":" in body[2:]:
            head, _, label = body.rpartition(":")
        else:
            head, label = body, Path(body).name
        directory = Path(head)
        if not directory.is_dir():
            raise SystemExit(f"{label}: not a directory: {directory}")
        if not any(directory.glob("*.npy")):
            raise SystemExit(f"{label}: no *.npy under {directory}")
        sources.append({"label": label, "dir": directory, "align": align})
    labels = [s["label"] for s in sources]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"Prediction labels must be unique, got {labels}")
    return sources


def read_rows(manifest: Path, data_root: Path, gt_view: str) -> list[dict]:
    depth_field = "rgb_depth_path" if gt_view == "rgb" else "thermal_depth_path"
    view = "rgb" if gt_view == "rgb" else "thr"
    rows = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            relative = row.get(depth_field) or row.get("depth_path")
            if not relative:
                raise SystemExit(f"Row {row.get('id')} has no {gt_view}-view GT path")
            if f"/{view}/" not in str(relative).replace("\\", "/"):
                raise SystemExit(f"Row {row.get('id')}: GT {relative} is not the {view} view")
            rows.append({
                "id": str(row["id"]),
                "gt": data_root / relative,
                "thermal": data_root / row["thermal_path"] if row.get("thermal_path") else None,
                "rgb": data_root / row["rgb_path"] if row.get("rgb_path") else None,
            })
    if not rows:
        raise SystemExit(f"{manifest} produced no rows")
    return rows


# ── scan ────────────────────────────────────────────────────────────────────

def aligned_map(source: dict, row: dict, gt: np.ndarray, valid: np.ndarray,
                min_depth: float) -> np.ndarray:
    raw = collapse_channels(np.load(prediction_path(source["dir"], row["id"]), allow_pickle=False))
    pred = resize_dense_prediction(raw, tuple(gt.shape))
    return np.clip(align_prediction(pred, gt, valid, source["align"]), min_depth, MAX_D)


def scan(sources: list[dict], rows: list[dict], args) -> dict:
    """Per-frame, per-model, per-stratum AbsRel -- the same numbers the tables use."""
    out: dict[str, dict] = {s["label"]: {} for s in sources}
    checked = {s["label"]: 0.0 for s in sources}
    for index, row in enumerate(rows):
        _, gt = load_ms2_gt(row["gt"], args.depth_scale)
        valid = official_valid_mask(gt, args.min_depth, args.max_depth)
        if not valid.any():
            continue
        # strata depend on GT only, so they are computed once per frame, not once per model
        strata = strata_for(gt, valid, include_boundary=args.with_boundary)
        for source in sources:
            raw = collapse_channels(
                np.load(prediction_path(source["dir"], row["id"]), allow_pickle=False))
            pred = resize_dense_prediction(raw, tuple(gt.shape))
            a = np.clip(align_prediction(pred, gt, valid, source["align"]),
                        args.min_depth, args.max_depth)
            error = np.abs(a - gt) / np.maximum(gt, 1e-6)
            # the mirror must not drift from the frozen protocol module, exactly as
            # analyze_prediction_regions checks it
            official = evaluate_sample(pred, gt, align=source["align"],
                                       min_depth=args.min_depth, max_depth=args.max_depth)
            gap = abs(float(error[valid].mean()) - float(official["abs_rel"]))
            checked[source["label"]] = max(checked[source["label"]], gap)
            if gap > args.tolerance:
                raise SystemExit(
                    f"{source['label']}/{row['id']}: local AbsRel {error[valid].mean():.8f} "
                    f"disagrees with evaluate_sample {official['abs_rel']:.8f} (gap {gap:.2e}); "
                    "the alignment mirror has drifted from ms2_eval.official_protocol.")
            out[source["label"]][row["id"]] = {
                name: float(error[mask].mean())
                for name, mask in strata.items() if mask.any()
            }
        if (index + 1) % 250 == 0:
            print(f"  [scan] {index + 1}/{len(rows)}", flush=True)
    for label, gap in checked.items():
        print(f"  [scan] {label}: worst alignment gap vs evaluate_sample {gap:.2e}", flush=True)
    return out


def frame_index(sample_id: str) -> int:
    tail = sample_id.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def spread(candidates: list[str], count: int, separation: int) -> list[str]:
    """Take the top `count` keeping frames at least `separation` apart.

    Ranking by any error metric puts consecutive frames of the same scene at the
    top, so the unfiltered top-6 is one moment shown six times.
    """
    picked: list[str] = []
    for sample_id in candidates:
        position = frame_index(sample_id)
        if all(abs(position - frame_index(other)) >= separation for other in picked):
            picked.append(sample_id)
        if len(picked) == count:
            break
    if len(picked) < count:                    # separation too strict for this set
        for sample_id in candidates:
            if sample_id not in picked:
                picked.append(sample_id)
            if len(picked) == count:
                break
    return sorted(picked, key=frame_index)


def rank_distance(a: list[str], b: list[str]) -> int:
    """Pairwise disagreements between two orderings of the same labels."""
    position = {label: i for i, label in enumerate(b)}
    return sum(1 for i in range(len(a)) for j in range(i + 1, len(a))
               if position[a[i]] > position[a[j]])


def pick_frames(mode: str, scanned: dict, rows: list[dict], args) -> tuple[list[str], str]:
    ids = [r["id"] for r in rows if all(r["id"] in scanned[label] for label in scanned)]
    if not ids:
        raise SystemExit("No frame is present in every prediction set")
    labels = list(scanned)
    stratum = args.stratum

    def value(label, sample_id):
        return scanned[label][sample_id].get(stratum, scanned[label][sample_id]["all"])

    if mode == "manual":
        missing = [f for f in args.frame_ids if f not in scanned[labels[0]]]
        if missing:
            raise SystemExit(f"--frame-ids not in the scan: {missing}")
        return list(args.frame_ids), "手动指定" if args.lang == "zh" else "hand-picked"

    if mode == "uniform":
        chosen = [ids[int(i)] for i in np.linspace(0, len(ids) - 1, args.frames)]
        return chosen, (f"等距抽样（{stratum} 无关）" if args.lang == "zh"
                        else f"evenly spaced (not stratum-driven)")

    if mode == "far":
        ref = args.pick_ref
        if ref == "mean":
            score = {f: float(np.mean([value(l, f) for l in labels])) for f in ids}
            what = "全部模型平均" if args.lang == "zh" else "mean over all models"
        else:
            if ref not in scanned:
                raise SystemExit(f"--pick-ref {ref!r} is not one of {labels} (or 'mean')")
            score = {f: value(ref, f) for f in ids}
            what = ref
        order = sorted(ids, key=lambda f: -score[f])
        note = (f"按 {stratum} 误差最大挑选（{what}）" if args.lang == "zh"
                else f"largest {stratum} error ({what})")
        return spread(order, args.frames, args.min_separation), note

    if mode == "gap":
        if not args.gap or ":" not in args.gap:
            raise SystemExit("--pick gap needs --gap WINNER:LOSER")
        winner, loser = args.gap.split(":", 1)
        for label in (winner, loser):
            if label not in scanned:
                raise SystemExit(f"--gap label {label!r} is not one of {labels}")
        score = {f: value(loser, f) - value(winner, f) for f in ids}
        order = sorted(ids, key=lambda f: -score[f])
        note = (f"按 {stratum} 上 {winner} 领先 {loser} 最多挑选" if args.lang == "zh"
                else f"largest {winner} advantage over {loser} in {stratum}")
        return spread(order, args.frames, args.min_separation), note

    if mode == "flip":
        if len(labels) < 2:
            raise SystemExit("--pick flip needs at least two prediction sets")
        overall = sorted(labels, key=lambda l: np.mean([value(l, f) for f in ids]))
        score = {f: rank_distance(sorted(labels, key=lambda l: value(l, f)), overall)
                 for f in ids}
        order = sorted(ids, key=lambda f: (-score[f], -value(overall[0], f)))
        note = ((f"按 {stratum} 上排序翻转最厉害挑选（全集排序 {' < '.join(overall)}）")
                if args.lang == "zh"
                else f"ranking flipped hardest in {stratum} (full-set order {' < '.join(overall)})")
        return spread(order, args.frames, args.min_separation), note

    raise SystemExit(f"unknown --pick {mode}")


# ── rendering ───────────────────────────────────────────────────────────────

def colour(depth, lo, hi, mask=None):
    n = np.clip((depth - lo) / max(hi - lo, 1e-9), 0, 1)
    rgb = (CMAP(n)[..., :3] * 255).astype(np.uint8)
    if mask is not None:
        rgb[~mask] = HOLE
    return rgb


def input_tile(path: Path) -> Image.Image:
    src = Image.open(path)
    if src.mode in ("I;16", "I"):
        a = np.asarray(src).astype(np.float64)
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        src = Image.fromarray(((((a - lo) / max(hi - lo, 1e-9)).clip(0, 1)) * 255).astype("uint8"))
    return src.convert("RGB")


def render(style: str, sources: list[dict], rows: list[dict], picked: list[str],
           note: str, scanned: dict, args, fonts: Fonts, text: dict) -> Image.Image:
    by_id = {r["id"]: r for r in rows}
    columns = [("input", m) for m in args.inputs] + [("gt", None)] + [("pred", s) for s in sources]

    probe = by_id[picked[0]]
    heights = []
    for kind, payload in columns:
        path = probe[payload] if kind == "input" else probe["gt"]
        w, h = Image.open(path).size
        heights.append(round(TILE_W * h / w))
    row_h = max(heights)

    width = ROWLAB + len(columns) * (TILE_W + GAP)
    height = HEAD + len(picked) * (row_h + CAP) + FOOT
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    draw.text((ROWLAB, 16), args.title, font=fonts(31), fill=INK)
    draw.text((ROWLAB, 58), note, font=fonts(18), fill=INK)
    key = "sub_official" if style == "official" else "sub_shared"
    draw.text((ROWLAB, 86), text[key], font=fonts(16), fill=DIM)
    draw.text((ROWLAB, 108), text[key + "2"], font=fonts(15), fill=DIM)

    for ci, (kind, payload) in enumerate(columns):
        x = ROWLAB + ci * (TILE_W + GAP)
        if kind == "input":
            label, sub = text["input_rgb" if payload == "rgb" else "input_thr"], ""
        elif kind == "gt":
            label, sub = text["gt"], f"GT view {args.gt_view}"
        else:
            label, sub = payload["label"], f"{text['align']} {payload['align']}"
        draw.text((x, HEAD - 44), label, font=fonts(19), fill=INK)
        if sub:
            draw.text((x, HEAD - 21), sub, font=fonts(14), fill=DIM)

    for ri, sample_id in enumerate(picked):
        row = by_id[sample_id]
        y = HEAD + ri * (row_h + CAP)
        _, gt = load_ms2_gt(row["gt"], args.depth_scale)
        valid = official_valid_mask(gt, args.min_depth, args.max_depth)
        clipped = np.clip(gt[valid], MIN_D, MAX_D)
        gt_lo, gt_hi = float(clipped.min()), float(clipped.max())

        draw.text((10, y + row_h // 2 - 34), f"{text['frame']}\n{frame_index(sample_id):06d}",
                  font=fonts(17), fill=INK)
        if style == "shared":
            draw.text((10, y + row_h // 2 + 14), f"{text['scale']}\n{gt_lo:.1f}–{gt_hi:.0f}m",
                      font=fonts(14), fill=DIM)

        for ci, (kind, payload) in enumerate(columns):
            x = ROWLAB + ci * (TILE_W + GAP)
            annotation = ""
            if kind == "input":
                image = input_tile(row[payload])
            elif kind == "gt":
                image = Image.fromarray(
                    colour(np.clip(gt, MIN_D, MAX_D), gt_lo, gt_hi, valid))
                annotation = f"{gt_lo:.1f}–{gt_hi:.0f} m　{text['valid']} {valid.mean()*100:.0f}%"
            else:
                depth = aligned_map(payload, row, gt, valid, args.min_depth)
                stats = scanned[payload["label"]][sample_id]
                far = stats.get(args.stratum)
                annotation = f"AbsRel {stats['all']:.4f}"
                if far is not None and args.stratum != "all":
                    annotation += f"　{args.stratum} {far:.4f}"
                if style == "official":
                    image = Image.fromarray(colour(depth, depth.min(), depth.max()))
                    annotation += f"\n{text['range']} {depth.min():.1f}–{depth.max():.0f} m"
                else:
                    image = Image.fromarray(colour(depth, gt_lo, gt_hi))
            w, h = image.size
            th = round(TILE_W * h / w)
            ty = y + (row_h - th) // 2
            canvas.paste(image.resize((TILE_W, th), Image.LANCZOS), (x, ty))
            draw.rectangle([x - 1, ty - 1, x + TILE_W, ty + th], outline=EDGE)
            if annotation:
                draw.text((x, y + row_h + 6), annotation, font=fonts(15), fill=DIM)

    draw.text((ROWLAB, height - 32),
              text["foot"].format(lo=MIN_D, hi=int(MAX_D)), font=fonts(15), fill=DIM)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", action="append", required=True,
                        metavar="DIR[:LABEL][@ALIGN]",
                        help="Raw *.npy directory. @ALIGN overrides --align for this column "
                             "only -- needed to put ssi and ssi_disparity models in one figure.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--align", default="ssi_disparity", choices=ALIGN_MODES,
                        help="Default alignment for columns without an @ suffix. "
                             "ssi_disparity for our route suite, ssi for the original AnyThermal.")
    parser.add_argument("--gt-view", default="thermal", choices=("thermal", "rgb"),
                        help="One view per figure: mixing views in one table is forbidden, "
                             "and a figure is a table with pictures.")
    parser.add_argument("--inputs", default="thermal",
                        help="Comma-separated input columns to show: thermal,rgb.")
    parser.add_argument("--pick", default="far",
                        choices=("far", "gap", "flip", "uniform", "manual"))
    parser.add_argument("--stratum", default=FAR,
                        help=f"Stratum that drives picking and the per-panel annotation "
                             f"(default {FAR!r}; 'all' for whole-frame).")
    parser.add_argument("--pick-ref", default="mean",
                        help="--pick far: which model's error to rank by, or 'mean'.")
    parser.add_argument("--gap", default=None, metavar="WINNER:LOSER",
                        help="--pick gap: maximise WINNER's advantage over LOSER in --stratum.")
    parser.add_argument("--frame-ids", nargs="+", default=[],
                        help="--pick manual: explicit sample ids.")
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--min-separation", type=int, default=30,
                        help="Minimum frame-number distance between picks, so the top of the "
                             "ranking is not the same moment six times. 0 disables.")
    parser.add_argument("--style", default="both", choices=("both", "official", "shared"))
    parser.add_argument("--title", default="")
    parser.add_argument("--lang", default="zh", choices=("zh", "en"))
    parser.add_argument("--font", default=None, help="Path to a TTF/TTC; needed for --lang zh.")
    parser.add_argument("--scan-cache", type=Path, default=None,
                        help="Reuse/write a scan JSON. Defaults to <output-dir>/scan.json.")
    parser.add_argument("--rescan", action="store_true", help="Ignore an existing scan cache.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Scan every Nth frame. Picking can only see scanned frames.")
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.inputs = [v.strip() for v in args.inputs.split(",") if v.strip()]
    for value in args.inputs:
        if value not in ("thermal", "rgb"):
            raise SystemExit(f"--inputs takes thermal and/or rgb, got {value!r}")

    # Only the two structure/* strata need torch (see ms2_eval.stratify). Dropping
    # them keeps the figures buildable on a plain numpy+PIL box; the depth bands
    # the slides actually argue from are unaffected.
    try:
        import torch  # noqa: F401
        args.with_boundary = True
    except ImportError:
        args.with_boundary = False
        print("[strata] torch 不可用，跳过 structure/* 两个分层（深度带与行带不受影响）"
              if args.lang == "zh" else
              "[strata] torch unavailable, skipping the two structure/* strata "
              "(depth bands and row bands are unaffected)", flush=True)
        if args.stratum.startswith("structure/"):
            raise SystemExit(f"--stratum {args.stratum} needs torch; "
                             "run this where the region tools run, or pick a depth/row stratum")

    font_path, args.lang = resolve_font(args.font, args.lang)
    fonts, text = Fonts(font_path), TEXT[args.lang]

    sources = parse_sources(args.predictions, args.align)
    rows = read_rows(args.manifest, args.data_root, args.gt_view)[:: max(1, args.stride)]
    for value in args.inputs:
        if any(row[value] is None for row in rows):
            raise SystemExit(f"--inputs {value}: the manifest has no {value}_path for every row")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = args.scan_cache or (args.output_dir / "scan.json")

    signature = {
        "manifest": str(args.manifest), "stride": args.stride, "gt_view": args.gt_view,
        "sources": [{"label": s["label"], "dir": str(s["dir"]), "align": s["align"]}
                    for s in sources],
        "min_depth": args.min_depth, "max_depth": args.max_depth,
        "depth_scale": args.depth_scale,
    }
    scanned = None
    if cache.is_file() and not args.rescan:
        payload = json.loads(cache.read_text(encoding="utf-8"))
        if payload.get("signature") == signature:
            scanned = payload["scan"]
            print(f"[scan] reusing {cache}", flush=True)
        else:
            print(f"[scan] {cache} was built for a different set of inputs -- rescanning",
                  flush=True)
    if scanned is None:
        print(f"[scan] {len(rows)} frames x {len(sources)} models", flush=True)
        scanned = scan(sources, rows, args)
        cache.write_text(json.dumps({"signature": signature, "scan": scanned}, ensure_ascii=False),
                         encoding="utf-8")
        print(f"[scan] wrote {cache}", flush=True)

    picked, note = pick_frames(args.pick, scanned, rows, args)
    print(f"[pick] {args.pick}: {note}")
    for sample_id in picked:
        parts = "  ".join(f"{s['label']} {scanned[s['label']][sample_id].get(args.stratum, float('nan')):.4f}"
                          for s in sources)
        print(f"       {sample_id}   {args.stratum}:  {parts}")

    if not args.title:
        args.title = " · ".join(s["label"] for s in sources)

    styles = ("official", "shared") if args.style == "both" else (args.style,)
    for style in styles:
        image = render(style, sources, rows, picked, note, scanned, args, fonts, text)
        path = args.output_dir / f"{style}.png"
        image.save(path)
        print(f"[out]  {path}   {image.size[0]}x{image.size[1]}")

    (args.output_dir / "picked_frames.json").write_text(
        json.dumps({"pick": args.pick, "note": note, "stratum": args.stratum,
                    "frame_ids": picked,
                    "per_frame": {f: {s["label"]: scanned[s["label"]][f] for s in sources}
                                  for f in picked}},
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
