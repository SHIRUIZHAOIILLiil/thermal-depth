"""Render side-by-side qualitative comparisons with Iris's own colouring.

Colour maps are not cosmetic here: a different normalisation or colormap makes
two models look different when they are not. This uses `colorize_depth_map`
from `lotus/utils/image_utils.py` — the function Iris itself calls in `infer.py`
and in its training visualisations — so the figures are directly comparable with
the ones in the paper.

Sparse LiDAR ground truth is dilated before colouring; at 29% density the raw
points are invisible at slide size, and an undilated GT panel reads as noise.

    python tools/export_qualitative.py \\
        --models b_thermal_unet:outputs/route_suite/b_thermal_unet_20ep/epoch05_weights.pt \\
                 c1_vae_adapter:outputs/route_suite/c1_vae_adapter_20ep/best_weights.pt \\
        --frames 3 --output-dir outputs/route_suite/qualitative
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = ROOT / "lotus"
for path in (ROOT, LOTUS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from utils.image_utils import colorize_depth_map  # noqa: E402  (Iris official)
from ms2_eval.official_protocol import fit_scale_shift, official_valid_mask  # noqa: E402
from train_route_suite import (  # noqa: E402
    ROUTES,
    RouteModel,
    load_input_tensor,
    read_manifest,
    rotate_captions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, metavar="ROUTE:CHECKPOINT[:LABEL]")
    parser.add_argument(
        "--extra-panel", nargs="*", default=[], metavar="LABEL:DIR",
        help="Add a column from metric depth already on disk, as "
             "<DIR>/<frame id>.npy in metres, which is what "
             "run_ms2_supdepth_baselines.py --save-pred-ids writes. Only the "
             "shared style renders these: colouring a baseline on its own "
             "scale beside ours would make the two differ for the wrong reason.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/route_suite/qualitative"))
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=Path(
            "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/"
            "ms2_fixed_sequence_internvl3_8b_filtered_caption_val_rgb_depth_v1_clip75_rerun_20260714.jsonl"
        ),
    )
    parser.add_argument("--ms2-root", type=Path, default=Path("/mnt/e/dataset/ms2"))
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--frame-ids", nargs="*", default=None, help="Explicit manifest ids instead of a stride.")
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--gt-dilate", type=int, default=3, help="Max-pool window used to make sparse GT visible.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frozen-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument(
        "--caption-override", default=None,
        help=(
            "Text used by models whose --caption-mode is 'override'. An Iris-Fig-9-style "
            "probe: hand-write a description that contradicts the scene and see whether "
            "the prediction visibly moves. A demonstration, not a measurement."
        ),
    )
    parser.add_argument(
        "--caption-mode",
        default="empty",
        help=(
            "empty | correct | shuffled | override, or one per model as a comma list. A "
            "caption-trained checkpoint rendered under 'empty' is being shown outside "
            "the input mode it was trained in. 'shuffled' keeps the text distribution "
            "and breaks only the image-text correspondence, so the same checkpoint "
            "listed twice (correct / shuffled) isolates caption content."
        ),
    )
    parser.add_argument(
        "--style",
        default="official",
        choices=("official", "shared", "both"),
        help=(
            "official = Iris's colorize_depth_map,每个面板按自己的量程归一化（与论文图可比，"
            "但两个模型的同一颜色不代表同一距离）。shared = 先按官方协议把预测对齐成米制深度，"
            "再让整行共用该帧真值的量程 —— 只有这个模式下'天空发黄'才能读作'距离判错'。"
        ),
    )
    parser.add_argument(
        "--panel-labels", choices=("on", "off"), default="on",
        help="off = 面板上不画任何文字。给幻灯用的图一律用 off：烘进像素的标注没法在"
             "排版时改，读图的人会把它当成结论的一部分。栏目名放到幻灯的图注里。",
    )
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    return parser.parse_args()


def align_to_metric_depth(prediction, gt, min_depth, max_depth):
    """Official ssi_disparity path: fit in disparity space, invert, clamp."""
    valid = official_valid_mask(gt, min_depth, max_depth)
    gt_disparity = np.zeros_like(gt, np.float64)
    gt_disparity[valid] = 1.0 / gt[valid].astype(np.float64)
    scale, shift = fit_scale_shift(prediction, gt_disparity.astype(np.float32), valid)
    aligned = 1.0 / np.clip(prediction.astype(np.float64) * scale + shift, 1e-3, None)
    return np.clip(aligned, min_depth, max_depth), valid


def colour_shared(disparity, lo, hi, mask=None):
    """Iris's own colorize_depth_map, forced onto a shared normalisation range.

    The official helper always min-max normalises the array it is handed, which
    is exactly what makes two panels incomparable. Rather than re-implement the
    colouring (and risk inverting the colormap), clip to the shared range and
    append a one-pixel-tall sentinel row carrying lo and hi: the official
    function then normalises against those bounds, and the row is cropped off.
    Every displayed pixel is genuine, and the colouring is literally Iris's --
    disparity in, reverse_color=True, i.e. 红=近 / 蓝=远, the same convention as
    `lotus/infer.py:211` (`reverse_color=args.disparity`).
    """
    clipped = np.clip(np.asarray(disparity, np.float32), lo, hi)
    sentinel = np.full((1, clipped.shape[1]), lo, np.float32)
    sentinel[0, 0] = hi
    padded = np.vstack([clipped, sentinel])
    if mask is not None:
        mask = np.vstack([mask, np.zeros((1, mask.shape[1]), bool)])
        image = colorize_depth_map(padded, mask=torch.from_numpy(mask), reverse_color=True)
    else:
        image = colorize_depth_map(padded, reverse_color=True)
    return image.crop((0, 0, image.width, image.height - 1))


def dilate(values: np.ndarray, valid: np.ndarray, window: int):
    if window <= 1:
        return values, valid
    filled = torch.from_numpy(np.where(valid, values, -np.inf))[None, None]
    pooled = F.max_pool2d(filled, window, stride=1, padding=window // 2)[0, 0].numpy()
    mask = np.isfinite(pooled)
    return np.where(mask, pooled, 0.0), mask


def strip(panels: list[tuple[str, Image.Image]], width: int, label_h: int = 34) -> Image.Image:
    from PIL import ImageDraw, ImageFont

    resized = []
    for name, im in panels:
        h = int(im.height * width / im.width)
        resized.append((name, im.resize((width, h), Image.BICUBIC)))
    row_h = max(im.height for _, im in resized) + label_h
    canvas = Image.new("RGB", (width * len(resized), row_h), (255, 255, 255))
    if label_h <= 0:
        for i, (_, im) in enumerate(resized):
            canvas.paste(im, (i * width, 0))
        return canvas
    draw = ImageDraw.Draw(canvas)
    # panel labels are ASCII on purpose: a missing CJK face silently renders
    # every Chinese glyph as a box, and the fallback bitmap font is unreadable
    font = ImageFont.load_default()
    for candidate in ("C:/Windows/Fonts/segoeui.ttf", "/mnt/c/Windows/Fonts/segoeui.ttf",
                      "C:/Windows/Fonts/arial.ttf", "/mnt/c/Windows/Fonts/arial.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            font = ImageFont.truetype(candidate, 22)
            break
        except OSError:
            continue
    for i, (name, im) in enumerate(resized):
        canvas.paste(im, (i * width, label_h))
        draw.text((i * width + 8, 7), name, fill=(31, 42, 68), font=font)
    return canvas


def parse_extra_panels(entries):
    out = []
    for entry in entries:
        if ":" not in entry:
            raise SystemExit(f"Bad --extra-panel {entry!r}; expected LABEL:DIR")
        label, directory = entry.split(":", 1)
        path = Path(directory)
        if not path.is_dir():
            raise SystemExit(f"--extra-panel {label}: no such directory {path}")
        out.append((label, path))
    return out


def main() -> None:
    args = parse_args()
    extra_panels = parse_extra_panels(args.extra_panel)
    args.val_caption_mode = "empty"
    args.gt_decode_fp32 = True
    args.input_max_edge = 0
    args.gt_min_depth, args.gt_max_depth = 0.1, 80.0

    modes = [token.strip() for token in args.caption_mode.split(",") if token.strip()]
    if len(modes) == 1:
        modes *= len(args.models)
    if len(modes) != len(args.models):
        raise SystemExit(
            f"--caption-mode takes one value or one per model; got {len(modes)} for {len(args.models)}"
        )
    unknown = sorted(set(modes) - {"empty", "correct", "shuffled", "override"})
    if unknown:
        raise SystemExit(f"Unknown caption mode(s): {unknown}")
    if "override" in modes and not (args.caption_override or "").strip():
        raise SystemExit("--caption-mode override needs --caption-override TEXT")

    specs = []
    for entry, mode in zip(args.models, modes):
        route, _, rest = entry.partition(":")
        # Splitting the whole entry on ":" breaks on Windows, where the checkpoint
        # carries a drive letter and "b_thermal_unet:E:/runs/x.pt:label" yields four
        # fields, silently taking "E" as the checkpoint and the path as the label.
        # Take the label off the right instead, and treat a lone letter before the
        # first slash as a drive rather than a checkpoint that was given no label.
        head, sep, tail = rest.rpartition(":")
        if sep and not (len(head) == 1 and head.isalpha()):
            checkpoint, label = head, tail
        else:
            checkpoint, label = rest, ""
        if route not in ROUTES or not checkpoint:
            raise SystemExit(f"Bad --models entry {entry!r}; expected ROUTE:CHECKPOINT[:LABEL]")
        path = Path(checkpoint)
        # Two arms of one experiment share a route name (b_thermal_unet empty vs
        # caption), so keying panels by route alone would overwrite one of them.
        specs.append((route, path, label or f"{path.parent.name}", mode))
    labels = [spec[2] for spec in specs]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"Model labels must be unique, got {labels}")

    device = torch.device(args.device)
    frozen_dtype = torch.float16 if args.frozen_dtype == "fp16" else torch.float32
    args.output_dir.mkdir(parents=True, exist_ok=True)

    modality = ROUTES[specs[0][0]][0]
    rows = read_manifest(args.val_manifest, args.ms2_root, modality, split=None, check_files=False)
    if args.frame_ids:
        wanted = set(args.frame_ids)
        available = {r["id"] for r in rows}
        rows = [r for r in rows if r["id"] in wanted]
        if not rows:
            # Otherwise both models load and run before an empty panel list
            # crashes the canvas -- a GPU job spent to learn the ids were wrong.
            # The usual cause is passing the numeric suffix: manifest ids carry
            # the sequence, e.g. 2021-08-13-16-08-46_000036.
            sample = sorted(available)[:2]
            raise SystemExit(
                f"None of --frame-ids {sorted(wanted)} are in {args.val_manifest.name}. "
                f"Ids look like {sample} -- pass them in full, not just the number."
            )
        missing = sorted(wanted - available)
        if missing:
            print(f"[data] not in this manifest, skipped: {missing}", flush=True)
    else:
        step = max(1, len(rows) // (args.frames + 1))
        rows = rows[step::step][: args.frames]
    print(f"[data] {len(rows)} frames: {[r['id'] for r in rows]}", flush=True)

    texts = {"correct": {row["id"]: row["caption"] for row in rows}}
    if "override" in modes:
        texts["override"] = {row["id"]: args.caption_override for row in rows}
    if "shuffled" in modes:
        # Rotate a copy so a 'correct' model in the same figure keeps its own text.
        # The selected frames are hundreds to thousands of frames apart, so the
        # donor caption describes a completely different place.
        rotated = [dict(row) for row in rows]
        rotation = rotate_captions(rotated)
        texts["shuffled"] = {row["id"]: row["caption"] for row in rotated}
        print(
            f"[data] shuffled captions: rotated by {rotation['rotation_offset']} of "
            f"{len(rows)} selected frames, {rotation['self_assignments']} self-assignments",
            flush=True,
        )

    predictions: dict[str, dict[str, np.ndarray]] = {}
    for route, checkpoint, label, mode in specs:
        print(f"[model] {label}: {route} <- {checkpoint.name}  (prompt: {mode})", flush=True)
        args.route = route
        args.caption_mode = mode
        model = RouteModel(args, device, frozen_dtype)
        payload = torch.load(checkpoint.resolve(), map_location="cpu", weights_only=False)
        if payload.get("route") != route:
            raise SystemExit(f"{checkpoint}: route {payload.get('route')!r} != {route!r}")
        model.load_state_dicts(payload["state_dicts"], str(checkpoint))
        model.set_train(False)
        empty_prompt = model.encode_prompt("")
        store = {}
        with torch.no_grad():
            for row in rows:
                if mode == "empty":
                    prompt = empty_prompt
                else:
                    text = texts[mode][row["id"]]
                    if not text.strip():
                        raise SystemExit(f"--caption-mode {mode} but {row['id']} has no caption")
                    prompt = model.encode_prompt(text)
                image_tensor, _ = load_input_tensor(row, ROUTES[route][0], args)
                store[row["id"]] = model.predict_disparity(row, image_tensor, prompt).float().cpu().numpy()
        predictions[label] = store
        del model
        torch.cuda.empty_cache()

    label_h = 34 if args.panel_labels == "on" else 0
    styles = ("official", "shared") if args.style == "both" else (args.style,)
    strips: dict[str, list] = {style: [] for style in styles}
    sky_report: list[dict] = []

    for row in rows:
        thermal = np.asarray(Image.open(row["image_path"])).astype(np.float32)
        t_lo, t_hi = np.percentile(thermal, [1, 99])
        grey = Image.fromarray(
            (np.clip((thermal - t_lo) / max(t_hi - t_lo, 1e-6), 0, 1) * 255).astype(np.uint8)
        ).convert("RGB")

        gt = np.asarray(Image.open(row["depth_path"])).astype(np.float32) / args.depth_scale
        valid = gt > 1e-3
        top = slice(0, gt.shape[0] // 3)

        aligned_by_label = {}
        for _, _, label, _ in specs:
            pred = predictions[label][row["id"]]
            if pred.shape != gt.shape:
                pred = F.interpolate(
                    torch.from_numpy(pred)[None, None], gt.shape, mode="bilinear", align_corners=False
                )[0, 0].numpy()
                predictions[label][row["id"]] = pred
            aligned, _ = align_to_metric_depth(pred, gt, args.min_depth, args.max_depth)
            aligned_by_label[label] = aligned
            # Keep the aligned maps: re-colouring a figure then costs nothing,
            # instead of re-running both models on the GPU.
            (args.output_dir / "aligned").mkdir(parents=True, exist_ok=True)
            np.save(args.output_dir / "aligned" / f"{label}__{row['id']}.npy", aligned.astype(np.float32))
            # And the network's own output, before the affine fit and before the
            # [min_depth, max_depth] clamp. The aligned map is the right thing to score
            # and the wrong thing to look at: the clamp pins whole far-field regions to
            # exactly max_depth -- 7% of one frame here -- so they print as a flat slab
            # and a figure drawn from it understates what the model resolves. The
            # relative-depth figures in this literature show this array, not that one.
            (args.output_dir / "raw").mkdir(parents=True, exist_ok=True)
            np.save(args.output_dir / "raw" / f"{label}__{row['id']}.npy", pred.astype(np.float32))
            gt_top = gt[top][valid[top]]
            sky_report.append(
                {
                    "frame": row["id"],
                    "model": label,
                    "top_third_pred_median_m": float(np.median(aligned[top])),
                    "top_third_pred_p90_m": float(np.percentile(aligned[top], 90)),
                    "top_third_gt_median_m": float(np.median(gt_top)) if gt_top.size else None,
                    "top_third_gt_valid_px": int(valid[top].sum()),
                }
            )

        if "official" in styles:
            disparity = np.zeros_like(gt)
            disparity[valid] = 1.0 / gt[valid]
            shown, mask = dilate(disparity, valid, args.gt_dilate)
            panels = [
                ("Thermal input", grey),
                (f"LiDAR GT  ({valid.mean():.0%} valid, dilated)", colorize_depth_map(
                    shown, mask=torch.from_numpy(mask), reverse_color=True)),
            ]
            for _, _, label, mode in specs:
                pred = predictions[label][row["id"]]
                panel = colorize_depth_map(pred, reverse_color=True)
                panels.append((f"{label}  [{mode}]", panel))
                panel.save(args.output_dir / f"{label}_pred_demo.png")
            strips["official"].append(strip(panels, args.panel_width, label_h))

        if "shared" in styles:
            # Iris colours disparity, so the shared range lives in disparity too.
            gt_disparity = np.zeros_like(gt)
            gt_disparity[valid] = 1.0 / gt[valid]
            lo = float(np.percentile(gt_disparity[valid], 1))
            hi = float(np.percentile(gt_disparity[valid], 99))
            shown, mask = dilate(gt_disparity, valid, args.gt_dilate)
            panels = [
                ("Thermal input", grey),
                (f"LiDAR GT  {1 / hi:.0f}-{1 / lo:.0f} m ({valid.mean():.0%} valid)",
                 colour_shared(shown, lo, hi, mask=mask)),
            ]
            for _, _, label, mode in specs:
                aligned = aligned_by_label[label]
                panels.append((
                    f"{label}  [{mode}]",
                    colour_shared(1.0 / np.maximum(aligned, 1e-6), lo, hi),
                ))
            for extra_label, extra_dir in extra_panels:
                path = extra_dir / f"{row['id']}.npy"
                if not path.exists():
                    raise SystemExit(
                        f"--extra-panel {extra_label}: nothing for {row['id']} at {path}"
                    )
                metric = np.load(path).astype(np.float32)
                panels.append((extra_label,
                               colour_shared(1.0 / np.maximum(metric, 1e-6), lo, hi)))
            strips["shared"].append(strip(panels, args.panel_width, label_h))

    for style, images in strips.items():
        total_h = sum(im.height for im in images) + 12 * (len(images) - 1)
        canvas = Image.new("RGB", (images[0].width, total_h), (255, 255, 255))
        y = 0
        for im in images:
            canvas.paste(im, (0, y))
            y += im.height + 12
        suffix = "" if args.style != "both" else f"_{style}"
        out = args.output_dir / f"comparison_strip{suffix}.png"
        canvas.save(out)
        print(f"[done] {out}  ({canvas.width}x{canvas.height})")

    # The sky question is a distance question, so print the distances too: the
    # top third is where LiDAR has almost no returns, which is exactly why a
    # wrong depth there costs nothing in the metric and shows up only in colour.
    print(f"\n{'frame':>28} {'model':>28} {'top1/3 pred median':>19} {'p90':>8} {'GT median':>10} {'GT px':>7}")
    for entry in sky_report:
        gt_median = f"{entry['top_third_gt_median_m']:.1f}" if entry["top_third_gt_median_m"] else "-"
        print(
            f"{entry['frame']:>28} {entry['model']:>28} {entry['top_third_pred_median_m']:>19.1f} "
            f"{entry['top_third_pred_p90_m']:>8.1f} {gt_median:>10} {entry['top_third_gt_valid_px']:>7}"
        )
    (args.output_dir / "sky_band_depths.json").write_text(
        json.dumps(sky_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] {args.output_dir / 'sky_band_depths.json'}")


if __name__ == "__main__":
    main()
