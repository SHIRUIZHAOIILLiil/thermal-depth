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
from train_route_suite import ROUTES, RouteModel, load_input_tensor, read_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True, metavar="ROUTE:CHECKPOINT")
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
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    args.val_caption_mode = "empty"
    args.caption_mode = "empty"
    args.gt_decode_fp32 = True
    args.input_max_edge = 0
    args.gt_min_depth, args.gt_max_depth = 0.1, 80.0

    specs = []
    for entry in args.models:
        route, _, checkpoint = entry.partition(":")
        if route not in ROUTES or not checkpoint:
            raise SystemExit(f"Bad --models entry {entry!r}; expected ROUTE:CHECKPOINT")
        specs.append((route, Path(checkpoint)))

    device = torch.device(args.device)
    frozen_dtype = torch.float16 if args.frozen_dtype == "fp16" else torch.float32
    args.output_dir.mkdir(parents=True, exist_ok=True)

    modality = ROUTES[specs[0][0]][0]
    rows = read_manifest(args.val_manifest, args.ms2_root, modality, split=None, check_files=False)
    if args.frame_ids:
        wanted = set(args.frame_ids)
        rows = [r for r in rows if r["id"] in wanted]
    else:
        step = max(1, len(rows) // (args.frames + 1))
        rows = rows[step::step][: args.frames]
    print(f"[data] {len(rows)} frames: {[r['id'] for r in rows]}", flush=True)

    predictions: dict[str, dict[str, np.ndarray]] = {}
    for route, checkpoint in specs:
        print(f"[model] {route} <- {checkpoint.name}", flush=True)
        args.route = route
        model = RouteModel(args, device, frozen_dtype)
        payload = torch.load(checkpoint.resolve(), map_location="cpu", weights_only=False)
        if payload.get("route") != route:
            raise SystemExit(f"{checkpoint}: route {payload.get('route')!r} != {route!r}")
        for name, module in model.trainable_modules().items():
            module.load_state_dict(payload["state_dicts"][name], strict=True)
        model.set_train(False)
        prompt = model.encode_prompt("")
        store = {}
        with torch.no_grad():
            for row in rows:
                image_tensor, _ = load_input_tensor(row, ROUTES[route][0], args)
                store[row["id"]] = model.predict_disparity(row, image_tensor, prompt).float().cpu().numpy()
        predictions[route] = store
        del model
        torch.cuda.empty_cache()

    strips = []
    for row in rows:
        thermal = np.asarray(Image.open(row["image_path"])).astype(np.float32)
        lo, hi = np.percentile(thermal, [1, 99])
        grey = Image.fromarray((np.clip((thermal - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)).convert("RGB")

        gt = np.asarray(Image.open(row["depth_path"])).astype(np.float32) / args.depth_scale
        valid = gt > 1e-3
        disparity = np.zeros_like(gt)
        disparity[valid] = 1.0 / gt[valid]
        shown, mask = dilate(disparity, valid, args.gt_dilate)
        gt_panel = colorize_depth_map(shown, mask=torch.from_numpy(mask), reverse_color=True)

        panels = [("Thermal input", grey),
                  (f"LiDAR GT  ({valid.mean():.0%} valid, dilated)", gt_panel)]
        for route, _ in specs:
            pred = predictions[route][row["id"]]
            panels.append((f"Route {route.split('_')[0]}  prediction", colorize_depth_map(pred, reverse_color=True)))
            single = colorize_depth_map(pred, reverse_color=True)
            single.save(args.output_dir / f"{route.split('_')[0]}_pred_demo.png")
        strips.append(strip(panels, args.panel_width))

    total_h = sum(im.height for im in strips) + 12 * (len(strips) - 1)
    canvas = Image.new("RGB", (strips[0].width, total_h), (255, 255, 255))
    y = 0
    for im in strips:
        canvas.paste(im, (0, y))
        y += im.height + 12
    out = args.output_dir / "comparison_strip.png"
    canvas.save(out)
    print(f"[done] {out}  ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
