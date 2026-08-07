"""Segment sky in MS2 thermal frames, and check the masks against the lidar.

Jasmine's sky loss (appendix C.3) needs a sky mask; it takes one from Mask2Former
run on RGB. Our input is thermal, and MS2's RGB sits at a different viewpoint, so
the mask has to come from the thermal image itself -- which Mask2Former has never
been trained on. Whether that works is an empirical question, so this tool
answers it before any training run depends on it.

The gate is physical rather than visual: **sky returns no lidar**. A mask pixel
that carries a return is a mask pixel that is not sky. So for every frame we
report how much of the predicted sky carries GT, and how far that GT is. A good
mask overlaps the valid mask almost nowhere; a mask that grabs buildings or road
will light this up immediately, without anyone squinting at overlays.

The thermal-to-uint8 conversion is the shared one the training path uses, so the
segmenter sees exactly the image the depth model sees.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.anythermal_lotus_v2 import thermal_to_lotus_input  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="facebook/mask2former-swin-large-cityscapes-semantic",
        help="Cityscapes-trained: its label set has 'sky', and its domain is driving.",
    )
    parser.add_argument("--max-samples", type=int, default=100,
                        help="0 = every frame. Uniformly spaced, so a subset still "
                             "covers the whole sequence.")
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--horizon-row", type=int, default=152,
                        help="Optical axis row from calib.npy (K_thrL cy = 151.63). "
                             "Sky below it is suspicious.")
    parser.add_argument("--preview-frames", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--save-masks", action="store_true",
                        help="Write masks/<id>.png (0/255). Off by default so a "
                             "quality probe does not litter scratch with inodes.")
    parser.add_argument("--save-labels", action="store_true",
                        help="Also write labels/<id>.png -- the FULL class map as "
                             "uint8 class ids, not just sky. Needed to ask where a "
                             "caption's effect lands: the caption names objects, and "
                             "a whole-image mean cannot see a correction confined to "
                             "one of them. id2label is recorded in the report.")
    return parser.parse_args()


def read_manifest(path: Path, max_samples: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            depth = row.get("thermal_depth_path") or row.get("depth_path")
            if not depth or "/thr/" not in str(depth).replace("\\", "/"):
                raise ValueError(f"Row {row.get('id')} lacks thermal-view GT")
            rows.append({"id": row["id"], "thermal_path": row["thermal_path"],
                         "depth_path": depth})
    if not rows:
        raise ValueError("Manifest is empty")
    if 0 < max_samples < len(rows):
        rows = [rows[int(i)] for i in np.linspace(0, len(rows) - 1, max_samples, dtype=int)]
    return rows


def find_sky_label(id2label: dict) -> int:
    matches = [int(k) for k, v in id2label.items() if str(v).strip().lower() == "sky"]
    if not matches:
        raise SystemExit(
            f"No 'sky' class in this checkpoint's label set: {sorted(id2label.values())}"
        )
    return matches[0]


def main() -> None:
    args = parse_args()
    # Resolve the class the checkpoint declares rather than naming one: Mask2Former
    # lives under universal segmentation and SegFormer under semantic, so no single
    # Auto* class covers both, and a fallback checkpoint should need only --model.
    # Every such processor exposes post_process_semantic_segmentation, which is all
    # this tool uses.
    import transformers
    from transformers import AutoConfig, AutoImageProcessor

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest, args.max_samples)
    device = torch.device(args.device)

    # Say where we are before the slow part: a failure inside from_pretrained is
    # otherwise indistinguishable from a failure importing torch.
    print(f"[init] loading {args.model} (local_files_only={args.local_files_only})", flush=True)
    config = AutoConfig.from_pretrained(args.model, local_files_only=args.local_files_only)
    architectures = list(getattr(config, "architectures", None) or [])
    if not architectures:
        raise SystemExit(f"{args.model}: config declares no architecture to load")
    model_class = getattr(transformers, architectures[0], None)
    if model_class is None:
        raise SystemExit(
            f"{args.model} declares {architectures[0]}, which transformers "
            f"{transformers.__version__} does not provide."
        )
    processor = AutoImageProcessor.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    model = model_class.from_pretrained(
        args.model, local_files_only=args.local_files_only
    ).to(device).eval()
    sky_id = find_sky_label(model.config.id2label)
    print(f"[init] {type(model).__name__}: sky = class {sky_id}, {len(rows)} frames", flush=True)

    mask_dir = args.output_dir / "masks"
    if args.save_masks:
        mask_dir.mkdir(exist_ok=True)
    label_dir = args.output_dir / "labels"
    if args.save_labels:
        label_dir.mkdir(exist_ok=True)

    records: list[dict] = []
    previews: list[tuple[np.ndarray, np.ndarray]] = []
    for index, row in enumerate(rows):
        thermal = thermal_to_lotus_input(args.ms2_root / row["thermal_path"], processing_res=0)
        image = thermal.rgb_image
        height, width = image.size[1], image.size[0]
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        labels = processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(height, width)]
        )[0].cpu().numpy()
        sky = labels == sky_id

        gt = np.asarray(Image.open(args.ms2_root / row["depth_path"]), dtype=np.float32)
        gt = gt / args.depth_scale
        valid = (gt > args.min_depth) & (gt < args.max_depth) & np.isfinite(gt)

        overlap = sky & valid
        rows_of_sky = np.nonzero(sky.any(axis=1))[0]
        records.append({
            "id": row["id"],
            "sky_fraction": float(sky.mean()),
            "sky_pixels": int(sky.sum()),
            # The gate: sky should carry no return at all.
            "sky_with_gt_fraction": float(overlap.sum() / max(sky.sum(), 1)),
            "sky_gt_median_m": float(np.median(gt[overlap])) if overlap.any() else None,
            "sky_below_horizon_fraction": float(
                sky[args.horizon_row:].sum() / max(sky.sum(), 1)
            ),
            "sky_lowest_row": int(rows_of_sky.max()) if rows_of_sky.size else None,
            # How much of the unsupervised area the mask actually claims.
            "unsupervised_covered_fraction": float(
                (sky & ~valid).sum() / max((~valid).sum(), 1)
            ),
        })
        if args.save_masks:
            Image.fromarray((sky * 255).astype(np.uint8)).save(mask_dir / f"{row['id']}.png")
        if args.save_labels:
            if labels.max() > 255:
                raise SystemExit(f"{len(model.config.id2label)} classes will not fit in uint8")
            Image.fromarray(labels.astype(np.uint8)).save(label_dir / f"{row['id']}.png")
        if len(previews) < args.preview_frames and index % max(1, len(rows) // max(args.preview_frames, 1)) == 0:
            previews.append((np.asarray(image), sky))
        if (index + 1) % 25 == 0:
            print(f"[{index + 1}/{len(rows)}]", flush=True)

    fractions = np.array([r["sky_fraction"] for r in records])
    with_gt = np.array([r["sky_with_gt_fraction"] for r in records])
    below = np.array([r["sky_below_horizon_fraction"] for r in records])
    covered = np.array([r["unsupervised_covered_fraction"] for r in records])
    empty = int((fractions == 0).sum())

    print(f"\n=== {len(records)} frames ===")
    print(f"  sky fraction of image        mean {fractions.mean()*100:6.2f}%  "
          f"median {np.median(fractions)*100:6.2f}%  (frames with no sky: {empty})")
    print(f"  GATE  sky pixels carrying GT mean {with_gt.mean()*100:6.2f}%  "
          f"max {with_gt.max()*100:6.2f}%   <- want ~0")
    print(f"  sky below the horizon row    mean {below.mean()*100:6.2f}%   <- want ~0")
    print(f"  of all unsupervised pixels,  {covered.mean()*100:6.2f}% claimed as sky")
    gt_depths = [r["sky_gt_median_m"] for r in records if r["sky_gt_median_m"]]
    if gt_depths:
        print(f"  where sky does carry GT, median depth {np.median(gt_depths):.1f} m "
              f"(near = the mask grabbed structure)")

    verdict = "PASS" if with_gt.mean() < 0.01 and below.mean() < 0.02 else "SUSPECT"
    print(f"\n  verdict: {verdict}  "
          f"(PASS needs <1% of sky pixels carrying GT and <2% below the horizon)")

    (args.output_dir / "sky_mask_report.json").write_text(
        json.dumps({"model": args.model, "sky_class": sky_id, "frames": len(records),
                    "id2label": {int(k): v for k, v in model.config.id2label.items()},
                    "manifest": str(args.manifest), "verdict": verdict,
                    "summary": {"sky_fraction_mean": float(fractions.mean()),
                                "sky_with_gt_fraction_mean": float(with_gt.mean()),
                                "sky_with_gt_fraction_max": float(with_gt.max()),
                                "sky_below_horizon_fraction_mean": float(below.mean()),
                                "unsupervised_covered_fraction_mean": float(covered.mean()),
                                "frames_without_sky": empty},
                    "per_frame": records}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    if previews:
        panels = []
        for image, sky in previews:
            shown = image.astype(np.float32).copy()
            shown[sky] = 0.5 * shown[sky] + 0.5 * np.array([255.0, 40.0, 40.0])
            panels.append(np.clip(shown, 0, 255).astype(np.uint8))
        Image.fromarray(np.concatenate(panels, axis=0)).save(
            args.output_dir / "sky_mask_preview.png"
        )
    print(f"\n-> {args.output_dir}/sky_mask_report.json")


if __name__ == "__main__":
    main()
