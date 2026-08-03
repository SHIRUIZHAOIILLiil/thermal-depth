"""Where does a route's error live? Stratified evaluation on the MS2 val set.

Global AbsRel collapses a whole image into one number, so it cannot say whether
an improvement came from near or far, from the road surface or from object
boundaries.  Iris's own claims are local ones (Findings 1-2: small objects,
ambiguous regions, occlusions), so answering them needs the error broken out by
region.

Three axes, all computed on the official protocol's aligned prediction so the
numbers stay comparable with `eval_*.json`:

  depth band     GT depth bins (near / mid / far). Where the LiDAR says the
                 pixel actually is.
  image row      vertical thirds. In a driving scene the top third is sky and
                 distant structure, the bottom third is road surface.
  structure      "boundary" = valid GT pixels whose local neighbourhood spans a
                 large depth range (a depth discontinuity, i.e. an object edge);
                 "interior" = the rest. This is the closest honest proxy for
                 Iris's small-object claim that sparse LiDAR GT allows.

Two or more checkpoints can be passed at once; the tool then also reports the
per-frame paired difference per stratum, so "epoch 5 beat epoch 1 mainly in the
far field" becomes a testable statement rather than an impression.

    python tools/analyze_route_regions.py --route b_thermal_unet \\
        --checkpoints outputs/route_suite/b_thermal_unet_20ep/epoch01_weights.pt \\
                      outputs/route_suite/b_thermal_unet_20ep/epoch05_weights.pt \\
        --output-dir outputs/route_suite/b_thermal_unet_20ep/regions --val-stride 1
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

from ms2_eval.official_protocol import fit_scale_shift, official_valid_mask  # noqa: E402
from train_route_suite import (  # noqa: E402
    ROUTES,
    RouteModel,
    load_input_tensor,
    read_manifest,
    rotate_captions,
)

DEPTH_BANDS = ((0.0, 10.0, "near <10m"), (10.0, 30.0, "mid 10-30m"), (30.0, 80.0, "far >30m"))
ROW_BANDS = ((0.0, 1 / 3, "top"), (1 / 3, 2 / 3, "middle"), (2 / 3, 1.0, "bottom"))
BOUNDARY_WINDOW = 9          # neighbourhood side length, pixels
BOUNDARY_RATIO = 1.25        # local max/min depth ratio that counts as a discontinuity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", required=True, choices=sorted(ROUTES))
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    parser.add_argument("--val-stride", type=int, default=1)
    parser.add_argument(
        "--caption-mode",
        default="empty",
        help=(
            "empty | correct | shuffled, or one per checkpoint as a comma list. A "
            "caption-trained checkpoint scored under 'empty' is being evaluated outside "
            "the input mode it was trained in, so the empty-vs-caption region comparison "
            "wants 'empty,correct' -- each arm in its own mode, same frames, so the "
            "paired bootstrap stays valid. 'shuffled' keeps the text distribution and "
            "breaks only the image-text correspondence."
        ),
    )
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frozen-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--input-max-edge", type=int, default=0)
    parser.add_argument("--gt-decode-fp32", action="store_true", default=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def boundary_mask(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Valid pixels sitting on a depth discontinuity.

    Sparse LiDAR makes a gradient operator meaningless, so instead of
    differentiating we ask a rank question over a window: does the local
    neighbourhood of valid samples span more than BOUNDARY_RATIO in depth?
    Implemented with max/min pooling so it costs one pass on the GPU-free path.
    """
    filled_high = np.where(valid, depth, -np.inf)
    filled_low = np.where(valid, depth, np.inf)
    tensor_high = torch.from_numpy(filled_high)[None, None]
    tensor_low = torch.from_numpy(filled_low)[None, None]
    pad = BOUNDARY_WINDOW // 2
    local_max = F.max_pool2d(tensor_high, BOUNDARY_WINDOW, stride=1, padding=pad)[0, 0].numpy()
    local_min = -F.max_pool2d(-tensor_low, BOUNDARY_WINDOW, stride=1, padding=pad)[0, 0].numpy()
    span = np.zeros_like(depth)
    usable = valid & np.isfinite(local_max) & np.isfinite(local_min) & (local_min > 0)
    span[usable] = local_max[usable] / np.maximum(local_min[usable], 1e-6)
    return valid & (span > BOUNDARY_RATIO)


def strata_for(gt: np.ndarray, valid: np.ndarray) -> dict[str, np.ndarray]:
    """name -> boolean mask (already intersected with `valid`)."""
    height = gt.shape[0]
    rows = np.arange(height)[:, None] / height
    strata: dict[str, np.ndarray] = {"all": valid}
    for low, high, name in DEPTH_BANDS:
        strata[f"depth/{name}"] = valid & (gt >= low) & (gt < high)
    for low, high, name in ROW_BANDS:
        band = (rows >= low) & (rows < high)
        strata[f"row/{name}"] = valid & np.broadcast_to(band, gt.shape)
    edge = boundary_mask(gt, valid)
    strata["structure/boundary"] = edge
    strata["structure/interior"] = valid & ~edge
    return strata


@torch.no_grad()
def score_checkpoint(model: RouteModel, checkpoint: Path, rows: list[dict], prompt_for, args) -> dict:
    payload = torch.load(checkpoint.resolve(), map_location="cpu", weights_only=False)
    if payload.get("route") != args.route:
        raise SystemExit(f"{checkpoint}: route {payload.get('route')!r} != --route {args.route!r}")
    model.load_state_dicts(payload["state_dicts"], str(checkpoint))
    model.set_train(False)

    per_frame: dict[str, list[float]] = {}
    pixel_totals: dict[str, list[float]] = {}
    for index, row in enumerate(rows):
        image_tensor, _ = load_input_tensor(row, model.modality, args)
        prediction = model.predict_disparity(row, image_tensor, prompt_for(index))
        gt = np.asarray(Image.open(row["depth_path"]), dtype=np.float32) / args.depth_scale
        pred = prediction[None, None]
        if pred.shape[-2:] != gt.shape:
            pred = F.interpolate(pred, gt.shape, mode="bilinear", align_corners=False)
        raw = pred[0, 0].float().cpu().numpy()

        valid = official_valid_mask(gt, args.min_depth, args.max_depth)
        if not valid.any():
            continue
        # official ssi_disparity alignment, then back to metric depth
        gt_disparity = np.zeros_like(gt, np.float64)
        gt_disparity[valid] = 1.0 / gt[valid].astype(np.float64)
        scale, shift = fit_scale_shift(raw, gt_disparity.astype(np.float32), valid)
        aligned = 1.0 / np.clip(raw.astype(np.float64) * scale + shift, 1e-3, None)
        aligned = np.clip(aligned, args.min_depth, args.max_depth)
        error = np.abs(aligned - gt) / np.maximum(gt, 1e-6)

        for name, mask in strata_for(gt, valid).items():
            count = int(mask.sum())
            if count == 0:
                continue
            per_frame.setdefault(name, []).append(float(error[mask].mean()))
            pixel_totals.setdefault(name, []).append(float(count))
        if (index + 1) % 500 == 0:
            print(f"    {index + 1}/{len(rows)}", flush=True)

    return {
        "per_frame": per_frame,
        "pixel_counts": {name: float(np.sum(values)) for name, values in pixel_totals.items()},
    }


def paired_difference(left: list[float], right: list[float], bootstrap: int, rng) -> dict:
    length = min(len(left), len(right))
    difference = np.array(right[:length]) - np.array(left[:length])
    indices = rng.integers(0, length, size=(bootstrap, length))
    samples = difference[indices].mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "mean_difference": float(difference.mean()),
        "ci95": [float(low), float(high)],
        "significant": bool(low > 0 or high < 0),
        "right_win_rate": float((difference < 0).mean()),
        "n": int(length),
    }


def main() -> None:
    args = parse_args()
    args.val_caption_mode = args.caption_mode
    args.gt_min_depth, args.gt_max_depth = 0.1, 80.0

    device = torch.device(args.device)
    frozen_dtype = torch.float16 if args.frozen_dtype == "fp16" else torch.float32
    args.output_dir.mkdir(parents=True, exist_ok=True)

    modality = ROUTES[args.route][0]
    rows = read_manifest(args.val_manifest, args.ms2_root, modality, split=None)
    rows = rows[:: max(1, args.val_stride)]
    print(f"[data] {len(rows)} val frames", flush=True)

    modes = [token.strip() for token in args.caption_mode.split(",") if token.strip()]
    if len(modes) == 1:
        modes *= len(args.checkpoints)
    if len(modes) != len(args.checkpoints):
        raise SystemExit(
            f"--caption-mode takes one value or one per checkpoint; got {len(modes)} "
            f"for {len(args.checkpoints)} checkpoints"
        )
    unknown = sorted(set(modes) - {"empty", "correct", "shuffled"})
    if unknown:
        raise SystemExit(f"Unknown caption mode(s): {unknown}")
    args.val_caption_mode = modes[0]

    captions = {"empty": None, "correct": None, "shuffled": None}
    rotation = None
    if set(modes) - {"empty"}:
        missing = [row["id"] for row in rows if not row["caption"].strip()]
        if missing:
            raise SystemExit(
                f"caption mode {sorted(set(modes))} but {len(missing)} rows lack captions "
                f"(first: {missing[:3]})"
            )
        captions["correct"] = [row["caption"] for row in rows]
        if "shuffled" in modes:
            # rotate a copy: the 'correct' arm in the same run must keep its own text
            rotated_rows = [dict(row) for row in rows]
            rotation = rotate_captions(rotated_rows)
            captions["shuffled"] = [row["caption"] for row in rotated_rows]
            print(
                f"[data] shuffled captions: rotated by {rotation['rotation_offset']} frames, "
                f"{rotation['self_assignments']} self-assignments",
                flush=True,
            )

    model = RouteModel(args, device, frozen_dtype)
    empty_prompt = model.encode_prompt("")

    def make_prompt_for(mode):
        if mode == "empty":
            return lambda index: empty_prompt
        # ~450 MB of VRAM to cache 3k prompts, so encode per frame instead; the
        # text encoder is negligible next to the U-Net forward.
        texts = captions[mode]
        return lambda index: model.encode_prompt(texts[index])

    # Two arms of the same experiment carry the same file name
    # (…/b_20ep/epoch05_weights.pt vs …/b_caption_20ep/epoch05_weights.pt), so
    # keying on the stem alone would silently drop one of them.
    stems = [checkpoint.stem for checkpoint in args.checkpoints]
    labels = [
        stem if stems.count(stem) == 1 else f"{checkpoint.parent.name}/{stem}"
        for checkpoint, stem in zip(args.checkpoints, stems)
    ]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"Checkpoint labels are not unique: {labels}")

    results = {}
    for checkpoint, mode, label in zip(args.checkpoints, modes, labels):
        print(f"[scan] {label}  (prompt: {mode})", flush=True)
        results[label] = score_checkpoint(model, checkpoint, rows, make_prompt_for(mode), args)
        results[label]["caption_mode"] = mode
        results[label]["checkpoint"] = str(checkpoint)

    names = list(results)
    strata = sorted(results[names[0]]["per_frame"])
    rng = np.random.default_rng(args.seed)

    # Labels can be long and share a prefix (…_20ep vs …_caption_20ep), which a
    # truncated column header would hide; alias them and print the key.
    aliases = {name: chr(ord("A") + index) for index, name in enumerate(names)}
    print("")
    for name in names:
        print(f"[{aliases[name]}] {name}  (prompt: {results[name]['caption_mode']})")
    print(f"\n{'stratum':24s} " + "".join(f"{aliases[name]:>16s}" for name in names))
    print("-" * (24 + 16 * len(names)))
    for stratum in strata:
        cells = "".join(f"{np.mean(results[name]['per_frame'][stratum]):>16.4f}" for name in names)
        print(f"{stratum:24s} {cells}")

    report: dict = {
        "strata": {},
        "checkpoints": names,
        "frames": len(rows),
        "route": args.route,
        "val_manifest": str(args.val_manifest),
        "caption_mode": {name: results[name]["caption_mode"] for name in results},
        "caption_rotation": rotation,
    }
    for stratum in strata:
        entry = {
            "means": {name: float(np.mean(results[name]["per_frame"][stratum])) for name in names},
            "pixels": results[names[0]]["pixel_counts"].get(stratum),
            "paired": {},
        }
        for i in range(len(names) - 1):
            for j in range(i + 1, len(names)):
                key = f"{names[j]} - {names[i]}"
                entry["paired"][key] = paired_difference(
                    results[names[i]]["per_frame"][stratum],
                    results[names[j]]["per_frame"][stratum],
                    args.bootstrap,
                    rng,
                )
        report["strata"][stratum] = entry

    if len(names) >= 2:
        print("\npaired differences (AbsRel, negative = the right-hand checkpoint better)")
        for stratum in strata:
            for key, stats in report["strata"][stratum]["paired"].items():
                marker = "*" if stats["significant"] else " "
                right, _, left = key.partition(" - ")
                short = f"{aliases.get(right, right)} - {aliases.get(left, left)}"
                print(
                    f"  {stratum:24s} {short:12s} {stats['mean_difference']:+.5f}{marker} "
                    f"CI[{stats['ci95'][0]:+.5f},{stats['ci95'][1]:+.5f}] win {stats['right_win_rate']:.1%}"
                )

    path = args.output_dir / "region_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
