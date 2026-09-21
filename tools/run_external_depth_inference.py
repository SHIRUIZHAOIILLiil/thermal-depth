"""Run a published RGB depth model over MS2 thermal frames and dump raw predictions.

The official evaluator (``tools/run_official_ms2_evaluation.py``) reads a
directory of ``<sample_id>.npy`` raw outputs, so an external model only has to
produce those: everything after -- masking, alignment, clamping, the metric
formulas, the macro aggregation -- is then the same code path our own model is
scored through. That is the point. A comparison table assembled from numbers
each author computed their own way is not a comparison.

Two decisions are deliberate and both are recorded in the provenance JSON.

**The thermal stretch defaults to percentile.** MS2 thermal is 16-bit and these
models want 8-bit RGB, so something has to do the conversion. Our own pipeline
used min-max, and measuring it showed a median frame loses 14.3 points of
adjacent-difference resolution that way. Handing the baselines the degraded
version while our line runs on the fixed one would manufacture a win. Both
modes are available so the gap can be reported, but the comparison runs on
percentile.

**Alignment is not chosen here.** It belongs to the evaluator, per model, in
that model's native output space -- depth-space affine and disparity-space
affine are different function families, and getting it wrong cost us a factor
of three and a half on Marigold once already.

    python tools/run_external_depth_inference.py --model depth_anything_v2 \
        --manifest $SCRATCH/manifests/ms2_test_official9.jsonl \
        --data-root $SCRATCH/data/ms2 \
        --out-dir $SCRATCH/runs/external/da2/predictions --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ms2_eval.io import load_manifest  # noqa: E402
from ms2_eval.resize import resize_dense_prediction  # noqa: E402


# What each model natively emits, so the evaluator can be told the right
# alignment space rather than having it guessed at the table-assembly stage.
MODELS = {
    "depth_anything_v2": {
        "hf_id": "depth-anything/Depth-Anything-V2-Large-hf",
        "output": "affine-invariant inverse depth (disparity)",
        "align": "ssi_disparity",
    },
    "pixel_perfect_depth": {
        # NeurIPS 2025, 500M, diffusion in pixel space with no VAE. Its repo is
        # cloned rather than pulled from the hub: the checkpoint is a bare
        # state_dict and the model class lives in that tree.
        "repo_env": "PPD_REPO",
        # The paper normalises its target as log depth between the 2nd and 98th
        # percentiles, and run.py min-max normalises the output for a colormap,
        # so neither says what infer_image returns. Left unset on purpose --
        # --probe-output-space measures it on train frames.
        "output": "unstated; measure it",
        "align": None,
    },
}

# Which spaces a raw output can be affine in, and the evaluator mode for each.
# A model is scored in the space its output is affine in; scoring across
# families is what reported Marigold at 0.265.
OUTPUT_SPACES = {
    "depth": ("ssi", lambda gt: gt),
    "disparity": ("ssi_disparity", lambda gt: 1.0 / np.maximum(gt, 1e-6)),
    "log_depth": ("ssi_log", lambda gt: np.log(np.maximum(gt, 1e-6))),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--stretch", default="percentile", choices=("percentile", "minmax"),
                        help="16-bit thermal -> 8-bit. See the module docstring "
                             "for why this is not min-max by default.")
    parser.add_argument("--limit", type=int, default=0, help="Smoke on the first N frames.")
    parser.add_argument(
        "--weights", type=Path,
        help="Local fine-tuned weights to load instead of the hub id -- a "
             "directory written by save_pretrained. Without it this evaluates "
             "the released model, which for a thermal table means a model that "
             "has never seen thermal, and the two must never share a row.",
    )
    parser.add_argument("--probe-output-space", action="store_true",
                        help="Measure which space the raw output is affine in and stop. "
                             "⚠️ Point this at a TRAIN manifest: it fits to GT, and "
                             "choosing an evaluation mode by looking at test would be "
                             "reading test GT to configure the evaluation.")
    parser.add_argument("--ppd-checkpoint", type=Path,
                        help="pixel_perfect_depth: the ppd.pth state_dict.")
    parser.add_argument("--ppd-semantics", type=Path,
                        help="pixel_perfect_depth: depth_anything_v2_vitl.pth.")
    parser.add_argument("--sampling-steps", type=int, default=4,
                        help="pixel_perfect_depth: run.py's default is 4.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def thermal_to_rgb(path: str, stretch: str) -> Image.Image:
    raw = np.asarray(Image.open(path), dtype=np.float32)
    if stretch == "percentile":
        low, high = (float(v) for v in np.percentile(raw, (1.0, 99.0)))
    else:
        low, high = float(raw.min()), float(raw.max())
    if high <= low:
        eight = np.zeros(raw.shape, np.uint8)
    else:
        eight = np.clip((raw - low) / (high - low) * 255.0, 0, 255).round().astype(np.uint8)
    return Image.fromarray(np.repeat(eight[:, :, None], 3, axis=2), mode="RGB")


def load_predictor(args: argparse.Namespace, spec: dict):
    """Return a callable mapping a PIL RGB image to a raw HxW float array."""
    if args.model == "depth_anything_v2":
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        weights = str(args.weights) if args.weights else spec["hf_id"]
        processor = AutoImageProcessor.from_pretrained(spec["hf_id"])
        model = AutoModelForDepthEstimation.from_pretrained(weights)
        if args.weights:
            print(f"[weights] {weights}  (预处理仍取自 {spec['hf_id']})", flush=True)
        model = model.to(args.device).eval()

        def predict(image):
            with torch.no_grad():
                inputs = processor(images=image, return_tensors="pt").to(args.device)
                return model(**inputs).predicted_depth[0].float().cpu().numpy()
        return predict

    if args.model == "pixel_perfect_depth":
        repo = os.environ.get(spec["repo_env"])
        if not repo:
            raise SystemExit(f"Set {spec['repo_env']} to the cloned pixel-perfect-depth tree")
        sys.path.insert(0, repo)
        from ppd.pixel_perfect_depth import PixelPerfectDepth  # type: ignore

        model = PixelPerfectDepth(semantics_model="da2",
                                  semantics_pth=str(args.ppd_semantics),
                                  sampling_steps=args.sampling_steps)
        model.load_state_dict(torch.load(args.ppd_checkpoint, map_location="cpu"),
                              strict=False)
        model = model.to(args.device).eval()

        def predict(image):
            # infer_image takes an OpenCV-style BGR array, as run.py feeds it.
            bgr = np.asarray(image)[:, :, ::-1].copy()
            with torch.no_grad():
                depth, _ = model.infer_image(bgr)
            array = depth.squeeze().float().cpu().numpy()
            # run.py min-max normalises here for its colormap. Not done: that is
            # a display step, and the evaluator wants what the network emitted.
            return array
        return predict

    raise SystemExit(f"No loader for {args.model!r}")


def probe_output_space(predict, samples, args) -> None:
    """Which space is the raw output affine in -- depth, disparity, or log depth?

    Every model here emits an unscaled map, and the evaluator has to fit two
    parameters in the space that map is affine in. Reading it off the paper is
    how Marigold was reported at 0.265: its own docs describe a depth output and
    the fit was run in the wrong family anyway.

    So it is measured. For each frame the raw output is regressed against each
    candidate transform of GT on the lidar pixels, and the space reported is the
    one whose residual is smallest -- per frame, then a vote, so one unusual
    frame cannot decide it.
    """
    from PIL import Image as PILImage

    wins = {name: 0 for name in OUTPUT_SPACES}
    residuals = {name: [] for name in OUTPUT_SPACES}
    for sample in samples:
        gt = np.asarray(PILImage.open(sample.thermal_gt_path), np.float32) / 256.0
        valid = np.isfinite(gt) & (gt > 1e-3) & (gt < 80.0)
        if not valid.any():
            continue
        raw = predict(thermal_to_rgb(sample.thermal_path, args.stretch))
        raw = resize_dense_prediction(raw, gt.shape[:2])
        x = raw[valid].astype(np.float64)
        best, best_residual = None, np.inf
        for name, (_mode, transform) in OUTPUT_SPACES.items():
            y = transform(gt[valid].astype(np.float64))
            if not np.isfinite(y).all():
                continue
            slope, intercept = np.polyfit(x, y, 1)
            # Normalised so the three spaces are comparable despite living on
            # different scales: a residual as a fraction of the target's spread.
            spread = y.std() or 1.0
            residual = float(np.sqrt(np.mean((slope * x + intercept - y) ** 2)) / spread)
            residuals[name].append(residual)
            if residual < best_residual:
                best, best_residual = name, residual
        if best:
            wins[best] += 1

    print()
    print(f"在 {sum(wins.values())} 帧上测量原始输出仿射于哪个空间")
    print()
    print(f"{'空间':<12}{'评估模式':<16}{'相对残差中位':>14}{'最优帧数':>10}")
    for name, (mode, _t) in OUTPUT_SPACES.items():
        median = float(np.median(residuals[name])) if residuals[name] else float("nan")
        print(f"{name:<12}{mode:<16}{median:>14.4f}{wins[name]:>10}")
    winner = max(wins, key=wins.get)
    print()
    print(f"判定：{winner}  ->  评估时用 --align {OUTPUT_SPACES[winner][0]}")
    print("⚠️ 这是在给定的 manifest 上拟合出来的。若它不是 train 划分，这个判定"
          "就是拿 test GT 配置评估，不能用。")


def main() -> None:
    args = parse_args()
    spec = MODELS[args.model]

    samples, manifest_info = load_manifest(args.manifest, args.data_root)
    if args.limit:
        samples = samples[:args.limit]
    print(f"[data] {len(samples)} 帧   manifest {manifest_info['sha256'][:12]}", flush=True)
    align = spec["align"] or "（未定，先跑 --probe-output-space）"
    print(f"[model] {args.model}   输出：{spec['output']}   评估用 --align {align}",
          flush=True)
    print(f"[input] 热像转 8 位：{args.stretch}", flush=True)

    predict = load_predictor(args, spec)

    if args.probe_output_space:
        probe_output_space(predict, samples, args)
        return

    if spec["align"] is None:
        raise SystemExit(
            f"{args.model} 的输出空间还没测过。先在 train 划分上跑 "
            "--probe-output-space，把结果填进 MODELS['align']，再出预测。"
            "猜错空间不会报错，只会给出一个错三十倍、看着合理的数。")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, sample in enumerate(samples):
        prediction = predict(thermal_to_rgb(sample.thermal_path, args.stretch))
        # The evaluator refuses a shape mismatch rather than resizing silently,
        # so the prediction goes back to the GT grid here.
        gt_hw = np.asarray(Image.open(sample.thermal_gt_path)).shape[:2]
        np.save(args.out_dir / f"{sample.sample_id}.npy",
                resize_dense_prediction(prediction, gt_hw))
        written += 1
        if index % 500 == 0:
            print(f"  {index}/{len(samples)}", flush=True)

    provenance = {
        "model": args.model,
        "hf_id": spec.get("hf_id"),
        "weights": str(args.weights) if args.weights else None,
        "native_output": spec["output"],
        "evaluator_align": spec["align"],
        "thermal_stretch": args.stretch,
        "frames": written,
        "manifest": manifest_info,
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "note": "RGB-trained model applied to thermal without adaptation.",
    }
    (args.out_dir.parent / "inference_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] {written} 帧 -> {args.out_dir}")
    print(f"[next] 评估时必须带 --align {spec['align']}")


if __name__ == "__main__":
    main()
