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


def main() -> None:
    args = parse_args()
    spec = MODELS[args.model]

    samples, manifest_info = load_manifest(args.manifest, args.data_root)
    if args.limit:
        samples = samples[:args.limit]
    print(f"[data] {len(samples)} 帧   manifest {manifest_info['sha256'][:12]}", flush=True)
    print(f"[model] {spec['hf_id']}   输出：{spec['output']}   评估用 --align {spec['align']}",
          flush=True)
    print(f"[input] 热像转 8 位：{args.stretch}", flush=True)

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    processor = AutoImageProcessor.from_pretrained(spec["hf_id"])
    model = AutoModelForDepthEstimation.from_pretrained(spec["hf_id"]).to(args.device).eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for index, sample in enumerate(samples):
        image = thermal_to_rgb(sample.thermal_path, args.stretch)
        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(args.device)
            prediction = model(**inputs).predicted_depth[0].float().cpu().numpy()
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
        "hf_id": spec["hf_id"],
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
