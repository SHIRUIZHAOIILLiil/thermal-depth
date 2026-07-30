"""Let the caption supply the scale the protocol currently takes from the GT.

Relative-depth models are scored after a per-image affine fit against the ground
truth: `evaluate_sample(align="ssi")` solves (scale, shift) on the valid GT
pixels and only then measures error. That fit is where "how far away is this
scene" enters, and it enters from the GT -- not from the model, and not from the
text. `probe_caption_scale_information.py` showed the captions we already have
explain ~23% of the variance of those two parameters (and ~55% of the per-image
median GT depth), with a random-permutation control at R^2 ~ -0.08.

So this asks the useful follow-up: if the alignment came from the CAPTION instead
of from the GT, how much of the accuracy survives? Three arms on identical
predictions:

    gt_affine      per-image (scale, shift) fitted on GT  -- the official number
    global_affine  one (scale, shift) for the whole set   -- no per-image scale
    caption_affine per-image (scale, shift) predicted from the caption, ridge,
                   out-of-fold, so no frame's own alignment trains its predictor

`caption_affine` between the other two means the text carries usable metric
information -- the ceiling is `gt_affine`, and `global_affine` is what you get
with no scale cue at all. Everything runs on saved *.npy predictions: CPU only.

    python tools/probe_caption_predicted_alignment.py \
        --prediction-dir outputs/lotus_line_v2/anythermal_midas_val_full_rerun/raw_predictions \
        --align ssi
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "lotus"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ms2_eval.official_protocol import (  # noqa: E402
    fit_scale_shift,
    official_depth_errors,
    official_valid_mask,
)
from ms2_eval.resize import resize_dense_prediction  # noqa: E402

DEFAULT_MANIFEST = Path(
    "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/"
    "ms2_fixed_sequence_internvl3_8b_filtered_caption_val_rgb_depth_v1_clip75_rerun_20260714.jsonl"
)
TOKEN_RE = re.compile(r"[a-z]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ms2-root", type=Path, default=Path("/mnt/e/dataset/ms2"))
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--align", default="ssi", choices=("ssi", "ssi_disparity"))
    parser.add_argument("--output", type=Path, default=Path("outputs/route_suite/caption_predicted_alignment.json"))
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def features_for(captions: list[str], vocab: list[str]) -> np.ndarray:
    index = {token: i for i, token in enumerate(vocab)}
    matrix = np.zeros((len(captions), len(vocab) + 2), np.float64)
    for row, caption in enumerate(captions):
        tokens = TOKEN_RE.findall(caption.lower())
        for token in tokens:
            position = index.get(token)
            if position is not None:
                matrix[row, position] += 1.0
        matrix[row, -2] = len(tokens)
        matrix[row, -1] = 1.0
    return matrix


def out_of_fold_predictions(features, targets, folds, alpha, seed):
    """Ridge, out-of-fold. targets: [n, k]. Returns predictions of the same shape."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(features))
    predictions = np.zeros_like(targets)
    for fold in range(folds):
        test = order[fold::folds]
        train = np.setdiff1d(order, test, assume_unique=False)
        x = features[train]
        gram = x.T @ x
        penalty = alpha * np.eye(gram.shape[0])
        penalty[-1, -1] = 0.0
        weights = np.linalg.solve(gram + penalty, x.T @ targets[train])
        predictions[test] = features[test] @ weights
    return predictions, order


def main() -> None:
    args = parse_args()
    rows = []
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not str(row.get("caption", "")).strip():
                continue
            if (args.prediction_dir / f"{row['id']}.npy").is_file():
                rows.append(row)
    if args.limit:
        rows = rows[: args.limit]
    if len(rows) < 200:
        raise SystemExit(f"Only {len(rows)} frames have both a caption and a saved prediction")
    print(f"[data] {len(rows)} frames", flush=True)

    # pass 1: the GT-fitted alignment (the official arm) and the per-frame pixels
    scales, shifts, gt_metrics = [], [], []
    cache = []
    for index, row in enumerate(rows):
        raw = np.load(args.prediction_dir / f"{row['id']}.npy", allow_pickle=False)
        raw = np.squeeze(raw).astype(np.float32)
        gt = np.asarray(Image.open(args.ms2_root / row["depth_path"]), np.float32) / args.depth_scale
        pred = resize_dense_prediction(raw, tuple(gt.shape))
        valid = official_valid_mask(gt, args.min_depth, args.max_depth)
        if not valid.any():
            continue
        if args.align == "ssi":
            scale, shift = fit_scale_shift(pred, gt, valid)
        else:
            disparity = np.zeros_like(gt, np.float64)
            disparity[valid] = 1.0 / gt[valid].astype(np.float64)
            scale, shift = fit_scale_shift(pred, disparity.astype(np.float32), valid)
        scales.append(float(scale))
        shifts.append(float(shift))
        cache.append((row["caption"], pred, gt, valid))
        aligned = apply_alignment(pred, scale, shift, args.align)
        gt_metrics.append(
            official_depth_errors(aligned, gt, valid, min_depth=args.min_depth, max_depth=args.max_depth)["abs_rel"]
        )
        if (index + 1) % 1000 == 0:
            print(f"    {index + 1}/{len(rows)}", flush=True)

    captions = [entry[0] for entry in cache]
    counts: dict[str, int] = {}
    for caption in captions:
        for token in set(TOKEN_RE.findall(caption.lower())):
            counts[token] = counts.get(token, 0) + 1
    vocab = [t for t, c in sorted(counts.items(), key=lambda kv: -kv[1]) if c >= args.min_count]
    features = features_for(captions, vocab)
    print(f"[features] {len(vocab)} tokens", flush=True)

    targets = np.stack([np.log(np.clip(scales, 1e-12, None)), np.array(shifts)], axis=1)
    predicted, order = out_of_fold_predictions(features, targets, args.folds, args.ridge, args.seed)

    # the global arm: one alignment for everything, taken out-of-fold too
    global_pred = np.zeros_like(targets)
    for fold in range(args.folds):
        test = order[fold::args.folds]
        train = np.setdiff1d(order, test, assume_unique=False)
        global_pred[test] = targets[train].mean(axis=0)

    arms = {"gt_affine": gt_metrics, "caption_affine": [], "global_affine": []}
    for i, (_, pred, gt, valid) in enumerate(cache):
        for name, source in (("caption_affine", predicted), ("global_affine", global_pred)):
            scale = float(np.exp(source[i, 0]))
            shift = float(source[i, 1])
            aligned = apply_alignment(pred, scale, shift, args.align)
            arms[name].append(
                official_depth_errors(
                    aligned, gt, valid, min_depth=args.min_depth, max_depth=args.max_depth
                )["abs_rel"]
            )

    print(f"\n{'arm':16s} {'AbsRel':>9s}  {'vs gt_affine':>13s}")
    print("-" * 42)
    report = {"frames": len(cache), "align": args.align, "prediction_dir": str(args.prediction_dir), "arms": {}}
    baseline = float(np.mean(arms["gt_affine"]))
    for name in ("gt_affine", "caption_affine", "global_affine"):
        mean = float(np.mean(arms[name]))
        report["arms"][name] = {"abs_rel": mean, "delta_vs_gt_affine": mean - baseline}
        print(f"{name:16s} {mean:>9.4f}  {mean - baseline:>+13.4f}")
    recovered = (
        (np.mean(arms["global_affine"]) - np.mean(arms["caption_affine"]))
        / max(np.mean(arms["global_affine"]) - baseline, 1e-12)
    )
    report["caption_recovers_fraction_of_gt_alignment"] = float(recovered)
    print(f"\ncaption 挽回了 GT 对齐相对全局对齐优势的 {recovered:.1%}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Written to {args.output}")


def apply_alignment(pred, scale, shift, align):
    if align == "ssi":
        return np.clip(pred.astype(np.float64) * scale + shift, 1e-6, None)
    return 1.0 / np.clip(pred.astype(np.float64) * scale + shift, 1e-3, None)


if __name__ == "__main__":
    main()
