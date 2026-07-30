"""Does a caption know how far away the scene is? A probe on the discarded axis.

Every number in this project is scale-shift invariant: `masked_ssi_l1` fits a
per-image affine (scale, shift) against the GT disparity, detaches it, and only
then takes the loss; the official protocol does the same before scoring. So the
two per-image degrees of freedom that carry "how far away everything is" are
removed by construction -- from the loss AND from the metric.

That matters because the language-and-depth literature puts language's value
mostly in exactly that axis (WorDepth CVPR'24, VGLD, "resolving scale
ambiguities through language descriptions"), and Iris's own largest gains are on
the two indoor datasets, where a sentence pins the range hardest.

This probe asks whether the captions we already have carry that information:
regress the per-image alignment parameters (recorded by the official evaluator)
and the per-image median GT depth from the caption text, cross-validated, with
two controls -- shuffled captions and caption length alone.

    R^2 near 0 for real captions  -> the text has no scale information; a
                                     metric-space loss would not unlock anything.
    R^2 clearly > 0, shuffled ~ 0 -> our objective throws away a real signal,
                                     and a scale-preserving loss is worth a day.

Pure CPU, no GPU, no sklearn (ridge has a closed form).

    python tools/probe_caption_scale_information.py \
        --per-sample outputs/route_suite/b_thermal_unet_20ep/eval_full_ep05_per_sample.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_MANIFEST = Path(
    "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/"
    "ms2_fixed_sequence_internvl3_8b_filtered_caption_val_rgb_depth_v1_clip75_rerun_20260714.jsonl"
)
TOKEN_RE = re.compile(r"[a-z]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ms2-root", type=Path, default=Path("/mnt/e/dataset/ms2"))
    parser.add_argument(
        "--per-sample",
        type=Path,
        required=True,
        help="eval_*_per_sample.csv or metrics/per_image.csv: supplies alignment_scale/shift.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/route_suite/caption_scale_probe.json"))
    parser.add_argument("--vocab-size", type=int, default=1500)
    parser.add_argument("--min-count", type=int, default=20, help="Drop tokens rarer than this.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def read_alignment(path: Path) -> dict[str, tuple[float, float]]:
    out = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row.get("id") or row.get("sample_id")
            out[key] = (float(row["alignment_scale"]), float(row["alignment_shift"]))
    if not out:
        raise SystemExit(f"No rows with alignment parameters in {path}")
    return out


def build_features(captions: list[str], vocab: list[str]) -> np.ndarray:
    index = {token: i for i, token in enumerate(vocab)}
    features = np.zeros((len(captions), len(vocab) + 2), np.float64)
    for row, caption in enumerate(captions):
        tokens = TOKEN_RE.findall(caption.lower())
        for token in tokens:
            position = index.get(token)
            if position is not None:
                features[row, position] += 1.0
        features[row, -2] = len(tokens)           # length, so we can price it separately
        features[row, -1] = 1.0                   # intercept
    return features


def ridge_cv(features: np.ndarray, target: np.ndarray, folds: int, alpha: float, seed: int) -> float:
    """Out-of-fold R^2 of a ridge fit. Closed form, no sklearn."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(target))
    predictions = np.zeros_like(target)
    for fold in range(folds):
        test = order[fold::folds]
        train = np.setdiff1d(order, test, assume_unique=False)
        x, y = features[train], target[train]
        gram = x.T @ x
        penalty = alpha * np.eye(gram.shape[0])
        penalty[-1, -1] = 0.0                     # never penalise the intercept
        weights = np.linalg.solve(gram + penalty, x.T @ y)
        predictions[test] = features[test] @ weights
    residual = float(np.sum((target - predictions) ** 2))
    total = float(np.sum((target - target.mean()) ** 2))
    return 1.0 - residual / total if total > 0 else float("nan")


def main() -> None:
    args = parse_args()
    alignment = read_alignment(args.per_sample)

    rows = []
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in alignment and str(row.get("caption", "")).strip():
                rows.append(row)
    if args.limit:
        rows = rows[: args.limit]
    if len(rows) < 100:
        raise SystemExit(f"Only {len(rows)} frames matched between manifest and {args.per_sample}")
    print(f"[data] {len(rows)} frames with both a caption and alignment parameters", flush=True)

    captions = [row["caption"] for row in rows]
    scale = np.array([alignment[row["id"]][0] for row in rows], np.float64)
    shift = np.array([alignment[row["id"]][1] for row in rows], np.float64)

    median_depth = np.zeros(len(rows), np.float64)
    for i, row in enumerate(rows):
        depth = np.asarray(Image.open(args.ms2_root / row["depth_path"]), np.float32) / args.depth_scale
        valid = (depth > 1e-3) & (depth < 80.0)
        median_depth[i] = float(np.median(depth[valid])) if valid.any() else np.nan
        if (i + 1) % 2000 == 0:
            print(f"    GT {i + 1}/{len(rows)}", flush=True)
    keep = np.isfinite(median_depth)

    counts: dict[str, int] = {}
    for caption in captions:
        for token in set(TOKEN_RE.findall(caption.lower())):
            counts[token] = counts.get(token, 0) + 1
    vocab = [t for t, c in sorted(counts.items(), key=lambda kv: -kv[1]) if c >= args.min_count][
        : args.vocab_size
    ]
    print(f"[features] vocabulary {len(vocab)} tokens (>= {args.min_count} frames)", flush=True)

    real = build_features(captions, vocab)
    # A half-set rotation is NOT a safe control here: this is one continuous
    # drive, and if the route doubles back, the donor 2905 frames away can sit at
    # nearly the same place -- so its caption still describes the recipient's
    # surroundings. A uniform random permutation has no such geometry.
    rng = np.random.default_rng(args.seed)
    permuted = [captions[i] for i in rng.permutation(len(captions))]
    shuffled = build_features(permuted, vocab)
    rotation = len(captions) // 2
    rotated = build_features(captions[rotation:] + captions[:rotation], vocab)
    length_only = real[:, -2:]                    # length + intercept

    targets = {
        "log_alignment_scale": np.log(np.clip(scale, 1e-9, None)),
        "alignment_shift": shift,
        "log_median_gt_depth": np.log(np.clip(median_depth, 1e-6, None)),
    }
    report = {
        "frames": len(rows),
        "per_sample": str(args.per_sample),
        "vocabulary": len(vocab),
        "folds": args.folds,
        "ridge_alpha": args.ridge,
        "r2": {},
    }
    print(f"\n{'target':24s} {'caption':>10s} {'permuted':>10s} {'rotated':>9s} {'length':>8s}")
    print("-" * 64)
    for name, target in targets.items():
        mask = keep if name == "log_median_gt_depth" else np.ones(len(target), bool)
        cells = {
            "caption": ridge_cv(real[mask], target[mask], args.folds, args.ridge, args.seed),
            "permuted": ridge_cv(shuffled[mask], target[mask], args.folds, args.ridge, args.seed),
            "rotated_half": ridge_cv(rotated[mask], target[mask], args.folds, args.ridge, args.seed),
            "length_only": ridge_cv(length_only[mask], target[mask], args.folds, args.ridge, args.seed),
        }
        report["r2"][name] = cells
        print(
            f"{name:24s} {cells['caption']:>10.4f} {cells['permuted']:>10.4f} "
            f"{cells['rotated_half']:>9.4f} {cells['length_only']:>8.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
