"""Score the metric head in metres, under the official protocol.

The head turns our relative output into depth without touching test GT:

    D = exp(A(thermal, caption) * y + B(thermal, caption))

so unlike every other number in this project it can be read with no alignment
at all. That is a stricter test than the published baselines pass -- DORN, BTS,
AdaBins and NeWCRF are all reported after a per-image median scaling fitted
against test GT (COMPARISON_PROTOCOL section 1) -- so both readings are computed
and written to separate files:

  none    what the head actually predicts. No test GT enters at any point.
  median  the same predictions under the baselines' own 1-parameter alignment,
          which is the only way to sit in their column on their terms.

They are not two views of one result and must not be merged into one table. The
file names say which is which, following the evaluator's own convention.

The arm's configuration is read from its checkpoint, not from a flag: a
no-caption head scored with real captions would be fed a distribution it never
trained on, and nothing downstream would report the mismatch.

    python tools/eval_metric_rescale_head.py \
        --head $SCRATCH/runs/metric_head/cap_s42/head_epoch1.pt \
        --manifest $SCRATCH/manifests/.../ms2_test_day3_common_thermalcap_....jsonl \
        --ms2-root $SCRATCH/data/ms2 \
        --relative-dir $SCRATCH/runs/eval/<test export>/raw_predictions \
        --out-dir $SCRATCH/runs/metric_head/cap_s42/eval_day3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ms2_eval.official_protocol import (  # noqa: E402
    DEFAULT_MAX_DEPTH_M, DEFAULT_MIN_DEPTH_M, evaluate_sample)
from tools.train_metric_rescale_head import RescaleHead, thermal_to_unit  # noqa: E402

METRICS = ("abs_rel", "rmse", "a1", "sq_rel", "rmse_log", "log10", "a2", "a3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ms2-root", required=True, type=Path)
    parser.add_argument("--relative-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sd2-repo", default="Manojb/stable-diffusion-2-1-base")
    parser.add_argument("--stretch", default="percentile", choices=("percentile", "minmax"),
                        help="Must match the arm that produced --relative-dir.")
    parser.add_argument("--align-modes", nargs="+", default=["none", "median"],
                        choices=("none", "median", "ssi", "ssi_log", "ssi_disparity"))
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_head(path: Path, text_dim: int, width: int, device: str):
    """Rebuild the arm the checkpoint describes.

    global_only is inferred from the weights rather than stored, so a checkpoint
    written before the flag existed still loads, and a stale flag cannot
    contradict what is actually in the file.
    """
    blob = torch.load(path, map_location="cpu")
    state = blob["model"]
    global_only = "to_ab.weight" not in state
    head = RescaleHead(width, text_dim, blob["a_init"], blob["b_init"],
                       global_only=global_only)
    head.load_state_dict(state)
    return head.to(device).eval(), blob, global_only


def main() -> None:
    args = parse_args()

    from transformers import CLIPTextModel, CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(args.sd2_repo, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.sd2_repo, subfolder="text_encoder").to(args.device).eval()

    head, blob, global_only = load_head(
        args.head, text_encoder.config.hidden_size, args.width, args.device)
    no_captions = bool(blob.get("no_captions", False))
    arm = "global-only" if global_only else (
        "per-pixel, caption off" if no_captions else "per-pixel, caption on")
    print(f"[head] {args.head.name}   臂 = {arm}   "
          f"a_init={blob['a_init']:.4f} b_init={blob['b_init']:.4f}", flush=True)
    if global_only or no_captions:
        print("[head] 该臂训练时没有真 caption，评估同样喂空串", flush=True)

    rows = []
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if (args.relative_dir / f"{row['id']}.npy").is_file():
                    rows.append(row)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"没有一帧在 {args.relative_dir} 里找到相对深度")
    print(f"[data] {len(rows)} 帧   thermal={args.stretch}", flush=True)

    per_sample: dict[str, list[dict]] = {mode: [] for mode in args.align_modes}
    for index, row in enumerate(rows):
        gt = np.asarray(Image.open(args.ms2_root / row["thermal_depth_path"]),
                        dtype=np.float32) / 256.0
        relative = np.load(args.relative_dir / f"{row['id']}.npy").astype(np.float32)
        if relative.shape != gt.shape:
            relative = F.interpolate(torch.from_numpy(relative)[None, None], gt.shape,
                                     mode="bilinear", align_corners=False)[0, 0].numpy()
        thermal = thermal_to_unit(args.ms2_root / row["thermal_path"], args.stretch)
        caption = "" if (no_captions or global_only) else str(row.get("caption") or "")

        with torch.no_grad():
            tokens = tokenizer([caption], padding="max_length", truncation=True,
                               max_length=tokenizer.model_max_length, return_tensors="pt")
            text = text_encoder(tokens.input_ids.to(args.device), return_dict=False)[0]
            a, b = head(torch.from_numpy(thermal)[None, None].to(args.device),
                        torch.from_numpy(relative)[None, None].to(args.device), text)
            metres = torch.exp((a * torch.from_numpy(relative)[None, None].to(args.device)
                                + b).clamp(-9.0, 9.0))[0, 0].cpu().numpy()

        for mode in args.align_modes:
            try:
                result = evaluate_sample(metres.astype(np.float32), gt, align=mode,
                                         min_depth=DEFAULT_MIN_DEPTH_M,
                                         max_depth=DEFAULT_MAX_DEPTH_M)
            except Exception as error:                      # noqa: BLE001
                raise SystemExit(f"{row['id']} 在 align={mode} 下失败：{error}") from error
            per_sample[mode].append({"id": row["id"], **result})

        if index % 500 == 0:
            print(f"  {index}/{len(rows)}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print()
    print(f"{'对齐':<10}{'用了 test GT':<14}{'AbsRel':>10}{'RMSE':>10}{'δ1':>10}")
    print("-" * 56)
    for mode in args.align_modes:
        samples = per_sample[mode]
        summary = {key: float(np.mean([s[key] for s in samples]))
                   for key in METRICS if key in samples[0]}
        used_gt = "否" if mode == "none" else "是"
        print(f"{mode:<10}{used_gt:<14}{summary['abs_rel']:>10.5f}"
              f"{summary['rmse']:>10.4f}{summary['a1']:>10.4f}")

        # Separate files per alignment, as the official evaluator does: these are
        # different claims, not two readings of one number.
        stem = args.out_dir / f"metric_head_{mode}"
        stem.with_suffix(".json").write_text(json.dumps({
            "head": str(args.head),
            "arm": arm,
            "align_mode": mode,
            "test_gt_used_for_fitting": mode != "none",
            "thermal_stretch": args.stretch,
            "relative_dir": str(args.relative_dir),
            "manifest": str(args.manifest),
            "frames": len(samples),
            "summary": summary,
            "written_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        with stem.with_name(f"metric_head_{mode}_per_sample.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)

    print()
    print(f"写入 {args.out_dir}")
    if "none" in args.align_modes:
        print("⚠️ align=none 那一行是零对齐的真米制，比已发表基线所用的口径更严格；"
              "与它们并排时两行都要摆出来。")


if __name__ == "__main__":
    main()
