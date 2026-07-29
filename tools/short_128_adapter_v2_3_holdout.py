"""Train Adapter V2.3 on 128 fixed Train frames and audit 16 held-out Train frames.

The holdout frames are frozen before optimization and never enter a training
batch.  AnyThermal and the Lotus VAE teacher remain frozen; Val/Test and the
Lotus U-Net are not used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Sequence

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import overfit_32_adapter_v2_distillation as base  # noqa: E402


DEFAULT_MANIFEST = Path(
    "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/"
    "ms2_fixed_sequence_internvl3_8b_filtered_caption_train.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ms2-root", type=Path, default=Path("/mnt/e/dataset/ms2"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lotus_line_v2/short_128_v2_3_holdout16"),
    )
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--holdout-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cosine-loss-weight", type=float, default=0.1)
    parser.add_argument("--channel-stats-loss-weight", type=float, default=0.1)
    parser.add_argument("--spatial-gradient-loss-weight", type=float, default=0.1)
    parser.add_argument("--multiscale-gradient-loss-weight", type=float, default=0.5)
    parser.add_argument("--processing-res", type=int, default=0)
    parser.add_argument("--teacher-posterior", choices=("mode",), default="mode")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vae-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    args.adapter_architecture = "v2_3_thermal_detail_skip"
    args.use_output_group_norm = False
    args.num_samples = args.train_samples + args.holdout_samples
    return args


def partition_uniform_cache(
    cache: Sequence[Dict[str, Any]],
    train_count: int,
    holdout_count: int,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Split a uniform cache into interleaved, disjoint train/holdout subsets."""

    total = train_count + holdout_count
    if min(train_count, holdout_count) <= 0 or len(cache) != total:
        raise ValueError(
            f"Expected {train_count}+{holdout_count}={total} cached rows, got {len(cache)}."
        )
    holdout_positions = {
        min(total - 1, int((index + 0.5) * total / holdout_count))
        for index in range(holdout_count)
    }
    if len(holdout_positions) != holdout_count:
        raise RuntimeError("Holdout binning produced duplicate positions.")
    train = [dict(item, short_split="train") for i, item in enumerate(cache) if i not in holdout_positions]
    holdout = [
        dict(item, short_split="holdout")
        for i, item in enumerate(cache)
        if i in holdout_positions
    ]
    if len(train) != train_count or len(holdout) != holdout_count:
        raise RuntimeError("Unexpected short-train partition sizes.")
    train_ids = {item["id"] for item in train}
    holdout_ids = {item["id"] for item in holdout}
    if train_ids & holdout_ids:
        raise RuntimeError("Train and holdout IDs overlap.")
    return train, holdout


def gate_metrics(
    initial: Dict[str, Any],
    final: Dict[str, Any],
    *,
    mse_ratio_limit: float,
) -> Dict[str, Any]:
    mse_ratio = final["mse"] / initial["mse"]
    passed = bool(
        math.isfinite(mse_ratio)
        and mse_ratio <= mse_ratio_limit
        and final["cosine"] >= 0.9
        and final["pearson"] >= 0.9
        and 0.5 <= final["gradient_energy_ratio"] <= 1.5
        and all(item["finite"] for item in final["fixed_eight"])
    )
    return {
        "mse_ratio": mse_ratio,
        "mse_ratio_limit": mse_ratio_limit,
        "cosine_minimum": 0.9,
        "pearson_minimum": 0.9,
        "gradient_energy_ratio_range": [0.5, 1.5],
        "passed": passed,
    }


def flat_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, float]:
    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
        if key != "fixed_eight"
    }


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_role_manifest(source: Path, cache: Sequence[Dict[str, Any]], target: Path) -> None:
    wanted = {int(item["manifest_index"]): item["short_split"] for item in cache}
    selected = []
    with source.open("r", encoding="utf-8") as handle:
        for manifest_index, line in enumerate(handle):
            if manifest_index not in wanted or not line.strip():
                continue
            row = json.loads(line)
            row["adapter_v2_short_split"] = wanted[manifest_index]
            selected.append((manifest_index, row))
    if len(selected) != len(cache):
        raise RuntimeError(f"Manifest write selected {len(selected)} rows, expected {len(cache)}.")
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for _, row in selected),
        encoding="utf-8",
    )


def public_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "fixed_eight"}


def main() -> None:
    args = parse_args()
    if (args.train_samples, args.holdout_samples, args.batch_size) != (128, 16, 4):
        raise ValueError("The frozen Phase E protocol requires 128 train, 16 holdout, batch size 4.")
    if args.steps <= 0 or args.log_interval <= 0:
        raise ValueError("steps and log-interval must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base.seed_everything(args.seed)
    cache = base.cache_samples(args, device)
    train_cache, holdout_cache = partition_uniform_cache(
        cache, args.train_samples, args.holdout_samples
    )
    source_manifest = args.train_manifest.resolve()
    write_role_manifest(source_manifest, train_cache, output_dir / "train_128_manifest.jsonl")
    write_role_manifest(source_manifest, holdout_cache, output_dir / "holdout_16_manifest.jsonl")

    adapter = base.make_adapter(args).to(device=device, dtype=torch.float32).train()
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    initial_train = base.evaluate(adapter, train_cache, device, args.batch_size, args)
    initial_holdout = base.evaluate(adapter, holdout_cache, device, args.batch_size, args)
    base.save_checkpoint(adapter, output_dir, 0, args)
    history = [
        {
            "step": 0,
            **flat_metrics("train", initial_train),
            **flat_metrics("holdout", initial_holdout),
        }
    ]
    print(json.dumps(history[-1]), flush=True)

    for step in range(1, args.steps + 1):
        indices = base.batch_indices(
            step - 1, len(train_cache), args.batch_size, args.seed
        )
        features, thermal, teacher = base.stack_batch(train_cache, indices, device)
        optimizer.zero_grad(set_to_none=True)
        output = base.prediction(adapter, features, thermal, teacher)
        losses = base.loss_components(output, teacher, args)
        loss = losses["total"]
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite loss at step {step}.")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in adapter.parameters()
            if parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise RuntimeError(f"Missing or non-finite Adapter gradient at step {step}.")
        optimizer.step()

        if step % args.log_interval == 0 or step == args.steps:
            train_metrics = base.evaluate(
                adapter, train_cache, device, args.batch_size, args
            )
            holdout_metrics = base.evaluate(
                adapter, holdout_cache, device, args.batch_size, args
            )
            record = {
                "step": step,
                **flat_metrics("train", train_metrics),
                **flat_metrics("holdout", holdout_metrics),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    final_train = base.evaluate(adapter, train_cache, device, args.batch_size, args)
    final_holdout = base.evaluate(adapter, holdout_cache, device, args.batch_size, args)
    checkpoint = base.save_checkpoint(adapter, output_dir, args.steps, args)
    train_gate = gate_metrics(initial_train, final_train, mse_ratio_limit=0.2)
    holdout_gate = gate_metrics(initial_holdout, final_holdout, mse_ratio_limit=0.25)
    gate_passed = bool(train_gate["passed"] and holdout_gate["passed"])

    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "phase": "Adapter V2.3 Phase E 128-frame short train with 16-frame holdout",
        "route": (
            "frozen AnyThermal + audited thermal detail + trainable Adapter; "
            "frozen VAE teacher; no U-Net"
        ),
        "source_split": "Train only, deterministically partitioned before optimization",
        "uses_val": False,
        "uses_test": False,
        "resume": None,
        "v1_checkpoint_used": False,
        "adapter_architecture": args.adapter_architecture,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": manifest_sha256(source_manifest),
        "train_manifest": str(output_dir / "train_128_manifest.jsonl"),
        "holdout_manifest": str(output_dir / "holdout_16_manifest.jsonl"),
        "train_ids": [item["id"] for item in train_cache],
        "holdout_ids": [item["id"] for item in holdout_cache],
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "initial_train": public_metrics(initial_train),
        "final_train": public_metrics(final_train),
        "initial_holdout": public_metrics(initial_holdout),
        "final_holdout": public_metrics(final_holdout),
        "fixed_eight_holdout_final": final_holdout["fixed_eight"],
        "train_gate": train_gate,
        "holdout_gate": holdout_gate,
        "gate_passed": gate_passed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "train_gate": train_gate,
                "holdout_gate": holdout_gate,
                "final_train": summary["final_train"],
                "final_holdout": summary["final_holdout"],
                "gate_passed": gate_passed,
                "checkpoint": str(checkpoint),
            },
            indent=2,
        )
    )
    print(output_dir / "summary.json")
    if not gate_passed:
        raise SystemExit("128/16 short-train holdout gate failed; do not proceed to Val.")


if __name__ == "__main__":
    main()
