"""Does the caption actually reach the feature map, and does it care what it says?

End-to-end metrics told us that correct and shuffled captions score identically
(paired difference -0.00006, win rate 49.5%).  That is consistent with two very
different mechanisms, and the fix differs depending on which one is true:

  (a) text never reaches the features   -> cross-attention output barely moves
  (b) text perturbs the features hard, but without regard to content
                                        -> output moves a lot, yet correct and
                                           shuffled move it to the same place

This probe reads the answer straight out of the network instead of inferring it
from AbsRel.  For each frame it runs the U-Net three times on an identical image
condition -- empty prompt, correct caption, shuffled caption -- and records, for
each of the 16 cross-attention layers:

  delta_vs_empty     ||out_text - out_empty|| / ||out_empty||
                     how far the caption moves the features at all
  delta_correct_vs_shuffled
                     ||out_correct - out_shuffled|| / ||out_correct||
                     how much of that movement depends on the content
  content_fraction   the ratio of the two: ~0 means the movement is content-blind
  attention_entropy  entropy of the attention over the 77 text tokens,
                     normalised by log(77); ~1.0 means no token selectivity

    python tools/probe_cross_attention_binding.py --route b_thermal_unet \\
        --checkpoint outputs/route_suite/b_thermal_unet_20ep/epoch05_weights.pt \\
        --output-dir outputs/route_suite/b_thermal_unet_20ep/attn_probe --frames 200
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = ROOT / "lotus"
for path in (ROOT, LOTUS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.anythermal_lotus_v2_4 import seeded_noise  # noqa: E402
from train_route_suite import (  # noqa: E402
    ROUTES,
    RouteModel,
    load_input_tensor,
    read_manifest,
    rotate_captions,
    task_embedding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", default="b_thermal_unet", choices=sorted(ROUTES))
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    parser.add_argument("--frames", type=int, default=200, help="Uniformly spaced frames to probe.")
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frozen-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--input-max-edge", type=int, default=0)
    parser.add_argument("--gt-decode-fp32", action="store_true", default=False)
    parser.add_argument("--no-attention-entropy", action="store_true",
                        help="Skip recomputing attention weights (saves memory on large feature maps).")
    return parser.parse_args()


class CrossAttentionRecorder:
    """Capture every attn2 output, and optionally its attention distribution."""

    def __init__(self, unet, want_entropy: bool):
        self.want_entropy = want_entropy
        self.outputs: dict[str, torch.Tensor] = {}
        self.entropies: dict[str, float] = {}
        self.handles = []
        self.layers = []
        for name, module in unet.named_modules():
            if name.endswith("attn2") and hasattr(module, "to_k"):
                self.layers.append(name)
                self.handles.append(
                    module.register_forward_hook(self._make_hook(name), with_kwargs=True)
                )

    def _make_hook(self, name):
        def hook(module, args, kwargs, output):
            self.outputs[name] = output.detach().float()
            if not self.want_entropy:
                return
            hidden = args[0] if args else kwargs.get("hidden_states")
            context = kwargs.get("encoder_hidden_states")
            if hidden is None or context is None:
                return
            with torch.no_grad():
                query = module.to_q(hidden)
                key = module.to_k(context)
                heads = module.heads
                batch, positions, inner = query.shape
                head_dim = inner // heads
                query = query.view(batch, positions, heads, head_dim).transpose(1, 2)
                key = key.view(batch, key.shape[1], heads, head_dim).transpose(1, 2)
                logits = torch.matmul(query.float(), key.float().transpose(-1, -2))
                logits = logits / math.sqrt(head_dim)
                probabilities = logits.softmax(dim=-1)
                entropy = -(probabilities.clamp_min(1e-12).log() * probabilities).sum(-1)
                self.entropies[name] = float(entropy.mean() / math.log(probabilities.shape[-1]))

        return hook

    def snapshot(self) -> tuple[dict, dict]:
        outputs = {name: tensor.clone() for name, tensor in self.outputs.items()}
        entropies = dict(self.entropies)
        self.outputs.clear()
        self.entropies.clear()
        return outputs, entropies

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def unet_forward(model: RouteModel, row: dict, image_tensor, prompt, args) -> None:
    condition = model.condition(row, image_tensor)
    noise = seeded_noise(
        (1, *condition.shape[1:]),
        seed=args.seed + int(row["manifest_index"]),
        device=model.device,
        dtype=torch.float32,
        scale=float(model.lotus.scheduler.init_noise_sigma),
    )
    timestep = torch.full((1,), args.timestep, device=model.device, dtype=torch.long)
    latent_input = model.lotus.scheduler.scale_model_input(noise, timestep)
    unet_dtype = next(model.unet.parameters()).dtype
    model.unet(
        torch.cat([condition, latent_input], dim=1).to(unet_dtype),
        timestep,
        encoder_hidden_states=prompt.to(unet_dtype),
        class_labels=task_embedding(1, model.device, unet_dtype),
        return_dict=False,
    )


def relative_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm().item()
    if denominator <= 0:
        return float("nan")
    return float((left - right).norm().item() / denominator)


def main() -> None:
    args = parse_args()
    args.val_caption_mode = "correct"
    args.caption_mode = "empty"

    device = torch.device(args.device)
    frozen_dtype = torch.float16 if args.frozen_dtype == "fp16" else torch.float32
    args.output_dir.mkdir(parents=True, exist_ok=True)

    modality = ROUTES[args.route][0]
    rows = read_manifest(args.val_manifest, args.ms2_root, modality, split=None)
    correct_captions = [row["caption"] for row in rows]
    rotation = rotate_captions(rows)          # rows now carry the donor caption
    for row, caption in zip(rows, correct_captions):
        row["shuffled_caption"] = row["caption"]
        row["caption"] = caption
    step = max(1, len(rows) // args.frames)
    rows = rows[::step][: args.frames]
    print(f"[data] probing {len(rows)} frames; donor rotation {rotation['rotation_offset']}", flush=True)

    model = RouteModel(args, device, frozen_dtype)
    payload = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    if payload.get("route") != args.route:
        raise SystemExit(f"checkpoint route {payload.get('route')!r} != --route {args.route!r}")
    for name, module in model.trainable_modules().items():
        module.load_state_dict(payload["state_dicts"][name], strict=True)
    model.set_train(False)
    print(f"[model] {args.checkpoint.name} (epoch {payload.get('epoch')})", flush=True)

    empty_prompt = model.encode_prompt("")
    recorder = CrossAttentionRecorder(model.unet, want_entropy=not args.no_attention_entropy)
    print(f"[probe] {len(recorder.layers)} cross-attention layers hooked", flush=True)

    accumulator: dict[str, dict[str, list[float]]] = {
        name: {"delta_vs_empty": [], "delta_correct_vs_shuffled": [], "entropy_empty": [], "entropy_correct": []}
        for name in recorder.layers
    }

    for index, row in enumerate(rows):
        image_tensor, _ = load_input_tensor(row, modality, args)

        unet_forward(model, row, image_tensor, empty_prompt, args)
        empty_outputs, empty_entropy = recorder.snapshot()

        unet_forward(model, row, image_tensor, model.encode_prompt(row["caption"]), args)
        correct_outputs, correct_entropy = recorder.snapshot()

        unet_forward(model, row, image_tensor, model.encode_prompt(row["shuffled_caption"]), args)
        shuffled_outputs, _ = recorder.snapshot()

        for name in recorder.layers:
            if name not in empty_outputs:
                continue
            accumulator[name]["delta_vs_empty"].append(
                relative_distance(empty_outputs[name], correct_outputs[name])
            )
            accumulator[name]["delta_correct_vs_shuffled"].append(
                relative_distance(correct_outputs[name], shuffled_outputs[name])
            )
            if name in empty_entropy:
                accumulator[name]["entropy_empty"].append(empty_entropy[name])
            if name in correct_entropy:
                accumulator[name]["entropy_correct"].append(correct_entropy[name])

        if (index + 1) % 50 == 0:
            print(f"    {index + 1}/{len(rows)}", flush=True)

    recorder.close()

    report: dict = {"checkpoint": str(args.checkpoint), "frames": len(rows), "layers": {}}
    print(f"\n{'layer':52s} {'Δ vs empty':>12s} {'Δ corr-shuf':>13s} {'content':>9s} {'entropy':>9s}")
    print("-" * 100)
    for name in recorder.layers:
        entry = accumulator[name]
        if not entry["delta_vs_empty"]:
            continue
        delta_empty = float(np.mean(entry["delta_vs_empty"]))
        delta_content = float(np.mean(entry["delta_correct_vs_shuffled"]))
        fraction = delta_content / delta_empty if delta_empty > 0 else float("nan")
        entropy = float(np.mean(entry["entropy_correct"])) if entry["entropy_correct"] else float("nan")
        report["layers"][name] = {
            "delta_vs_empty": delta_empty,
            "delta_correct_vs_shuffled": delta_content,
            "content_fraction": fraction,
            "attention_entropy_correct": entropy,
            "attention_entropy_empty": (
                float(np.mean(entry["entropy_empty"])) if entry["entropy_empty"] else float("nan")
            ),
        }
        print(f"{name:52s} {delta_empty:12.4f} {delta_content:13.4f} {fraction:9.3f} {entropy:9.3f}")

    values = list(report["layers"].values())
    summary = {
        "mean_delta_vs_empty": float(np.mean([v["delta_vs_empty"] for v in values])),
        "mean_delta_correct_vs_shuffled": float(np.mean([v["delta_correct_vs_shuffled"] for v in values])),
        "mean_content_fraction": float(np.mean([v["content_fraction"] for v in values])),
        "mean_attention_entropy": float(np.nanmean([v["attention_entropy_correct"] for v in values])),
    }
    report["summary"] = summary
    print("\n" + json.dumps(summary, indent=2))

    path = args.output_dir / "cross_attention_probe.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWritten to {path}")


if __name__ == "__main__":
    main()
