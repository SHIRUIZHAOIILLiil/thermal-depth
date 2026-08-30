"""Carry a `train_iris_ms2_g.py` checkpoint into the format our evaluators read.

Iris's trainer saves what `accelerate` and `diffusers` save: a `checkpoint-N/`
directory holding the prepared model, and at the end a full pipeline directory.
Every evaluator we have -- the official BridgeMSD protocol runner, the region
analyser, `eval.sbatch` -- reads `train_route_suite.py`'s payload instead: a
single `.pt` carrying `route`, `epoch` and a `state_dicts` mapping. This moves
the U-Net across so the new line is scored by exactly the tooling that produced
every number it will be compared against.

Nothing is adapted or renamed on the way. `train_route_suite.py` builds its
U-Net as `copy.deepcopy(LotusGPipeline.from_pretrained(...).unet)`, and the MS2
trainer's `ckpt` arm loads that same class from that same repository, so the two
state dicts are expected to agree key for key. That expectation is checked
rather than assumed: a silent mismatch here would surface as a mysteriously bad
AbsRel hours later, which is the most expensive way to find it.

    python tools/convert_iris_ms2_checkpoint.py \
        --source $SCRATCH/runs/iris_ms2/iris_ms2_ckpt/checkpoint-5000 \
        --output $SCRATCH/runs/iris_ms2/iris_ms2_ckpt/converted/step05000_weights.pt

Then score it the usual way:

    ROUTE=b_thermal_unet CAPTION_MODE=empty TAG=irisms2_s5000_empty \
    CKPT=<the .pt above> VAL_MANIFEST=<test manifest> sbatch ... eval.sbatch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# train_route_suite.py writes this string into every checkpoint it saves; the
# evaluator does not check it, but keeping it identical means a converted file
# and a native one cannot be told apart by anything downstream.
CHECKPOINT_FORMAT = "route_suite_multi_epoch_pure_gt"
# The MS2 line trains a thermal input against the frozen VAE condition with the
# whole U-Net unfrozen -- route b, whose persisted modules are exactly {"unet"}.
ROUTE = "b_thermal_unet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True,
                        help="A checkpoint-N/ directory, or the pipeline directory saved at the end.")
    parser.add_argument("--output", type=Path, required=True, help="Destination .pt")
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity",
                        help="Reference U-Net the key check compares against.")
    parser.add_argument("--backbone", choices=("g", "d", "marigold"), default="g",
                        help="Which Lotus variant this checkpoint is. G takes the concatenated "
                             "[condition, latent] input (conv_in 8); D is the direct variant and "
                             "takes the condition alone (conv_in 4). Must agree with "
                             "--lotus-model-path, which is what the key check compares against.")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--skip-reference-check", action="store_true",
                        help="Skip loading the reference U-Net. Only for a machine without the weights.")
    parser.add_argument("--caption-mode", default="correct", choices=("empty", "correct"),
                        help="What the run trained with. Recorded for provenance; the evaluator "
                             "takes its own --val-caption-mode and never reads this.")
    parser.add_argument("--step", type=int, default=None,
                        help="Training step this checkpoint is from. Read off the directory name when omitted.")
    return parser.parse_args()


def find_state_file(source: Path) -> tuple[Path, str]:
    """Locate the U-Net weights inside either directory layout."""
    if not source.exists():
        raise SystemExit(f"No such source: {source}")
    candidates = [
        # A full pipeline, from pipeline.save_pretrained() at the end of training.
        (source / "unet" / "diffusion_pytorch_model.safetensors", "pipeline"),
        (source / "unet" / "diffusion_pytorch_model.bin", "pipeline"),
        # An accelerate save_state() checkpoint. Only the U-Net is prepared, so
        # the single model file it wrote is the U-Net.
        (source / "model.safetensors", "accelerate"),
        (source / "pytorch_model.bin", "accelerate"),
        (source / "model.bin", "accelerate"),
    ]
    for path, kind in candidates:
        if path.is_file():
            return path, kind
    listing = "\n  ".join(sorted(p.name for p in source.iterdir())) if source.is_dir() else "(not a directory)"
    raise SystemExit(
        f"Found no U-Net weights under {source}. It holds:\n  {listing}\n"
        "Expected either unet/diffusion_pytorch_model.* (a pipeline directory) "
        "or model.safetensors / pytorch_model.bin (an accelerate checkpoint)."
    )


def load_state_dict(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def strip_wrappers(state: dict) -> dict:
    """Drop the prefixes accelerate and torch.compile can leave behind."""
    for prefix in ("module.", "_orig_mod."):
        if state and all(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: value for key, value in state.items()}
    return state


def reference_state_dict(model_path: str, local_files_only: bool) -> dict:
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(
        model_path, subfolder="unet", local_files_only=local_files_only
    )
    return unet.state_dict()


def check_against_reference(state: dict, reference: dict, backbone: str = "g") -> None:
    missing = sorted(set(reference) - set(state))
    extra = sorted(set(state) - set(reference))
    mismatched = [
        (key, tuple(reference[key].shape), tuple(state[key].shape))
        for key in sorted(set(state) & set(reference))
        if tuple(state[key].shape) != tuple(reference[key].shape)
    ]
    if missing or extra or mismatched:
        lines = ["The converted U-Net does not match the reference architecture."]
        if missing:
            lines.append(f"  missing {len(missing)}: {missing[:5]}")
        if extra:
            lines.append(f"  unexpected {len(extra)}: {extra[:5]}")
        for key, want, got in mismatched[:5]:
            lines.append(f"  shape {key}: reference {want}, checkpoint {got}")
        if mismatched:
            lines.append(f"  ({len(mismatched)} shape mismatches in total)")
        raise SystemExit("\n".join(lines))
    want_in = 4 if backbone == "d" else 8
    conv_in = state.get("conv_in.weight")
    if conv_in is not None and conv_in.shape[1] != want_in:
        raise SystemExit(
            f"conv_in accepts {conv_in.shape[1]} channels, not {want_in}: with --backbone "
            f"{backbone} this U-Net cannot take the input the route feeds it "
            "(g: the concatenated [condition, latent]; d: the condition alone)."
        )
    print(f"[check] {len(state)} tensors, keys and shapes match the reference U-Net", flush=True)


def infer_step(source: Path, given: int | None) -> int | None:
    if given is not None:
        return given
    name = source.name
    if name.startswith("checkpoint-"):
        suffix = name.split("-", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return None


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    state_file, kind = find_state_file(source)
    print(f"[source] {kind} layout: {state_file}", flush=True)

    state = strip_wrappers(load_state_dict(state_file))
    if not state:
        raise SystemExit(f"{state_file} carried no tensors")

    if args.skip_reference_check:
        print("[check] skipped by request -- a key mismatch will surface as a bad metric instead",
              flush=True)
    else:
        check_against_reference(
            state,
            reference_state_dict(args.lotus_model_path, args.local_files_only),
            args.backbone,
        )

    step = infer_step(source, args.step)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "route": ROUTE,
        # The evaluator prints this as "epoch"; this line counts in steps, so the
        # step goes here and the honest name is kept beside it.
        "epoch": step,
        "global_step": step,
        "manifest_sha256": None,
        "caption_mode": args.caption_mode,
        "val_metrics": None,
        "trainable_modules": ["unet"],
        "state_dicts": {"unet": {key: value.detach().cpu() for key, value in state.items()}},
        "converted_from": {
            "source": str(source),
            "state_file": str(state_file),
            "layout": kind,
            "trainer": "lotus/train_iris_ms2_g.py",
            "reference_unet": None if args.skip_reference_check else args.lotus_model_path,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    size_gb = args.output.stat().st_size / (1 << 30)
    print(f"[done] step {step} -> {args.output} ({size_gb:.2f} GB)")
    print("[next] score it with slurm/eval.sbatch, ROUTE=b_thermal_unet")
    print(json.dumps({k: v for k, v in payload.items() if k != "state_dicts"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
