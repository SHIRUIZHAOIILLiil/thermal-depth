"""One-batch frozen-U-Net response-consistency smoke gate for Adapter V2.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = ROOT / "lotus"
for path in (ROOT, LOTUS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.anythermal_encoder import AnyThermalEncoder  # noqa: E402
from models.anythermal_lotus_adapter_v2 import AnyThermalLotusAdapterV2  # noqa: E402
from models.anythermal_lotus_model import extract_anythermal_feature_pyramid  # noqa: E402
from models.anythermal_lotus_v2 import (  # noqa: E402
    condition_distillation_losses,
    encode_condition_latent,
    thermal_to_lotus_input,
)
from pipeline import LotusGPipeline  # noqa: E402


DEFAULT_MANIFEST = Path(
    "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b/"
    "ms2_fixed_sequence_internvl3_8b_filtered_caption_train.jsonl"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ms2-root", type=Path, default=Path("/mnt/e/dataset/ms2"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/lotus_line_v2/smoke_1batch_v2_1_unet_response"),
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--response-weight", type=float, default=1.0)
    parser.add_argument("--latent-weight", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def read_sample(path: Path, index: int):
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index == index:
                row = json.loads(line)
                if row.get("split") != "train":
                    raise ValueError("Response smoke accepts Train samples only.")
                return {"id": row["id"], "thermal_path": row["thermal_path"]}
    raise IndexError(f"sample-index {index} is outside the manifest.")


def task_embedding(device, dtype):
    task = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
    return torch.cat([torch.sin(task), torch.cos(task)], dim=-1)


def stats(tensor):
    value = tensor.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "finite": bool(torch.isfinite(value).all()),
    }


def module_gradients(module):
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    return {
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        ),
        "has_gradient": bool(gradients),
        "finite": bool(all(torch.isfinite(gradient).all() for gradient in gradients)),
        "l2": float(
            torch.sqrt(sum((gradient.detach().float() ** 2).sum() for gradient in gradients)).cpu()
        )
        if gradients
        else 0.0,
    }


def main():
    args = parse_args()
    if min(args.response_weight, args.latent_weight) < 0:
        raise ValueError("Loss weights must be non-negative.")
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    sample = read_sample(args.train_manifest.resolve(), args.sample_index)
    thermal_path = args.ms2_root.resolve() / sample["thermal_path"]
    thermal = thermal_to_lotus_input(thermal_path, processing_res=0)
    lotus = LotusGPipeline.from_pretrained(
        args.lotus_model_path,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    anythermal = AnyThermalEncoder(
        model_path=args.anythermal_model_path,
        device=str(device),
        local_files_only=args.local_files_only,
    )
    for module in (lotus.vae, lotus.text_encoder, lotus.unet, anythermal.model):
        module.requires_grad_(False).eval()
    adapter = AnyThermalLotusAdapterV2().to(device=device, dtype=torch.float32).train()

    teacher_condition = encode_condition_latent(
        lotus.vae,
        thermal.tensor,
        posterior="mode",
    ).to(device=device, dtype=torch.float32)
    features, info, diagnostics = extract_anythermal_feature_pyramid(
        anythermal,
        thermal_path,
        enable_grad=False,
    )
    features = [feature.to(device=device, dtype=torch.float32) for feature in features]
    student_condition = adapter(features, target_size=tuple(teacher_condition.shape[-2:]))
    if student_condition.shape != teacher_condition.shape:
        raise RuntimeError("Student/teacher condition shapes differ.")

    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(
        teacher_condition.shape,
        generator=generator,
        device=device,
        dtype=dtype,
    ) * lotus.scheduler.init_noise_sigma
    timestep = torch.full(
        (teacher_condition.shape[0],),
        args.timestep,
        device=device,
        dtype=torch.long,
    )
    latent_input = lotus.scheduler.scale_model_input(noise, timestep)
    prompt, _ = lotus.encode_prompt(
        prompt="",
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=None,
    )
    class_labels = task_embedding(device, dtype)
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
    ):
        teacher_response = lotus.unet(
            torch.cat([teacher_condition.to(dtype), latent_input], dim=1),
            timestep,
            encoder_hidden_states=prompt.to(dtype=dtype),
            class_labels=class_labels,
            return_dict=False,
        )[0]
    with torch.autocast(
        device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
    ):
        student_response = lotus.unet(
            torch.cat([student_condition.to(dtype), latent_input], dim=1),
            timestep,
            encoder_hidden_states=prompt.to(dtype=dtype),
            class_labels=class_labels,
            return_dict=False,
        )[0]

    response_mse = F.mse_loss(student_response.float(), teacher_response.float())
    latent_losses = condition_distillation_losses(
        student_condition,
        teacher_condition,
        spatial_gradient_weight=0.02,
    )
    total = args.response_weight * response_mse + args.latent_weight * latent_losses["total"]
    if not bool(torch.isfinite(total)):
        raise RuntimeError("Response-consistency total loss is not finite.")
    total.backward()

    gradient_owners = {
        "adapter": module_gradients(adapter),
        "anythermal": module_gradients(anythermal.model),
        "vae": module_gradients(lotus.vae),
        "text_encoder": module_gradients(lotus.text_encoder),
        "unet": module_gradients(lotus.unet),
    }
    if not gradient_owners["adapter"]["has_gradient"] or gradient_owners["adapter"]["l2"] <= 0:
        raise RuntimeError("Adapter received no response gradient.")
    if not gradient_owners["adapter"]["finite"]:
        raise RuntimeError("Adapter response gradient is non-finite.")
    if any(gradient_owners[name]["has_gradient"] for name in ("anythermal", "vae", "text_encoder", "unet")):
        raise RuntimeError("A frozen module unexpectedly owns gradients.")

    summary = {
        "phase": "Adapter V2.1 frozen-U-Net response-consistency 1-batch smoke",
        "gate_passed": True,
        "training": False,
        "optimizer_constructed": False,
        "optimizer_step": False,
        "sample": {**sample, "thermal_path_resolved": str(thermal_path)},
        "settings": {
            "seed": args.seed,
            "timestep": args.timestep,
            "prompt": "",
            "teacher_posterior": "mode",
            "processing_res": 0,
            "response_weight": args.response_weight,
            "latent_weight": args.latent_weight,
        },
        "thermal": thermal.diagnostics,
        "feature_grid": list(info.grid_size),
        "anythermal_thermal_diagnostics": diagnostics,
        "teacher_condition": stats(teacher_condition),
        "student_condition": stats(student_condition),
        "teacher_response": stats(teacher_response),
        "student_response": stats(student_response),
        "losses": {
            "total": float(total.detach().cpu()),
            "unet_response_mse": float(response_mse.detach().cpu()),
            **{
                f"latent_{name}": float(value.detach().cpu())
                for name, value in latent_losses.items()
            },
        },
        "gradient_owners": gradient_owners,
    }
    path = output / "summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "gate_passed": True,
        "losses": summary["losses"],
        "gradient_owners": gradient_owners,
        "summary": str(path),
    }, indent=2))


if __name__ == "__main__":
    main()
