"""Run Lotus-D once and audit its conditioning tensor shapes."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from pipeline import LotusDPipeline


Shape = Tuple[int, ...]
ShapeRecords = Dict[str, Optional[Shape]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the original Lotus-D inference path.")
    parser.add_argument("--image", required=True, help="Path to one RGB input image.")
    parser.add_argument(
        "--model-path",
        default="jingheya/lotus-depth-d-v2-0-disparity",
        help="Local Diffusers model directory or Hugging Face repository id.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu.")
    parser.add_argument("--processing-res", type=int, default=None)
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--half-precision", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--debug-shapes",
        action="store_true",
        help="Print the tensors observed along the Lotus-D inference path.",
    )
    return parser.parse_args()


def tensor_shape(value: Any) -> Shape:
    if isinstance(value, (tuple, list)):
        value = value[0]
    if hasattr(value, "sample") and torch.is_tensor(value.sample):
        value = value.sample
    if not torch.is_tensor(value):
        raise TypeError(f"Expected a tensor-like model value, got {type(value)!r}.")
    return tuple(value.shape)


def register_shape_hooks(
    pipeline: LotusDPipeline,
    records: ShapeRecords,
) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def vae_encoder_pre_hook(module: torch.nn.Module, args: Tuple[Any, ...]) -> None:
        del module
        records["vae_encoder_input"] = tensor_shape(args[0])

    def vae_encoder_output_hook(
        module: torch.nn.Module,
        args: Tuple[Any, ...],
        output: Any,
    ) -> None:
        del module, args
        records["vae_encoder_output"] = tensor_shape(output)

    def vae_distribution_parameters_hook(
        module: torch.nn.Module,
        args: Tuple[Any, ...],
        output: Any,
    ) -> None:
        del module, args
        records["vae_distribution_parameters"] = tensor_shape(output)
    def unet_pre_hook(
        module: torch.nn.Module,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> None:
        del module
        records["unet_sample_input"] = tensor_shape(args[0])
        records["prompt_embedding"] = tensor_shape(kwargs["encoder_hidden_states"])
        records["task_embedding"] = tensor_shape(kwargs["class_labels"])

    def unet_output_hook(
        module: torch.nn.Module,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        output: Any,
    ) -> None:
        del module, args, kwargs
        records["unet_model_prediction"] = tensor_shape(output)

    def vae_decoder_input_hook(module: torch.nn.Module, args: Tuple[Any, ...]) -> None:
        del module
        records["vae_decoder_input"] = tensor_shape(args[0])
    def vae_decoder_output_hook(
        module: torch.nn.Module,
        args: Tuple[Any, ...],
        output: Any,
    ) -> None:
        del module, args
        records["vae_decoder_output"] = tensor_shape(output)

    handles.append(pipeline.vae.encoder.register_forward_pre_hook(vae_encoder_pre_hook))
    handles.append(pipeline.vae.encoder.register_forward_hook(vae_encoder_output_hook))
    handles.append(pipeline.vae.quant_conv.register_forward_hook(vae_distribution_parameters_hook))
    handles.append(pipeline.unet.register_forward_pre_hook(unet_pre_hook, with_kwargs=True))
    handles.append(pipeline.unet.register_forward_hook(unet_output_hook, with_kwargs=True))
    handles.append(pipeline.vae.post_quant_conv.register_forward_pre_hook(vae_decoder_input_hook))
    handles.append(pipeline.vae.decoder.register_forward_hook(vae_decoder_output_hook))
    return handles


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"Image does not exist: {image_path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"CUDA is unavailable; cannot use --device {args.device}.")

    device = torch.device(args.device)
    dtype = torch.float16 if args.half_precision else torch.float32
    pipeline = LotusDPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)

    image = Image.open(image_path).convert("RGB")
    image_array = np.asarray(image, dtype=np.float32)
    rgb_in = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    rgb_in = (rgb_in / 127.5 - 1.0).to(device)
    task_emb = torch.tensor([[1.0, 0.0]], device=device)
    task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1)

    records: ShapeRecords = {
        "original_image": tuple(image_array.shape),
        "pipeline_input": tuple(rgb_in.shape),
        "scheduler_output": None,
    }
    handles = register_shape_hooks(pipeline, records)
    autocast_context = torch.autocast(device.type) if device.type == "cuda" else nullcontext()
    try:
        with torch.inference_mode(), autocast_context:
            output = pipeline(
                rgb_in=rgb_in,
                prompt="",
                num_inference_steps=1,
                output_type="np",
                timesteps=[args.timestep],
                task_emb=task_emb,
                processing_res=args.processing_res,
                match_input_res=True,
                resample_method="bilinear",
            ).images
    finally:
        for handle in handles:
            handle.remove()

    records["postprocessed_output"] = tuple(output.shape)
    if args.debug_shapes:
        print(f"原始输入图像 shape: {records['original_image']}")
        print(f"pipeline 输入 tensor shape: {records['pipeline_input']}")
        print(f"vae_encoder_input shape: {records['vae_encoder_input']}")
        print(f"vae_encoder_output shape: {records['vae_encoder_output']}")
        print(f"vae_distribution_parameters shape: {records['vae_distribution_parameters']}")
        print(f"rgb_latents / unet_sample_input shape: {records['unet_sample_input']}")
        print("noisy_depth_latent: None (Lotus-D 不创建随机 noise/depth latent)")
        print(f"unet_sample_input shape: {records['unet_sample_input']}")
        print(f"prompt_embedding shape: {records['prompt_embedding']}")
        print(f"task_embedding shape: {records['task_embedding']}")
        print(f"unet_model_prediction shape: {records['unet_model_prediction']}")
        print("scheduler_output: None (Lotus-D 不调用 scheduler.step)")
        print(f"vae_decoder_input shape: {records['vae_decoder_input']}")
        print(f"vae_decoder_output shape: {records['vae_decoder_output']}")
        print(f"postprocessed_output shape: {records['postprocessed_output']}")
        print(f"U-Net 配置输入 channels: {pipeline.unet.config.in_channels}")

    print("Lotus-D inference completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
