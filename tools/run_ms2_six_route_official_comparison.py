"""Six-route Lotus-G comparison on the first MS2 test sample.

Every route is evaluated and colorized by the upstream Lotus evaluator.  The
final montage only places the untouched official ``vis/*.png`` files on a
labelled canvas.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = ROOT / "lotus"
for path in (ROOT, LOTUS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluation.evaluation import evaluation_depth
from models.anythermal_encoder import AnyThermalEncoder
from models.anythermal_lotus_bridge import AnyThermalLotusBridge
from pipeline import LotusGPipeline
from utils.image_utils import resize_back, resize_max_res


ROUTES = (
    ("rgb_no_caption", "RGB", "No caption", "rgb"),
    ("thermal_no_caption", "Thermal", "No caption", "thermal"),
    ("anythermal_no_caption", "AnyThermal features", "No caption", "thermal"),
    ("rgb_caption", "RGB", "Caption", "rgb"),
    ("thermal_caption", "Thermal", "Caption", "thermal"),
    ("anythermal_caption", "AnyThermal features", "Caption", "thermal"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--processing-res", type=int, default=768)
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_first_sample(manifest: Path):
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if str(row.get("image_id", "")) != "000000":
                    raise ValueError(f"First test sample is not image_id 000000: {row.get('id')}")
                return row
    raise ValueError("Manifest is empty")


def load_image_tensor(path: Path, *, thermal: bool = False):
    with Image.open(path) as image:
        if thermal:
            raw = np.array(image, copy=True)
            uint8 = AnyThermalEncoder._array_to_uint8(raw)
            if uint8.ndim == 3:
                uint8 = uint8[..., 0]
            if float(uint8.std()) <= 0.0 or float(np.mean(uint8 == 255)) >= 0.99:
                raise ValueError("Thermal conversion is constant or saturated")
            rgb = np.repeat(uint8[..., np.newaxis], 3, axis=-1)
        else:
            rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float()
    return tensor / 127.5 - 1.0, rgb


def task_embedding(device, dtype):
    task = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype)
    return torch.cat([torch.sin(task), torch.cos(task)], dim=-1)


def raw_noise(pipeline, tensor, processing_res, seed, device, dtype):
    processed = resize_max_res(tensor, max_edge_resolution=processing_res)
    height, width = processed.shape[-2:]
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        (1, int(pipeline.unet.config.in_channels) // 2,
         height // int(pipeline.vae_scale_factor), width // int(pipeline.vae_scale_factor)),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return noise, (height, width)


def lotus_image_prediction(pipeline, tensor, prompt, noise, args, device, dtype):
    seed_everything(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
        result = pipeline(
            rgb_in=tensor.to(device),
            prompt=prompt,
            num_inference_steps=1,
            timesteps=[args.timestep],
            generator=generator,
            latents=noise.clone(),
            output_type="np",
            task_emb=task_embedding(device, dtype),
            processing_res=args.processing_res,
            match_input_res=True,
            resample_method="bilinear",
        ).images[0]
    return np.asarray(result, np.float32).mean(axis=-1)


def anythermal_prediction(anythermal, bridge, pipeline, thermal_path, prompt, noise,
                          processed_hw, original_hw, args, device, dtype):
    seed_everything(args.seed)
    encoded = anythermal.encode(thermal_path)
    target = (
        processed_hw[0] // int(pipeline.vae_scale_factor),
        processed_hw[1] // int(pipeline.vae_scale_factor),
    )
    condition = bridge(
        encoded["spatial_features"].to(device=device, dtype=dtype),
        target_size=target,
        output_channels=int(pipeline.unet.config.in_channels) // 2,
    )
    prompt_embeds, _ = pipeline.encode_prompt(
        prompt,
        device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=None,
    )
    timestep = torch.tensor(args.timestep, device=device, dtype=torch.long)
    latents = noise.clone() * pipeline.scheduler.init_noise_sigma
    latent_input = pipeline.scheduler.scale_model_input(latents, timestep)
    unet_input = torch.cat([condition, latent_input], dim=1)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
        x0 = pipeline.unet(
            unet_input,
            timestep,
            encoder_hidden_states=prompt_embeds.to(dtype=dtype),
            class_labels=task_embedding(device, dtype),
            return_dict=False,
        )[0]
        decoded = pipeline.vae.decode(
            x0 / pipeline.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        image = pipeline.image_processor.postprocess(
            decoded,
            output_type="np",
            do_denormalize=[True],
        )
    image = resize_back(
        np.asarray(image, np.float32),
        original_hw,
        InterpolationMode.NEAREST,
    )[0]
    return np.asarray(image, np.float32).mean(axis=-1)


def write_dataset_config(output, name, display, image_rel, depth_rel, args):
    filename_list = output / f"{name}_filename_list.txt"
    filename_list.write_text(f"{image_rel} {depth_rel}\n", encoding="utf-8")
    config = output / f"{name}_dataset.yaml"
    config.write_text(
        "\n".join([
            f"name: {name}",
            f"disp_name: {display}",
            "dir: .",
            f"filenames: {filename_list.as_posix()}",
            f"depth_scale: {args.depth_scale}",
            f"min_depth: {args.min_depth}",
            f"max_depth: {args.max_depth}",
            "processing_res: null",
            "",
        ]),
        encoding="utf-8",
    )
    return config


def save_prediction(prediction_root, image_rel, prediction):
    image_path = Path(image_rel)
    target = prediction_root / image_path.parent / f"pred_{image_path.stem}.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, np.asarray(prediction, np.float32))


def build_montage(route_vis, output_path):
    images = {name: Image.open(path).convert("RGB") for name, path in route_vis.items()}
    cell_w = max(image.width for image in images.values())
    cell_h = max(image.height for image in images.values())
    header_h, row_label_w = 58, 150
    canvas = Image.new("RGB", (row_label_w + 3 * cell_w, header_h + 2 * (cell_h + 38)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    small = ImageFont.load_default()
    columns = (("RGB", "rgb"), ("Thermal", "thermal"), ("AnyThermal features", "anythermal"))
    for col, (label, _) in enumerate(columns):
        x = row_label_w + col * cell_w
        draw.text((x + 12, 16), label, fill="black", font=font)
    for row, row_name in enumerate(("No caption", "Caption")):
        y = header_h + row * (cell_h + 38)
        draw.text((10, y + 12), row_name, fill="black", font=small)
        suffix = "no_caption" if row == 0 else "caption"
        for col, (_, prefix) in enumerate(columns):
            key = f"{prefix}_{suffix}"
            image = images[key]
            x = row_label_w + col * cell_w + (cell_w - image.width) // 2
            canvas.paste(image, (x, y + 38))
    canvas.save(output_path)
    for image in images.values():
        image.close()


def main():
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest.resolve()
    ms2_root = args.ms2_root.resolve()
    sample = load_first_sample(manifest)
    caption = str(sample.get("caption", "")).strip()
    if not caption:
        raise ValueError("Test sample has no caption")
    rgb_path = ms2_root / sample["rgb_path"]
    thermal_path = ms2_root / sample["thermal_path"]
    rgb_tensor, rgb_u8 = load_image_tensor(rgb_path)
    thermal_tensor, thermal_u8 = load_image_tensor(thermal_path, thermal=True)
    inputs = output / "inputs"
    inputs.mkdir()
    Image.fromarray(rgb_u8).save(inputs / "rgb.png")
    Image.fromarray(thermal_u8).save(inputs / "thermal.png")

    device, dtype = torch.device("cuda"), torch.float16
    pipeline = LotusGPipeline.from_pretrained(
        args.lotus_model_path,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)
    for module in (pipeline.vae, pipeline.text_encoder, pipeline.unet):
        module.requires_grad_(False).eval()
    anythermal = AnyThermalEncoder(
        model_path=args.anythermal_model_path,
        device="cuda",
        local_files_only=args.local_files_only,
    )
    bridge = AnyThermalLotusBridge().to(device).eval()
    rgb_noise, _ = raw_noise(pipeline, rgb_tensor, args.processing_res, args.seed, device, dtype)
    thermal_noise, thermal_processed_hw = raw_noise(
        pipeline, thermal_tensor, args.processing_res, args.seed, device, dtype
    )
    prompts = {"no_caption": "", "caption": caption}
    predictions = {}
    for mode, prompt in prompts.items():
        predictions[f"rgb_{mode}"] = lotus_image_prediction(
            pipeline, rgb_tensor, prompt, rgb_noise, args, device, dtype
        )
        predictions[f"thermal_{mode}"] = lotus_image_prediction(
            pipeline, thermal_tensor, prompt, thermal_noise, args, device, dtype
        )
        predictions[f"anythermal_{mode}"] = anythermal_prediction(
            anythermal,
            bridge,
            pipeline,
            thermal_path,
            prompt,
            thermal_noise,
            thermal_processed_hw,
            thermal_u8.shape[:2],
            args,
            device,
            dtype,
        )

    rgb_config = write_dataset_config(
        output,
        "ms2_rgb",
        "MS2 RGB-view filtered LiDAR",
        sample["rgb_path"],
        sample["rgb_depth_path"],
        args,
    )
    thermal_config = write_dataset_config(
        output,
        "ms2_thermal",
        "MS2 thermal-view filtered LiDAR",
        sample["thermal_path"],
        sample["thermal_depth_path"],
        args,
    )
    route_vis, metrics = {}, {}
    for route_name, _, _, view in ROUTES:
        prediction_root = output / "raw_predictions" / route_name
        image_rel = sample["rgb_path"] if view == "rgb" else sample["thermal_path"]
        save_prediction(prediction_root, image_rel, predictions[route_name])
        eval_dir = output / "official_eval" / route_name
        config = rgb_config if view == "rgb" else thermal_config
        tracker = evaluation_depth(
            str(eval_dir),
            str(config),
            str(ms2_root),
            eval_mode="load_prediction",
            pred_suffix=".npy",
            prediction_dir=str(prediction_root),
            alignment="least_square_disparity",
            save_pred_vis=True,
            processing_res=None,
        )
        vis_files = list((eval_dir / "vis").glob("*.png"))
        if len(vis_files) != 1:
            raise RuntimeError(f"Expected one official vis for {route_name}, got {vis_files}")
        route_vis[route_name] = vis_files[0]
        metrics[route_name] = tracker.result()

    build_montage(route_vis, output / "comparison_official_vis.png")
    metadata = {
        "sample_id": sample["id"],
        "split": sample.get("split"),
        "caption": caption,
        "seed": args.seed,
        "timestep": args.timestep,
        "processing_res": args.processing_res,
        "lotus_model": args.lotus_model_path,
        "anythermal_model": args.anythermal_model_path,
        "evaluator": "lotus/evaluation/evaluation.py::evaluation_depth",
        "visualizer": "lotus/utils/image_utils.py::colorize_depth_map",
        "metric_comparison_rule": "RGB-view metrics and thermal-view metrics are not ranked together",
        "route_metrics": metrics,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output": str(output), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
