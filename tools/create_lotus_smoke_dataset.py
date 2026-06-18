#!/usr/bin/env python3
"""Create a tiny Lotus-compatible training dataset for pipeline smoke tests."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


SCENES = ("02", "06", "18", "20")
CONDITIONS = (
    "15-deg-left",
    "15-deg-right",
    "30-deg-left",
    "30-deg-right",
    "clone",
    "fog",
    "morning",
    "overcast",
    "rain",
    "sunset",
)
CAMERAS = ("0", "1")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-images", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    return parser.parse_args()


def source_images(directory, count):
    paths = sorted(directory.glob("*.jpg")) + sorted(directory.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No source images found in {directory}")
    return [paths[index % len(paths)] for index in range(count)]


def resized_rgb(path, size):
    with Image.open(path) as image:
        return image.convert("RGB").resize(size, Image.Resampling.BILINEAR)


def make_depth(height, width, near=2.0, far=12.0):
    x = np.linspace(near, far, width, dtype=np.float32)
    y = np.linspace(0.0, 2.0, height, dtype=np.float32)[:, None]
    return x[None, :] + y


def create_hypersim(output, images):
    root = output / "hypersim" / "train" / "ai_001_001"
    rgb_dir = root / "scene_cam_00_final_preview"
    geometry_dir = root / "scene_cam_00_geometry_hdf5"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    geometry_dir.mkdir(parents=True, exist_ok=True)

    descriptions = {}
    height, width = 768, 1024
    depth = make_depth(height, width)
    normal = np.zeros((height, width, 3), dtype=np.float32)
    normal[..., 2] = -1.0

    for index, source in enumerate(images):
        stem = f"frame.{index:04d}"
        image_path = rgb_dir / f"{stem}.tonemap.jpg"
        depth_path = geometry_dir / f"{stem}.depth_meters.hdf5"
        normal_path = geometry_dir / f"{stem}.normal_cam.hdf5"

        resized_rgb(source, (width, height)).save(image_path, quality=95)
        with h5py.File(depth_path, "w") as handle:
            handle.create_dataset("dataset", data=depth, compression="gzip")
        with h5py.File(normal_path, "w") as handle:
            handle.create_dataset("dataset", data=normal, compression="gzip")

        descriptions[str(image_path)] = {
            "description": "An indoor scene with foreground objects and a deeper background."
        }

    return descriptions


def create_vkitti(output, source):
    root = output / "vkitti"
    descriptions = {}

    for scene in SCENES:
        for condition in CONDITIONS:
            for camera in CAMERAS:
                base = root / f"Scene{scene}" / condition / "frames"
                (base / "rgb" / f"Camera_{camera}").mkdir(parents=True, exist_ok=True)
                (base / "depth" / f"Camera_{camera}").mkdir(parents=True, exist_ok=True)
                (base / "normal" / f"Camera_{camera}").mkdir(parents=True, exist_ok=True)

    base = root / "Scene02" / "clone" / "frames"
    image_path = base / "rgb" / "Camera_0" / "rgb_00000.jpg"
    depth_path = base / "depth" / "Camera_0" / "depth_00000.png"
    normal_path = base / "normal" / "Camera_0" / "normal_00000.png"

    height, width = 352, 1216
    resized_rgb(source, (width, height)).save(image_path, quality=95)
    depth_cm = np.clip(make_depth(height, width, near=5.0, far=40.0) * 100, 1, 65535)
    Image.fromarray(depth_cm.astype(np.uint16), mode="I;16").save(depth_path)
    normal = np.zeros((height, width, 3), dtype=np.uint8)
    normal[..., 0] = 127
    normal[..., 1] = 127
    normal[..., 2] = 255
    Image.fromarray(normal, mode="RGB").save(normal_path)

    descriptions[str(image_path)] = {
        "description": "A road scene with vehicles and structures extending into the distance."
    }
    return descriptions


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images = source_images(args.source_images, args.samples)

    hypersim_descriptions = create_hypersim(args.output, images)
    vkitti_descriptions = create_vkitti(args.output, images[0])
    description_dir = args.output / "descriptions"
    write_json(description_dir / "hypersim_depth_descriptions.json", hypersim_descriptions)
    write_json(description_dir / "vkitti_depth_descriptions.json", vkitti_descriptions)

    manifest = {
        "purpose": "Lotus/Iris pipeline smoke test only; not a benchmark dataset.",
        "hypersim_samples": len(hypersim_descriptions),
        "vkitti_samples": len(vkitti_descriptions),
        "source_images": [str(path) for path in images],
    }
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
