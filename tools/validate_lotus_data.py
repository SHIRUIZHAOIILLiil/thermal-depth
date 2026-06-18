#!/usr/bin/env python3
"""Load one sample from each Lotus dataset used by the smoke pipeline."""

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lotus-root", type=Path, required=True)
    parser.add_argument("--smoke-root", type=Path, required=True)
    parser.add_argument("--eval-depth-root", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, str(args.lotus_root))

    from evaluation.dataset_depth import DatasetMode, get_dataset
    from utils.hypersim_dataset import get_hypersim_dataset_depth_normal
    from utils.vkitti_dataset import VKITTIDataset, VKITTITransform

    descriptions = args.smoke_root / "descriptions"
    hypersim, transform, _ = get_hypersim_dataset_depth_normal(
        str(args.smoke_root / "hypersim"),
        resolution=256,
        random_flip=False,
        norm_type="trunc_disparity",
        truncnorm_min=0.02,
        text_descriptions_path=str(
            descriptions / "hypersim_depth_descriptions.json"
        ),
    )
    hypersim_sample = hypersim.with_transform(transform)[0]
    print(
        "hypersim:",
        tuple(hypersim_sample["pixel_values"].shape),
        tuple(hypersim_sample["depth_values"].shape),
        repr(hypersim_sample["text_descriptions"]),
    )

    vkitti = VKITTIDataset(
        str(args.smoke_root / "vkitti"),
        VKITTITransform(random_flip=False),
        norm_type="trunc_disparity",
        text_descriptions_path=str(descriptions / "vkitti_depth_descriptions.json"),
    )
    vkitti_sample = vkitti[0]
    print(
        "vkitti:",
        tuple(vkitti_sample["pixel_values"].shape),
        tuple(vkitti_sample["depth_values"].shape),
        repr(vkitti_sample["text_description"]),
    )

    config = OmegaConf.load(
        args.lotus_root / "datasets/eval/depth/configs/data_scannet_val.yaml"
    )
    config.filenames = str(args.lotus_root / config.filenames)
    scannet = get_dataset(
        config,
        base_data_dir=str(args.eval_depth_root),
        mode=DatasetMode.EVAL,
    )
    scannet_sample = scannet[0]
    print(
        "scannet:",
        len(scannet),
        tuple(scannet_sample["rgb_int"].shape),
        tuple(scannet_sample["depth_raw_linear"].shape),
        scannet_sample["rgb_relative_path"],
    )


if __name__ == "__main__":
    main()
