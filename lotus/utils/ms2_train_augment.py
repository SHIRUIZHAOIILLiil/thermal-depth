"""Stronger training augmentation, kept apart from the checksummed dataset module.

Geometry zooms each axis independently and crops back. Intensity uses gamma, because a linear
gain before the per-frame percentile stretch is undone by that stretch. Training only.
"""

import os

import torch

from utils.ms2_thermal_dataset import MS2ThermalTransform


def random_scale_crop(image, depth, *extra, max_scale):
    """Image is bilinear; depth-like maps are nearest and keep their metres."""
    h, w = image.shape[-2:]
    scale_x, scale_y = torch.empty(2).uniform_(1.0, max_scale).tolist()
    scaled_h, scaled_w = int(h * scale_y), int(w * scale_x)
    top = int(torch.randint(scaled_h - h + 1, (1,)))
    left = int(torch.randint(scaled_w - w + 1, (1,)))

    def crop(item, mode, **kwargs):
        scaled = torch.nn.functional.interpolate(item[None], (scaled_h, scaled_w), mode=mode, **kwargs)[0]
        return scaled[..., top:top + h, left:left + w]

    return (crop(image, "bilinear", align_corners=False), crop(depth, "nearest"),
            *(crop(item, "nearest") for item in extra))


def random_gamma(image, low, high):
    """Log-uniform gamma on an image in [-1, 1]; one value for all channels."""
    gamma = torch.empty(1).uniform_(torch.log(torch.tensor(low)), torch.log(torch.tensor(high))).exp()
    return ((image + 1) / 2).clamp(0, 1).pow(gamma) * 2 - 1


class StrongTransform(MS2ThermalTransform):
    """Flip as before, then scale-crop (skipped at max_scale 1), then gamma."""

    def __init__(self, random_flip):
        super().__init__(random_flip)
        self.max_scale = float(os.environ["IRIS_AUG_MAX_SCALE"])
        self.gamma = tuple(float(x) for x in os.environ["IRIS_AUG_GAMMA"].split(","))
        if self.max_scale < 1 or len(self.gamma) != 2 or not 0 < self.gamma[0] < 1 < self.gamma[1]:
            raise ValueError(f"Invalid augmentation: {self.max_scale} {self.gamma}")
        print(f"[ms2-aug] flip={random_flip} max_scale={self.max_scale} gamma={self.gamma}", flush=True)

    def __call__(self, image, depth, *extra):
        image, depth, *extra = super().__call__(image, depth, *extra)
        if self.max_scale > 1:
            image, depth, *extra = random_scale_crop(image, depth, *extra, max_scale=self.max_scale)
        return (random_gamma(image, *self.gamma), depth, *extra)
