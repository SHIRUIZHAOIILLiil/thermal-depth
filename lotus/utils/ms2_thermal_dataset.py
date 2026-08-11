"""MS2 thermal + completed pseudo GT, in the shape `train_iris_g.py` already eats.

This is a drop-in replacement for `VKITTIDataset` so Iris's own trainer can run
unchanged on MS2. Two things differ from VKITTI, and only two:

  * the image is a high-bit-depth thermal frame, decoded through the shared
    `thermal_to_lotus_input` (AGENTS.md forbids `convert("RGB")` on raw `I;16`);
  * the depth map is the completed pseudo GT -- this frame's real lidar written
    back over calibrated AnyThermal pseudo depth -- rather than a rendered map.

The completion is what makes this dataset usable here at all. Iris encodes the
depth map as a whole image and masks latent cells by 8x8 max-pooling of the
invalid mask (`train_iris_g.py:1054`), so one missing pixel voids a whole cell;
MS2's lidar reaches about a quarter of the frame and would void nearly all of
them. Filled in, every cell survives, which is exactly the supervision this
line is built on.

The normalisation block below is copied verbatim from `vkitti_dataset.py` so the
five `--norm_type` options behave identically; Iris's scripts select
`trunc_disparity`. Sky lands on this contract for free: the completed map is
clipped at `d_max`, so sky sits at exactly 80 m, which is how VKITTI encodes it
too -- `_get_valid_mask` excludes it from the quantiles and `_get_sky_mask`
hands it back to the trainer, which counts it as valid for the latent mask.

Note on `--random_flip`: Iris flips VKITTI and Hypersim while leaving the text
description untouched, so a caption saying "on the left" can end up describing
the right. We keep that behaviour rather than silently diverging; pass
`random_flip=False` if you want the captions to stay truthful.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.anythermal_lotus_v2 import thermal_to_lotus_input  # noqa: E402


class MS2ThermalTransform:
    """Random horizontal flip. MS2 thermal is 640x256, already a multiple of 8,
    so there is no crop or resize to mirror VKITTI's KB-crop with."""

    def __init__(self, random_flip):
        self.random_flip = random_flip

    def __call__(self, image, depth):
        """image (3,h,w) in [-1,1]; depth (1,h,w) in metres."""
        if self.random_flip and torch.rand(1) > 0.5:
            image = TF.hflip(image)
            depth = torch.flip(depth, [-1])
        return image, depth


class MS2ThermalDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        ms2_root,
        pseudo_gt_dir,
        transform=None,
        norm_type="trunc_disparity",
        truncnorm_min=0.02,
        depth_scale=256.0,
        d_min=1e-3,
        d_max=80.0,
        use_captions=True,
    ):
        """
        Args:
            manifest_path: jsonl with id / thermal_path / thermal_depth_path / caption.
            ms2_root: the single root every manifest path is relative to.
            pseudo_gt_dir: calibrated_pseudo_depth/*.npy from build_anythermal_pseudo_gt.py.
            norm_type: as in vkitti_dataset; Iris's scripts pass trunc_disparity.
            use_captions: False feeds every frame the empty string, which is the
                no-text ablation, not a missing-data fallback.
        """
        self.ms2_root = Path(ms2_root)
        self.pseudo_gt_dir = Path(pseudo_gt_dir)
        self.transform = transform
        self.norm_type = norm_type
        self.truncnorm_min = truncnorm_min
        self.truncnorm_max = 1 - truncnorm_min
        self.depth_scale = depth_scale
        self.d_min = d_min
        self.d_max = d_max
        self.use_captions = use_captions

        self.rows = []
        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                image_field = row.get("thermal_path")
                depth_field = row.get("thermal_depth_path") or row.get("depth_path")
                if not image_field or not depth_field:
                    raise ValueError(f"Row {row.get('id')} lacks a thermal input or its thermal-view GT")
                # Conclusion 15: thermal-view and RGB-view GT are different exams.
                if row.get("rgb_depth_path") and "/thr/" not in str(depth_field).replace("\\", "/"):
                    raise ValueError(f"Row {row.get('id')}: GT {depth_field} is not the thermal view")
                self.rows.append({
                    "id": str(row["id"]),
                    "image_path": str(self.ms2_root / image_field),
                    "depth_path": str(self.ms2_root / depth_field),
                    "caption": str(row.get("caption", "")),
                })
        if not self.rows:
            raise ValueError(f"{manifest_path} produced no rows")
        if not self.pseudo_gt_dir.is_dir():
            raise FileNotFoundError(f"No pseudo GT directory at {self.pseudo_gt_dir}")
        print(f"[ms2] {len(self.rows)} frames from {os.path.basename(str(manifest_path))}, "
              f"completed depth from {self.pseudo_gt_dir}, norm_type={self.norm_type}, "
              f"captions={'on' if self.use_captions else 'off'}", flush=True)

    def __len__(self):
        return len(self.rows)

    def _completed_depth(self, row):
        """Real lidar written back over calibrated pseudo depth, in metres.

        Where lidar spoke, lidar wins; the pseudo map fills the rest. Out-of-range
        pseudo depth is clipped rather than dropped: this map is encoded as one
        image, so there is nowhere to drop a pixel to.
        """
        gt = np.asarray(Image.open(row["depth_path"]), dtype=np.float32) / self.depth_scale
        real = np.isfinite(gt) & (gt > self.d_min) & (gt < self.d_max)
        pseudo_path = self.pseudo_gt_dir / f"{row['id']}.npy"
        if not pseudo_path.is_file():
            # A missing file would quietly turn the target back into sparse lidar.
            raise FileNotFoundError(f"No pseudo depth for {row['id']}: {pseudo_path}")
        pseudo = np.load(pseudo_path, allow_pickle=False).astype(np.float32)
        if pseudo.shape != gt.shape:
            raise ValueError(f"{row['id']}: pseudo {pseudo.shape} vs GT {gt.shape}")
        dense = np.clip(np.where(real, gt, pseudo), self.d_min, self.d_max)
        return torch.from_numpy(dense)[None]  # (1, h, w)

    def __getitem__(self, idx):
        row = self.rows[idx]
        example = {"image_path": row["image_path"], "depth_path": row["depth_path"]}

        # processing_res=0 keeps MS2's native 640x256; both edges divide by 8.
        thermal = thermal_to_lotus_input(row["image_path"], processing_res=0)
        if thermal.diagnostics["converted_uint8_std"] <= 0:
            raise RuntimeError(f"Constant thermal conversion: {row['id']}")
        image = thermal.tensor[0]  # (3, h, w) in [-1, 1]

        depth = self._completed_depth(row)
        if self.transform:
            image, depth = self.transform(image, depth)

        valid_mask_raw = self._get_valid_mask(depth).clone()
        sky_mask_raw = self._get_sky_mask(depth).clone()
        example["valid_mask_values"] = valid_mask_raw
        example["sky_mask_values"] = sky_mask_raw
        example["pixel_values"] = image

        # --- verbatim from vkitti_dataset.py so every norm_type matches Iris ---
        valid_mask = valid_mask_raw & (depth > 0)
        if self.norm_type == "instnorm":
            dmin = depth[valid_mask].min()
            dmax = depth[valid_mask].max()
            depth_norm = ((depth - dmin) / (dmax - dmin + 1e-5) - 0.5) * 2.0
        elif self.norm_type == "truncnorm":
            dmin = torch.quantile(depth[valid_mask], self.truncnorm_min)
            dmax = torch.quantile(depth[valid_mask], self.truncnorm_max)
            depth_norm = ((depth - dmin) / (dmax - dmin + 1e-5) - 0.5) * 2.0
        elif self.norm_type == "perscene_norm":
            depth_norm = ((depth / self.d_max) - 0.5) * 2.0
        elif self.norm_type == "disparity":
            disparity = 1 / depth
            disparity_min = disparity[valid_mask].min()
            disparity_max = disparity[valid_mask].max()
            disparity_norm = ((disparity - disparity_min) / (disparity_max - disparity_min + 1e-5) - 0.5) * 2.0
            depth_norm = disparity_norm
        elif self.norm_type == "trunc_disparity":
            disparity = 1 / depth
            disparity_min = torch.quantile(disparity[valid_mask], self.truncnorm_min)
            disparity_max = torch.quantile(disparity[valid_mask], self.truncnorm_max)
            disparity_norm = ((disparity - disparity_min) / (disparity_max - disparity_min + 1e-5) - 0.5) * 2.0
            depth_norm = disparity_norm
        else:
            raise TypeError(f"Not supported normalization type: {self.norm_type}. ")

        depth_norm = depth_norm.clip(-1, 1)
        depth_norm = depth_norm.repeat(3, 1, 1)  # (3, h, w)
        example["depth_values"] = depth_norm

        example["text_description"] = row["caption"] if self.use_captions else ""
        return example

    def _get_valid_mask(self, depth: torch.Tensor):
        # `>=` where VKITTI has `>`, and the one character matters here: the
        # completed map is clipped into [d_min, d_max], so a pixel whose pseudo
        # depth came out non-positive sits at exactly d_min. Under a strict `>` it
        # would belong to neither mask, and 8x8 pooling would then void its whole
        # latent cell -- 64 supervised pixels lost to one. VKITTI's depth is
        # rendered rather than clipped and never lands on the bound.
        return torch.logical_and((depth >= self.d_min), (depth < self.d_max)).bool()

    def _get_sky_mask(self, depth: torch.Tensor):
        # Completion clipped everything past d_max to exactly d_max, which is how
        # VKITTI encodes sky; the trainer adds this back into the latent's valid mask.
        return torch.logical_and((depth > self.d_min), (depth >= self.d_max)).bool()


def collate_fn_ms2(examples):
    """Same keys as `collate_fn_vkitti`, minus `normal_values`.

    The depth task reads `batch["depth_values"]` and never touches the normal
    map; emitting a zero tensor under that name would only make an absent
    modality look present.
    """
    image_pathes = [example["image_path"] for example in examples]
    depth_pathes = [example["depth_path"] for example in examples]
    text_descriptions = [example["text_description"] for example in examples]

    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    depth_values = torch.stack([example["depth_values"] for example in examples])
    depth_values = depth_values.to(memory_format=torch.contiguous_format).float()

    valid_mask_values = torch.stack([example["valid_mask_values"] for example in examples])
    valid_mask_values = valid_mask_values.to(memory_format=torch.contiguous_format).float()

    sky_mask_values = torch.stack([example["sky_mask_values"] for example in examples])
    sky_mask_values = sky_mask_values.to(memory_format=torch.contiguous_format).float()

    return {
        "pixel_values": pixel_values,
        "depth_values": depth_values,
        "valid_mask_values": valid_mask_values,
        "sky_mask_values": sky_mask_values,
        "image_pathes": image_pathes,
        "depth_pathes": depth_pathes,
        "text_descriptions": text_descriptions,
    }
