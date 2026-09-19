"""MS2 thermal as a Pixel-Perfect Depth dataset.

Copied into the cloned tree as ppd/data/ms2_thermal.py; the config refers to it
as ppd.data.ms2_thermal.Dataset. It lives in our repository rather than theirs
so that what we changed stays under our own review, and so a fresh clone of
theirs is never something we have edited.

Three overrides, because their base class already does the rest:

read_rgb  -- theirs reads 8-bit colour with cv2 and divides by 255. Ours is a
             16-bit single-channel thermal frame, and the stretch to 8 bits is
             a measured choice, not a formality: min-max costs the median frame
             14.3 points of adjacent-difference resolution, so the 1%/99% clip
             is what our own line trains under and what the comparison has to
             use on both sides.
read_depth -- the completed target: AnyThermal pseudo depth everywhere with the
             real lidar returns written over it, which is the target our line
             trains on. Supervising PPD on anything else would compare two
             models trained on different things.
build_metas -- their split file, two parallel lists of paths, generated from our
             manifest by integrations/ppd/prepare_ms2.py.
"""

from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

from ppd.data.depth_estimation import Dataset as BaseDataset

D_MIN, D_MAX = 1e-3, 80.0


class Dataset(BaseDataset):
    def build_metas(self):
        with open(self.cfg.split_path, "r", encoding="utf-8") as handle:
            split = json.load(handle)
        root = self.cfg.data_root
        name = self.cfg.split
        self.rgb_files = [os.path.join(root, p) for p in split[f"{name}_rgb_paths"]]
        self.depth_files = [os.path.join(root, p) for p in split[f"{name}_dpt_paths"]]
        # The lidar sits beside the pseudo depth rather than in depth_files,
        # because the target is the two combined and their base class reads one
        # file per frame.
        self.lidar_files = [os.path.join(root, p) for p in split[f"{name}_lidar_paths"]]
        assert len(self.rgb_files) == len(self.depth_files) == len(self.lidar_files)
        self.stretch = self.cfg.get("stretch", "percentile")
        self.depth_scale = self.cfg.get("depth_scale", 256.0)
        # Off means supervise on the raw lidar alone -- the other arm worth
        # having, and the one that needs no pseudo depth at all.
        self.overwrite_with_lidar = self.cfg.get("overwrite_with_lidar", True)

    def read_rgb(self, index):
        raw = np.asarray(Image.open(self.rgb_files[index]), dtype=np.float32)
        if self.stretch == "percentile":
            low, high = (float(v) for v in np.percentile(raw, (1.0, 99.0)))
        else:
            low, high = float(raw.min()), float(raw.max())
        unit = (np.zeros(raw.shape, np.float32) if high <= low
                else np.clip((raw - low) / (high - low), 0.0, 1.0).astype(np.float32))
        # Three identical channels: the network expects colour, and a thermal
        # frame has one band. Their base class returns HWC float32 in [0,1].
        return np.repeat(unit[:, :, None], 3, axis=2)

    def read_depth(self, index, depth=None):
        lidar = np.asarray(Image.open(self.lidar_files[index]),
                           dtype=np.float32) / self.depth_scale
        real = np.isfinite(lidar) & (lidar > D_MIN) & (lidar < D_MAX)
        if not self.overwrite_with_lidar:
            return np.clip(lidar, D_MIN, D_MAX).astype(np.float32), real.astype(np.uint8)
        pseudo = np.load(self.depth_files[index]).astype(np.float32)
        dense = np.clip(np.where(real, lidar, pseudo), D_MIN, D_MAX)
        # Dense by construction, so every pixel is supervised -- the same
        # coverage our own line trains under.
        return dense.astype(np.float32), np.ones(dense.shape, np.uint8)
