"""Training target that never sees a fixed 20% of each frame's LiDAR cells.

The cells are those of job 9229162 (8 px, one latent pixel; seed 20261017; 20% of the
cells that hold LiDAR). AnyThermal is recalibrated on the other 80%, only that 80% is
written back over it, and only that 80% reaches the image-space LiDAR loss. Inside a
held-out cell the target is therefore the calibrated pseudo depth alone, as it is
everywhere LiDAR is absent. Kept apart from the checksummed dataset module.

The raw AnyThermal maps were not migrated. The stored map is an affine function of the
resized raw map, so an affine refit of the stored map on the fit LiDAR is the same
least-squares problem as refitting the raw map, up to float rounding.
"""

import hashlib
import json

import numpy as np
import torch
from PIL import Image

from utils.ms2_thermal_dataset import MS2ThermalDataset
from tools.build_anythermal_pseudo_gt import calibrate_pseudo_depth

SEED, FRACTION, CELL = 20261017, .2, 8
CHECK_FRAMES = 8
# A refit on all LiDAR must give back the stored map: scale 1, shift 0, within these.
SCALE_TOL, SHIFT_TOL_M, MAP_TOL_M = 1e-4, 1e-3, 1e-3


def frame_seed(sample_id, seed):
    return (seed + int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16)) % (2**31)


def holdout_mask(valid, sample_id, seed=SEED, fraction=FRACTION, cell=CELL):
    """Same cells as tools/diagnose_far_target_source.holdout_mask."""
    h, w = valid.shape
    rows, cols = np.indices((h, w))
    index = rows//cell*((w+cell-1)//cell) + cols//cell
    occupied = np.unique(index[valid])
    if not len(occupied):
        raise ValueError(f'{sample_id}: no valid LiDAR')
    rng = np.random.default_rng(frame_seed(sample_id, seed))
    count = max(1, round(fraction*len(occupied)))
    return valid & np.isin(index, rng.choice(occupied, count, replace=False))


class HoldoutDataset(MS2ThermalDataset):
    """The plain completed target, built from the fit LiDAR only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.sky_mask_dir is not None or self.pure_pseudo_target or self.norm_type == 'global_metric_disparity':
            raise ValueError('The holdout target supports only the plain completed target')
        config = json.loads((self.pseudo_gt_dir.parent/'build_config.json').read_text())
        if (config['depth_scale'], config['gt_min_depth'], config['gt_max_depth']) != (self.depth_scale, self.d_min, self.d_max):
            raise ValueError(f'Pseudo build range differs from the dataset: {config}')
        self.dtype = np.dtype(config['dtype'])
        self.min_fit_pixels = int(config['min_fit_pixels'])
        held, worst = [], 0.
        for row in self.rows[:CHECK_FRAMES]:
            gt, real = self._lidar(row)
            stored = np.load(self.pseudo_gt_dir/f"{row['id']}.npy", allow_pickle=False)
            refit, info = self._refit(row, gt, real)
            error = float(np.abs(refit.astype(np.float64)-stored).max())
            if (stored.dtype != self.dtype or abs(info['scale']-1) > SCALE_TOL or abs(info['shift']) > SHIFT_TOL_M
                    or error > MAP_TOL_M):
                raise ValueError(f"{row['id']}: stored pseudo map is not this frame's LiDAR calibration "
                                 f"(scale {info['scale']}, shift {info['shift']}, max change {error} m)")
            worst = max(worst, error)
            held.append(holdout_mask(real, row['id']).sum()/real.sum())
        print(f'[ms2-holdout] refit {len(held)} stored pseudo maps on all LiDAR: max change {worst:.2e} m; '
              f'held LiDAR share {np.mean(held):.3f}; cell={CELL} fraction={FRACTION} seed={SEED}', flush=True)

    def _lidar(self, row):
        gt = np.asarray(Image.open(row['depth_path']), dtype=np.float64)/self.depth_scale
        return gt, np.isfinite(gt) & (gt > self.d_min) & (gt < self.d_max)

    def _refit(self, row, gt, mask):
        stored = np.load(self.pseudo_gt_dir/f"{row['id']}.npy", allow_pickle=False)
        pseudo, info = calibrate_pseudo_depth(stored, gt, mask, min_fit_pixels=self.min_fit_pixels)
        return pseudo.astype(self.dtype), info

    def _pseudo(self, row, gt, mask):
        return self._refit(row, gt, mask)[0]

    def _completed_depth(self, row):
        gt, real = self._lidar(row)
        fit = real & ~holdout_mask(real, row['id'])
        if fit.sum() < self.min_fit_pixels:
            raise ValueError(f"{row['id']}: only {int(fit.sum())} fit LiDAR pixels")
        pseudo = self._pseudo(row, gt, fit).astype(np.float32)
        gt = gt.astype(np.float32)
        dense = np.clip(np.where(fit, gt, pseudo), self.d_min, self.d_max)
        return (torch.from_numpy(dense)[None],
                torch.from_numpy(np.where(fit, gt, np.float32(0)))[None],
                torch.from_numpy(fit.astype(np.float32))[None])
