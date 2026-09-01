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
from tools.metric_depth_norm import depth_to_unit  # noqa: E402


class MS2ThermalTransform:
    """Random horizontal flip. MS2 thermal is 640x256, already a multiple of 8,
    so there is no crop or resize to mirror VKITTI's KB-crop with."""

    def __init__(self, random_flip):
        self.random_flip = random_flip

    def __call__(self, image, depth, *extra):
        """image (3,h,w) in [-1,1]; depth (1,h,w) in metres.

        `extra` holds any further pixel-aligned map that must move with the
        image -- the sparse lidar depth and its mask, once the metric loss needs
        them.  A map left out of this call would keep the unflipped geometry and
        supervise the prediction against a mirrored scene on half the steps,
        which is the kind of bug that costs a training run rather than a crash.
        """
        if self.random_flip and torch.rand(1) > 0.5:
            image = TF.hflip(image)
            depth = torch.flip(depth, [-1])
            extra = tuple(torch.flip(item, [-1]) for item in extra)
        return (image, depth, *extra)


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
        metric_norm=None,
        sky_mask_dir=None,
        pure_pseudo_target=False,
    ):
        """
        Args:
            manifest_path: jsonl with id / thermal_path / thermal_depth_path / caption.
            ms2_root: the single root every manifest path is relative to.
            pseudo_gt_dir: calibrated_pseudo_depth/*.npy from build_anythermal_pseudo_gt.py.
            norm_type: as in vkitti_dataset; Iris's scripts pass trunc_disparity.
            use_captions: False feeds every frame the empty string, which is the
                no-text ablation, not a missing-data fallback.
            metric_norm: a `tools.metric_depth_norm.MetricNorm`. Required by, and
                only used by, norm_type="global_metric_disparity". Its two
                constants come from the training split and never move, which is
                what gives the target -- and therefore the model -- absolute
                scale. Every other norm_type normalises each frame separately
                and cannot.
            sky_mask_dir: Mask2Former sky masks from tools/build_sky_masks.py.
                Given one, the completed target's sky pixels are overwritten with
                `d_max` rather than keeping the pseudo network's guess there.

                This corrects the target, it does not add a term. Measured over
                312 training frames (tools/audit_sky_vs_pseudo.py), the completed
                target puts its sky at a median of 26.8 m -- 36.9% of sky pixels
                under 20 m, only 0.3% at the cap -- and the trained model reads
                24.5 m up there. The model is reproducing what it was taught, not
                drifting. Sky is where a monocular estimate has least to go on and
                where the physical prior is strongest, so the order is: lidar,
                then the segmentation, then the estimate.

                Precedence rather than a weighted average, because the two signals
                are not both right and there is no ratio worth picking between
                them. A pixel carrying lidar keeps its lidar value whatever the
                mask says.
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
        self.metric_norm = metric_norm
        self.sky_mask_dir = Path(sky_mask_dir) if sky_mask_dir else None
        self.pure_pseudo_target = bool(pure_pseudo_target)
        if self.sky_mask_dir is not None and not self.sky_mask_dir.is_dir():
            raise FileNotFoundError(f"No sky mask directory at {self.sky_mask_dir}")
        # How much of the target the sky rule actually rewrote, accumulated over
        # the epoch. A rule that fires on almost nothing and a rule that fires on
        # a quarter of the frame are different experiments, and only the log can
        # tell them apart afterwards.
        self.sky_override_pixels = 0
        self.sky_override_total = 0
        self.sky_frames_without_sky = 0
        if norm_type == "global_metric_disparity":
            if metric_norm is None:
                raise ValueError(
                    "norm_type='global_metric_disparity' needs metric_norm; fit it "
                    "with tools/fit_metric_norm.py on the training split."
                )
            if (metric_norm.min_depth, metric_norm.max_depth) != (d_min, d_max):
                # The constants were estimated under one validity range; using
                # them under another silently changes which pixels they describe.
                raise ValueError(
                    f"metric_norm range ({metric_norm.min_depth}, {metric_norm.max_depth}) "
                    f"does not match the dataset's ({d_min}, {d_max})"
                )
            if metric_norm.depth_scale != depth_scale:
                raise ValueError(
                    f"metric_norm depth_scale {metric_norm.depth_scale} != {depth_scale}"
                )
        # Clip accounting for the global target, accumulated over the epoch.
        # The quantiles are 2/98, so about 4% of pixels are expected to land on a
        # bound; a much larger share means the constants no longer describe what
        # is being fed, and that has to be visible rather than inferred.
        self.clip_low = 0
        self.clip_high = 0
        self.clip_total = 0
        self.clip_frames = 0

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

        Returns `(dense, real_depth, real_mask)`. The last two are the *sparse*
        lidar, which the latent objective never needed but the metric loss does:
        the metric anchor may only be read where the sensor actually measured,
        and the completed map cannot say where that was once the two are mixed.
        """
        gt = np.asarray(Image.open(row["depth_path"]), dtype=np.float32) / self.depth_scale
        real = np.isfinite(gt) & (gt > self.d_min) & (gt < self.d_max)
        pseudo_path = self.pseudo_gt_dir / f"{row['id']}.npy" if self.pseudo_gt_dir else None
        if pseudo_path is None or not pseudo_path.is_file():
            # A missing file would quietly turn the target back into sparse lidar.
            raise FileNotFoundError(f"No pseudo depth for {row['id']}: {pseudo_path}")
        pseudo = np.load(pseudo_path, allow_pickle=False).astype(np.float32)
        if pseudo.shape != gt.shape:
            raise ValueError(f"{row['id']}: pseudo {pseudo.shape} vs GT {gt.shape}")
        # `pure_pseudo_target` leaves the pseudo map alone. The latent objective
        # then learns from one continuous surface rather than from a smooth map
        # with a measured value stamped into a quarter of its pixels, and the
        # real returns are left for whichever term is meant to read them --
        # which is the only place they can be told apart, since after this line
        # the completed map cannot say which pixels were measured.
        dense = np.clip(pseudo if self.pure_pseudo_target else np.where(real, gt, pseudo),
                        self.d_min, self.d_max)
        if self.sky_mask_dir is not None:
            mask_path = self.sky_mask_dir / f"{row['id']}.png"
            if not mask_path.is_file():
                # build_sky_masks.py --save-masks writes a file for every frame,
                # all-zero ones included, so a missing file means the mask set
                # does not cover this manifest rather than "this frame has no sky".
                raise FileNotFoundError(f"No sky mask for {row['id']}: {mask_path}")
            sky = np.asarray(Image.open(mask_path)) > 127
            if sky.shape != dense.shape:
                raise ValueError(f"{row['id']}: sky mask {sky.shape} vs target {dense.shape}")
            # Lidar first: where the sensor spoke, the mask does not get a vote.
            override = sky & ~real
            dense = np.where(override, self.d_max, dense)
            self.sky_override_pixels += int(override.sum())
            self.sky_override_total += int(override.size)
            if not sky.any():
                self.sky_frames_without_sky += 1
        return (
            torch.from_numpy(dense)[None],                       # (1, h, w) metres
            torch.from_numpy(np.where(real, gt, 0.0))[None],     # (1, h, w) metres, 0 = no lidar
            torch.from_numpy(real.astype(np.float32))[None],     # (1, h, w) 1 = lidar
        )

    def __getitem__(self, idx):
        row = self.rows[idx]
        example = {"image_path": row["image_path"], "depth_path": row["depth_path"]}

        # processing_res=0 keeps MS2's native 640x256; both edges divide by 8.
        thermal = thermal_to_lotus_input(row["image_path"], processing_res=0)
        if thermal.diagnostics["converted_uint8_std"] <= 0:
            raise RuntimeError(f"Constant thermal conversion: {row['id']}")
        image = thermal.tensor[0]  # (3, h, w) in [-1, 1]

        depth, gt_depth, gt_mask = self._completed_depth(row)
        if self.transform:
            image, depth, gt_depth, gt_mask = self.transform(image, depth, gt_depth, gt_mask)

        valid_mask_raw = self._get_valid_mask(depth).clone()
        sky_mask_raw = self._get_sky_mask(depth).clone()
        example["valid_mask_values"] = valid_mask_raw
        example["sky_mask_values"] = sky_mask_raw
        example["pixel_values"] = image
        # Sparse real lidar, in metres, and where it exists. The metric loss
        # reads these two and nothing else; every other target in this file has
        # been through a normalisation that removed absolute scale.
        example["gt_depth_values"] = gt_depth
        example["gt_valid_values"] = gt_mask

        # --- verbatim from vkitti_dataset.py so every norm_type matches Iris ---
        valid_mask = valid_mask_raw & (depth > 0)
        # The two constants this frame was normalised by. Normalisation is
        # otherwise a one-way trip: a loss that wants to score the decoded
        # prediction in metres has to undo it, and after this block the bounds
        # are gone. NaN marks a norm_type whose inverse is not this pair, and
        # the trainer refuses to score in metres under one.
        norm_lo = float("nan")
        norm_hi = float("nan")
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
            norm_lo, norm_hi = float(disparity_min), float(disparity_max)
        elif self.norm_type == "global_metric_disparity":
            # The one branch whose two constants do not come from this frame.
            # Same functional form as trunc_disparity directly above -- the
            # quantiles are simply taken once, on the training split, and frozen
            # (tools/fit_metric_norm.py). That is what leaves absolute scale in
            # the target, and it is the whole of the representational change.
            depth_norm, _, report = depth_to_unit(depth, self.metric_norm)
            self.clip_low += report.clipped_low
            self.clip_high += report.clipped_high
            self.clip_total += report.total
            self.clip_frames += 1
        else:
            raise TypeError(f"Not supported normalization type: {self.norm_type}. ")

        depth_norm = depth_norm.clip(-1, 1)
        depth_norm = depth_norm.repeat(3, 1, 1)  # (3, h, w)
        example["depth_values"] = depth_norm
        example["norm_bounds"] = torch.tensor([norm_lo, norm_hi], dtype=torch.float32)

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

    def reset_clip_counters(self) -> None:
        """Zero the clip accounting, so a caller can measure a known set of frames.

        The counters cannot accumulate across a training epoch: with
        `num_workers > 0` each worker gets its own copy of this object and its
        increments never return to the main process. So the only honest use is
        to reset, read a bounded set of frames in the main process, and report
        that -- which is what the startup probe does.
        """
        self.clip_low = self.clip_high = self.clip_total = self.clip_frames = 0

    def sky_summary(self) -> str:
        """What the sky precedence rule rewrote. Empty when the rule is off."""
        if self.sky_mask_dir is None:
            return "sky rule off: the completed target keeps the pseudo depth everywhere"
        if not self.sky_override_total:
            return f"sky rule on ({self.sky_mask_dir.name}): nothing read yet"
        share = self.sky_override_pixels / self.sky_override_total * 100
        return (
            f"sky rule on ({self.sky_mask_dir.name}): rewrote "
            f"{self.sky_override_pixels:,} of {self.sky_override_total:,} target pixels "
            f"({share:.2f}%) to {self.d_max:.0f} m; "
            f"{self.sky_frames_without_sky} frames had no sky at all"
        )

    def clip_summary(self) -> str:
        """How much of the target the global normalisation pinned to a bound.

        ⚠️ Counts only the frames read **in this process**. With
        `num_workers > 0` each worker holds its own copy of this dataset and its
        increments never come back, so calling this from the training loop
        reports the startup probe's frames for ever and looks like a counter
        that has frozen. Read it once, after a known set of frames, and say how
        many that was -- which is what the frame count below is for.
        """
        if self.norm_type != "global_metric_disparity":
            return f"{self.norm_type}: per-frame normalisation, no global clip to report"
        if not self.clip_total:
            return "global_metric_disparity: nothing normalised yet"
        share = (self.clip_low + self.clip_high) / self.clip_total * 100
        return (
            f"global_metric_disparity target clip to [-1,1] over {self.clip_frames} frames "
            f"read in this process (q outside "
            f"[{self.metric_norm.q_lo:.6f}, {self.metric_norm.q_hi:.6f}] 1/m, i.e. depth "
            f"outside [{self.metric_norm.depth_at_q_hi:.2f}, "
            f"{self.metric_norm.depth_at_q_lo:.2f}] m): "
            f"{self.clip_low:,} far + {self.clip_high:,} near of {self.clip_total:,} "
            f"pixels ({share:.2f}%)"
        )

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

    gt_depth_values = torch.stack([example["gt_depth_values"] for example in examples])
    gt_depth_values = gt_depth_values.to(memory_format=torch.contiguous_format).float()

    gt_valid_values = torch.stack([example["gt_valid_values"] for example in examples])
    gt_valid_values = gt_valid_values.to(memory_format=torch.contiguous_format).float()

    norm_bounds = torch.stack([example["norm_bounds"] for example in examples]).float()

    return {
        "pixel_values": pixel_values,
        "depth_values": depth_values,
        "valid_mask_values": valid_mask_values,
        "sky_mask_values": sky_mask_values,
        "gt_depth_values": gt_depth_values,
        "gt_valid_values": gt_valid_values,
        "norm_bounds": norm_bounds,
        "image_pathes": image_pathes,
        "depth_pathes": depth_pathes,
        "text_descriptions": text_descriptions,
    }
