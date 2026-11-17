"""Per-frame log-depth target shared by the synthetic pretraining loaders.

The MS2 fine-tune trains on `log_truncnorm` (lotus/utils/ms2_thermal_dataset.py): log depth,
per-frame quantile bounds, mapped to [-1, 1] with near at -1 and far at +1. Lotus's
`trunc_disparity` maps 1/depth the same way, so near is +1 there: a U-Net pretrained on
disparity starts the MS2 fine-tune with its output reversed as well as re-spaced. This is the
same formula as the MS2 dataset, for Hypersim and vKITTI (docs/IRIS_SYNTH_PRETRAIN_V1.md).
"""
import torch


def log_truncnorm(depth, valid, q_min, q_max):
    """Map depth to [-1, 1] by its log, between the q_min and q_max quantiles of the valid pixels.

    The caller clips to [-1, 1] as it does for every other norm_type.
    """
    if not 0.0 <= q_min < 0.5 < q_max <= 1.0:
        raise ValueError(f"log_truncnorm bounds ({q_min}, {q_max}) out of range")
    log_depth = torch.log(depth.clamp(min=1e-6))
    values = log_depth[valid]
    if values.numel() == 0:
        raise ValueError("log_truncnorm needs at least one valid depth pixel")
    dmin = torch.quantile(values, q_min)
    dmax = torch.quantile(values, q_max)
    return ((log_depth - dmin) / (dmax - dmin + 1e-5) - 0.5) * 2.0
