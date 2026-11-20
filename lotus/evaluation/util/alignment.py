# Author: Bingxin Ke
# Last modified: 2024-01-11

import numpy as np
import torch


def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred


# ******************** disparity space ********************
def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity


def disparity2depth(disparity, **kwargs):
    return depth2disparity(disparity, **kwargs)


# ******************** log space (not part of Lotus / Iris) ********************
def align_depth_least_square_log(gt_arr, pred_arr, valid_mask_arr, max_resolution=None):
    """Affine fit of the raw prediction to log GT, then exp: the MS2 `ssi_log` protocol.

    For models trained on a log-depth target (docs/IRIS_SYNTH_PRETRAIN_V1.md). Lotus and Iris
    evaluate in disparity space; numbers from this mode are not comparable with their tables.
    The exponent is clipped to [-9, 9] before exp, as ms2_eval/official_protocol.py does.
    """
    gt = np.asarray(gt_arr)
    valid = np.asarray(valid_mask_arr).astype(bool) & (gt > 0)
    gt_log = np.zeros(gt.shape, np.float64)
    gt_log[valid] = np.log(gt[valid].astype(np.float64))
    log_pred, scale, shift = align_depth_least_square(
        gt_arr=gt_log, pred_arr=pred_arr, valid_mask_arr=valid,
        return_scale_shift=True, max_resolution=max_resolution,
    )
    return np.exp(np.clip(log_pred, -9.0, 9.0)), scale, shift
