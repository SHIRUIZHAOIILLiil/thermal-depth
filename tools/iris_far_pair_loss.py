"""Local LiDAR pair selection and the trainable scope of the SD2.1 continuation."""
import numpy as np
import torch
from torch import nn


def trainable_convs(unet):
    unet.requires_grad_(False)
    start = len(unet.up_blocks) - 2
    if start < 0:
        raise ValueError('Expected at least two U-Net up blocks')
    params = {}
    for i in range(start, start+2):
        count = len(params)
        for name, module in unet.up_blocks[i].named_modules():
            if name.startswith(('resnets.', 'upsamplers.')) and isinstance(module, nn.Conv2d):
                for key, p in module.named_parameters(recurse=False):
                    params[f'up_blocks.{i}.{name}.{key}'] = p.requires_grad_(True)
        if len(params) == count:
            raise ValueError(f'No eligible convolution in up_blocks.{i}')
    return params


def local_pairs(gt, seed):
    """Sample unique unordered pairs using LiDAR geometry alone, in native pixels."""
    if gt.ndim != 2:
        raise ValueError('Expected native HxW GT')
    h,w = gt.shape
    valid = np.isfinite(gt) & (gt > .001) & (gt < 80.)
    far = np.flatnonzero(valid & (gt >= 40.))
    rng = np.random.default_rng(seed)
    if len(far) > 1024:
        far = rng.choice(far, 1024, replace=False)
    ys, xs = np.divmod(far, w)
    pairs = []
    for d in (8,16,32):
        for dy,dx in ((0,d),(0,-d),(d,0),(-d,0),(d,d),(d,-d),(-d,d),(-d,-d)):
            y,x = ys+dy, xs+dx
            inside = (y>=0) & (y<h) & (x>=0) & (x<w)
            a,b = far[inside], y[inside]*w+x[inside]
            keep = valid.flat[b] & (np.abs(gt.flat[a]-gt.flat[b]) >= 5.)
            pairs.append(np.column_stack((np.minimum(a[keep],b[keep]),np.maximum(a[keep],b[keep]))))
    selected = np.unique(np.concatenate(pairs), axis=0)
    if len(selected) > 256:
        selected = selected[rng.choice(len(selected), 256, replace=False)]
    return selected


def signed_gap_loss(aligned, gt, pairs, usable):
    """Mean absolute error of the signed metric differences, averaged over eligible frames."""
    losses, count = [], 0
    for i, indices in enumerate(pairs):
        if not usable[i] or not len(indices):
            continue
        a,b = torch.as_tensor(indices.T.copy(), device=gt.device, dtype=torch.long)
        p = aligned[i].flatten().clamp(.001, 80.)
        g = gt[i].flatten()
        losses.append(((p[a]-p[b]) - (g[a]-g[b])).abs().mean())
        count += len(indices)
    return (torch.stack(losses).mean() if losses else aligned.new_zeros(())), count
