"""Real-LiDAR SiLog plus fixed 0.1 * missing-pixel pseudo-depth log-L1."""
import torch
from models.da2_silog import masked_silog

BETA = 0.1


def pseudo_mask(gt, pseudo):
    # Finite measured depths >=80 m are out of scope, NOT missing returns.
    missing = ~torch.isfinite(gt) | (gt <= .001)
    return missing & torch.isfinite(pseudo) & (pseudo > .001) & (pseudo < 80.)


def mixed_depth_loss(prediction, gt, pseudo, beta=BETA):
    if prediction.shape != gt.shape or pseudo.shape != gt.shape:
        raise ValueError('Prediction/LiDAR/pseudo shapes must match; never resize GT')
    if beta not in (0., BETA):
        raise ValueError('Only fixed beta=.1, or beta=0 for equivalence tests')
    real, empty = masked_silog(prediction, gt)
    mask = pseudo_mask(gt, pseudo)
    count = int(mask.sum())
    # Avoid 0 * NaN and preserve EXACT baseline behavior in the beta=0 test.
    if beta == 0 or count == 0:
        auxiliary = prediction.new_zeros((), dtype=torch.float32)
    else:
        pred = prediction[mask].float()
        target = pseudo[mask].detach().float()
        if not torch.isfinite(pred).all() or not (pred > 0).all():
            raise ValueError('Pseudo log-L1 requires finite positive predictions')
        auxiliary = (pred.log()-target.log()).abs().mean()
    total = real if beta == 0 else real + beta*auxiliary
    return total, dict(real=real.detach(), pseudo=auxiliary.detach(), empty_gt_frames=empty,
                      pseudo_pixels=count, lidar_pixels=int((torch.isfinite(gt) & (gt>.001) & (gt<80)).sum()))
