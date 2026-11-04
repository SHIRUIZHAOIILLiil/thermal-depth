"""Local removal of censored far-depth latent targets and frozen selection rules."""
import numpy as np
import torch
from torch.nn import functional as F

ARMS = ('control', 'masked')
START = 4000
STEPS = (4250, 4500, 5000)
BANDS = ((0,20), (20,40), (40,60))
ABS_TOL, A1_TOL, BAND_TOL, GAIN = .0005, .001, .0005, .02


def release_cells(gt, valid, bounds):
    if gt.ndim != 4 or gt.shape[1] != 1 or gt.shape != valid.shape:
        raise ValueError('Expected matched Bx1xHxW LiDAR and validity tensors')
    if gt.shape[-2] % 8 or gt.shape[-1] % 8 or bounds.shape != (len(gt),2):
        raise ValueError('Expected native geometry divisible by 8 and per-frame log bounds')
    if not torch.isfinite(bounds).all():
        raise ValueError('Nonfinite normalization bounds')
    far = valid.bool() & torch.isfinite(gt) & (gt >= 60.) & (gt < 80.)
    censored = far & (gt.clamp_min(1e-6).log() > bounds[:,1,None,None,None])
    return F.max_pool2d(censored.float(), 8, 8).bool()


def latent_loss(pred, target, original_keep, released, arm):
    if arm not in ARMS or pred.shape != target.shape or original_keep.shape != pred.shape:
        raise ValueError('Unexpected arm or latent geometry')
    denominator = original_keep.sum()
    if not denominator:
        raise ValueError('No original valid latent elements')
    keep = original_keep & ~released.expand_as(pred) if arm == 'masked' else original_keep
    # Preserve the original denominator, including removed elements and channels.
    return (pred[keep].float()-target[keep].float()).square().sum()/denominator


def gates(candidate, baseline, original):
    m, b = candidate['metrics'], baseline['metrics']
    return dict(
        rmse=m['rmse'] <= b['rmse']-GAIN,
        absrel=m['abs_rel'] <= min(b['abs_rel'], original['abs_rel'])+ABS_TOL,
        a1=m['a1'] >= max(b['a1'], original['a1'])-A1_TOL,
        near=all(candidate['bands'][f'{lo}-{hi}']['abs_rel']
                 <= baseline['bands'][f'{lo}-{hi}']['abs_rel']+BAND_TOL for lo,hi in BANDS))


def select(curve, original):
    if set(curve) != {START,*STEPS}:
        raise ValueError('Incomplete validation curve')
    for row in curve.values():
        values = [row['metrics'][k] for k in ('rmse','abs_rel','a1')]
        values += [row['bands'][f'{lo}-{hi}']['abs_rel'] for lo,hi in BANDS]
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite validation scores')
        if any(row['bands'][k]['frames'] != curve[START]['bands'][k]['frames']
               for k in curve[START]['bands']):
            raise ValueError('Band support changed between validation checkpoints')
    eligible = [step for step in STEPS if all(gates(curve[step],curve[START],original).values())]
    return min(eligible, key=lambda s:(curve[s]['metrics']['rmse'],s), default=START)
