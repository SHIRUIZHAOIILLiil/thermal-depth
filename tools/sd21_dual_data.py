"""Reuse audited pseudo files, reconstruct the l100 dense log target in memory."""
import torch
from tools.sd21_pseudo_data import ThermalMixed


def log_target(gt,pseudo):
    real=torch.isfinite(gt)&(gt>.001)&(gt<80)
    dense=torch.where(real,gt,pseudo).clamp(.001,80.)
    if not torch.isfinite(dense).all(): raise ValueError('Dense latent target has holes/nonfinite values')
    valid=(dense>=.001)&(dense<80)
    if not valid.any(): raise ValueError('No valid pixels for l100 target quantiles')
    log=dense.clamp(min=1e-6).log()
    lo=torch.quantile(log[valid],.02); hi=torch.quantile(log[valid],.98)
    target=((log-lo)/(hi-lo+1e-5)-.5)*2
    return target.clamp(-1,1).unsqueeze(0).repeat(3,1,1),torch.stack((lo,hi))


class ThermalDual(ThermalMixed):
    def __getitem__(self,index):
        item=super().__getitem__(index)
        item['target'],item['bounds']=log_target(item['gt'],item.pop('pseudo'))
        return item
