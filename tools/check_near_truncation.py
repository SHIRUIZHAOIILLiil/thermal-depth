"""Does the near-side 2% quantile clip of the training target cost AbsRel at 0-10 m? CPU only, train frames.

DIAGNOSTIC ONLY (train targets, no model, no val/test). docs/IRIS_SYNTH_PRETRAIN_V1.md.

`log_truncnorm` maps each frame's log depth between its q_min and q_max quantiles and clips to [-1, 1]
(lotus/utils/ms2_thermal_dataset.py). tmax1 moved the far bound to the frame maximum; the near bound is
still the 2% quantile, so every pixel nearer than it is taught the same depth. A model that learns the
target exactly is then too far on exactly those pixels.

Per frame, the target the dataset actually builds (depth_values, norm_bounds) is inverted to metres and
compared with the real LiDAR at its pixels. The cost is reported in the evaluator's AbsRel units: per
frame the mean over valid LiDAR pixels of |target - lidar| / lidar, then the mean over frames, split by
GT band so each band's number adds up to the frame mean (as in tools/val_pixel_gap.py v2). The same is
computed for alternative near quantiles (0.01, 0.005, 0) from the same dense map, to size a fix.

Control: the q=0.02 bound computed here equals the dataset's own norm_bounds[0], and the q=0.02 cost
from the formula equals the cost from the dataset's own clipped target, on every frame.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ms2_eval.official_protocol import official_valid_mask

BANDS = ((0, 10), (10, 20), (20, 40), (40, 80))
QUANTILES = (.02, .01, .005, 0.)
SUGGESTABLE = (.01, .005)
TRAINED_Q = .02
BOUND_TOL = 1e-5     # log metres, float32 quantile of the same tensor
COST_TOL = 1e-5      # per-frame AbsRel units, float32 target round trip
LOG_EPS = 1e-6       # float32 log of the dense map against float64 log of the LiDAR
# Pre-registered reading on the 0-10 m share of the q=0.02 cost (the val AbsRel gap at 0-10 m is 0.00242).
MATERIAL, EXCLUDED = .0010, .0003


def band_keys():
    return [f'{lo}-{hi}m' for lo, hi in BANDS]


def frame_costs(log_dense, valid_dense, lidar, lidar_valid, q_list, q_max=1.):
    """Per-band AbsRel cost of the clipped target at the LiDAR pixels, frame-normalised, for each q."""
    import torch
    values = log_dense[valid_dense]
    hi = float(torch.quantile(values, q_max))
    d = lidar[lidar_valid].astype(np.float64)
    n = d.size
    out, bounds = {}, {}
    for q in q_list:
        lo = float(torch.quantile(values, q))
        bounds[q] = lo
        target = np.exp(np.clip(np.log(d), lo, hi))
        err = np.abs(target-d)/d
        out[q] = {k: float(err[(d > a) & (d <= b)].sum()/n) for k, (a, b) in zip(band_keys(), BANDS)}
        # The bound is a float32 quantile of the same pixels: a point sitting on it is not below it.
        below = np.log(d) < lo-LOG_EPS
        out[q]['clipped_share_0-10m'] = float((below & (d <= 10)).sum()/max(((d <= 10)).sum(), 1))
        out[q]['clipped_points'] = int(below.sum())
    return out, bounds, n


def dataset_cost(depth_values, norm_bounds, lidar, lidar_valid):
    """Same q=0.02 cost, but from the dataset's own clipped target (the control)."""
    lo, hi = (float(x) for x in norm_bounds)
    v = depth_values[0].numpy().astype(np.float64)
    target = np.exp((v/2+.5)*(hi-lo+1e-5)+lo)[lidar_valid]
    d = lidar[lidar_valid].astype(np.float64)
    return float((np.abs(target-d)/d).sum()/d.size)


def reading(cost_0_10):
    if cost_0_10 >= MATERIAL:
        return 'material'
    if cost_0_10 <= EXCLUDED:
        return 'excluded'
    return 'partial'


def suggested_q(means):
    """If material: the largest alternative q whose 0-10 m cost is at most 20% of the trained one.

    q=0 is reported for sizing only and never suggested: the bound would be the frame minimum, the
    single-outlier failure that instnorm showed (memory of the log-target line, 2026-09-16).
    """
    base = means[TRAINED_Q]['0-10m']
    for q in SUGGESTABLE:
        if means[q]['0-10m'] <= .2*base:
            return q
    return None


def run(a):
    import torch
    from lotus.utils.ms2_thermal_dataset import MS2ThermalDataset
    ds = MS2ThermalDataset(a.train_manifest, a.data_root, a.pseudo_dir, transform=None,
                           norm_type='log_truncnorm', truncnorm_min=TRAINED_Q, truncnorm_max=1., use_captions=False)
    rng = np.random.default_rng(a.seed)
    picks = sorted(rng.choice(len(ds), size=min(a.frames, len(ds)), replace=False).tolist())
    sums = {q: {k: 0. for k in band_keys()} for q in QUANTILES}
    clipped_share = {q: [] for q in QUANTILES}
    lo_m = {q: [] for q in QUANTILES}
    worst = dict(bound=0., cost=0.)
    used = 0
    for i, idx in enumerate(picks, 1):
        ex = ds[idx]
        target = ex['depth_values']  # the dataset's own clipped, normalised target
        dense_m, _, _ = ds._completed_depth(ds.rows[idx])
        valid_dense = ds._get_valid_mask(dense_m) & (dense_m > 0)
        log_dense = torch.log(dense_m.clamp(min=1e-6))
        lidar = ex['gt_depth_values'][0].numpy()
        lidar_valid = (ex['gt_valid_values'][0].numpy() > 0) & official_valid_mask(lidar, .001, 80.)
        if not lidar_valid.any():
            continue
        costs, bounds, _ = frame_costs(log_dense, valid_dense, lidar, lidar_valid, QUANTILES)
        worst['bound'] = max(worst['bound'], abs(bounds[TRAINED_Q]-float(ex['norm_bounds'][0])))
        worst['cost'] = max(worst['cost'], abs(sum(costs[TRAINED_Q][k] for k in band_keys())
                                               - dataset_cost(target, ex['norm_bounds'], lidar, lidar_valid)))
        for q in QUANTILES:
            for k in band_keys():
                sums[q][k] += costs[q][k]
            clipped_share[q].append(costs[q]['clipped_share_0-10m'])
            lo_m[q].append(float(np.exp(bounds[q])))
        used += 1
        if i % 250 == 0:
            print(f'[frames] {i}/{len(picks)}', flush=True)
    control = dict(max_bound_diff=worst['bound'], bound_tol=BOUND_TOL, max_cost_diff=worst['cost'], cost_tol=COST_TOL,
                   passed=worst['bound'] <= BOUND_TOL and worst['cost'] <= COST_TOL)
    print('[control]', json.dumps(control), flush=True)
    if not control['passed']:
        raise SystemExit('the reconstructed target does not match the dataset')
    means = {q: {k: v/used for k, v in sums[q].items()} for q in QUANTILES}
    rows = {}
    for q in QUANTILES:
        lm = np.asarray(lo_m[q])
        rows[str(q)] = dict(cost_by_band=means[q], cost_total=float(sum(means[q].values())),
                            near_bound_m=dict(p10=float(np.percentile(lm, 10)), p50=float(np.median(lm)),
                                              p90=float(np.percentile(lm, 90))),
                            lidar_0_10_below_bound_share=float(np.mean(clipped_share[q])))
        print(f'[q={q:g}] near bound m p10/p50/p90 {rows[str(q)]["near_bound_m"]["p10"]:.2f}/'
              f'{rows[str(q)]["near_bound_m"]["p50"]:.2f}/{rows[str(q)]["near_bound_m"]["p90"]:.2f}  '
              f'0-10 m LiDAR below bound {100*rows[str(q)]["lidar_0_10_below_bound_share"]:.1f}%  '
              f'AbsRel cost total {rows[str(q)]["cost_total"]:.5f}  '
              + '  '.join(f'{k} {v:.5f}' for k, v in means[q].items()), flush=True)
    verdict = reading(means[TRAINED_Q]['0-10m'])
    result = dict(control=control, frames=used, seed=a.seed, quantiles=rows, reading=verdict,
                  thresholds=dict(material=MATERIAL, excluded=EXCLUDED),
                  suggested_truncnorm_min=suggested_q(means) if verdict == 'material' else None)
    print('[reading]', json.dumps(dict(reading=verdict, cost_0_10=means[TRAINED_Q]['0-10m'],
                                       suggested_truncnorm_min=result['suggested_truncnorm_min'])), flush=True)
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('[completed]', a.out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('data-root', 'train-manifest', 'pseudo-dir', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--frames', type=int, default=2000)
    p.add_argument('--seed', type=int, default=20261014)
    run(p.parse_args())
