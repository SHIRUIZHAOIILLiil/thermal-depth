"""Split near-range error by spatial scale: frame, large regions, mid structure, fine detail.

q98 decoder against NeWCRF on the 96 validation frames, SSI-log with the fit-only rule.
Blocks are 32 and 8 pixels; the parts of each frame's squared error add up exactly.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ms2_eval.io import sha256_file
from ms2_eval.official_protocol import official_valid_mask
from tools import diagnose_depth_chain as chain
from tools import diagnose_far_target_source as source_tool
from tools import diagnose_pred_binned_far as pred_bins
from tools import diagnose_same_alignment as alignment
from tools.build_anythermal_pseudo_gt import resize_to
from tools.diagnose_model_far_gap import DIAGNOSTIC_JOB

BANDS = (('0-40', 0., 40.), ('10-20', 10., 20.), ('20-40', 20., 40.))
SCALES = ('frame', 'large', 'mid', 'fine')
LARGE, FINE = 32, 8


def group_means(values, labels):
    _, index = np.unique(labels, return_inverse=True)
    return (np.bincount(index, values)/np.bincount(index))[index]


def decompose(residual, rows, cols):
    """Nested squared-error parts: frame mean, 32 px blocks, 8 px blocks, inside 8 px."""
    frame = np.full(residual.shape, residual.mean())
    large = group_means(residual, rows//LARGE*10000+cols//LARGE)
    fine = group_means(residual, rows//FINE*10000+cols//FINE)
    # An 8 px cell always sits inside one 32 px block, so these means nest.
    return np.array(((frame**2).sum(), ((large-frame)**2).sum(), ((fine-large)**2).sum(), ((residual-fine)**2).sum()))


def frame_parts(aligned, gt, sample_id):
    valid = official_valid_mask(gt, .001, 80.)
    held = source_tool.holdout_mask(valid, sample_id)
    residual = np.clip(aligned.astype(np.float64), .001, 80.)-gt.astype(np.float64)
    parts = {}
    for band, lo, hi in BANDS:
        rows, cols = np.nonzero(held & (gt >= lo) & (gt < hi))
        if rows.size:
            parts[band] = (rows.size, decompose(residual[rows, cols], rows, cols))
    return parts


def merge(totals, source, parts):
    for band, (count, values) in parts.items():
        row = totals.setdefault((source, band), [0, np.zeros(len(SCALES))])
        row[0] += count
        row[1] += values


def table(totals):
    rows = []
    for band, _, _ in BANDS:
        count, ours = totals['q98_decoder', band]
        other, newcrf = totals['newcrf', band]
        if count != other:
            raise ValueError(f'{band}: point counts differ')
        difference = (ours-newcrf).sum()
        rows.extend(dict(band=band, scale=scale, points=int(count), ours_rmse_m=float(np.sqrt(ours[i]/count)),
                         newcrf_rmse_m=float(np.sqrt(newcrf[i]/count)),
                         difference_share=float((ours[i]-newcrf[i])/difference)) for i, scale in enumerate(SCALES))
    return rows


def run(a):
    import torch
    from tools import eval_iris_decoder_only as prior
    from tools.da2_caption_data import VAL

    if not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA')
    job = a.runs/f'outputs/lotus_line_v2/iris_far_target_source_fitonly_v2/job_{DIAGNOSTIC_JOB}'
    reference = json.loads((job/'summary.json').read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    contract, _, step, weights = prior.selection(a.decoder_run)
    c = prior.parent.parent_inputs(a.runs)['contract']
    frames, _, manifest, _ = source_tool.select_frames(c, 'val', source_tool.FRAMES, a.data_root, VAL, a.out)
    if sha256_file(manifest) != reference['subsets']['val']:
        raise ValueError('val subset differs from the diagnostic job')
    model, recipe, _ = prior.build(SimpleNamespace(runs=a.runs), contract, weights)
    with (a.decoder_run/'train/test_selected_v1/val_frames.csv').open(newline='') as f:
        cached = {r['id']: float(r['rmse']) for r in csv.DictReader(f)}
    totals, far_bias, decoder_rmse = {}, [], []
    for s, (ident, raw, gt) in zip(frames, prior.pilot.iris.predict_samples(model, frames, 'frame', recipe), strict=True):
        if ident != s.sample_id:
            raise ValueError('Prediction order differs')
        newcrf = resize_to(np.load(job/'raw/val/newcrf'/f'{ident}.npy', allow_pickle=False), gt.shape).astype(np.float32)
        log_newcrf = np.log(np.maximum(newcrf, 1e-6)).astype(np.float32)
        valid = official_valid_mask(gt, .001, 80.)
        fit = valid & ~source_tool.holdout_mask(valid, ident)
        for source, value in (('q98_decoder', raw), ('newcrf', log_newcrf)):
            merge(totals, source, frame_parts(chain.aligned_depth(value, gt, fit, 'ssi_log'), gt, ident))
        pred, truth = pred_bins.points(log_newcrf, gt, ident)
        far = (truth >= 60.) & (truth < 80.)
        if far.any():
            far_bias.append(float((pred[far]-truth[far]).mean()))
        decoder_rmse.append(alignment.scores(raw, gt, 'ssi_log')['rmse'])
    checked = pred_bins.control(float(np.mean(far_bias)), float(np.mean(decoder_rmse)),
                                float(np.mean([cached[s.sample_id] for s in frames])))
    rows = table(totals)
    (a.out/'summary.json').write_text(json.dumps(dict(control=checked, scales=rows, decoder_step=step,
        script_sha256=sha256_file(__file__)), indent=2, allow_nan=False))
    print('[control]', json.dumps(checked), flush=True)
    for r in rows:
        print('[scale]', r['band'], r['scale'], r['points'], round(r['ours_rmse_m'], 3), round(r['newcrf_rmse_m'], 3),
              round(r['difference_share'], 3), flush=True)
    print('[completed]', a.out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'out', 'data-root', 'decoder-run'):
        p.add_argument('--'+name, type=Path, required=True)
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
    run(p.parse_args())
