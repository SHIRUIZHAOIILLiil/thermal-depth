"""Where is the 0-40 m error: depth edges, flat interiors, or whole-frame offset?

q98 decoder against NeWCRF on the 96 validation frames, both SSI-log with the fit-only rule.
Regions come from tools/diagnose_error_sources.py. Squared errors are pooled so shares add up.
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
from tools.diagnose_error_sources import boundary_regions
from tools.diagnose_model_far_gap import DIAGNOSTIC_JOB

BANDS = (('0-10', 0., 10.), ('10-20', 10., 20.), ('20-40', 20., 40.))
REGIONS = ('edge', 'between', 'interior', 'sparse')
WINDOW, EDGE_RATIO, FLAT_RATIO, MIN_NEIGHBORS = 7, 1.25, 1.05, 10


def frame_stats(aligned, gt, sample_id):
    """Region and band of every held-out point below 40 m, plus the whole-frame offset."""
    valid = official_valid_mask(gt, .001, 80.)
    held = source_tool.holdout_mask(valid, sample_id)
    regions = boundary_regions(gt, valid, WINDOW, EDGE_RATIO, FLAT_RATIO, MIN_NEIGHBORS)
    residual = np.clip(aligned.astype(np.float64), .001, 80.)-gt.astype(np.float64)
    near = held & (gt < 40.)
    rows = {}
    for region in REGIONS:
        for band, lo, hi in BANDS:
            chosen = near & regions[region] & (gt >= lo) & (gt < hi)
            if chosen.any():
                values = residual[chosen]
                rows[region, band] = (values.size, float(values.sum()), float((values**2).sum()))
    values = residual[near]
    offset = (values.size, float(values.size*values.mean()**2), float((values**2).sum()))
    return rows, offset


def merge(totals, key, rows):
    for inner, values in rows.items():
        row = totals.setdefault((*key, *inner), np.zeros(3))
        row += values


def table(totals, source_a='q98_decoder', source_b='newcrf', split='val'):
    """Each region/band row, with its share of the total squared-error difference."""
    def pick(source):
        return {k[2:]: v for k, v in totals.items() if k[:2] == (source, split)}
    a, b = pick(source_a), pick(source_b)
    if set(a) != set(b):
        raise ValueError('Models cover different region rows')
    total = sum(a[k][2]-b[k][2] for k in a)
    rows = []
    for key in sorted(a):
        n = a[key][0]
        if n != b[key][0]:
            raise ValueError(f'{key}: point counts differ')
        difference = a[key][2]-b[key][2]
        rows.append(dict(region=key[0], band=key[1], points=int(n), ours_rmse_m=float(np.sqrt(a[key][2]/n)),
                         newcrf_rmse_m=float(np.sqrt(b[key][2]/n)), ours_bias_m=float(a[key][1]/n),
                         newcrf_bias_m=float(b[key][1]/n), difference_share=float(difference/total) if total else 0.))
    return rows


def offset_share(offsets):
    """Split summed squared error into whole-frame offset and the rest."""
    n, common, whole = np.array(offsets, np.float64).sum(0)
    return dict(points=int(n), rmse_m=float(np.sqrt(whole/n)), frame_offset_rmse_m=float(np.sqrt(common/n)),
                within_rmse_m=float(np.sqrt((whole-common)/n)))


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
    totals, offsets, far_bias, decoder_rmse = {}, dict(q98_decoder=[], newcrf=[]), [], []
    for s, (ident, raw, gt) in zip(frames, prior.pilot.iris.predict_samples(model, frames, 'frame', recipe), strict=True):
        if ident != s.sample_id:
            raise ValueError('Prediction order differs')
        newcrf = resize_to(np.load(job/'raw/val/newcrf'/f'{ident}.npy', allow_pickle=False), gt.shape).astype(np.float32)
        log_newcrf = np.log(np.maximum(newcrf, 1e-6)).astype(np.float32)
        valid = official_valid_mask(gt, .001, 80.)
        fit = valid & ~source_tool.holdout_mask(valid, ident)
        for source, value in (('q98_decoder', raw), ('newcrf', log_newcrf)):
            rows, offset = frame_stats(chain.aligned_depth(value, gt, fit, 'ssi_log'), gt, ident)
            merge(totals, (source, 'val'), rows)
            offsets[source].append(offset)
        pred, truth = pred_bins.points(log_newcrf, gt, ident)
        far = (truth >= 60.) & (truth < 80.)
        if far.any():
            far_bias.append(float((pred[far]-truth[far]).mean()))
        decoder_rmse.append(alignment.scores(raw, gt, 'ssi_log')['rmse'])
    checked = pred_bins.control(float(np.mean(far_bias)), float(np.mean(decoder_rmse)),
                                float(np.mean([cached[s.sample_id] for s in frames])))
    rows = table(totals)
    split = {k: offset_share(v) for k, v in offsets.items()}
    (a.out/'summary.json').write_text(json.dumps(dict(control=checked, regions=rows, frame_offset=split,
        decoder_step=step, script_sha256=sha256_file(__file__)), indent=2, allow_nan=False))
    print('[control]', json.dumps(checked), flush=True)
    for name, row in split.items():
        print('[offset]', name, json.dumps({k: round(v, 3) for k, v in row.items()}), flush=True)
    for r in rows:
        print('[region]', r['region'], r['band'], r['points'], round(r['ours_rmse_m'], 3), round(r['newcrf_rmse_m'], 3),
              round(r['ours_bias_m'], 3), round(r['newcrf_bias_m'], 3), round(r['difference_share'], 3), flush=True)
    print('[completed]', a.out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'out', 'data-root', 'decoder-run'):
        p.add_argument('--'+name, type=Path, required=True)
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
    run(p.parse_args())
