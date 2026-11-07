"""Is the far bias systematic, or regression to the mean from binning by GT?

Same frames, held-out cells and fit-only SSI-log as the earlier diagnostics. Every held-out
point is binned twice, by GT depth and by predicted depth, with pooled statistics. No test.
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
from tools import diagnose_same_alignment as alignment
from tools.build_anythermal_pseudo_gt import resize_to
from tools.diagnose_model_far_gap import DIAGNOSTIC_JOB

NEWCRF_VAL_FAR_BIAS_M = -9.459420964125627  # run_20261107_014506, ssi_log, frame mean
CONTROL_TOLERANCE_M = .01


def points(raw_log, gt, sample_id):
    """Held-out (pred, gt) after a fit that cannot see them."""
    valid = official_valid_mask(gt, .001, 80.)
    held = source_tool.holdout_mask(valid, sample_id)
    aligned = chain.aligned_depth(raw_log, gt, valid & ~held, 'ssi_log')
    return np.clip(aligned[held].astype(np.float64), .001, 80.), gt[held].astype(np.float64)


def add(totals, key, pred, gt, axis):
    for band, lo, hi in source_tool.BANDS:
        selected = (axis >= lo) & ((axis < hi) | (hi == source_tool.BANDS[-1][2]))
        if selected.any():
            residual = pred[selected]-gt[selected]
            row = totals.setdefault((*key, band), np.zeros(3))
            row += (residual.size, residual.sum(), (residual**2).sum())


def rows(totals):
    return [dict(zip(('source', 'split', 'binned_by', 'band'), key), points=int(n), bias_m=float(s/n),
                 rmse_m=float(np.sqrt(q/n))) for key, (n, s, q) in sorted(totals.items())]


def accumulate(totals, source, split, pred, gt):
    add(totals, (source, split, 'gt'), pred, gt, gt)
    add(totals, (source, split, 'prediction'), pred, gt, pred)


def control(frame_bias, decoder_rmse, cached):
    far = abs(frame_bias-NEWCRF_VAL_FAR_BIAS_M) <= CONTROL_TOLERANCE_M
    rmse = abs(decoder_rmse-cached) <= alignment.TOLERANCE_M
    return dict(newcrf_val_far_bias_m=frame_bias, decoder_rmse=decoder_rmse, decoder_cached_rmse=cached, passed=far and rmse)


def run(a):
    import torch
    from tools import eval_iris_decoder_only as prior
    from tools.da2_caption_data import TRAIN, VAL

    if not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA')
    job = a.runs/f'outputs/lotus_line_v2/iris_far_target_source_fitonly_v2/job_{DIAGNOSTIC_JOB}'
    reference = json.loads((job/'summary.json').read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    contract, _, step, weights = prior.selection(a.decoder_run)
    c = prior.parent.parent_inputs(a.runs)['contract']
    splits = {}
    for split, allowed in (('train', TRAIN), ('val', VAL)):
        frames, _, manifest, _ = source_tool.select_frames(c, split, source_tool.FRAMES, a.data_root, allowed, a.out)
        if sha256_file(manifest) != reference['subsets'][split]:
            raise ValueError(f'{split} subset differs from the diagnostic job')
        splits[split] = frames
    model, recipe, _ = prior.build(SimpleNamespace(runs=a.runs), contract, weights)
    with (a.decoder_run/'train/test_selected_v1/val_frames.csv').open(newline='') as f:
        cached = {r['id']: float(r['rmse']) for r in csv.DictReader(f)}
    totals, far_bias, decoder_rmse = {}, [], []
    for split, frames in splits.items():
        for s, (ident, raw, gt) in zip(frames, prior.pilot.iris.predict_samples(model, frames, 'frame', recipe), strict=True):
            if ident != s.sample_id:
                raise ValueError('Prediction order differs')
            accumulate(totals, 'q98_decoder', split, *points(raw, gt, ident))
            newcrf = resize_to(np.load(job/'raw'/split/'newcrf'/f'{ident}.npy', allow_pickle=False), gt.shape).astype(np.float32)
            log_newcrf = np.log(np.maximum(newcrf, 1e-6)).astype(np.float32)
            pred, truth = points(log_newcrf, gt, ident)
            accumulate(totals, 'newcrf', split, pred, truth)
            if split == 'val':
                far = (truth >= 60.) & (truth < 80.)
                if far.any():
                    far_bias.append(float((pred[far]-truth[far]).mean()))
                decoder_rmse.append(alignment.scores(raw, gt, 'ssi_log')['rmse'])
        print(f'[scored] {split}', flush=True)
    table = rows(totals)
    checked = control(float(np.mean(far_bias)), float(np.mean(decoder_rmse)),
                      float(np.mean([cached[s.sample_id] for s in splits['val']])))
    (a.out/'summary.json').write_text(json.dumps(dict(control=checked, table=table, decoder_step=step,
        decoder_sha256=sha256_file(weights), script_sha256=sha256_file(__file__)), indent=2, allow_nan=False))
    print('[control]', json.dumps(checked), flush=True)
    for r in table:
        print('[bin]', r['source'], r['split'], r['binned_by'], r['band'], r['points'],
              round(r['bias_m'], 2), round(r['rmse_m'], 2), flush=True)
    print('[completed]', a.out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'out', 'data-root', 'decoder-run'):
        p.add_argument('--'+name, type=Path, required=True)
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
    run(p.parse_args())
