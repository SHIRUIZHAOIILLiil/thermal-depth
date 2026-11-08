"""Does the model copy the pseudo target where training never saw the LiDAR?

The holdout arm (lotus/utils/ms2_holdout.py) trained without 20% of each training frame's
LiDAR cells, the cells job 9229162 scores. On that job's 96 training frames, compare the
arm's held-out RMSE with two numbers from the same cells: the target it was given there
(9229162, AnyThermal q100 after the VAE) and tmax1, which did see that LiDAR (9229261,
re-measured here as the control).

Rule fixed before the run, 0-40 m: r = (holdout - tmax1) / (target - tmax1).
r >= 2/3 copies the target, r <= 1/3 learns beyond it, otherwise partial.
Inference only, no test.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'lotus'))
from ms2_eval.io import sha256_file
from ms2_eval.official_protocol import official_valid_mask
from tools import diagnose_far_gap_curve as curve
from tools import diagnose_far_target_source as source_tool
from tools import diagnose_model_far_gap as gap_tool

BASELINE_RUN = 'sd2_cap_pct_logtn_img_l0100_tmax1_s43'
HOLDOUT_RUN = 'sd2_cap_pct_logtn_img_l0100_tmax1_holdout_s43'
STEP = 16000
MODEL_JOB = 9229261
BAND = '0-40'
COPY, LEARN = 2/3, 1/3
TOLERANCE_M = .002
HELD = source_tool.HELDOUT_ALIGN


def decision(target, seen, holdout):
    span = target-seen
    if span <= 0:
        raise ValueError(f'Target {target} is not worse than the model that saw the LiDAR ({seen})')
    r = (holdout-seen)/span
    verdict = 'copies_target' if r >= COPY-1e-12 else 'learns_beyond_target' if r <= LEARN+1e-12 else 'partial'
    return dict(band=BAND, target_m=target, tmax1_m=seen, holdout_m=holdout, r=r,
                copy_limit=COPY, learn_limit=LEARN, verdict=verdict)


def control(measured, expected):
    return dict(expected=expected, measured=measured, tolerance_m=TOLERANCE_M,
                passed=abs(measured-expected) <= TOLERANCE_M+1e-12)


def check_cells(frames, min_fit_pixels):
    """The training mask and the scoring mask must be the same cells on every scored frame."""
    from ms2_eval.io import load_ms2_gt
    from utils import ms2_holdout
    for s in frames:
        gt = load_ms2_gt(s.thermal_gt_path, 256.)[1]
        valid = official_valid_mask(gt, .001, 80.)
        held = source_tool.holdout_mask(valid, s.sample_id)
        if not np.array_equal(held, ms2_holdout.holdout_mask(valid, s.sample_id)):
            raise ValueError(f'{s.sample_id}: training and scoring cells differ')
        if (valid & ~held).sum() < min_fit_pixels:
            raise ValueError(f'{s.sample_id}: too few fit pixels')


def run(a):
    import torch
    from tools import eval_iris_decoder_only as prior
    from tools.da2_caption_data import TRAIN, VAL
    from tools.train_route_suite import RouteModel

    if not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA')
    base = a.runs/'outputs/lotus_line_v2'
    target_path = base/f'iris_far_target_source_fitonly_v2/job_{gap_tool.DIAGNOSTIC_JOB}/summary.json'
    model_path = base/f'iris_model_far_gap_v1/job_{MODEL_JOB}/summary.json'
    target_ref, model_ref = (json.loads(p.read_text()) for p in (target_path, model_path))
    a.out.mkdir(parents=True, exist_ok=False)
    c = prior.parent.parent_inputs(a.runs)['contract']
    config = json.loads((Path(c['pseudo']).parent/'build_config.json').read_text())
    splits = {}
    for split, allowed in (('train', TRAIN), ('val', VAL)):
        frames, _, manifest, _ = source_tool.select_frames(c, split, source_tool.FRAMES, a.data_root, allowed, a.out)
        if sha256_file(manifest) != target_ref['subsets'][split]:
            raise ValueError(f'{split} subset differs from job {gap_tool.DIAGNOSTIC_JOB}')
        splits[split] = frames
    check_cells(splits['train'], int(config['min_fit_pixels']))
    recipe = SimpleNamespace(**c['recipe'])
    model = RouteModel(recipe, torch.device('cuda'), torch.float16)
    runs = dict(tmax1=BASELINE_RUN, holdout=HOLDOUT_RUN)
    records, weights = [], {}
    for name, run_name in runs.items():
        path = a.runs/'iris_ms2'/run_name/'converted'/f'step{STEP}_weights.pt'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        curve.check_payload(payload, STEP)
        model.load_state_dicts(payload['state_dicts'], str(path))
        model.unet.requires_grad_(False)
        model.set_train(False)
        del payload
        weights[name] = dict(path=str(path), sha256=sha256_file(path))
        for split, frames in splits.items():
            for s, (ident, raw, gt) in zip(frames, prior.pilot.iris.predict_samples(model, frames, 'frame', recipe), strict=True):
                if ident != s.sample_id:
                    raise ValueError('Prediction order differs')
                records.extend(dict(split=split, id=ident, source=name, q=None, stage='model', **r)
                               for r in gap_tool.frame_records(raw, gt, ident))
        print(f'[scored] {name}', flush=True)
    table = source_tool.aggregate(records)
    row = lambda name, split, band: gap_tool.model_row(table, name, split, band)
    bands = {}
    for band, _, _ in source_tool.BANDS:
        target = gap_tool.target_row(target_ref['table'], 'train', band)
        for name in runs:
            if (row(name, 'train', band)['frames'], row(name, 'train', band)['points']) != (target['frames'], target['points']):
                raise ValueError(f'{name} {band}: held-out points differ from the target rows')
        bands[band] = dict(target_train=target['rmse_m'],
                           **{f'{name}_{split}': row(name, split, band)['rmse_m'] for name in runs for split in splits},
                           **{f'{name}_train_fit': source_tool.lookup(table, split='train', source=name, q=None, stage='model',
                              mode=HELD, mask='fit', band=band)['rmse_m'] for name in runs})
    checked = control(row('tmax1', 'train', BAND)['rmse_m'],
                      gap_tool.model_row(model_ref['table'], gap_tool.TMAX1, 'train', BAND)['rmse_m'])
    result = decision(bands[BAND]['target_train'], bands[BAND]['tmax1_train'], bands[BAND]['holdout_train'])
    if not checked['passed']:
        result['verdict'] = 'control_failed'
    (a.out/'summary.json').write_text(json.dumps(dict(control=checked, decision=result, bands=bands, table=table,
        weights=weights, references={str(p): sha256_file(p) for p in (target_path, model_path)},
        script_sha256=sha256_file(__file__)), indent=2, allow_nan=False))
    print('[control]', json.dumps(checked), flush=True)
    for band, values in bands.items():
        print('[band]', band, json.dumps({k: round(v, 4) for k, v in values.items()}), flush=True)
    print('[decision]', json.dumps(result), flush=True)
    print('[completed]', a.out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'out', 'data-root'):
        p.add_argument('--'+name, type=Path, required=True)
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
    run(p.parse_args())
