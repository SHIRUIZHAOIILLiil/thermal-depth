"""Does the train-to-validation far gap open with training steps?

Scores tmax1 checkpoints on the frames, held-out cells and alignment of job 9229261.
Inference only, no test. Temporary conversions stay in --out; the source run is read-only.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ms2_eval.io import sha256_file
from ms2_eval.official_protocol import evaluate_sample
from tools import diagnose_far_target_source as source_tool
from tools import diagnose_model_far_gap as gap_tool

STEPS = (1000, 2000, 4000, 8000, 12000, 16000, 20000)
ANCHOR = 16000
PREVIOUS_JOB = 9229261
ANCHOR_TOLERANCE_M = .05
HEADLINE = ('abs_rel', 'rmse', 'a1')


def check_payload(payload, step):
    if (payload.get('route') != 'b_thermal_unet' or payload.get('caption_mode') != 'correct'
            or payload.get('global_step') != step or set(payload.get('state_dicts', {})) != {'unet'}):
        raise ValueError(f'Wrong converted checkpoint for step {step}')


def curve_rows(table, headline):
    """One row per step, using the 60-80 and 40-60 held-out rows of each split."""
    rows = []
    for step in sorted({r['step'] for r in table}):
        row = dict(step=step)
        for split in ('train', 'val'):
            for band, label in (('60-80', 'far'), ('40-60', 'mid')):
                found = [r for r in table if r['step'] == step and r['split'] == split and r['band'] == band
                         and r['mode'] == gap_tool.HELD and r['mask'] == 'heldout']
                if len(found) != 1:
                    raise ValueError(f'Expected one row for {step} {split} {band}')
                row[f'{split}_{label}_bias_m'] = found[0]['bias_m']
                row[f'{split}_{label}_rmse_m'] = found[0]['rmse_m']
            row[split] = headline[step][split]
        row['far_gap_m'] = row['val_far_bias_m']-row['train_far_bias_m']
        row['far_rmse_gap_m'] = row['val_far_rmse_m']-row['train_far_rmse_m']
        rows.append(row)
    return rows


def anchor_check(rows, reference):
    """The converted-checkpoint path must land on the numbers of the earlier job."""
    decomposition = {r['band']: r for r in reference['decomposition']}['60-80']
    row = next(r for r in rows if r['step'] == ANCHOR)
    measured = dict(train=row['train_far_bias_m'], val=row['val_far_bias_m'])
    expected = dict(train=decomposition['model_train_m'], val=decomposition['model_val_m'])
    passed = all(abs(measured[k]-expected[k]) <= ANCHOR_TOLERANCE_M+1e-12 for k in measured)
    return dict(step=ANCHOR, expected=expected, measured=measured, tolerance_m=ANCHOR_TOLERANCE_M, passed=passed)


def weights(step, a):
    if step == ANCHOR:
        return a.source/'converted/step16000_weights.pt', False
    path = a.out/f'step{step}.tmp.pt'
    subprocess.run([sys.executable, str(ROOT/'tools/convert_iris_ms2_checkpoint.py'), '--source',
                    str(a.source/f'checkpoint-{step}'), '--output', str(path), '--caption-mode', 'correct',
                    '--step', str(step)], check=True)
    return path, True


def run(a):
    import torch
    from tools import eval_iris_decoder_only as prior
    from tools.da2_caption_data import TRAIN, VAL
    from tools.train_route_suite import RouteModel

    if not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA')
    base = a.runs/'outputs/lotus_line_v2'
    target = json.loads((base/f'iris_far_target_source_fitonly_v2/job_{gap_tool.DIAGNOSTIC_JOB}/summary.json').read_text())
    reference = json.loads((base/f'iris_model_far_gap_v1/job_{PREVIOUS_JOB}/summary.json').read_text())
    if a.source.name != 'sd2_cap_pct_logtn_img_l0100_tmax1_s43':
        raise ValueError('Expected the tmax1 caption run')
    a.out.mkdir(parents=True, exist_ok=False)
    contract = prior.parent.parent_inputs(a.runs)['contract']
    splits = {}
    for split, allowed in (('train', TRAIN), ('val', VAL)):
        frames, _, manifest, _ = source_tool.select_frames(contract, split, source_tool.FRAMES, a.data_root, allowed, a.out)
        if sha256_file(manifest) != target['subsets'][split]:
            raise ValueError(f'{split} subset differs from job {gap_tool.DIAGNOSTIC_JOB}')
        splits[split] = frames
    recipe = SimpleNamespace(**contract['recipe'])
    model = RouteModel(recipe, torch.device('cuda'), torch.float16)
    records, headline = [], {}
    for step in STEPS:
        path, temporary = weights(step, a)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        check_payload(payload, step)
        model.load_state_dicts(payload['state_dicts'], str(path))
        model.unet.requires_grad_(False)
        model.set_train(False)
        del payload
        if temporary:
            path.unlink()
        headline[step] = {}
        for split, frames in splits.items():
            scores = {k: [] for k in HEADLINE}
            predictions = prior.pilot.iris.predict_samples(model, frames, 'frame', recipe)
            for s, (ident, raw, gt) in zip(frames, predictions, strict=True):
                if ident != s.sample_id:
                    raise ValueError('Prediction order differs')
                records.extend(dict(step=step, split=split, id=ident, **r) for r in gap_tool.frame_records(raw, gt, ident))
                metrics = evaluate_sample(raw, gt, align='ssi_log', min_depth=.001, max_depth=80.)
                for key in HEADLINE:
                    scores[key].append(float(metrics[key]))
            headline[step][split] = {k: float(np.mean(v)) for k, v in scores.items()}
        print(f'[scored] step {step}', flush=True)
    with (a.out/'per_frame.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    groups = {}
    for r in records:
        groups.setdefault((r['step'], r['split'], r['mode'], r['mask'], r['band']), []).append(r)
    table = [dict(step=k[0], split=k[1], mode=k[2], mask=k[3], band=k[4], frames=len(v),
                  points=sum(r['points'] for r in v), bias_m=float(np.mean([r['bias_m'] for r in v])),
                  rmse_m=float(np.mean([r['rmse_m'] for r in v]))) for k, v in sorted(groups.items())]
    rows = curve_rows(table, headline)
    checked = anchor_check(rows, reference)
    (a.out/'summary.json').write_text(json.dumps(dict(anchor=checked, curve=rows, table=table, steps=list(STEPS),
        source=str(a.source), subsets=target['subsets'], script_sha256=sha256_file(__file__)), indent=2, allow_nan=False))
    print('[anchor]', json.dumps(checked), flush=True)
    for row in rows:
        print('[curve]', json.dumps(row), flush=True)
    print('[completed]', a.out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'out', 'data-root', 'source'):
        p.add_argument('--'+name, type=Path, required=True)
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
    run(p.parse_args())
