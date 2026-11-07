"""Does the model miss far depth on the frames it trained on, or only on validation?

Same 96 train and 96 val frames, same held-out cells and fit-only SSI-log as job 9229162,
so model rows line up with that job's AnyThermal q100 target rows. Inference only; no test.
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

DIAGNOSTIC_JOB = 9229162
TMAX1 = 'tmax1_16k'
CONTROL_MODEL = 'conv4000'
# Job 9174706, model stage: 60-80 m, whole-frame SSI-log, same validation frames.
CONTROL_BIAS_M, CONTROL_TOLERANCE_M = -12.69, 1.
HELD = source_tool.HELDOUT_ALIGN


def frame_records(raw, gt, sample_id):
    """Held-out and fit rows share one fit; the control row refits on all LiDAR."""
    valid = official_valid_mask(gt, .001, 80.)
    held = source_tool.holdout_mask(valid, sample_id)
    fit = valid & ~held
    aligned = chain.aligned_depth(raw, gt, fit, 'ssi_log')
    records = []
    for mask, chosen in (('heldout', held), ('fit', fit)):
        records.extend(dict(mode=HELD, mask=mask, band=band, **row)
                       for band, row in source_tool.score(aligned, gt, chosen).items())
    whole = chain.aligned_depth(raw, gt, valid, 'ssi_log')
    records.extend(dict(mode='ssi_log', mask='control_all', band=band, **row)
                   for band, row in source_tool.score(whole, gt, valid).items())
    return records


def target_row(table, split, band):
    return source_tool.lookup(table, split=split, source='anythermal', q=1., stage='vae',
                              mode=HELD, mask='heldout', band=band)


def model_row(table, name, split, band):
    return source_tool.lookup(table, split=split, source=name, q=None, stage='model',
                              mode=HELD, mask='heldout', band=band)


def decomposition(table, reference, name=TMAX1):
    """Split the signed path from the train target to validation output, band by band."""
    rows = []
    for band, _, _ in source_tool.BANDS:
        model = {s: model_row(table, name, s, band) for s in ('train', 'val')}
        target = {s: target_row(reference, s, band) for s in ('train', 'val')}
        for split in model:
            if (model[split]['frames'], model[split]['points']) != (target[split]['frames'], target[split]['points']):
                raise ValueError(f'{split} {band}: frames or held-out points differ from the target rows')
        total = model['val']['bias_m']-target['train']['bias_m']
        rows.append(dict(band=band, target_train_m=target['train']['bias_m'], target_val_m=target['val']['bias_m'],
                         model_train_m=model['train']['bias_m'], model_val_m=model['val']['bias_m'],
                         model_train_rmse_m=model['train']['rmse_m'], model_val_rmse_m=model['val']['rmse_m'],
                         total_m=total, train_shortfall_m=model['train']['bias_m']-target['train']['bias_m'],
                         train_to_val_m=model['val']['bias_m']-model['train']['bias_m']))
    return rows


def control(table):
    measured = source_tool.lookup(table, split='val', source=CONTROL_MODEL, q=None, stage='model',
                                  mode='ssi_log', mask='control_all', band='60-80')['bias_m']
    return dict(expected=CONTROL_BIAS_M, measured=measured, tolerance_m=CONTROL_TOLERANCE_M,
                passed=abs(measured-CONTROL_BIAS_M) <= CONTROL_TOLERANCE_M+1e-12)


def run(a):
    import torch
    from tools import eval_iris_decoder_only as prior
    from tools.da2_caption_data import TRAIN, VAL
    from tools.train_iris_tmax1_bias import check_source
    from tools.train_route_suite import RouteModel

    if not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA')
    reference_path = a.runs/f'outputs/lotus_line_v2/iris_far_target_source_fitonly_v2/job_{DIAGNOSTIC_JOB}/summary.json'
    reference = json.loads(reference_path.read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    inputs = prior.parent.parent_inputs(a.runs)
    contract = inputs['contract']
    splits = {}
    for split, allowed in (('train', TRAIN), ('val', VAL)):
        frames, _, manifest, _ = source_tool.select_frames(contract, split, source_tool.FRAMES, a.data_root, allowed, a.out)
        if sha256_file(manifest) != reference['subsets'][split]:
            raise ValueError(f'{split} subset differs from job {DIAGNOSTIC_JOB}')
        splits[split] = frames
    recipe = SimpleNamespace(**contract['recipe'])
    tmax1 = check_source(a.source)
    model = RouteModel(recipe, torch.device('cuda'), torch.float16)
    loaders = {TMAX1: lambda: prior.pilot.iris.load_arm(model, tmax1, 'frame'),
               CONTROL_MODEL: lambda: prior.pilot.load_selected(model, 'frame', inputs['source'], inputs['conv'], contract)}
    records = []
    for name, load in loaders.items():
        load()
        for split, frames in splits.items():
            predictions = prior.pilot.iris.predict_samples(model, frames, 'frame', recipe)
            for s, (ident, raw, gt) in zip(frames, predictions, strict=True):
                if ident != s.sample_id:
                    raise ValueError('Prediction order differs')
                records.extend(dict(split=split, id=ident, source=name, q=None, stage='model', **r)
                               for r in frame_records(raw, gt, ident))
            print(f'[scored] {name} {split}', flush=True)
    with (a.out/'per_frame.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    table = source_tool.aggregate(records)
    checked = control(table)
    gap = decomposition(table, reference['table'])
    (a.out/'summary.json').write_text(json.dumps(dict(control=checked, decomposition=gap, table=table,
        reference=str(reference_path), reference_sha256=sha256_file(reference_path), subsets=reference['subsets'],
        tmax1=str(tmax1), tmax1_sha256=sha256_file(tmax1), control_checkpoint=str(inputs['conv']),
        script_sha256=sha256_file(__file__)), indent=2, allow_nan=False))
    print('[control]', json.dumps(checked), flush=True)
    for row in gap:
        print('[gap]', json.dumps(row), flush=True)
    print('[completed]', a.out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'out', 'data-root', 'source'):
        p.add_argument('--'+name, type=Path, required=True)
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
    run(p.parse_args())
