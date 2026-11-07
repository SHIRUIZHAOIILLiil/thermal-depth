"""Train-to-validation far gap of each source's own prediction, without calibration or VAE.

Reads the raw predictions and frame subsets of job 9229162 and the tmax1 rows of job 9229261,
then aligns with the same held-out cells and fit-only rule. CPU only; no test frames.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ms2_eval.io import load_ms2_gt, sha256_file
from ms2_eval.official_protocol import evaluate_sample, official_valid_mask
from tools import diagnose_depth_chain as chain
from tools import diagnose_far_target_source as source_tool
from tools.build_anythermal_pseudo_gt import resize_to

TARGET_JOB, MODEL_JOB = 9229162, 9229261
METRIC_SOURCES = ('newcrf', 'bts', 'adabins')
ALIGNMENTS = dict(newcrf=('ssi_log', 'ssi'), bts=('ssi_log', 'ssi'), adabins=('ssi_log', 'ssi'), anythermal=('ssi',))
MODEL = 'tmax1_16k'
CONTROL_TOLERANCE_M = .01


def frame_records(raw, gt, sample_id, alignments):
    valid = official_valid_mask(gt, .001, 80.)
    held = source_tool.holdout_mask(valid, sample_id)
    fit = valid & ~held
    records = []
    for mode in alignments:
        value = np.log(np.maximum(raw, 1e-6)).astype(np.float32) if mode == 'ssi_log' else raw
        aligned = chain.aligned_depth(value, gt, fit, mode)
        records.extend(dict(mode=mode, band=band, **row)
                       for band, row in source_tool.score(aligned, gt, held).items())
    return records


def summarise(records):
    """Frame-mean rows for every sequence and for the whole split."""
    groups = {}
    for r in records:
        for sequence in ('ALL', r['id'][:19]):
            groups.setdefault((r['source'], r['mode'], r['split'], sequence, r['band']), []).append(r)
    return [dict(source=k[0], mode=k[1], split=k[2], sequence=k[3], band=k[4], frames=len(v),
                 points=sum(r['points'] for r in v), bias_m=float(np.mean([r['bias_m'] for r in v])),
                 rmse_m=float(np.mean([r['rmse_m'] for r in v])))
            for k, v in sorted(groups.items())]


def gaps(table):
    found = {(r['source'], r['mode'], r['split'], r['band']): r for r in table if r['sequence'] == 'ALL'}
    rows = []
    for source, mode, split, band in sorted(found):
        if split != 'train' or (source, mode, 'val', band) not in found:
            continue
        train, val = found[source, mode, 'train', band], found[source, mode, 'val', band]
        rows.append(dict(source=source, mode=mode, band=band, train_bias_m=train['bias_m'], val_bias_m=val['bias_m'],
                         gap_m=val['bias_m']-train['bias_m'], train_rmse_m=train['rmse_m'], val_rmse_m=val['rmse_m']))
    return rows


def model_records(path):
    with Path(path).open(newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['source'] == MODEL and r['mode'] == source_tool.HELDOUT_ALIGN
                and r['mask'] == 'heldout']
    if not rows:
        raise ValueError('No tmax1 rows in the earlier job')
    return [dict(source=MODEL, mode='ssi_log', split=r['split'], id=r['id'], band=r['band'], points=int(r['points']),
                 bias_m=float(r['bias_m']), rmse_m=float(r['rmse_m'])) for r in rows]


def check_masks(records, model):
    """Both must score identical frames and held-out points, band by band."""
    key = lambda r: (r['split'], r['id'], r['band'], r['points'])
    expected = {key(r) for r in model}
    for source, mode in {(r['source'], r['mode']) for r in records}:
        if {key(r) for r in records if (r['source'], r['mode']) == (source, mode)} != expected:
            raise ValueError(f'{source} {mode}: frames or held-out points differ from the model rows')


def control(measured, cached):
    if set(measured) != set(cached):
        raise ValueError('Control frames differ from the NeWCRF cache')
    a, b = float(np.mean(list(measured.values()))), float(np.mean([cached[k] for k in measured]))
    return dict(measured_rmse_m=a, cached_rmse_m=b, tolerance_m=CONTROL_TOLERANCE_M,
                passed=abs(a-b) <= CONTROL_TOLERANCE_M+1e-12)


def run(a):
    base = a.runs/'outputs/lotus_line_v2'
    job = base/f'iris_far_target_source_fitonly_v2/job_{TARGET_JOB}'
    summary = json.loads((job/'summary.json').read_text())
    archive = json.loads((base/'iris_pair_rmse_val_v1/job_9167854/summary.json').read_text())
    analysis = Path(archive['paths']['newcrf_cache']).parents[2]/'analysis/per_frame.csv'
    with analysis.open(newline='') as f:
        cached = {r['id']: float(r['newcrf_rmse']) for r in csv.DictReader(f)}
    a.out.mkdir(parents=True, exist_ok=False)
    records, measured = [], {}
    for split in ('train', 'val'):
        subset = job/f'{split}_subset.jsonl'
        if sha256_file(subset) != summary['subsets'][split]:
            raise ValueError(f'{split} subset changed')
        for row in chain.read_jsonl(subset):
            ident = row['id']
            gt = load_ms2_gt(a.data_root/row['thermal_depth_path'], 256.)[1]
            for source, alignments in ALIGNMENTS.items():
                folder = job/'raw'/split/source
                folder = folder/'raw_predictions' if source == 'anythermal' else folder
                raw = resize_to(np.load(folder/f'{ident}.npy', allow_pickle=False), gt.shape).astype(np.float32)
                records.extend(dict(split=split, id=ident, source=source, **r)
                               for r in frame_records(raw, gt, ident, alignments))
                if split == 'val' and source == 'newcrf':
                    measured[ident] = float(evaluate_sample(raw, gt, align='ssi', min_depth=.001, max_depth=80.)['rmse'])
        print(f'[scored] {split}', flush=True)
    model = model_records(base/f'iris_model_far_gap_v1/job_{MODEL_JOB}/per_frame.csv')
    check_masks(records, model)
    checked = control(measured, {k: cached[k] for k in measured})
    table = summarise(records+model)
    gap = gaps(table)
    (a.out/'summary.json').write_text(json.dumps(dict(control=checked, gaps=gap, table=table,
        script_sha256=sha256_file(__file__)), indent=2, allow_nan=False))
    print('[control]', json.dumps(checked), flush=True)
    for row in gap:
        if row['band'] != '0-40':
            print('[gap]', json.dumps(row), flush=True)
    for r in table:
        if r['mode'] == 'ssi_log' and r['source'] in (MODEL, 'newcrf') and r['band'] == '60-80' and r['sequence'] != 'ALL':
            print('[sequence]', r['source'], r['split'], r['sequence'], r['frames'],
                  round(r['bias_m'], 2), round(r['rmse_m'], 2), flush=True)
    print('[completed]', a.out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'data-root', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
