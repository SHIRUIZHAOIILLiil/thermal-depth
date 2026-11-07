"""Score the q98 decoder model under the same SSI alignment as NeWCRF, on validation frames.

The model outputs log depth in a per-frame unit. Depth needs a log range, so one constant is
measured on training frames and frozen: depth = exp(range * output), then the 2-parameter
depth-space fit. The sensitivity rows are reported only and never select anything. No test.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ms2_eval.io import load_ms2_gt, sha256_file
from ms2_eval.official_protocol import evaluate_sample
from tools import diagnose_far_target_source as source_tool
from tools.build_anythermal_pseudo_gt import resize_to
from tools.diagnose_model_far_gap import DIAGNOSTIC_JOB

KEYS = ('abs_rel', 'rmse', 'a1')
SENSITIVITY = (.8, 1., 1.2)
NEWCRF_CONTROL_M = 2.457972237888758  # same 96 frames, cached and direct agree exactly
TOLERANCE_M = .002


def depth_from_output(raw, log_range):
    return np.exp(np.clip(log_range*raw.astype(np.float64), -20., 20.)).astype(np.float32)


def scores(pred, gt, align):
    metrics = evaluate_sample(pred, gt, align=align, min_depth=.001, max_depth=80.)
    return {k: float(metrics[k]) for k in KEYS}


def mean_rows(rows):
    return {name: {k: float(np.mean([r[k] for r in group])) for k in KEYS} for name, group in rows.items()}


def compare(headline):
    """Each model row against NeWCRF under `ssi`, the alignment of its main-table entry."""
    reference = headline['newcrf ssi']
    return {name: dict(row, rmse_gap_m=row['rmse']-reference['rmse'], absrel_gap=row['abs_rel']-reference['abs_rel'],
                       a1_gap=row['a1']-reference['a1']) for name, row in headline.items()}


def control(newcrf_rmse, decoder_rmse, cached):
    newcrf = abs(newcrf_rmse-NEWCRF_CONTROL_M) <= TOLERANCE_M
    decoder = abs(decoder_rmse-cached) <= TOLERANCE_M
    return dict(newcrf_rmse=newcrf_rmse, decoder_rmse=decoder_rmse, decoder_cached_rmse=cached,
                tolerance_m=TOLERANCE_M, passed=newcrf and decoder)


def run(a):
    import torch
    from types import SimpleNamespace
    from tools import eval_iris_decoder_only as prior
    from tools.da2_caption_data import TRAIN, VAL

    if not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA')
    base = a.runs/'outputs/lotus_line_v2'
    job = base/f'iris_far_target_source_fitonly_v2/job_{DIAGNOSTIC_JOB}'
    reference = json.loads((job/'summary.json').read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    contract, _, step, weights = prior.selection(a.decoder_run)
    inputs = prior.parent.parent_inputs(a.runs)
    c = inputs['contract']
    train, _, train_manifest, _ = source_tool.select_frames(c, 'train', source_tool.FRAMES, a.data_root, TRAIN, a.out)
    val, _, val_manifest, _ = source_tool.select_frames(c, 'val', source_tool.FRAMES, a.data_root, VAL, a.out)
    for split, manifest in (('train', train_manifest), ('val', val_manifest)):
        if sha256_file(manifest) != reference['subsets'][split]:
            raise ValueError(f'{split} subset differs from the diagnostic job')
    dataset = source_tool.MS2ThermalDataset(train_manifest, a.data_root, Path(c['pseudo']), norm_type='log_truncnorm',
                                            truncnorm_min=.02, truncnorm_max=.98)
    bounds = np.array([dataset[i]['norm_bounds'].numpy() for i in range(len(dataset))], np.float64)
    log_range = float((bounds[:, 1]-bounds[:, 0]).mean())
    model, recipe, _ = prior.build(SimpleNamespace(runs=a.runs), contract, weights)
    with (a.decoder_run/'train/test_selected_v1/val_frames.csv').open(newline='') as f:
        cached = {r['id']: float(r['rmse']) for r in csv.DictReader(f)}
    rows = {'decoder ssi_log': [], 'newcrf ssi': [], 'newcrf ssi_log': []}
    rows.update({f'decoder ssi x{factor}': [] for factor in SENSITIVITY})
    per_frame = []
    for s, (ident, raw, gt) in zip(val, prior.pilot.iris.predict_samples(model, val, 'frame', recipe), strict=True):
        if ident != s.sample_id:
            raise ValueError('Prediction order differs')
        found = {'decoder ssi_log': scores(raw, gt, 'ssi_log')}
        for factor in SENSITIVITY:
            found[f'decoder ssi x{factor}'] = scores(depth_from_output(raw, log_range*factor), gt, 'ssi')
        newcrf = resize_to(np.load(job/'raw/val/newcrf'/f'{ident}.npy', allow_pickle=False), gt.shape).astype(np.float32)
        found['newcrf ssi'] = scores(newcrf, gt, 'ssi')
        found['newcrf ssi_log'] = scores(np.log(np.maximum(newcrf, 1e-6)).astype(np.float32), gt, 'ssi_log')
        for name, row in found.items():
            rows[name].append(row)
            per_frame.append(dict(id=ident, name=name, **row))
    with (a.out/'per_frame.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, list(per_frame[0]))
        writer.writeheader()
        writer.writerows(per_frame)
    headline = mean_rows(rows)
    checked = control(headline['newcrf ssi']['rmse'], headline['decoder ssi_log']['rmse'],
                      float(np.mean([cached[s.sample_id] for s in val])))
    table = compare(headline)
    (a.out/'summary.json').write_text(json.dumps(dict(control=checked, log_range=log_range, table=table,
        decoder_step=step, decoder=str(weights), decoder_sha256=sha256_file(weights),
        range_source='mean q02-q98 log depth range on the 96 training frames',
        script_sha256=sha256_file(__file__)), indent=2, allow_nan=False))
    print('[control]', json.dumps(checked), flush=True)
    print('[range]', log_range, flush=True)
    for name, row in table.items():
        print('[row]', name, json.dumps({k: round(v, 5) for k, v in row.items()}), flush=True)
    print('[completed]', a.out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'out', 'data-root', 'decoder-run'):
        p.add_argument('--'+name, type=Path, required=True)
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
    run(p.parse_args())
