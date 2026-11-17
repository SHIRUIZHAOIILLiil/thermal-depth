"""Synth-start caption arm -> frozen U-Net -> original depth decoder adaptation (docs/SYNTH_POSTHOC_V1.md).

The decoder stage is train_iris_tmax1_decoder.py unchanged (budget, loss, data, 0/500/1000 selection);
only the frozen U-Net differs. It is the synth arm's step 16000, or its 12000-20000 weight average
when diagnose_checkpoint_average.py ruled `use_average` (no headline metric worse on its 96 val frames).
Validation only. The verdict against NeWCRF under the same ssi_log alignment is written to
<run>/train/verdict.json; it is a report, it does not change the selection.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import train_iris_tmax1_decoder as stage
from tools.diagnose_checkpoint_average import SYNTH_RUN, WINDOW
from tools.iris_tmax1_selection import checked_metrics

pair = stage.pair
# Full val, step 16000, val job 9431554: the doc rounds AbsRel/δ1 to 4 places, RMSE 2.65101 comes from
# tools/val_gap_breakdown.py (docs/IRIS_SYNTH_PRETRAIN_V1.md). Tolerances cover that rounding.
EXPECTED_16K = dict(abs_rel=.0763, rmse=2.65101, a1=.9491)
TOLERANCE = dict(abs_rel=.00006, rmse=.002, a1=.00006)
NEWCRF_SSI_LOG_JOB = 9488948
KEYS = ('abs_rel', 'rmse', 'a1')


def require_16k(metrics):
    checked_metrics(metrics)
    for key, tol in TOLERANCE.items():
        if abs(metrics[key]-EXPECTED_16K[key]) > tol:
            raise ValueError(f'synth step 16000 does not reproduce {key}: {metrics[key]} vs {EXPECTED_16K[key]}')


def resolve(summary_path, runs):
    """(weights path, kind) chosen by the averaging job's pre-registered rule."""
    summary = json.loads(Path(summary_path).read_text())
    if summary.get('source_run') != SYNTH_RUN or not summary['control']['passed']:
        raise ValueError('Averaging job is not a passed synth control')
    if summary['use_average']:
        path = Path(summary['weights'])
        if pair.sha256_file(path) != summary['weights_sha256']:
            raise ValueError('Averaged weights changed since the averaging job')
        return path, 'average'
    return Path(runs)/'iris_ms2'/SYNTH_RUN/'converted/step16000_weights.pt', 'step16000'


def load_average(model, path):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload.get('route') != 'b_thermal_unet' or payload.get('caption_mode') != 'correct'
            or payload.get('averaged_steps') != list(WINDOW) or payload.get('source_run') != SYNTH_RUN
            or set(payload.get('state_dicts', {})) != {'unet'}):
        raise ValueError(f'Not the synth 12000-20000 average: {path}')
    model.load_state_dicts(payload['state_dicts'], str(path))
    model.unet.requires_grad_(False)
    model.set_train(False)


def start_for(a):
    path, kind = resolve(a.average_summary, a.runs)
    code = ('tools/train_iris_synth_decoder.py', 'tools/diagnose_checkpoint_average.py')
    if kind == 'average':
        # No full-val number exists for the average yet: the smoke's step-0 validation is that number.
        return stage.Start(experiment='iris_synth_decoder_v1', label='synth 12000-20000 average',
            resolve=lambda _: path, load=load_average, require_start=checked_metrics,
            source_step='average 12000-20000', code=code)
    return stage.Start(experiment='iris_synth_decoder_v1', label='synth original 16000',
        resolve=lambda _: path, load=lambda model, p: pair.old.load_arm(model, p, 'frame'),
        require_start=require_16k, source_step=16000, code=code)


def newcrf_ssi_log(runs):
    path = Path(runs)/f'outputs/lotus_line_v2/newcrf_val_ssi_log_v1/job_{NEWCRF_SSI_LOG_JOB}/summary.json'
    summary = json.loads(path.read_text())
    if not summary['control']['passed'] or summary['frames'] != 14249:
        raise ValueError(f'NeWCRF ssi_log reference not usable: {path}')
    return {k: float(summary['means']['ssi_log'][k]) for k in KEYS}, {k: float(summary['means']['ssi'][k]) for k in KEYS}


def verdict(selected, start, same_alignment, own_alignment):
    """Pre-registered: all three headline metrics strictly better than NeWCRF under ssi_log."""
    def versus(m, r):
        return dict(abs_rel=m['abs_rel']-r['abs_rel'], rmse_m=m['rmse']-r['rmse'], a1=m['a1']-r['a1'],
                    beats_all_three=m['abs_rel'] < r['abs_rel'] and m['rmse'] < r['rmse'] and m['a1'] > r['a1'])
    return dict(selected=selected, start=start, newcrf_ssi_log=same_alignment, newcrf_ssi=own_alignment,
                vs_newcrf_ssi_log=versus(selected, same_alignment), vs_newcrf_ssi=versus(selected, own_alignment),
                vs_start=versus(selected, start), passed=versus(selected, same_alignment)['beats_all_three'])


def run(a):
    start = start_for(a)
    same, own = newcrf_ssi_log(a.runs)  # fail before any GPU work if the reference is missing
    stage.run(a, start)
    if a.mode != 'train':
        return
    summary = pair.read(a.run/'train/summary.json')
    chosen = summary['selected']
    result = verdict(summary['curve'][str(chosen)]['metrics'], summary['curve']['0']['metrics'], same, own)
    result.update(source_kind=start.label, source=str(start.resolve(a)), selected_updates=chosen)
    pair.json_write(a.run/'train/verdict.json', result)
    print('[verdict]', json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('smoke', 'train'))
    for name in ('runs', 'run', 'data-root', 'average-summary', 'comparison-run'):
        p.add_argument('--'+name, type=Path, required=True)
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_TEXT_PADDING='legacy', IRIS_DECODE_CLAMP='0')
    run(p.parse_args())
