"""Did an augmentation arm shrink the train-to-validation gap at 16k?

tmax1 and one candidate arm, same 96 + 96 frames and SSI-log as the earlier
diagnostics. The rule below was fixed before this script ran. Inference only, no test.
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
from ms2_eval.io import sha256_file
from ms2_eval.official_protocol import evaluate_sample
from tools import diagnose_far_gap_curve as curve
from tools import diagnose_far_target_source as source_tool
from tools import diagnose_model_far_gap as gap_tool

BASELINE_RUN = 'sd2_cap_pct_logtn_img_l0100_tmax1_s43'
STEP = 16000
# job 9229268, tmax1 16k on these frames
ANCHOR = dict(train=1.4837547212281368, val=2.7980045794905712)
TOLERANCE_M = .002
GAP_REDUCTION_M = .20  # the gap must shrink by at least this much to keep tuning augmentation
KEYS = ('abs_rel', 'rmse', 'a1')


def decision(rows):
    gaps = {name: row['val']['rmse']-row['train']['rmse'] for name, row in rows.items()}
    reduction = gaps['tmax1']-gaps['candidate']
    return dict(gaps=gaps, reduction_m=reduction, limit_m=GAP_REDUCTION_M,
                next='tune_augmentation' if reduction >= GAP_REDUCTION_M-1e-12 else 'close_augmentation')


def control(row):
    measured = {split: row[split]['rmse'] for split in ANCHOR}
    return dict(expected=ANCHOR, measured=measured, tolerance_m=TOLERANCE_M,
                passed=all(abs(measured[k]-ANCHOR[k]) <= TOLERANCE_M for k in ANCHOR))


def far_rows(records):
    table = source_tool.aggregate(records)
    return {r['split']: dict(bias_m=r['bias_m'], rmse_m=r['rmse_m']) for r in table
            if r['mode'] == gap_tool.HELD and r['mask'] == 'heldout' and r['band'] == '60-80'}


def run(a):
    import torch
    from tools import eval_iris_decoder_only as prior
    from tools.da2_caption_data import TRAIN, VAL
    from tools.train_route_suite import RouteModel

    if not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA')
    job = a.runs/f'outputs/lotus_line_v2/iris_far_target_source_fitonly_v2/job_{gap_tool.DIAGNOSTIC_JOB}'
    reference = json.loads((job/'summary.json').read_text())
    if not a.candidate.startswith('sd2_cap_pct_logtn_img_l0100_tmax1_') or a.candidate == BASELINE_RUN:
        raise ValueError(f'Unexpected candidate run {a.candidate}')
    runs = dict(tmax1=BASELINE_RUN, candidate=a.candidate)
    a.out.mkdir(parents=True, exist_ok=False)
    c = prior.parent.parent_inputs(a.runs)['contract']
    splits = {}
    for split, allowed in (('train', TRAIN), ('val', VAL)):
        frames, _, manifest, _ = source_tool.select_frames(c, split, source_tool.FRAMES, a.data_root, allowed, a.out)
        if sha256_file(manifest) != reference['subsets'][split]:
            raise ValueError(f'{split} subset differs from the diagnostic job')
        splits[split] = frames
    recipe = SimpleNamespace(**c['recipe'])
    model = RouteModel(recipe, torch.device('cuda'), torch.float16)
    rows, far, weights = {}, {}, {}
    for name, run_name in runs.items():
        path = a.runs/'iris_ms2'/run_name/'converted'/f'step{STEP}_weights.pt'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        curve.check_payload(payload, STEP)
        model.load_state_dicts(payload['state_dicts'], str(path))
        model.unet.requires_grad_(False)
        model.set_train(False)
        del payload
        weights[name] = dict(path=str(path), sha256=sha256_file(path))
        rows[name], records = {}, []
        for split, frames in splits.items():
            scores = {k: [] for k in KEYS}
            for s, (ident, raw, gt) in zip(frames, prior.pilot.iris.predict_samples(model, frames, 'frame', recipe), strict=True):
                if ident != s.sample_id:
                    raise ValueError('Prediction order differs')
                metrics = evaluate_sample(raw, gt, align='ssi_log', min_depth=.001, max_depth=80.)
                for key in KEYS:
                    scores[key].append(float(metrics[key]))
                records.extend(dict(split=split, id=ident, source=name, q=None, stage='model', **r)
                               for r in gap_tool.frame_records(raw, gt, ident))
            rows[name][split] = {k: float(np.mean(v)) for k, v in scores.items()}
        far[name] = far_rows(records)
        print(f'[scored] {name}', flush=True)
    checked = control(rows['tmax1'])
    result = decision(rows)
    (a.out/'summary.json').write_text(json.dumps(dict(candidate=a.candidate, control=checked, rows=rows, far_60_80=far, decision=result,
        weights=weights, script_sha256=sha256_file(__file__)), indent=2, allow_nan=False))
    print('[control]', json.dumps(checked), flush=True)
    for name in runs:
        for split in ('train', 'val'):
            print('[row]', name, split, json.dumps({k: round(v, 5) for k, v in rows[name][split].items()}),
                  'far60-80', json.dumps({k: round(v, 2) for k, v in far[name][split].items()}), flush=True)
    print('[decision]', json.dumps(result), flush=True)
    print('[completed]', a.out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'out', 'data-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--candidate', required=True)
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
    run(p.parse_args())
