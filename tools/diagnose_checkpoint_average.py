"""Does averaging tmax1 U-Net weights over late checkpoints reduce generalisation error?

One frozen window, steps 12000 to 20000. The result is still one model and one forward pass.
Scored on the 96 + 96 frames of the earlier diagnostics, SSI-log. No training, no test.
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
from tools.diagnose_model_far_gap import DIAGNOSTIC_JOB

WINDOW = tuple(range(12000, 20001, 1000))
BASELINE = 16000
BASELINE_VAL_RMSE = 2.7980045794905712  # job 9229268, same frames and recipe
TOLERANCE_M = .002
RMSE_GAIN_M, ABSREL_INCREASE = .05, .002
KEYS = ('abs_rel', 'rmse', 'a1')


def average_states(states):
    """Arithmetic mean of float tensors; anything else must already agree."""
    states = list(states)
    if not states:
        raise ValueError('Nothing to average')
    keys = set(states[0])
    if any(set(s) != keys for s in states):
        raise ValueError('Checkpoints differ in keys')
    averaged = {}
    for key, first in states[0].items():
        if first.is_floating_point():
            total = sum(s[key].double() for s in states)
            averaged[key] = (total/len(states)).to(first.dtype)
        else:
            if any(not s[key].equal(first) for s in states):
                raise ValueError(f'{key}: non-float values differ')
            averaged[key] = first
    return averaged


def gate(average, baseline):
    gain = baseline['rmse']-average['rmse']
    increase = average['abs_rel']-baseline['abs_rel']
    return dict(rmse_gain_m=gain, absrel_increase=increase, rmse_gain_limit_m=RMSE_GAIN_M, absrel_increase_limit=ABSREL_INCREASE,
                passed=gain >= RMSE_GAIN_M-1e-12 and increase <= ABSREL_INCREASE+1e-12)


def run(a):
    import torch
    from tools import eval_iris_decoder_only as prior
    from tools.da2_caption_data import TRAIN, VAL
    from tools.train_route_suite import RouteModel

    if not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA')
    if a.source.name != 'sd2_cap_pct_logtn_img_l0100_tmax1_s43':
        raise ValueError('Expected the tmax1 caption run')
    job = a.runs/f'outputs/lotus_line_v2/iris_far_target_source_fitonly_v2/job_{DIAGNOSTIC_JOB}'
    reference = json.loads((job/'summary.json').read_text())
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

    def score(state):
        model.load_state_dicts(dict(unet=state), 'checkpoint average diagnostic')
        model.unet.requires_grad_(False)
        model.set_train(False)
        result = {}
        for split, frames in splits.items():
            rows = {k: [] for k in KEYS}
            predictions = prior.pilot.iris.predict_samples(model, frames, 'frame', recipe)
            for s, (ident, raw, gt) in zip(frames, predictions, strict=True):
                if ident != s.sample_id:
                    raise ValueError('Prediction order differs')
                metrics = evaluate_sample(raw, gt, align='ssi_log', min_depth=.001, max_depth=80.)
                for key in KEYS:
                    rows[key].append(float(metrics[key]))
            result[split] = {k: float(np.mean(v)) for k, v in rows.items()}
        return result

    states, baseline = [], None
    for step in WINDOW:
        path, temporary = curve.weights(step, a)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        curve.check_payload(payload, step)
        states.append(payload['state_dicts']['unet'])
        if temporary:
            path.unlink()
        if step == BASELINE:
            baseline = score(states[-1])
    averaged = average_states(states)
    del states
    result = score(averaged)
    checked = dict(expected=BASELINE_VAL_RMSE, measured=baseline['val']['rmse'], tolerance_m=TOLERANCE_M,
                   passed=abs(baseline['val']['rmse']-BASELINE_VAL_RMSE) <= TOLERANCE_M)
    decision = gate(result['val'], baseline['val'])
    weights = None
    if decision['passed'] and checked['passed']:
        weights = a.out/'average_12000_20000.pt'
        torch.save(dict(route='b_thermal_unet', caption_mode='correct', global_step=20000, averaged_steps=list(WINDOW),
                        state_dicts=dict(unet=averaged)), weights)
    (a.out/'summary.json').write_text(json.dumps(dict(control=checked, baseline=baseline, average=result, gate=decision,
        window=list(WINDOW), weights=str(weights) if weights else None, script_sha256=sha256_file(__file__)),
        indent=2, allow_nan=False))
    print('[control]', json.dumps(checked), flush=True)
    for name, row in (('step16000', baseline), ('average', result)):
        for split in ('train', 'val'):
            print('[row]', name, split, json.dumps({k: round(v, 5) for k, v in row[split].items()}), flush=True)
    print('[gate]', json.dumps(decision), flush=True)
    print('[completed]', a.out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'out', 'data-root', 'source'):
        p.add_argument('--'+name, type=Path, required=True)
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
    run(p.parse_args())
