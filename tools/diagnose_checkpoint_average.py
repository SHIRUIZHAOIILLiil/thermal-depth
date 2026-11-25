"""Does averaging a run's U-Net weights over late checkpoints reduce generalisation error?

One frozen window, steps 12000 to 20000. The result is still one model and one forward pass.
Scored on the 96 + 96 frames of the earlier diagnostics, SSI-log. No training, no test.

Two sources, each with its own control (step 16000 on the same 96 val frames):
- tmax1 (job 9229303, 2026-11-07): the original 5 cm gate decides whether weights are saved.
- synth (docs/SYNTH_POSTHOC_V1.md, pre-registered 2026-11-17): the average is saved and used for
  the decoder stage iff it is no worse than step 16000 on all three val metrics (`use_average`).
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
TMAX1_RUN = 'sd2_cap_pct_logtn_img_l0100_tmax1_s43'
SYNTH_RUN = 'sd2_cap_pct_logtn_img_l0100_tmax1_synth_s43'
NOCAP_RUN = 'sd2_nocap_pct_logtn_img_l0100_tmax1_s43'
# Step-16000 val RMSE on the same 96 frames: tmax1 job 9229268, synth job 9488823. The empty-caption
# tmax1 arm (docs/SD2_POSTHOC_V1.md) is new, so it has no registered number: control is None and the
# decoder stage reproduces its full-val step 16000 from the arm's own val job instead.
# The l100 pair (docs/SD2_L100_POSTHOC_V1.md): the same post-hoc chain without tmax1. Neither has a
# registered 96-frame number either.
L100_RUN = 'sd2_cap_pct_logtn_img_l0100_s43'
NOCAP_L100_RUN = 'sd2_nocap_pct_logtn_img_l0100_s43'
SOURCES = {TMAX1_RUN: dict(control_val_rmse=BASELINE_VAL_RMSE, save='gate'),
           SYNTH_RUN: dict(control_val_rmse=2.67386, save='not_worse'),
           NOCAP_RUN: dict(control_val_rmse=None, save='not_worse', prompt='empty'),
           L100_RUN: dict(control_val_rmse=None, save='not_worse'),
           NOCAP_L100_RUN: dict(control_val_rmse=None, save='not_worse', prompt='empty')}
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


def not_worse(average, baseline):
    """The synth rule: the average replaces step 16000 only if no headline metric gets worse."""
    return (average['abs_rel'] <= baseline['abs_rel'] and average['rmse'] <= baseline['rmse']
            and average['a1'] >= baseline['a1'])


def prompt_of(policy):
    """'frame' (the arm's own caption) or 'empty'; also the converted checkpoints' caption_mode."""
    return policy.get('prompt', 'frame')


def caption_mode(prompt):
    return 'empty' if prompt == 'empty' else 'correct'


def control(expected, measured):
    if expected is None:
        return dict(expected=None, measured=measured, tolerance_m=TOLERANCE_M, passed=True,
                    note='new arm: no registered 96-frame number')
    return dict(expected=expected, measured=measured, tolerance_m=TOLERANCE_M,
                passed=abs(measured-expected) <= TOLERANCE_M)


def window_weights(step, a, mode):
    """diagnose_far_gap_curve.weights with the arm's caption mode."""
    import subprocess
    if step == BASELINE:
        return a.source/'converted'/f'step{BASELINE}_weights.pt', False
    path = a.out/f'step{step}.tmp.pt'
    subprocess.run([sys.executable, str(ROOT/'tools/convert_iris_ms2_checkpoint.py'), '--source',
                    str(a.source/f'checkpoint-{step}'), '--output', str(path), '--caption-mode', mode,
                    '--step', str(step)], check=True)
    return path, True


def check_window_payload(payload, step, mode):
    if (payload.get('route') != 'b_thermal_unet' or payload.get('caption_mode') != mode
            or payload.get('global_step') != step or set(payload.get('state_dicts', {})) != {'unet'}):
        raise ValueError(f'Wrong converted checkpoint for step {step} ({mode})')


def missing_checkpoints(source):
    """Everything the window needs, checked before any GPU work."""
    need = [source/'converted'/f'step{BASELINE}_weights.pt']
    need += [source/f'checkpoint-{s}' for s in WINDOW if s != BASELINE]
    return [str(p) for p in need if not p.exists()]


def run(a):
    import torch
    from tools import eval_iris_decoder_only as prior
    from tools.da2_caption_data import TRAIN, VAL
    from tools.train_route_suite import RouteModel

    if not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA')
    if a.source.name not in SOURCES:
        raise ValueError(f'No control registered for {a.source.name}')
    policy = dict(SOURCES[a.source.name])
    if a.save_rule == 'not_worse':
        # 2026-11-23 (docs/SD2_POSTHOC_V1.md): the tmax1 rerun for the ablation's final SD2-start model
        # uses the synth rule; the default keeps each source's registered rule (tmax1 job 9229303: gate).
        policy['save'] = 'not_worse'
    prompt = prompt_of(policy)
    mode = caption_mode(prompt)
    missing = missing_checkpoints(a.source)
    if missing:
        raise FileNotFoundError(f'Window checkpoints missing: {missing}')
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
            predictions = prior.pilot.iris.predict_samples(model, frames, prompt, recipe)
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
        path, temporary = window_weights(step, a, mode)
        payload = torch.load(path, map_location='cpu', weights_only=False)
        check_window_payload(payload, step, mode)
        states.append(payload['state_dicts']['unet'])
        if temporary:
            path.unlink()
        if step == BASELINE:
            baseline = score(states[-1])
    averaged = average_states(states)
    del states
    result = score(averaged)
    expected = policy['control_val_rmse']
    checked = control(expected, baseline['val']['rmse'])
    decision = gate(result['val'], baseline['val'])
    use_average = checked['passed'] and not_worse(result['val'], baseline['val'])
    save = decision['passed'] if policy['save'] == 'gate' else use_average
    weights = None
    if save and checked['passed']:
        weights = a.out/'average_12000_20000.pt'
        torch.save(dict(route='b_thermal_unet', caption_mode=mode, global_step=20000, averaged_steps=list(WINDOW),
                        source_run=a.source.name, state_dicts=dict(unet=averaged)), weights)
    (a.out/'summary.json').write_text(json.dumps(dict(control=checked, baseline=baseline, average=result, gate=decision,
        source=str(a.source), source_run=a.source.name, policy=policy, use_average=use_average,
        window=list(WINDOW), weights=str(weights) if weights else None,
        weights_sha256=sha256_file(weights) if weights else None, script_sha256=sha256_file(__file__)),
        indent=2, allow_nan=False))
    print('[control]', json.dumps(checked), flush=True)
    for name, row in (('step16000', baseline), ('average', result)):
        for split in ('train', 'val'):
            print('[row]', name, split, json.dumps({k: round(v, 5) for k, v in row[split].items()}), flush=True)
    print('[gate]', json.dumps(decision), flush=True)
    print('[use_average]', use_average, flush=True)
    print('[completed]', a.out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'out', 'data-root', 'source'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--save-rule', choices=('registered', 'not_worse'), default='registered')
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
    run(p.parse_args())
