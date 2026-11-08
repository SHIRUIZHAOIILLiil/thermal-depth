"""Do our predictions lean on regional temperature more than NeWCRF's do?

On the 96 validation frames of job 9229162, a smooth field (4 x 10 coarse grid, bicubic up to
256 x 640, so about 64 px across) is added to the raw thermal counts at a fraction of each
frame's 1-99% range. Both models stretch every frame to its own 1-99% range, so a uniform shift
would vanish; only the spatial part reaches them. Every model reads the same perturbed PNGs and
Iris keeps its per-frame noise seed. Each model keeps its main-table alignment.

Rule fixed before the run, amplitude 0.10, 0-40 m frame-mean RMSE rise over 4 fields:
sensitive if tmax1 rises by at least 5 cm and by at least twice NeWCRF's rise. Inference only,
validation frames only, no test.
"""
import argparse
import csv
import dataclasses
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ms2_eval.io import load_ms2_gt, sha256_file
from ms2_eval.official_protocol import evaluate_sample, official_valid_mask
from tools import diagnose_far_gap_curve as curve
from tools import diagnose_far_target_source as source_tool
from tools.diagnose_depth_chain import frame_seed
from tools.diagnose_error_scales import decompose
from tools.diagnose_error_sources import aligned_depth

AMPLITUDES = (.05, .10)
DECISION_AMPLITUDE = .10
FIELDS, GRID, SEED = 4, (4, 10), 20261108
MIN_RISE_M, RATIO = .05, 2.
STEP = 16000
RUNS = dict(tmax1='sd2_cap_pct_logtn_img_l0100_tmax1_s43', strongaug='sd2_cap_pct_logtn_img_l0100_tmax1_strongaug_s43')
ALIGN = dict(tmax1='ssi_log', strongaug='ssi_log', newcrf='ssi')
# Clean 96 validation frames: tmax1 16k (job 9229268) and NeWCRF (job 0787261b8 control).
CONTROL = dict(tmax1=(2.7980045794905712, .002), newcrf=(2.457972, .01))
NEAR = 40.
TARGET_JOB = 9229162


def low_field(sample_id, k, shape):
    rng = np.random.default_rng(frame_seed(sample_id, SEED+k))
    coarse = rng.standard_normal(GRID).astype(np.float32)
    field = np.asarray(Image.fromarray(coarse, mode='F').resize((shape[1], shape[0]), Image.BICUBIC), np.float64)
    return (field-field.mean())/field.std()


def perturb(raw, field, amplitude):
    lo, hi = np.percentile(raw, (1, 99))
    return np.clip(np.round(raw.astype(np.float64)+amplitude*(hi-lo)*field), 0, 65535).astype(np.uint16)


def conditions():
    return [('clean', 0., None)] + [(f'a{amp:.2f}_k{k}', amp, k) for amp in AMPLITUDES for k in range(FIELDS)]


def write_inputs(frames, out):
    """One folder of perturbed 16-bit PNGs per condition; the clean condition is the original."""
    sets = {}
    for name, amp, k in conditions():
        if k is None:
            sets[name] = list(frames)
            continue
        folder = out/'inputs'/name
        folder.mkdir(parents=True)
        chosen = []
        for s in frames:
            raw = np.asarray(Image.open(s.thermal_path))
            if raw.dtype != np.uint16 or raw.ndim != 2:
                raise ValueError(f'{s.sample_id}: expected 2-D uint16 thermal, got {raw.dtype} {raw.shape}')
            moved = perturb(raw, low_field(s.sample_id, k, raw.shape), amp)
            path = folder/f'{s.sample_id}.png'
            Image.fromarray(moved).save(path)
            if not np.array_equal(np.asarray(Image.open(path)), moved):
                raise ValueError(f'{path}: 16-bit round trip failed')
            chosen.append(dataclasses.replace(s, thermal_path=str(path)))
        sets[name] = chosen
    return sets


def frame_row(raw, gt, align, clean):
    valid = official_valid_mask(gt, .001, 80.)
    metrics = evaluate_sample(raw, gt, align=align, min_depth=.001, max_depth=80.)
    aligned = np.clip(aligned_depth(raw, gt, valid, align), .001, 80.)
    near = source_tool.score(aligned, gt, valid)['0-40']
    row = dict(abs_rel=float(metrics['abs_rel']), rmse=float(metrics['rmse']), a1=float(metrics['a1']),
               rmse_near=near['rmse_m'])
    if clean is not None:
        rows, cols = np.nonzero(valid & (gt < NEAR))
        change = aligned[rows, cols]-clean[rows, cols]
        row.update(change_near=float(np.sqrt(np.mean(change**2))), parts=decompose(change, rows, cols))
    return row, aligned


def summarise(records):
    """Per model and amplitude: mean rise over frames and fields, pooled scale parts of the change."""
    clean = {(r['model'], r['id']): r for r in records if r['condition'] == 'clean'}
    out = {}
    for model in sorted({r['model'] for r in records}):
        base = [r for k, r in clean.items() if k[0] == model]
        out[model] = dict(clean={k: float(np.mean([r[k] for r in base])) for k in ('abs_rel', 'rmse', 'a1', 'rmse_near')})
        for amp in AMPLITUDES:
            rows = [r for r in records if r['model'] == model and r['amplitude'] == amp]
            rise = {k: float(np.mean([r[k]-clean[model, r['id']][k] for r in rows])) for k in ('abs_rel', 'rmse', 'a1', 'rmse_near')}
            parts = np.sum([r['parts'] for r in rows], axis=0)
            out[model][f'{amp:.2f}'] = dict(rise=rise, change_near_m=float(np.mean([r['change_near'] for r in rows])),
                                             large_share=float((parts[0]+parts[1])/parts.sum()), samples=len(rows))
    return out


def decision(summary):
    key = f'{DECISION_AMPLITUDE:.2f}'
    ours, ref = (summary[m][key]['rise']['rmse_near'] for m in ('tmax1', 'newcrf'))
    sensitive = ours >= MIN_RISE_M-1e-12 and ours >= RATIO*max(ref, 0.)-1e-12
    return dict(amplitude=DECISION_AMPLITUDE, tmax1_rise_m=ours, newcrf_rise_m=ref, min_rise_m=MIN_RISE_M, ratio=RATIO,
                verdict='sensitive' if sensitive else 'not_sensitive',
                next='lowfreq_augmentation_arm' if sensitive else 'drop_lowfreq_hypothesis')


def control(summary):
    rows = {}
    for model, (expected, tolerance) in CONTROL.items():
        measured = summary[model]['clean']['rmse']
        rows[model] = dict(expected=expected, measured=measured, tolerance_m=tolerance,
                           passed=abs(measured-expected) <= tolerance+1e-12)
    return dict(rows=rows, passed=all(r['passed'] for r in rows.values()))


def run(a):
    import torch
    from tools import eval_iris_decoder_only as prior
    from tools.da2_caption_data import VAL
    from tools.train_route_suite import RouteModel

    if not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA')
    job = a.runs/f'outputs/lotus_line_v2/iris_far_target_source_fitonly_v2/job_{TARGET_JOB}'
    reference = json.loads((job/'summary.json').read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    c = prior.parent.parent_inputs(a.runs)['contract']
    frames, _, manifest, _ = source_tool.select_frames(c, 'val', source_tool.FRAMES, a.data_root, VAL, a.out)
    if sha256_file(manifest) != reference['subsets']['val']:
        raise ValueError('val subset differs from job 9229162')
    sets = write_inputs(frames, a.out)
    print(f'[inputs] {len(sets)} conditions x {len(frames)} frames', flush=True)
    records, provenance, clean = [], {}, {}

    def add(model, name, amp, ident, raw, gt):
        row, aligned = frame_row(raw, gt, ALIGN[model], None if name == 'clean' else clean[model, ident])
        if name == 'clean':
            clean[model, ident] = aligned
        records.append(dict(model=model, condition=name, amplitude=amp, id=ident, **row))

    for name, amp, _ in conditions():
        folder = a.out/'newcrf'/name
        provenance['newcrf'] = source_tool.run_baseline('newcrf', sets[name], a, folder)
        for s in sets[name]:
            gt = load_ms2_gt(s.thermal_gt_path, 256.)[1]
            add('newcrf', name, amp, s.sample_id, np.load(folder/f'{s.sample_id}.npy'), gt)
    print('[scored] newcrf', flush=True)
    recipe = SimpleNamespace(**c['recipe'])
    model = RouteModel(recipe, torch.device('cuda'), torch.float16)
    for key, run_name in RUNS.items():
        path = a.runs/'iris_ms2'/run_name/'converted'/f'step{STEP}_weights.pt'
        payload = torch.load(path, map_location='cpu', weights_only=False)
        curve.check_payload(payload, STEP)
        model.load_state_dicts(payload['state_dicts'], str(path))
        model.unet.requires_grad_(False)
        model.set_train(False)
        del payload
        provenance[key] = dict(path=str(path), sha256=sha256_file(path))
        for name, amp, _ in conditions():
            for s, (ident, raw, gt) in zip(sets[name], prior.pilot.iris.predict_samples(model, sets[name], 'frame', recipe), strict=True):
                if ident != s.sample_id:
                    raise ValueError('Prediction order differs')
                add(key, name, amp, ident, raw, gt)
        print(f'[scored] {key}', flush=True)
    with (a.out/'per_frame.csv').open('w', newline='') as f:
        fields = [k for k in records[-1] if k != 'parts']
        writer = csv.DictWriter(f, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    summary = summarise(records)
    checked = control(summary)
    result = decision(summary)
    if not checked['passed']:
        result['verdict'] = result['next'] = 'control_failed'
    (a.out/'summary.json').write_text(json.dumps(dict(control=checked, decision=result, summary=summary,
        provenance=provenance, subset_sha256=reference['subsets']['val'], script_sha256=sha256_file(__file__)),
        indent=2, allow_nan=False))
    print('[control]', json.dumps(checked), flush=True)
    for model, rows in summary.items():
        print('[clean]', model, json.dumps({k: round(v, 4) for k, v in rows['clean'].items()}), flush=True)
        for amp in AMPLITUDES:
            r = rows[f'{amp:.2f}']
            print('[rise]', model, f'{amp:.2f}', json.dumps({k: round(v, 4) for k, v in r['rise'].items()}),
                  'change_near', round(r['change_near_m'], 3), 'large_share', round(r['large_share'], 3), flush=True)
    print('[decision]', json.dumps(result), flush=True)
    print('[completed]', a.out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'out', 'data-root', 'bmsd-root', 'ckpt-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
    run(p.parse_args())
