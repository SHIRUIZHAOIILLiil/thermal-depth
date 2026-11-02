"""Fit a bounded upper-tail slope on train LiDAR; compare paired validation only."""
import argparse
import csv
from pathlib import Path
import sys
import os
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT/'tools'))

import numpy as np
from ms2_eval.io import load_ms2_gt, sha256_file
from ms2_eval.official_protocol import OFFICIAL_METRICS, evaluate_sample, official_valid_mask
from tools.da2_caption_data import TRAIN, checked_manifest
from tools.train_da2_caption import json_write

RUNS = {'frame': 'sd2_cap_pct_logtn_img_s43', 'empty': 'sd2_nocap_pct_logtn_img_s43'}
TRAIN_FILE = 'ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl'
VAL_FILE = 'ms2_val_official3_thermalcap_20260821.jsonl'
GRID = tuple(i/40 for i in range(21))  # 0..0.50: upper-tail slope 1..1.50.
QUANTILE = .80
ABS_TOL, A1_TOL, RMSE_GAIN = .0005, .001, .05


def sample_fit(train, per_sequence=256):
    """Seeded, stratified train-only selection; keep original manifest indices for noise."""
    groups = {seq: [] for seq in sorted(TRAIN)}
    for s in train:
        seq = s.sample_id.rsplit('_', 1)[0]
        if seq not in groups:
            raise ValueError(f'Non-training frame in fit: {s.sample_id}')
        groups[seq].append(s)
    rng = np.random.default_rng(43)
    selected = [group[int(i)] for group in groups.values()
                for i in rng.choice(len(group), per_sequence, replace=False)]
    return sorted(selected, key=lambda s: s.manifest_line)


def tail(raw):
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError('Expected one finite native-resolution log-depth prediction')
    return np.maximum(raw - np.float32(np.quantile(raw, QUANTILE)), 0.)


def stretch(raw, excess, beta):
    if not 0 <= beta <= GRID[-1]:
        raise ValueError('Out-of-range stretch')
    return raw if beta == 0 else raw + np.float32(beta)*excess


def score_pixels(raw, gt, betas):
    if raw.shape != gt.shape:
        raise ValueError('Native prediction/LiDAR geometry mismatch')
    excess = tail(raw)  # Use all predicted pixels, never the LiDAR mask, for the knot.
    mask = official_valid_mask(gt, .001, 80.)
    raw, excess, gt = (x[mask][None, :] for x in (raw, excess, gt))
    return [evaluate_sample(stretch(raw, excess, b), gt, align='ssi_log',
                            min_depth=.001, max_depth=80.) for b in betas]


def choose_beta(scores):
    zero = scores[0.]
    eligible = [b for b,m in scores.items()
                if m['abs_rel'] <= zero['abs_rel'] + ABS_TOL and m['a1'] >= zero['a1'] - A1_TOL]
    return min(eligible, key=lambda b: (scores[b]['rmse'], b))


def gates(zero, corrected):
    # Metrics are numpy floats; JSON cannot serialize numpy.bool_.
    return dict(rmse=bool(corrected['rmse'] <= zero['rmse'] - RMSE_GAIN),
                abs_rel=bool(corrected['abs_rel'] <= zero['abs_rel'] + ABS_TOL),
                a1=bool(corrected['a1'] >= zero['a1'] - A1_TOL))


def checkpoint(roots, arm):
    candidates = [r/RUNS[arm]/'converted/step16000_weights.pt' for r in roots]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f'Missing {arm} checkpoint: {candidates}')


def load_arm(model, path, arm):
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=False)
    mode = 'correct' if arm == 'frame' else 'empty'
    if (payload.get('route') != 'b_thermal_unet' or payload.get('caption_mode') != mode
            or payload.get('global_step') != 16000 or set(payload.get('state_dicts', {})) != {'unet'}):
        raise ValueError(f'Wrong converted {arm} checkpoint: {path}')
    model.load_state_dicts(payload['state_dicts'], str(path))
    model.unet.requires_grad_(False)
    model.set_train(False)


def predict_samples(model, samples, arm, recipe):
    import torch
    from tools.train_route_suite import load_input_tensor
    with torch.inference_mode():
        empty = model.encode_prompt('') if arm == 'empty' else None
        for s in samples:
            row = dict(id=s.sample_id, image_path=s.thermal_path, manifest_index=s.manifest_line-1)
            image, _ = load_input_tensor(row, model.modality, recipe)
            prompt = empty if arm == 'empty' else model.encode_prompt(s.caption)
            raw = model.predict_disparity(row, image, prompt).float().cpu().numpy()
            _, gt = load_ms2_gt(s.thermal_gt_path, 256.)
            yield s.sample_id, raw, gt


def measure(samples, predictions, betas, csv_file=None):
    totals = np.zeros((len(betas), len(OFFICIAL_METRICS)), dtype=np.float64)
    writer = csv.writer(csv_file) if csv_file is not None else None
    if writer:
        writer.writerow(('id', 'beta', *OFFICIAL_METRICS))
    for i, (sample, (ident, raw, gt)) in enumerate(zip(samples, predictions, strict=True), 1):
        if ident != sample.sample_id:
            raise ValueError('Frame ID/order mismatch')
        metrics = score_pixels(raw, gt, betas)
        for j, m in enumerate(metrics):
            values = [m[k] for k in OFFICIAL_METRICS]
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite metrics for {ident}')
            totals[j] += values
            if writer:
                writer.writerow((ident, betas[j], *values))
        if i % 500 == 0:
            print(f'[scored] {i}/{len(samples)}', flush=True)
    return {b: dict(zip(OFFICIAL_METRICS, row/len(samples))) for b,row in zip(betas, totals)}


def run(a):
    import torch
    from huggingface_hub import snapshot_download
    from tools.train_route_suite import RouteModel
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
    out = a.out
    out.mkdir(parents=True, exist_ok=False)
    train, ti = checked_manifest(a.manifests/TRAIN_FILE, a.data_root, 'train', 75688)
    fit = sample_fit(train)
    val, vi = checked_manifest(a.manifests/VAL_FILE, a.data_root, 'val', 14249)
    if not vi['sha256'].startswith('c26a5859'):
        raise ValueError('Wrong validation manifest')
    sources = {arm:checkpoint(a.checkpoints, arm) for arm in RUNS}
    snapshot = snapshot_download('jingheya/lotus-depth-g-v2-1-disparity', local_files_only=True)
    recipe = SimpleNamespace(route='b_thermal_unet', backbone='g', lotus_model_path=snapshot,
                             local_files_only=True, gt_decode_fp32=True, condition_latent='mode',
                             timestep=999, num_inference_steps=1, seed=20260703)
    contract = dict(train=ti, val=vi, fit_ids=[s.sample_id for s in fit], grid=GRID, quantile=QUANTILE,
                    gates=dict(rmse_gain=RMSE_GAIN, absrel_tol=ABS_TOL, a1_tol=A1_TOL),
                    recipe=vars(recipe), runner_sha256=sha256_file(Path(__file__)),
                    checkpoints={k:{'path':str(p), 'sha256':sha256_file(p)} for k,p in sources.items()})
    if len({s['sha256'] for s in contract['checkpoints'].values()}) != 2:
        raise ValueError('Expected distinct frame/empty checkpoints')
    json_write(out/'contract.json', contract)
    model = RouteModel(recipe, torch.device('cuda'), torch.float16)
    betas = {}
    for arm,path in sources.items():
        load_arm(model, path, arm)
        scores = measure(fit, predict_samples(model, fit, arm, recipe), GRID)
        betas[arm] = choose_beta(scores)
        json_write(out/f'{arm}_fit.json', dict(beta=betas[arm], train_scores=scores, frames=len(fit)))
    json_write(out/'frozen_betas.json', betas)
    print('[frozen-fit]', betas, flush=True)
    results = {}
    for arm,path in sources.items():
        load_arm(model, path, arm)
        beta = betas[arm]
        with (out/f'{arm}_val.csv').open('w', encoding='utf8', newline='') as f:
            scores = measure(val, predict_samples(model, val, arm, recipe), sorted({0., beta}), f)
        zero, corrected = scores[0.], scores[beta]
        rules = gates(zero, corrected)
        results[arm] = dict(beta=beta, zero=zero, corrected=corrected, gates=rules, pass_self=all(rules.values()))
        print('[validation]', arm, results[arm], flush=True)
    # Published full-validation l050 control (rounded); catch changed input/decoding/seed.
    zero = results['frame']['zero']
    for k, expected, tol in [('abs_rel', .07936, .00005), ('rmse', 2.9457, .002), ('a1', .94227, .00005)]:
        if abs(zero[k]-expected) > tol:
            raise ValueError(f'Historical frame zero not reproduced: {k} {zero[k]} vs {expected}')
    comparisons = {stage: {k:results['frame'][stage][k]-results['empty'][stage][k] for k in OFFICIAL_METRICS}
                   for stage in ('zero', 'corrected')}
    summary = dict(validation_frames=len(val), results=results, frame_minus_empty=comparisons,
                   adopt_frame_stretch=results['frame']['pass_self'])
    json_write(out/'summary.json', summary)
    print('[frame-minus-empty]', comparisons, flush=True)
    print('[completed]', out/'summary.json', 'adopt_frame_stretch=', summary['adopt_frame_stretch'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out', 'data-root', 'manifests'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    run(p.parse_args())
