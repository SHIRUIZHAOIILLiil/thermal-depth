"""Read-only VAE roundtrip of restored, censored real 60–80 m LiDAR targets.

Reuses job 9174706's validation subset and calibrated diagnostic pseudo maps.
No model inference, pseudo regeneration, training or test evaluation.
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
from ms2_eval.io import sha256_file
from ms2_eval.official_protocol import official_valid_mask
from tools import diagnose_depth_chain as chain

STAGES = ('original_unit', 'restored_unit', 'original_vae', 'restored_vae')
METRICS = ('bias_m', 'rmse_m', 'abs_rel', 'a1', 'above_80_pct')


def restore_far(target, gt, valid, bounds):
    """Keep the original bounds and every non-restored target value exactly."""
    target = np.asarray(target, np.float32)
    gt = np.asarray(gt, np.float32)
    valid = np.asarray(valid, bool)
    lo, hi = map(float, bounds)
    if (target.shape != (3, *gt.shape) or valid.shape != gt.shape
            or not np.isfinite(target).all() or not np.isfinite([lo, hi]).all() or hi <= lo):
        raise ValueError('Invalid target, GT shape or normalization bounds')
    if not np.array_equal(target, np.repeat(target[:1], 3, axis=0)):
        raise ValueError('Expected the original replicated grayscale target')
    if target.min() < -1 or target.max() > 1:
        raise ValueError('Original target must be clipped to [-1, 1]')
    far = valid & official_valid_mask(gt, .001, 80.) & (gt >= 60.)
    raw = np.zeros_like(gt)
    raw[far] = 2 * ((np.log(gt[far]) - lo) / (hi - lo + 1e-5) - .5)
    mask = far & (raw > 1.)
    if not np.all(target[:, mask] == 1.):
        raise ValueError('Censored real LiDAR does not match the original target')
    result = target.copy()
    result[:, mask] = raw[mask]
    return result, mask


def score_frame(gt, maps, bounds, restored):
    valid = official_valid_mask(gt, .001, 80.)
    bins = [('all', valid)]
    bins.extend((f'{lo}-{hi}m', valid & (gt >= lo) & (gt < hi))
                for lo, hi in ((0,20), (20,40), (40,60), (60,80)))
    bins.append(('restored_points', restored & valid))
    records = []
    for stage in STAGES:
        unit = np.asarray(maps[stage], np.float32)
        if unit.shape != gt.shape or not np.isfinite(unit).all():
            raise ValueError('Invalid VAE/target map')
        # Fit over the whole official mask, then bin. Never fit on far points alone.
        for mode, prediction in (
                ('target_scale', chain.inverse_log_unit(unit, bounds)),
                ('ssi_log', chain.aligned_depth(unit, gt, valid, 'ssi_log'))):
            for label, mask in bins:
                if not mask.any():
                    continue
                p = prediction[mask].astype(np.float64)
                g = gt[mask].astype(np.float64)
                clipped = np.clip(p, .001, 80.)
                error = clipped - g
                records.append(dict(stage=stage, mode=mode, depth=label, points=int(mask.sum()),
                    bias_m=float(error.mean()), rmse_m=float(np.sqrt(np.mean(error**2))),
                    abs_rel=float(np.mean(np.abs(error)/g)),
                    a1=float(np.mean(np.maximum(clipped/g, g/clipped) < 1.25)),
                    above_80_pct=float(100*np.mean(p > 80.))))
    return records


def aggregate(records):
    table, changes = [], []
    for seq in ['ALL', *sorted({r['sequence'] for r in records})]:
        for mode, depth in sorted({(r['mode'], r['depth']) for r in records}):
            grouped = {}
            for stage in STAGES:
                group = [r for r in records if r['mode'] == mode and r['depth'] == depth
                         and r['stage'] == stage and seq in ('ALL', r['sequence'])]
                if not group:
                    continue
                ids = {r['id'] for r in group}
                if len(ids) != len(group):
                    raise ValueError('Duplicate frame in aggregate')
                grouped[stage] = (ids, dict(sequence=seq, mode=mode, depth=depth, stage=stage,
                    frames=len(group), points=sum(r['points'] for r in group),
                    **{k:float(np.mean([r[k] for r in group])) for k in METRICS}))
                table.append(grouped[stage][1])
            if not grouped:
                continue
            if set(grouped) != set(STAGES) or any(grouped[s][0] != grouped[STAGES[0]][0] for s in STAGES):
                raise ValueError('Unpaired stages')
            values = {s:v[1] for s,v in grouped.items()}
            for suffix in ('unit', 'vae'):
                changes.append(dict(sequence=seq, mode=mode, depth=depth, transition=suffix,
                    **{'delta_'+k:values['restored_'+suffix][k]-values['original_'+suffix][k] for k in METRICS}))
            lift = values['restored_unit']['bias_m'] - values['original_unit']['bias_m']
            kept = values['restored_vae']['bias_m'] - values['original_vae']['bias_m']
            # A signed diagnostic ratio, NOT bounded accuracy or causal attribution.
            if mode == 'target_scale' and depth == 'restored_points':
                changes.append(dict(sequence=seq, mode=mode, depth=depth, transition='retention',
                    input_lift_m=lift, decoded_lift_m=kept,
                    signed_retention_ratio=kept/lift if lift > 1e-8 else None))
    return table, changes


def check_reproduction(records, reference):
    """Require the old target and old VAE stages to reproduce on identical frames."""
    for mode in ('target_scale', 'ssi_log'):
        for stage, old_stage in (('original_unit','target_unit'), ('original_vae','vae_mode')):
            group = [r for r in records if r['mode']==mode and r['stage']==stage and r['depth']=='60-80m']
            refs = [r for r in reference['table'] if r['sequence']=='ALL' and r['mode']==mode and r['stage']==old_stage]
            if len(refs) != 1 or not group:
                raise ValueError('Missing reference stage')
            ref = refs[0]
            if ref['frames'] != len(group) or ref['points'] != sum(r['points'] for r in group):
                raise ValueError('Reference far-point coverage changed')
            for key in ('bias_m', 'rmse_m'):
                actual = float(np.mean([r[key] for r in group]))
                if not np.isclose(actual, ref[key], rtol=0, atol=5e-4):
                    raise ValueError(f'Reference mismatch: {mode}/{stage}/{key}: {actual} != {ref[key]}')


def run(a):
    import torch
    from diffusers import AutoencoderKL
    from tools.da2_caption_data import VAL, checked_manifest
    from lotus.utils.ms2_thermal_dataset import MS2ThermalDataset

    source = a.source.resolve()
    reference = json.loads((source/'summary.json').read_text())
    manifest = source/'val96.jsonl'
    if sha256_file(manifest) != reference['subset_sha256']:
        raise ValueError('Reference subset hash changed')
    full = Path(reference['validation']['path'])
    if (sha256_file(full) != reference['validation']['sha256']
            or not reference['validation']['sha256'].startswith('c26a5859')
            or reference['validation']['sample_count'] != 14249):
        raise ValueError('Full validation manifest hash changed')
    expected = chain.select_rows(chain.read_jsonl(full), 96, 20261017, VAL)
    for r in expected:
        r['thermal_depth_path'] = r.get('thermal_depth_path') or r['depth_path']
    if chain.read_jsonl(manifest) != expected:
        raise ValueError('Not the frozen official validation subset')
    samples, _ = checked_manifest(manifest, a.data_root, 'val', 96)
    pseudo_dir = source/'diagnostic_pseudo'
    pseudo_hashes = {s.sample_id:sha256_file(pseudo_dir/f'{s.sample_id}.npy') for s in samples}
    dataset = MS2ThermalDataset(manifest, a.data_root, pseudo_dir,
        norm_type='log_truncnorm', truncnorm_min=.02, truncnorm_max=.98)
    a.out.mkdir(parents=True, exist_ok=False)
    vae = AutoencoderKL.from_pretrained(reference['vae'], subfolder='vae', local_files_only=True,
                                       torch_dtype=torch.float32).to('cuda').eval().requires_grad_(False)
    records, coverage = [], []
    with torch.inference_mode():
        for i, sample in enumerate(samples):
            ex = dataset[i]
            if dataset.rows[i]['id'] != sample.sample_id:
                raise ValueError('Frame order changed')
            gt = ex['gt_depth_values'][0].numpy()
            valid = ex['gt_valid_values'][0].numpy().astype(bool)
            np.testing.assert_array_equal(valid, official_valid_mask(gt, .001, 80.))
            target = ex['depth_values'].numpy()
            bounds = ex['norm_bounds'].numpy()
            restored, mask = restore_far(target, gt, valid, bounds)
            np.testing.assert_array_equal(restored[:,~mask], target[:,~mask])
            np.testing.assert_allclose(chain.inverse_log_unit(restored[0]/2+.5, bounds)[mask],
                                       gt[mask], rtol=0, atol=1e-4)
            inputs = torch.from_numpy(np.stack([target, restored])).cuda()
            decoded = vae.decode(vae.encode(inputs).latent_dist.mode()).sample
            if not torch.isfinite(decoded).all():
                raise ValueError('Nonfinite VAE reconstruction')
            units = (inputs[:,0]/2+.5).cpu().numpy()
            decoded_units = (decoded.mean(1)/2+.5).cpu().numpy()
            maps = dict(zip(STAGES, (units[0], units[1], decoded_units[0], decoded_units[1])))
            records.extend(dict(id=sample.sample_id, sequence=sample.sample_id.rsplit('_',1)[0], **r)
                           for r in score_frame(gt, maps, bounds, mask))
            coverage.append(dict(id=sample.sample_id, restored_points=int(mask.sum()),
                far_points=int((valid & (gt>=60)).sum()), input_max=float(restored.max()),
                decoded_max=float(decoded[1].max()), bounds=bounds.tolist()))
            if (i+1)%16 == 0:
                print(f'[progress] {i+1}/{len(samples)}', flush=True)
    if not sum(r['restored_points'] for r in coverage):
        raise ValueError('No censored far LiDAR points found')
    if [{'id':r['id'], 'points':r['far_points']} for r in coverage] != reference['coverage']:
        raise ValueError('Per-frame far LiDAR coverage differs from the source diagnostic')
    check_reproduction(records, reference)
    print('[reproduced] original target and VAE agree with job 9174706 within 0.0005 m', flush=True)
    table, changes = aggregate(records)
    with (a.out/'per_frame.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = dict(diagnostic_only=True, training=False, test=False,
        source=str(source), source_summary_sha256=sha256_file(source/'summary.json'),
        validation=reference['validation'], subset_sha256=reference['subset_sha256'],
        pseudo_sha256=pseudo_hashes, vae=reference['vae'], posterior='mode, float32; no output unit clamp',
        aggregation='mean over frames with GT in each bin; identical paired masks; whole-frame SSI-log',
        warning='GT-built targets only: not inference performance or evidence of caption gain. Target scale uses known bounds.',
        recipe='Restore only valid 60<=GT<80 with original normalized target>1; unchanged q02/q98 and all other pixels.',
        coverage=coverage, table=table, changes=changes)
    (a.out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    for row in table:
        if row['sequence']=='ALL':
            print('[stage]', json.dumps(row))
    for row in changes:
        if row['sequence']=='ALL':
            print('[change]', json.dumps(row))
    print('[completed]', a.out/'summary.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'out', 'data-root'):
        p.add_argument('--'+name, type=Path, required=True)
    os.environ['IRIS_THERMAL_STRETCH'] = 'percentile'
    run(p.parse_args())
