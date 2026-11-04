"""Validation-only attribution of the pure Iris RMSE gaps (positive means A is worse)."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import diagnose_error_sources as diag
from tools import test_iris_far_pairs_frame_4k as prior
from tools.test_iris_cross_region_pairs import checkpoint

EDGES = (0, 5, 10, 20, 30, 40, 60, 80)
BANDS = [f'{lo}-{hi}m' for lo, hi in zip(EDGES, EDGES[1:])]
PAIRS = [('base', 'local'), ('local', 'cross'), ('base', 'newcrf'),
         ('local', 'newcrf'), ('cross', 'newcrf')]


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-10)


def moments(raw, gt, align):
    valid = diag.official_valid_mask(gt, .001, 80.)
    if not valid.any():
        raise ValueError('Validation frame has no valid GT')
    pred = diag.resize_dense_prediction(diag.collapse_channels(raw), gt.shape)
    official = diag.evaluate_sample(pred, gt, align=align)
    error = np.clip(diag.aligned_depth(pred, gt, valid, align)[valid], .001, 80.) - gt[valid]
    bins = np.searchsorted(EDGES[1:-1], gt[valid], side='right')
    count = np.bincount(bins, minlength=len(BANDS))
    signed = np.bincount(bins, weights=error, minlength=len(BANDS))
    squared = np.bincount(bins, weights=error**2, minlength=len(BANDS))
    close(np.sqrt(squared.sum()/count.sum()), official['rmse'])
    return count, signed, squared, official['rmse']


def parts(stats, idx, weights):
    n, signed, squared, _ = (x[idx] for x in stats)
    means = signed / np.maximum(n, 1)
    mse = squared / np.maximum(n, 1)
    mass = weights.sum(axis=0)
    mean = (weights * means).sum(axis=0) / np.where(mass > 0, mass, 1.)
    shared = mass * mean**2
    between = (weights * (means - mean)**2).sum(axis=0)
    within = (weights * (mse - means**2)).sum(axis=0)
    return np.stack([shared, between, within]) / len(idx), mean


def breakdown(a, b, sequences):
    n = a[0]
    np.testing.assert_array_equal(n, b[0])
    denom = n.sum(axis=1) * (a[3] + b[3])
    inv = np.divide(1., denom, out=np.zeros_like(denom), where=denom > 0)
    weights = n * inv[:, None]
    contrib = (a[2] - b[2]) * inv[:, None]
    frame_gap = a[3] - b[3]
    close(contrib.sum(axis=1), frame_gap)
    rows = []
    groups = [('ALL', np.arange(len(n)))]
    groups.extend((seq, np.flatnonzero(sequences == seq)) for seq in np.unique(sequences))
    for seq, idx in groups:
        band_gap = contrib[idx].mean(axis=0)
        left, left_mean = parts(a, idx, weights[idx])
        right, right_mean = parts(b, idx, weights[idx])
        delta = left - right
        close(delta.sum(axis=0), band_gap)
        gap = frame_gap[idx].mean()
        close(band_gap.sum(), gap)
        for i, band in enumerate(BANDS):
            supported = weights[idx, i].sum() > 0
            rows.append(dict(sequence=seq, frames=len(idx), depth=band, pixels=int(n[idx,i].sum()),
                frames_with_bin=int((n[idx,i] > 0).sum()), sequence_rmse_gap_m=gap,
                sequence_contribution_to_all_m=gap*len(idx)/len(n),
                bin_gap_m=band_gap[i], contribution_to_all_m=band_gap[i]*len(idx)/len(n),
                mean_error_a_m=left_mean[i] if supported else None,
                mean_error_b_m=right_mean[i] if supported else None,
                common_bias_gap_m=delta[0,i], between_frames_gap_m=delta[1,i],
                within_frame_gap_m=delta[2,i]))
    close(sum(r['contribution_to_all_m'] for r in rows if r['sequence'] != 'ALL'), frame_gap.mean())
    return rows


def collect(samples, predictions, align, expected, name, out):
    rows = []
    for i, (sample, (ident, raw, gt)) in enumerate(zip(samples, predictions, strict=True), 1):
        if ident != sample.sample_id:
            raise ValueError(f'{name}: prediction ID/order differs at {ident}')
        rows.append(moments(raw, gt, align))
        if i % 1000 == 0:
            print(f'[validation] {name} {i}/{len(samples)}', flush=True)
    n, signed, squared, rmse = map(np.asarray, zip(*rows))
    if abs(rmse.mean() - expected) > .002:
        raise ValueError(f'{name}: validation RMSE {rmse.mean()} differs from {expected}')
    np.savez(out/f'{name}_moments.npz', ids=[s.sample_id for s in samples],
             counts=n, signed=signed, squared=squared, official_rmse=rmse)
    print(f'[reproduced] {name} RMSE={rmse.mean():.6f} expected={expected:.6f}', flush=True)
    return n, signed, squared, rmse


def reference_cache(a, info, samples):
    weights = a.baselines/'supdepth4thermal'/'MS2_MD_NeWCRF_THR_ckpt.ckpt'
    digest = diag.sha256_file(weights)
    attempts = [path for job in (8263904, 8285836) for path in sorted(a.diag_root.glob(f'*/attempt_{job}_*'))]
    for attempt in attempts:
        report = attempt/'analysis'/'summary.json'
        if not report.is_file():
            continue
        saved = prior.read(report)
        raw = attempt/'newcrf'/'raw_predictions'/'newcrf'
        if (saved['manifest']['sha256'] == info['sha256'] and saved['frames'] == len(samples)
                and saved.get('checkpoints', {}).get('newcrf', {}).get('sha256') == digest
                and all((raw/f'{s.sample_id}.npy').is_file() for s in samples)):
            print('[reference] Reusing verified official NeWCRF validation cache:', raw, flush=True)
            return raw, saved['headline']['newcrf']['rmse'], digest
    print('[reference] Verified old cache unavailable; generating official NeWCRF validation predictions', flush=True)
    folder = a.out/'newcrf_val'
    subprocess.run([sys.executable, str(diag.ROOT/'tools/run_ms2_supdepth_baselines.py'),
        '--models', 'newcrf', '--bmsd-root', str(a.baselines/'SupDepth4Thermal'),
        '--ckpt-dir', str(weights.parent), '--ms2-root', str(a.data_root),
        '--test-env', 'val', '--sample-step', '1', '--expect-frames', str(len(samples)),
        '--save-all-preds', '--output-dir', str(folder)], check=True)
    meta = prior.read(folder/'newcrf_val.json')
    if meta['frames'] != len(samples):
        raise ValueError('NeWCRF validation crawl differs from frozen manifest')
    return folder/'raw_predictions'/'newcrf', meta['by_alignment']['ssi']['rmse'], digest


def run(a):
    from tools.train_route_suite import RouteModel
    from tools import train_iris_far_pairs as train
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
    parent, contract, source, local, step = prior.selection(a.local_run)
    cross_settings = prior.read(a.cross_run/'contract.json')
    cross_summary = prior.read(a.cross_run/'summary.json')
    local_summary = prior.read(a.local_run/'summary.json')
    if (step != 2000 or Path(cross_settings['parent']) != a.local_run
            or cross_settings['selection_sha256'] != diag.sha256_file(a.local_run/'summary.json')
            or cross_settings['start'] != step or cross_settings['budget'] != 2000
            or Path(cross_settings['source']) != source or Path(cross_settings['start_conv']) != local
            or cross_settings['start_conv_sha256'] != diag.sha256_file(local)):
        raise ValueError('Cross-region run did not start from the requested local 2k model')
    checkpoint(a.cross_run, step, local, local_summary)
    candidate = a.cross_run/'conv_4000.pt'  # +2000 candidate, even when the selector kept its start.
    val_path = Path(contract['val']['path'])
    if val_path.name != train.old.VAL_FILE:
        raise ValueError('Expected frozen official validation manifest')
    samples, info = train.checked_manifest(val_path, a.data_root, 'val', 14249)
    if info != contract['val'] or cross_settings['val'] != info:
        raise ValueError('Validation manifest changed')
    a.out.mkdir(parents=True, exist_ok=False)
    raw_dir, expected_crf, crf_hash = reference_cache(a, info, samples)
    def cached_predictions():
        for s in samples:
            _, gt = diag.load_ms2_gt(s.thermal_gt_path, 256.)
            yield s.sample_id, np.load(raw_dir/f'{s.sample_id}.npy', allow_pickle=False), gt
    stats = {'newcrf': collect(samples, cached_predictions(), 'ssi', expected_crf, 'newcrf', a.out)}
    recipe = SimpleNamespace(**contract['recipe'])
    model = RouteModel(recipe, train.torch.device('cuda'), train.torch.float16)
    original_rmse = prior.read(parent/'train/frame/summary.json')['full']['0']['rmse']
    for name, conv, expected in [('base', None, original_rmse), ('local', local, local_summary['full']['2000']['rmse']),
                                  ('cross', candidate, cross_summary['full_by_added_steps']['2000']['rmse'])]:
        prior.pilot.load_selected(model, 'frame', source, conv, contract)
        preds = train.old.predict_samples(model, samples, 'frame', recipe)
        stats[name] = collect(samples, preds, 'ssi_log', expected, name, a.out)
    sequences = np.array([s.sample_id.rsplit('_', 1)[0] for s in samples])
    rows = []
    for left, right in PAIRS:
        records = breakdown(stats[left], stats[right], sequences)
        rows.extend(dict(a=left, b=right, **r) for r in records)
        overall = [r for r in records if r['sequence'] == 'ALL']
        print(f'[gap] {left} minus {right}: {100*overall[0]["sequence_rmse_gap_m"]:+.3f} cm; '
              + ' '.join(f'{r["depth"]}={100*r["bin_gap_m"]:+.3f}' for r in overall), flush=True)
        for r in records:
            if r['sequence'] != 'ALL' and r['depth'] == BANDS[0]:
                print(f'[sequence] {left} minus {right} {r["sequence"]} ({r["frames"]} frames): '
                      f'{100*r["sequence_rmse_gap_m"]:+.3f} cm within sequence, '
                      f'{100*r["sequence_contribution_to_all_m"]:+.3f} cm in full validation', flush=True)
    with (a.out/'attribution.csv').open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    train.json_write(a.out/'summary.json', dict(validation=info, frames=len(samples),
        official_rmse_m={name: s[3].mean() for name,s in stats.items()},
        paths=dict(base=str(source), local=str(local), cross=str(candidate), newcrf_cache=str(raw_dir)),
        sha256=dict(base=diag.sha256_file(source), local=diag.sha256_file(local),
                    cross=diag.sha256_file(candidate), newcrf=crf_hash),
        align=dict(base='ssi_log', local='ssi_log', cross='ssi_log', newcrf='ssi'),
        definition='A minus B in metres after native whole-frame alignment and clamp. Within each sequence or ALL, '
                   'decompose with weight n_bin / (N_frame * (RMSE_A + RMSE_B)). Bias sign is pred minus GT. '
                   'contribution_to_all_m includes the sequence frame fraction. ALL bias terms are recomputed across frames.',
        attribution=rows))
    print('[completed]', a.out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('local-run', 'cross-run', 'diag-root', 'data-root', 'baselines', 'out'):
        p.add_argument('--'+k, type=Path, required=True)
    run(p.parse_args())
