"""CPU-only alignment sensitivity on the cached NeWCRF validation maps from job 9167854."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import diagnose_pair_rmse as attribution
from tools import diagnose_error_sources as diag
from tools.da2_caption_data import checked_manifest
from tools.train_da2_caption import json_write
from tools.iris_far_stretch import VAL_FILE


def compare_frame(raw, gt):
    valid = diag.official_valid_mask(gt, .001, 80.)
    pred = diag.resize_dense_prediction(diag.collapse_channels(raw), gt.shape)
    if not valid.any() or not np.isfinite(pred[valid]).all() or (pred[valid] <= 0).any():
        raise ValueError('Need positive, finite NeWCRF depths at every valid LiDAR pixel')
    log_pred = np.zeros_like(pred)
    np.log(pred, out=log_pred, where=valid)
    far = valid & (gt >= 60)
    rows, too_near = [], []
    for x, mode in ((pred, 'ssi'), (log_pred, 'ssi_log')):
        rows.append(attribution.moments(x, gt, mode))
        aligned = np.clip(diag.aligned_depth(x, gt, valid, mode)[far], .001, 80.)
        too_near.append(np.count_nonzero(aligned < gt[far]))
    return rows, too_near


def archived_stats(path, ids):
    with np.load(path, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], ids)
        return tuple(z[k] for k in ('counts', 'signed', 'squared', 'official_rmse'))


def run(a):
    before = attribution.prior.read(a.previous/'summary.json')
    val = Path(before['validation']['path'])
    if val.name != VAL_FILE or not before['validation']['sha256'].startswith('c26a5859'):
        raise ValueError('Expected the frozen official validation manifest')
    samples, info = checked_manifest(val, a.data_root, 'val', 14249)
    if info != before['validation'] or before['align']['newcrf'] != 'ssi':
        raise ValueError('Prior diagnostic manifest or reference alignment changed')
    ids = [s.sample_id for s in samples]
    seqs = np.asarray([i.rsplit('_', 1)[0] for i in ids])
    raw_dir = Path(before['paths']['newcrf_cache'])
    diag.require_predictions(samples, [('newcrf', raw_dir, 'ssi')])
    reference = archived_stats(a.previous/'newcrf_moments.npz', ids)
    iris = archived_stats(a.previous/'local_moments.npz', ids)
    attribution.close(iris[3].mean(), before['official_rmse_m']['local'])
    a.out.mkdir(parents=True, exist_ok=False)
    native, alternative = [], []
    near = np.zeros(2, dtype=np.int64)
    for i, s in enumerate(samples, 1):
        _, gt = diag.load_ms2_gt(s.thermal_gt_path, 256.)
        try:
            rows, counts = compare_frame(np.load(raw_dir/f'{s.sample_id}.npy', allow_pickle=False), gt)
        except ValueError as e:
            raise ValueError(f'{s.sample_id}: {e}') from e
        native.append(rows[0])
        alternative.append(rows[1])
        near += counts
        if i % 1000 == 0:
            print(f'[validation] {i}/{len(samples)}', flush=True)
    native, alternative = [tuple(map(np.asarray, zip(*rows))) for rows in (native, alternative)]
    for actual, expected in zip(native, reference):
        attribution.close(actual, expected)
    attribution.close(native[3].mean(), before['official_rmse_m']['newcrf'])
    paired = {}
    for label, a_stats, b_stats in [('log_minus_ssi', alternative, native),
                                    ('iris_minus_ssi', iris, native), ('iris_minus_log', iris, alternative)]:
        paired[label] = attribution.breakdown(a_stats, b_stats, seqs)

    def headline(rows):
        far = next(r for r in rows if r['sequence'] == 'ALL' and r['depth'] == '60-80m')
        return dict(gap_cm=100*far['sequence_rmse_gap_m'], far_contribution_cm=100*far['bin_gap_m'])

    headlines = {k:headline(v) for k,v in paired.items()}
    outcomes = {}
    for label, stats, count in zip(('ssi', 'log_then_ssi_log'), (native, alternative), near):
        far_count = stats[0][:,-1].sum()
        if not far_count:
            raise ValueError('No valid far pixels')
        outcomes[label] = dict(rmse_m=stats[3].mean(), far_pixels=int(far_count),
            far_mean_error_m=stats[1][:,-1].sum()/far_count, far_too_near_pct=100*count/far_count)
        print('[alignment]', label, outcomes[label], flush=True)
    change = {k:outcomes['log_then_ssi_log'][k]-outcomes['ssi'][k]
              for k in ('rmse_m','far_mean_error_m','far_too_near_pct')}
    print('[change] log minus ssi', change, flush=True)
    for label, values in headlines.items():
        print('[gap]', label, values, flush=True)
    json_write(a.out/'summary.json', dict(validation=info, previous_summary_sha256=diag.sha256_file(a.previous/'summary.json'),
        newcrf_checkpoint_sha256=before['sha256']['newcrf'], raw_cache=str(raw_dir), alignments=outcomes,
        log_minus_ssi=change, headlines=headlines, attribution=paired,
        notes='GT mask shared; log of native-resolution positive predictions before whole-frame fitting. '
              'Far mean error is pixel-pooled pred minus GT; too-near percentage is pixel-pooled. '
              'gap_cm and far_contribution_cm are macro per-frame RMSE differences.'))
    print('[completed]', a.out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('previous','data-root','out'):
        p.add_argument('--'+key, type=Path, required=True)
    run(p.parse_args())
