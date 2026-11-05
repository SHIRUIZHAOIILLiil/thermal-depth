"""Frozen validation-only selection toward NeWCRF; no training dependencies."""
import csv
import json
import math
from pathlib import Path

import numpy as np

from ms2_eval.io import sha256_file

POLICY = dict(version='newcrf_primary3_v2', absrel_increase=.002, a1_decrease=.003,
              rmse_gain_m=.02, reference_align='ssi', candidate_align='ssi_log',
              ranking='mean positive relative deficit in AbsRel, RMSE, (1-a1); then RMSE, AbsRel, step')
KEYS = ('abs_rel', 'rmse', 'a1')


def checked_metrics(metrics):
    values = {k:float(metrics[k]) for k in KEYS}
    if (not all(math.isfinite(x) for x in values.values()) or values['abs_rel'] <= 0
            or values['rmse'] <= 0 or not 0 < values['a1'] < 1):
        raise ValueError('Nonfinite or invalid selection metrics')
    return values


def comparison(metrics, reference):
    m, r = checked_metrics(metrics), checked_metrics(reference)
    ratios = dict(abs_rel=m['abs_rel']/r['abs_rel'], rmse=m['rmse']/r['rmse'],
                  a1_error=(1-m['a1'])/(1-r['a1']))
    return dict(deficit_score=sum(max(0., x-1.) for x in ratios.values())/3,
                relative_errors=ratios, matched_primary3=all(x <= 1.+1e-12 for x in ratios.values()),
                rmse_gap_m=m['rmse']-r['rmse'], absrel_gap=m['abs_rel']-r['abs_rel'],
                a1_gap=m['a1']-r['a1'])


def select(full, reference, start=2000):
    if set(full) != {start, 3000, 4000}:
        raise ValueError('Expected start, +1000 and +2000 validation results')
    base = checked_metrics(full[start])
    initial = comparison(base, reference)
    table = {}
    for step, raw in sorted(full.items()):
        m = checked_metrics(raw)
        row = comparison(m, reference)
        gates = dict(absrel=m['abs_rel'] <= base['abs_rel']+POLICY['absrel_increase']+1e-12,
                     a1=m['a1'] >= base['a1']-POLICY['a1_decrease']-1e-12,
                     rmse=m['rmse'] <= base['rmse']-POLICY['rmse_gain_m']+1e-12,
                     overall=row['deficit_score'] < initial['deficit_score']-1e-12
                             or row['matched_primary3'])
        row.update(metrics=m, gates=gates, eligible=step != start and all(gates.values()),
                   rmse_gain_m=base['rmse']-m['rmse'])
        table[step] = row
    eligible = [s for s,r in table.items() if r['eligible']]
    chosen = min(eligible, key=lambda s:(table[s]['deficit_score'], full[s]['rmse'], full[s]['abs_rel'], s),
                 default=start)
    status = ('retained_start' if chosen == start else
              'matched_newcrf_primary3_on_val' if table[chosen]['matched_primary3'] else 'narrowed_gap_on_val')
    return chosen, dict(policy=POLICY, reference=checked_metrics(reference), status=status,
                        selected=chosen, candidates=table,
                        note='Selection score is internal, not a paper metric. Primary3 validation parity is not a test/all-seven win.')


def read_reference(archive_dir, archive, validation, ids):
    """Reuse the exact verified NeWCRF cache, including frame IDs and weight hash."""
    if archive['validation'] != validation or archive['align']['newcrf'] != 'ssi':
        raise ValueError('Reference validation/alignment mismatch')
    raw = Path(archive['paths']['newcrf_cache'])
    report_path = raw.parents[2]/'analysis'/'summary.json'
    csv_path = report_path.with_name('per_frame.csv')
    if not report_path.is_file() or not csv_path.is_file():
        raise FileNotFoundError(f'Need the verified diagnostic summary/CSV at {report_path.parent}; no new inference allowed')
    report = json.loads(report_path.read_text())
    models = [x for x in report['models'] if x['name']=='newcrf']
    if (report['manifest']['sha256'] != validation['sha256']
            or report['manifest']['sample_count'] != len(ids) or report['frames'] != len(ids)
            or report['skipped_missing_prediction'] != 0
            or report['checkpoints']['newcrf']['sha256'] != archive['sha256']['newcrf']
            or len(models) != 1 or models[0]['align'] != 'ssi'
            or Path(models[0]['prediction_dir']) != raw
            or any(report['settings'][k] != v for k,v in
                   dict(min_depth=.001,max_depth=80.,depth_scale=256.,limit=0).items())):
        raise ValueError('NeWCRF reference provenance mismatch')
    with csv_path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if [r['id'] for r in rows] != ids or len(set(ids)) != len(ids):
        raise ValueError('NeWCRF reference IDs/order differ from validation')
    metrics = {k:float(np.mean([float(r['newcrf_'+k]) for r in rows])) for k in KEYS}
    checked_metrics(metrics)
    for k in KEYS:
        if not np.isclose(metrics[k], report['headline']['newcrf'][k], atol=1e-9, rtol=1e-10):
            raise ValueError('NeWCRF reference CSV/headline mismatch')
    stats = archive_dir/'newcrf_moments.npz'
    with np.load(stats, allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], ids)
        np.testing.assert_allclose(z['official_rmse'].mean(), metrics['rmse'], atol=1e-9, rtol=1e-10)
    np.testing.assert_allclose(archive['official_rmse_m']['newcrf'], metrics['rmse'], atol=1e-9, rtol=1e-10)
    return dict(metrics=metrics, align='ssi', validation=validation,
                checkpoint_sha256=archive['sha256']['newcrf'], report=str(report_path),
                report_sha256=sha256_file(report_path), per_frame_sha256=sha256_file(csv_path),
                archive_summary_sha256=sha256_file(archive_dir/'summary.json'),
                moments_sha256=sha256_file(stats))
