"""NeWCRF on the full official validation split under both alignments, CPU only.

Our arms are scored with `ssi_log`, NeWCRF's main-table entry with `ssi`. This scores the
cached NeWCRF maps of job 9167854 (the cache every NeWCRF validation number since has used)
under both, with the same `evaluate_sample` the validation jobs use, so the two lines can
be compared under one alignment. Control: the `ssi` RMSE must reproduce the archived
2.471 m within 2 mm. No test split, no selection.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ms2_eval.io import load_ms2_gt, sha256_file
from ms2_eval.official_protocol import collapse_channels, evaluate_sample
from ms2_eval.resize import resize_dense_prediction

KEYS = ('abs_rel', 'rmse', 'a1')
TOLERANCE_M = .002
FRAMES = 14249


def score_frame(raw, gt):
    pred = resize_dense_prediction(collapse_channels(raw), gt.shape).astype(np.float32)
    rows = {}
    for align, x in (('ssi', pred), ('ssi_log', np.log(np.maximum(pred, 1e-6)).astype(np.float32))):
        m = evaluate_sample(x, gt, align=align, min_depth=.001, max_depth=80.)
        rows[align] = {k: float(m[k]) for k in KEYS}
    return rows


def run(a):
    from tools.da2_caption_data import checked_manifest  # imports torch; kept out of score_frame

    before = json.loads((a.previous/'summary.json').read_text())
    val = Path(before['validation']['path'])
    if not before['validation']['sha256'].startswith('c26a5859'):
        raise ValueError('Expected the frozen official validation manifest')
    samples, _ = checked_manifest(val, a.data_root, 'val', FRAMES)
    raw_dir = Path(before['paths']['newcrf_cache'])
    expected = float(before['official_rmse_m']['newcrf'])
    a.out.mkdir(parents=True, exist_ok=False)
    per_frame = []
    for i, s in enumerate(samples, 1):
        _, gt = load_ms2_gt(s.thermal_gt_path, 256.)
        rows = score_frame(np.load(raw_dir/f'{s.sample_id}.npy', allow_pickle=False), gt)
        per_frame.append(dict(id=s.sample_id, **{f'{al}_{k}': v for al, r in rows.items() for k, v in r.items()}))
        if i % 1000 == 0:
            print(f'[validation] {i}/{len(samples)}', flush=True)
    means = {al: {k: float(np.mean([r[f'{al}_{k}'] for r in per_frame])) for k in KEYS} for al in ('ssi', 'ssi_log')}
    control = dict(expected_ssi_rmse=expected, measured=means['ssi']['rmse'], tolerance_m=TOLERANCE_M,
                   passed=abs(means['ssi']['rmse']-expected) <= TOLERANCE_M)
    with (a.out/'per_frame.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, list(per_frame[0]))
        w.writeheader()
        w.writerows(per_frame)
    (a.out/'summary.json').write_text(json.dumps(dict(
        control=control, means=means, frames=len(per_frame), validation=str(val),
        raw_cache=str(raw_dir), previous=str(a.previous), script_sha256=sha256_file(__file__)), indent=2))
    print('[control]', json.dumps(control), flush=True)
    for al, r in means.items():
        print('[newcrf]', al, json.dumps({k: round(v, 5) for k, v in r.items()}), flush=True)
    if not control['passed']:
        raise SystemExit('control failed: the ssi RMSE does not reproduce the archived value')
    print('[completed]', a.out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('previous', 'data-root', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    run(p.parse_args())
