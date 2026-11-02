"""Rebuild the final summary from completed validation rows, without inference."""
import argparse
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from ms2_eval.io import sha256_file
from tools import iris_far_stretch as stretch
from tools.test_iris_far_stretch import read, selection
from tools.train_da2_caption import json_write


def fold(path, ids, beta):
    totals = {b:np.zeros(len(stretch.OFFICIAL_METRICS)) for b in sorted({0.,beta})}
    with path.open(encoding='utf8',newline='') as f:
        rows = csv.DictReader(f)
        for ident in ids:
            for b, total in totals.items():
                row = next(rows, None)
                if row is None or row['id'] != ident or float(row['beta']) != b:
                    raise ValueError(f'Missing or mismatched paired row: {path}, {ident}, {b}')
                values = [float(row[k]) for k in stretch.OFFICIAL_METRICS]
                if not np.isfinite(values).all():
                    raise ValueError(f'Nonfinite validation metrics: {ident}')
                total += values
        if next(rows,None) is not None:
            raise ValueError(f'Unexpected trailing CSV rows: {path}')
    return {b:dict(zip(stretch.OFFICIAL_METRICS,map(float,total/len(ids)))) for b,total in totals.items()}


def recover(parent):
    contract = read(parent/'contract.json')
    manifest = Path(contract['val']['path'])
    if (contract['runner_sha256'] != sha256_file(stretch.__file__) or
            sha256_file(manifest) != contract['val']['sha256'] or not contract['val']['sha256'].startswith('c26a5859')):
        raise ValueError('Wrong validation manifest or runner')
    with manifest.open(encoding='utf8') as f:
        ids = [json.loads(line)['id'] for line in f if line.strip()]
    if len(ids)!=14249 or len(set(ids))!=len(ids):
        raise ValueError('Expected exactly 14249 unique validation frames')
    betas = read(parent/'frozen_betas.json')
    results = {}
    for arm in stretch.RUNS:
        beta = betas[arm]
        fit = read(parent/f'{arm}_fit.json')
        if fit['beta'] != beta or beta != stretch.choose_beta({float(k):v for k,v in fit['train_scores'].items()}):
            raise ValueError('Frozen coefficient disagrees with saved training fit')
        scores = fold(parent/f'{arm}_val.csv', ids, beta)
        rules = stretch.gates(scores[0.],scores[beta])
        results[arm] = dict(beta=beta,zero=scores[0.],corrected=scores[beta],gates=rules,pass_self=all(rules.values()))
    zero = results['frame']['zero']
    if any(abs(zero[k]-v)>tol for k,v,tol in [('abs_rel',.07936,.00005),('rmse',2.9457,.002),('a1',.94227,.00005)]):
        raise ValueError('Historical zero not reproduced')
    diffs = {stage:{k:results['frame'][stage][k]-results['empty'][stage][k] for k in stretch.OFFICIAL_METRICS}
             for stage in ('zero','corrected')}
    summary = dict(validation_frames=len(ids),results=results,frame_minus_empty=diffs,
                   adopt_frame_stretch=results['frame']['pass_self'])
    dest = parent/'summary.json'
    if dest.exists() and read(dest) != summary:
        raise ValueError('Existing summary differs; refusing to replace it')
    json_write(dest,summary)
    adopt, _, _ = selection(parent)
    print('[recovered]',dest,'adopt_frame_stretch=',adopt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('parent',type=Path)
    recover(p.parse_args().parent)
