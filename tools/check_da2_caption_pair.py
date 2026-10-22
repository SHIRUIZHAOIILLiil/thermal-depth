"""Read-only pair audit and pilot decision. Never submits additional training."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check(a, b, smoke=False):
    ca, cb = read(a / 'contract.json'), read(b / 'contract.json')
    if ca['arm'] != 'no-caption' or cb['arm'] != 'caption':
        raise ValueError('Expected A=no-caption and B=caption')
    if {k: v for k, v in ca.items() if k != 'arm'} != {k: v for k, v in cb.items() if k != 'arm'}:
        raise ValueError('Paired contracts differ beyond caption arm')
    ra = [json.loads(x) for x in (a / 'train.jsonl').read_text().splitlines()]
    rb = [json.loads(x) for x in (b / 'train.jsonl').read_text().splitlines()]
    # Replayed steps after recovery can occur in append-only logs. Last occurrence wins.
    ra, rb = ({r['step']: r for r in rows} for rows in (ra, rb))
    end = 32 if smoke else 8000
    for rows in (ra, rb):
        if set(range(1, end + 1)) - rows.keys():
            raise ValueError('Pair is incomplete')
    if any(ra[s]['data_ids_sha256'] != rb[s]['data_ids_sha256'] for s in range(1, end + 1)):
        raise ValueError('Training image order differs')
    for directory in (a, b):
        done = read(directory / 'completed.json')
        if done['step'] != end or done['mode'] != ('smoke' if smoke else 'train'):
            raise ValueError('No matching successful completion marker')
    if smoke:
        print('[pair-gate] BOTH GPU smokes completed; image order and frozen configs match')
        return True
    reports = [(read(a / f'val_step{s:06d}.json'), read(b / f'val_step{s:06d}.json')) for s in (6000, 8000)]
    for x, y in reports:
        if x['samples'] != 14249 or y['samples'] != 14249:
            raise ValueError('Full validation required')
    x, y = reports[-1]
    am, bm = x['metrics']['ssi'], y['metrics']['ssi']
    conditions = dict(absrel_better_both=all(y['metrics']['ssi']['abs_rel'] < x['metrics']['ssi']['abs_rel'] for x, y in reports),
                      absrel_gain_0005=am['abs_rel'] - bm['abs_rel'] >= .0005,
                      rmse_guard=bm['rmse'] <= 1.005 * am['rmse'],
                      a1_guard=bm['a1'] >= am['a1'] - .001,
                      sequences=sum(y['sequences']['ssi'][s]['abs_rel'] < v['abs_rel']
                                    for s, v in x['sequences']['ssi'].items()) >= 2)
    print(json.dumps(dict(conditions=conditions, eligible_to_consider_continuation=all(conditions.values()),
                          note='resource gate, not statistical significance; no training auto-submitted'), indent=2))
    return all(conditions.values())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a', type=Path, required=True); p.add_argument('--b', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    check(args.a, args.b, args.smoke)
