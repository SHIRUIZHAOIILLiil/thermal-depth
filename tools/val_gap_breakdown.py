"""Where on the validation split does the synth arm trail NeWCRF? Per-frame CSVs only, no inference.

All three rows are full official val (14249 frames) under `ssi_log`:
  tmax1   eval/<tmax1 val tag>/eval_eval_affine_invariant_log_space_per_sample.csv
  synth   eval/<synth val tag>/...
  newcrf  outputs/lotus_line_v2/newcrf_val_ssi_log_v1/job_9488948/per_frame.csv (ssi_log_* columns)
Frames are paired by id. The means must reproduce each job's printed headline.

Reported, for gap = synth - newcrf and for gain = tmax1 - synth:
  - by sequence: frames, mean per model, contribution to the overall mean gap;
  - concentration: share of the summed per-frame gap carried by the worst 10% / 25% of frames;
  - quantiles of the per-frame gap.
The reading rule is in docs/IRIS_SYNTH_PRETRAIN_V1.md and was written before this ran.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

KEYS = ('abs_rel', 'rmse', 'a1')
VAL_FP = 'c26a5859'
TAGS = dict(tmax1='sd2_cap_pct_logtn_img_l0100_tmax1_s43', synth='sd2_cap_pct_logtn_img_l0100_tmax1_synth_s43')
HEADLINE = dict(tmax1=dict(abs_rel=.08137, rmse=2.7571, a1=.93937), synth=dict(abs_rel=.0763, rmse=2.651, a1=.9491),
                newcrf=dict(abs_rel=.07286, rmse=2.5625, a1=.95039))
TOLERANCE = dict(abs_rel=6e-5, rmse=6e-4, a1=6e-5)  # printed to 4-5 digits


def read(path, prefix=''):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return {r['id']: {k: float(r[prefix+k]) for k in KEYS} for r in rows}


def concentration(gap, shares=(.10, .25)):
    order = np.sort(gap)[::-1]
    total = order.sum()
    return {f'worst_{int(100*s)}pct': float(order[:int(round(s*len(order)))].sum()/total) for s in shares}


def breakdown(ids, seqs, a, b):
    """a - b per frame, positive means a is worse (for a1 the sign is flipped)."""
    out = {}
    for k in KEYS:
        sign = -1. if k == 'a1' else 1.
        gap = sign*(np.array([a[i][k] for i in ids])-np.array([b[i][k] for i in ids]))
        by_seq = {}
        for s in sorted(set(seqs)):
            m = seqs == s
            by_seq[s] = dict(frames=int(m.sum()), mean_gap=float(gap[m].mean()),
                             contribution=float(gap[m].sum()/len(gap)))
        out[k] = dict(mean_gap=float(gap.mean()), by_sequence=by_seq,
                      quantiles={q: float(np.quantile(gap, q)) for q in (.1, .25, .5, .75, .9)},
                      concentration=concentration(gap) if gap.sum() > 0 else None)
    return out


def run(a):
    tables = {name: read(a.runs/'eval'/f'{tag}_val_{VAL_FP}_stride1_correct_s16000'/
                         'eval_eval_affine_invariant_log_space_per_sample.csv') for name, tag in TAGS.items()}
    tables['newcrf'] = read(a.newcrf, 'ssi_log_')
    ids = sorted(tables['newcrf'])
    for name, t in tables.items():
        if sorted(t) != ids or len(ids) != 14249:
            raise ValueError(f'{name}: frame ids differ from NeWCRF or not 14249 frames')
    means = {n: {k: float(np.mean([t[i][k] for i in ids])) for k in KEYS} for n, t in tables.items()}
    control = {n: all(abs(means[n][k]-HEADLINE[n][k]) <= TOLERANCE[k] for k in KEYS) for n in means}
    print('[means]', json.dumps({n: {k: round(v, 5) for k, v in m.items()} for n, m in means.items()}))
    print('[control]', json.dumps(control))
    if not all(control.values()):
        raise SystemExit('a mean does not reproduce its job headline')
    seqs = np.array([i.rsplit('_', 1)[0] for i in ids])
    result = dict(means=means, gap_synth_minus_newcrf=breakdown(ids, seqs, tables['synth'], tables['newcrf']),
                  gain_tmax1_minus_synth=breakdown(ids, seqs, tables['tmax1'], tables['synth']))
    for label in ('gap_synth_minus_newcrf', 'gain_tmax1_minus_synth'):
        for k in KEYS:
            r = result[label][k]
            ratio = r['quantiles'][.5]/r['mean_gap'] if r['mean_gap'] else float('nan')
            print(f'[{label}] {k} mean {r["mean_gap"]:+.5f}  median/mean {ratio:.2f}  concentration {json.dumps(r["concentration"])}')
            print(f'    quantiles {json.dumps({q: round(v, 4) for q, v in r["quantiles"].items()})}')
            for s, v in r['by_sequence'].items():
                print(f'    {s}  frames {v["frames"]:5d}  mean {v["mean_gap"]:+.5f}  contributes {v["contribution"]:+.5f}')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print('[completed]', a.out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--newcrf', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())
