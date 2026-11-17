"""Is the synth arm's per-sequence gap to NeWCRF explained by what the frames contain? CPU only.

val_gap_breakdown.py found the RMSE gap (synth - NeWCRF, both ssi_log) concentrated in two of
the three validation sequences. Per frame this reads the official GT (valid 0.001-80 m) and the
raw 16-bit thermal image and records: mean GT depth, share of valid pixels beyond 40 m and 60 m,
valid-pixel count, thermal p1/p99 and their spread. Frames are binned into pooled quintiles of a
statistic; each sequence's gap is then re-averaged with equal weight per quintile, i.e. as if every
sequence had the same frame mix for that statistic. Reading rule: docs/IRIS_SYNTH_PRETRAIN_V1.md.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ms2_eval.io import load_ms2_gt
from ms2_eval.official_protocol import official_valid_mask
from tools import val_gap_breakdown as breakdown

STATS = ('gt_mean_m', 'gt_frac_gt40', 'gt_frac_gt60', 'gt_valid', 'thermal_p1', 'thermal_p99', 'thermal_spread')
BINS = 5
MIN_FRAMES = 30  # a sequence-bin cell with fewer frames is reported but left out of the re-average


def frame_stats(gt, thermal):
    valid = official_valid_mask(gt, .001, 80.)
    d = gt[valid].astype(np.float64)
    p1, p99 = np.percentile(thermal.astype(np.float64), (1, 99))
    return dict(gt_mean_m=float(d.mean()), gt_frac_gt40=float((d > 40).mean()), gt_frac_gt60=float((d > 60).mean()),
                gt_valid=int(valid.sum()), thermal_p1=float(p1), thermal_p99=float(p99), thermal_spread=float(p99-p1))


def reweighted(values, gap, seqs):
    """Equal weight per quintile, over the quintiles every sequence fills with >= MIN_FRAMES frames."""
    edges = np.quantile(values, np.linspace(0, 1, BINS+1)[1:-1])
    bins = np.searchsorted(edges, values, side='right')
    names = sorted(set(seqs))
    common = [b for b in range(BINS) if all(((seqs == s) & (bins == b)).sum() >= MIN_FRAMES for s in names)]
    out = {}
    for s in names:
        cells = []
        for b in range(BINS):
            m = (seqs == s) & (bins == b)
            cells.append(dict(bin=b, frames=int(m.sum()), mean_gap=float(gap[m].mean()) if m.any() else None))
        out[s] = dict(raw_mean=float(gap[seqs == s].mean()),
                      reweighted_mean=float(np.mean([gap[(seqs == s) & (bins == b)].mean() for b in common]))
                      if common else None, cells=cells)
    return dict(edges=[float(e) for e in edges], common_bins=common, by_sequence=out)


def run(a):
    from tools.da2_caption_data import checked_manifest  # imports torch

    before = json.loads((a.previous/'summary.json').read_text())
    if not before['validation']['sha256'].startswith('c26a5859'):
        raise ValueError('Expected the frozen official validation manifest')
    samples, _ = checked_manifest(Path(before['validation']['path']), a.data_root, 'val', 14249)
    synth = breakdown.read(a.runs/'eval'/f'{breakdown.TAGS["synth"]}_val_{breakdown.VAL_FP}_stride1_correct_s16000'/
                           'eval_eval_affine_invariant_log_space_per_sample.csv')
    newcrf = breakdown.read(a.newcrf, 'ssi_log_')
    ids = [s.sample_id for s in samples]
    if sorted(ids) != sorted(synth) or sorted(ids) != sorted(newcrf):
        raise ValueError('frame ids differ between the manifest and the per-frame files')
    gap = np.array([synth[i]['rmse']-newcrf[i]['rmse'] for i in ids])
    if abs(gap.mean()-.08851) > 6e-5:
        raise SystemExit(f'mean RMSE gap {gap.mean():.5f} does not reproduce 0.08851')
    seqs = np.array([i.rsplit('_', 1)[0] for i in ids])
    rows = []
    for k, s in enumerate(samples, 1):
        _, gt = load_ms2_gt(s.thermal_gt_path, 256.)
        with Image.open(s.thermal_path) as im:
            thermal = np.asarray(im)
        rows.append(frame_stats(gt, thermal))
        if k % 2000 == 0:
            print(f'[frames] {k}/{len(samples)}', flush=True)
    table = {k: np.array([r[k] for r in rows]) for k in STATS}
    per_seq = {s: {k: float(table[k][seqs == s].mean()) for k in STATS} for s in sorted(set(seqs))}
    corr = {k: float(np.corrcoef(table[k], gap)[0, 1]) for k in STATS}
    result = dict(sequence_means=per_seq, correlation_with_gap=corr,
                  reweighted={k: reweighted(table[k], gap, seqs) for k in STATS})
    print('[sequence means]')
    for s, v in per_seq.items():
        print('   ', s, json.dumps({k: round(x, 3) for k, x in v.items()}))
    print('[corr with per-frame RMSE gap]', json.dumps({k: round(v, 3) for k, v in corr.items()}))
    for k in STATS:
        r = result['reweighted'][k]['by_sequence']
        line = '  '.join(f'{s[-8:]} raw {v["raw_mean"]:+.4f} rew {v["reweighted_mean"] if v["reweighted_mean"] is None else round(v["reweighted_mean"], 4)}'
                         for s, v in r.items())
        print(f'[reweighted by {k}] bins {result["reweighted"][k]["common_bins"]}  {line}')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print('[completed]', a.out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('runs', 'newcrf', 'previous', 'data-root', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    run(p.parse_args())
