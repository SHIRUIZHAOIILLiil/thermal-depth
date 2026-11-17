"""Inside the frames: where does the synth arm lose to NeWCRF? Every 10th frame of each val sequence.

Diagnostic, val only, no selection. Per frame both models are aligned to GT with the evaluator's
`ssi_log` fit (synth: raw network output; NeWCRF: log of the cached depth of job 9167854), clipped
to the official 0.001-80 m, and every valid LiDAR pixel is put into a cell:
    image band  top / middle / bottom third of the rows
    GT depth    0-10 / 10-20 / 20-40 / 40-80 m
Per sequence and cell: point share, pooled RMSE of each model, share of the pooled squared-error
difference (synth - NeWCRF) and the mean signed log error of each model (positive = too far).
Pooled pixel errors weight frames by their point count, unlike the per-frame mean RMSE.

AbsRel and delta1 split exactly (v2, docs/IRIS_SYNTH_PRETRAIN_V1.md): both are per-frame means over the
frame's valid pixels, then means over frames, so cell c contributes mean_f( sum_{p in f, c} x_p / N_f ).
The cells of a frame add up to that frame's evaluator value (checked to 1e-9 on every frame), and the
cells of a sequence add up to its frame-mean AbsRel / delta1. Reading: share of the AbsRel gap
(synth - NeWCRF) per GT band, rows summed.

Controls: each frame's ssi_log RMSE must match the full validation job's per-frame CSV (synth) and
job 9488948's per-frame CSV (NeWCRF). Reading rule: docs/IRIS_SYNTH_PRETRAIN_V1.md.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ms2_eval.io import sha256_file
from ms2_eval.official_protocol import collapse_channels, evaluate_sample, official_valid_mask
from ms2_eval.resize import resize_dense_prediction
from tools import val_gap_breakdown as breakdown
from tools.diagnose_error_sources import aligned_depth

STRIDE = 10
STEP = 16000
ROWS = ('top', 'middle', 'bottom')
BANDS = ((0, 10), (10, 20), (20, 40), (40, 80))
SYNTH_TOL_M = 1e-3   # fp16 inference against the stored validation run
NEWCRF_TOL_M = 1e-6  # same cached maps, same arithmetic
# AbsRel / delta1 against the stored per-frame CSVs (same reasons); delta1 moves 1/N per flipped pixel.
STORED_TOL = dict(synth=dict(abs_rel=1e-4, a1=2e-3), newcrf=dict(abs_rel=1e-6, a1=1e-6))
EXACT_TOL = 1e-9     # cells of one frame against the evaluator on the same arrays
MODELS = ('synth', 'newcrf')


def band_reading(shares):
    """Pre-registered: which GT band holds at least half of the AbsRel gap (rows summed)."""
    groups = dict(near_0_10=shares['0-10m'], mid_10_40=shares['10-20m']+shares['20-40m'], far_40_80=shares['40-80m'])
    hit = [k for k, v in groups.items() if v >= .5]
    return dict(groups=groups, reading=hit[0] if len(hit) == 1 else ('spread' if not hit else 'ambiguous'))


class Cells:
    def __init__(self):
        shape = (len(ROWS), len(BANDS))
        self.n = np.zeros(shape, np.int64)
        self.sq = {m: np.zeros(shape) for m in ('synth', 'newcrf')}
        self.log = {m: np.zeros(shape) for m in ('synth', 'newcrf')}
        self.rel = {m: np.zeros(shape) for m in MODELS}  # sum over frames of the frame-normalised cell AbsRel
        self.hit = {m: np.zeros(shape) for m in MODELS}  # same for delta1 (ratio < 1.25)
        self.frames = 0

    def add(self, gt, valid, aligned):
        """Adds one frame; returns that frame's AbsRel / delta1 per model, summed from the cells."""
        h = gt.shape[0]
        rows = np.minimum(np.arange(h)*3//h, 2)[:, None].repeat(gt.shape[1], 1)
        g = gt.astype(np.float64)
        n_frame = int(valid.sum())
        covered = 0
        frame = {name: dict(abs_rel=0., a1=0.) for name in aligned}
        self.frames += 1
        for i in range(len(ROWS)):
            for j, (lo, hi) in enumerate(BANDS):
                m = valid & (rows == i) & (g > lo) & (g <= hi)
                if not m.any():
                    continue
                self.n[i, j] += int(m.sum())
                covered += int(m.sum())
                for name, d in aligned.items():
                    self.sq[name][i, j] += float(((d[m]-g[m])**2).sum())
                    self.log[name][i, j] += float((np.log(d[m])-np.log(g[m])).sum())
                    rel = float((np.abs(g[m]-d[m])/g[m]).sum())/n_frame
                    hit = float((np.maximum(g[m]/d[m], d[m]/g[m]) < 1.25).sum())/n_frame
                    self.rel[name][i, j] += rel
                    self.hit[name][i, j] += hit
                    frame[name]['abs_rel'] += rel
                    frame[name]['a1'] += hit
        if covered != n_frame:
            raise ValueError(f'cells cover {covered} of {n_frame} valid pixels')
        return frame

    def table(self):
        diff = self.sq['synth']-self.sq['newcrf']
        out = []
        for i, r in enumerate(ROWS):
            for j, (lo, hi) in enumerate(BANDS):
                n = self.n[i, j]
                out.append(dict(rows=r, band=f'{lo}-{hi}m', point_share=float(n/self.n.sum()),
                                diff_share=float(diff[i, j]/diff.sum()),
                                rmse_synth=float(np.sqrt(self.sq['synth'][i, j]/n)) if n else None,
                                rmse_newcrf=float(np.sqrt(self.sq['newcrf'][i, j]/n)) if n else None,
                                log_bias_synth=float(self.log['synth'][i, j]/n) if n else None,
                                log_bias_newcrf=float(self.log['newcrf'][i, j]/n) if n else None))
        rel_gap = (self.rel['synth']-self.rel['newcrf'])/self.frames
        a1_gap = (self.hit['newcrf']-self.hit['synth'])/self.frames  # positive = synth worse, like AbsRel
        for k, r in enumerate(out):
            i, j = divmod(k, len(BANDS))
            r.update(absrel_gap=float(rel_gap[i, j]), a1_gap=float(a1_gap[i, j]),
                     absrel_gap_share=float(rel_gap[i, j]/rel_gap.sum()) if rel_gap.sum() else None)
        bands = {f'{lo}-{hi}m': dict(point_share=float(self.n[:, j].sum()/self.n.sum()),
                                     absrel_gap=float(rel_gap[:, j].sum()), a1_gap=float(a1_gap[:, j].sum()),
                                     absrel_gap_share=float(rel_gap[:, j].sum()/rel_gap.sum()) if rel_gap.sum() else None,
                                     a1_gap_share=float(a1_gap[:, j].sum()/a1_gap.sum()) if a1_gap.sum() else None)
                 for j, (lo, hi) in enumerate(BANDS)}
        if rel_gap.sum() > 0:
            reading = band_reading({k: v['absrel_gap_share'] for k, v in bands.items()})
        else:
            reading = dict(groups=None, reading='no_absrel_gap')
        return dict(points=int(self.n.sum()), frames=self.frames,
                    pooled_rmse={k: float(np.sqrt(v.sum()/self.n.sum())) for k, v in self.sq.items()},
                    frame_mean={m: dict(abs_rel=float(self.rel[m].sum()/self.frames), a1=float(self.hit[m].sum()/self.frames))
                                for m in MODELS},
                    bands=bands, absrel_reading=reading, cells=out)


def run(a):
    import torch
    from tools import diagnose_far_gap_curve as curve
    from tools import eval_iris_decoder_only as prior
    from tools.da2_caption_data import checked_manifest
    from tools.train_route_suite import RouteModel

    if not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA')
    c = prior.parent.parent_inputs(a.runs)['contract']
    samples, info = checked_manifest(Path(c['val']['path']), a.data_root, 'val', 14249)
    if info != c['val'] or not info['sha256'].startswith(breakdown.VAL_FP):
        raise ValueError('validation manifest differs from the frozen recipe')
    before = json.loads((a.previous/'summary.json').read_text())
    raw_dir = Path(before['paths']['newcrf_cache'])
    stored = breakdown.read(a.runs/'eval'/f'{breakdown.TAGS["synth"]}_val_{breakdown.VAL_FP}_stride1_correct_s16000'/
                            'eval_eval_affine_invariant_log_space_per_sample.csv')
    stored_crf = breakdown.read(a.newcrf, 'ssi_log_')
    by_seq = {}
    for s in samples:
        by_seq.setdefault(s.sample_id.rsplit('_', 1)[0], []).append(s)
    chosen = []
    for seq in sorted(by_seq):
        frames = sorted(by_seq[seq], key=lambda s: int(s.sample_id.rsplit('_', 1)[1]))
        chosen.extend(frames[::STRIDE])
    recipe = SimpleNamespace(**c['recipe'])
    model = RouteModel(recipe, torch.device('cuda'), torch.float16)
    path = a.runs/'iris_ms2'/breakdown.TAGS['synth']/'converted'/f'step{STEP}_weights.pt'
    payload = torch.load(path, map_location='cpu', weights_only=False)
    curve.check_payload(payload, STEP)
    model.load_state_dicts(payload['state_dicts'], str(path))
    model.unet.requires_grad_(False)
    model.set_train(False)
    del payload
    cells, worst = {}, dict(synth=0., newcrf=0.)
    worst_stored = {m: dict(abs_rel=0., a1=0.) for m in MODELS}
    worst_exact = 0.
    for k, (s, (ident, raw, gt)) in enumerate(zip(chosen, prior.pilot.iris.predict_samples(model, chosen, 'frame', recipe), strict=True), 1):
        if ident != s.sample_id:
            raise ValueError('Prediction order differs')
        crf = resize_dense_prediction(collapse_channels(np.load(raw_dir/f'{ident}.npy', allow_pickle=False)), gt.shape).astype(np.float32)
        crf_log = np.log(np.maximum(crf, 1e-6)).astype(np.float32)
        official = {}
        for name, x, ref in (('synth', raw, stored), ('newcrf', crf_log, stored_crf)):
            official[name] = evaluate_sample(x, gt, align='ssi_log', min_depth=.001, max_depth=80.)
            worst[name] = max(worst[name], abs(float(official[name]['rmse'])-ref[ident]['rmse']))
            for key in ('abs_rel', 'a1'):
                worst_stored[name][key] = max(worst_stored[name][key], abs(float(official[name][key])-ref[ident][key]))
        valid = official_valid_mask(gt, .001, 80.)
        aligned = {name: np.clip(aligned_depth(x, gt, valid, 'ssi_log'), .001, 80.) for name, x in (('synth', raw), ('newcrf', crf_log))}
        frame = cells.setdefault(ident.rsplit('_', 1)[0], Cells()).add(gt, valid, aligned)
        cells.setdefault('all', Cells()).add(gt, valid, aligned)
        for name in MODELS:
            for key in ('abs_rel', 'a1'):
                worst_exact = max(worst_exact, abs(frame[name][key]-float(official[name][key])))
        if k % 200 == 0:
            print(f'[frames] {k}/{len(chosen)}', flush=True)
    control = dict(max_abs_rmse_diff=worst, tolerance_m=dict(synth=SYNTH_TOL_M, newcrf=NEWCRF_TOL_M),
                   max_abs_stored_diff=worst_stored, stored_tolerance=STORED_TOL,
                   max_cells_vs_evaluator=worst_exact, exact_tolerance=EXACT_TOL,
                   passed=(worst['synth'] <= SYNTH_TOL_M and worst['newcrf'] <= NEWCRF_TOL_M and worst_exact <= EXACT_TOL
                           and all(worst_stored[m][k] <= STORED_TOL[m][k] for m in MODELS for k in ('abs_rel', 'a1'))))
    print('[control]', json.dumps(control), flush=True)
    if not control['passed']:
        raise SystemExit('per-frame RMSE does not reproduce the stored validation results')
    result = dict(control=control, frames=len(chosen), stride=STRIDE, weights=str(path), weights_sha256=sha256_file(path),
                  sequences={seq: cl.table() for seq, cl in sorted(cells.items())}, script_sha256=sha256_file(__file__))
    for seq, t in result['sequences'].items():
        fm = t['frame_mean']
        print(f'[absrel-split] {seq} frames {t["frames"]}  AbsRel synth {fm["synth"]["abs_rel"]:.5f} newcrf {fm["newcrf"]["abs_rel"]:.5f}  '
              f'd1 synth {fm["synth"]["a1"]:.5f} newcrf {fm["newcrf"]["a1"]:.5f}  reading {t["absrel_reading"]["reading"]}', flush=True)
        for band, b in t['bands'].items():
            share = 'n/a' if b['absrel_gap_share'] is None else f'{100*b["absrel_gap_share"]:6.1f}%'
            d1share = 'n/a' if b['a1_gap_share'] is None else f'{100*b["a1_gap_share"]:6.1f}%'
            print(f'    {band:7s} pts {100*b["point_share"]:5.1f}%  AbsRel gap {b["absrel_gap"]:+.5f} ({share})  '
                  f'd1 gap {b["a1_gap"]:+.5f} ({d1share})', flush=True)
        print(f'[{seq}] points {t["points"]}  pooled RMSE synth {t["pooled_rmse"]["synth"]:.3f}  newcrf {t["pooled_rmse"]["newcrf"]:.3f}')
        print('    rows    band     pts%  diff%  rmse_s  rmse_n  logbias_s  logbias_n')
        for r in t['cells']:
            if r['rmse_synth'] is None:
                continue
            print(f'    {r["rows"]:7s} {r["band"]:7s} {100*r["point_share"]:5.1f} {100*r["diff_share"]:6.1f} '
                  f'{r["rmse_synth"]:7.2f} {r["rmse_newcrf"]:7.2f} {r["log_bias_synth"]:+9.3f} {r["log_bias_newcrf"]:+9.3f}')
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('[completed]', a.out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('runs', 'newcrf', 'previous', 'data-root', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
    run(p.parse_args())
