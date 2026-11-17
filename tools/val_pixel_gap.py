"""Inside the frames: where does the synth arm lose to NeWCRF? Every 10th frame of each val sequence.

Diagnostic, val only, no selection. Per frame both models are aligned to GT with the evaluator's
`ssi_log` fit (synth: raw network output; NeWCRF: log of the cached depth of job 9167854), clipped
to the official 0.001-80 m, and every valid LiDAR pixel is put into a cell:
    image band  top / middle / bottom third of the rows
    GT depth    0-10 / 10-20 / 20-40 / 40-80 m
Per sequence and cell: point share, pooled RMSE of each model, share of the pooled squared-error
difference (synth - NeWCRF) and the mean signed log error of each model (positive = too far).
Pooled pixel errors weight frames by their point count, unlike the per-frame mean RMSE.

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


class Cells:
    def __init__(self):
        shape = (len(ROWS), len(BANDS))
        self.n = np.zeros(shape, np.int64)
        self.sq = {m: np.zeros(shape) for m in ('synth', 'newcrf')}
        self.log = {m: np.zeros(shape) for m in ('synth', 'newcrf')}

    def add(self, gt, valid, aligned):
        h = gt.shape[0]
        rows = np.minimum(np.arange(h)*3//h, 2)[:, None].repeat(gt.shape[1], 1)
        g = gt.astype(np.float64)
        for i in range(len(ROWS)):
            for j, (lo, hi) in enumerate(BANDS):
                m = valid & (rows == i) & (g > lo) & (g <= hi)
                if not m.any():
                    continue
                self.n[i, j] += int(m.sum())
                for name, d in aligned.items():
                    self.sq[name][i, j] += float(((d[m]-g[m])**2).sum())
                    self.log[name][i, j] += float((np.log(d[m])-np.log(g[m])).sum())

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
        return dict(points=int(self.n.sum()),
                    pooled_rmse={k: float(np.sqrt(v.sum()/self.n.sum())) for k, v in self.sq.items()}, cells=out)


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
    for k, (s, (ident, raw, gt)) in enumerate(zip(chosen, prior.pilot.iris.predict_samples(model, chosen, 'frame', recipe), strict=True), 1):
        if ident != s.sample_id:
            raise ValueError('Prediction order differs')
        crf = resize_dense_prediction(collapse_channels(np.load(raw_dir/f'{ident}.npy', allow_pickle=False)), gt.shape).astype(np.float32)
        crf_log = np.log(np.maximum(crf, 1e-6)).astype(np.float32)
        for name, x, ref in (('synth', raw, stored), ('newcrf', crf_log, stored_crf)):
            got = evaluate_sample(x, gt, align='ssi_log', min_depth=.001, max_depth=80.)['rmse']
            worst[name] = max(worst[name], abs(float(got)-ref[ident]['rmse']))
        valid = official_valid_mask(gt, .001, 80.)
        aligned = {name: np.clip(aligned_depth(x, gt, valid, 'ssi_log'), .001, 80.) for name, x in (('synth', raw), ('newcrf', crf_log))}
        cells.setdefault(ident.rsplit('_', 1)[0], Cells()).add(gt, valid, aligned)
        if k % 200 == 0:
            print(f'[frames] {k}/{len(chosen)}', flush=True)
    control = dict(max_abs_rmse_diff=worst, tolerance_m=dict(synth=SYNTH_TOL_M, newcrf=NEWCRF_TOL_M),
                   passed=worst['synth'] <= SYNTH_TOL_M and worst['newcrf'] <= NEWCRF_TOL_M)
    print('[control]', json.dumps(control), flush=True)
    if not control['passed']:
        raise SystemExit('per-frame RMSE does not reproduce the stored validation results')
    result = dict(control=control, frames=len(chosen), stride=STRIDE, weights=str(path), weights_sha256=sha256_file(path),
                  sequences={seq: cl.table() for seq, cl in sorted(cells.items())}, script_sha256=sha256_file(__file__))
    for seq, t in result['sequences'].items():
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
