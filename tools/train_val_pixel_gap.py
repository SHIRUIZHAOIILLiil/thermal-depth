"""Is the synth arm's near-range shortfall against NeWCRF there on training frames too?

Same cell breakdown as val_pixel_gap.py (image third x GT depth band, both models `ssi_log`
aligned and clipped to 0.001-80 m), on the 96 train + 96 val frames of the 11-07 diagnostics,
whose NeWCRF maps job 9229162 cached for both splits. Headline per split: pooled RMSE of each
model over the 0-20 m points and their ratio. Reading rule: docs/IRIS_SYNTH_PRETRAIN_V1.md.

Controls (2 mm): synth per-split mean RMSE 1.48442 / 2.67386 (job 9488823); NeWCRF `ssi` val
mean RMSE 2.457972 (the 96-frame control every 11-07 diagnostic used).
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
from ms2_eval.official_protocol import evaluate_sample, official_valid_mask
from tools import val_gap_breakdown as breakdown
from tools import val_pixel_gap as px
from tools.build_anythermal_pseudo_gt import resize_to
from tools.diagnose_error_sources import aligned_depth

TOLERANCE_M = .002
SYNTH_CONTROL = dict(train=1.48442, val=2.67386)  # job 9488823, printed to 5 decimals
NEWCRF_SSI_VAL = 2.457972237888758
NEAR = ('0-10m', '10-20m')


def near_range(cells):
    """Pooled RMSE over every 0-20 m point, all image rows."""
    j = [k for k, (lo, hi) in enumerate(px.BANDS) if f'{lo}-{hi}m' in NEAR]
    n = cells.n[:, j].sum()
    rmse = {m: float(np.sqrt(cells.sq[m][:, j].sum()/n)) for m in cells.sq}
    return dict(points=int(n), rmse=rmse, ratio=rmse['synth']/rmse['newcrf'])


def run(a):
    import torch
    from tools import diagnose_far_gap_curve as curve
    from tools import diagnose_far_target_source as source_tool
    from tools import diagnose_model_far_gap as gap_tool
    from tools import eval_iris_decoder_only as prior
    from tools.da2_caption_data import TRAIN, VAL
    from tools.train_route_suite import RouteModel

    if not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA')
    job = a.runs/f'outputs/lotus_line_v2/iris_far_target_source_fitonly_v2/job_{gap_tool.DIAGNOSTIC_JOB}'
    reference = json.loads((job/'summary.json').read_text())
    c = prior.parent.parent_inputs(a.runs)['contract']
    a.out.mkdir(parents=True, exist_ok=False)
    splits = {}
    for split, allowed in (('train', TRAIN), ('val', VAL)):
        frames, _, manifest, _ = source_tool.select_frames(c, split, source_tool.FRAMES, a.data_root, allowed, a.out)
        if sha256_file(manifest) != reference['subsets'][split]:
            raise ValueError(f'{split} subset differs from the diagnostic job')
        splits[split] = frames
    recipe = SimpleNamespace(**c['recipe'])
    model = RouteModel(recipe, torch.device('cuda'), torch.float16)
    path = a.runs/'iris_ms2'/breakdown.TAGS['synth']/'converted'/f'step{px.STEP}_weights.pt'
    payload = torch.load(path, map_location='cpu', weights_only=False)
    curve.check_payload(payload, px.STEP)
    model.load_state_dicts(payload['state_dicts'], str(path))
    model.unet.requires_grad_(False)
    model.set_train(False)
    del payload
    cells, synth_rmse, crf_ssi = {}, {}, []
    for split, frames in splits.items():
        cells[split], synth_rmse[split] = px.Cells(), []
        for s, (ident, raw, gt) in zip(frames, prior.pilot.iris.predict_samples(model, frames, 'frame', recipe), strict=True):
            if ident != s.sample_id:
                raise ValueError('Prediction order differs')
            crf = resize_to(np.load(job/'raw'/split/'newcrf'/f'{ident}.npy', allow_pickle=False), gt.shape).astype(np.float32)
            crf_log = np.log(np.maximum(crf, 1e-6)).astype(np.float32)
            synth_rmse[split].append(float(evaluate_sample(raw, gt, align='ssi_log', min_depth=.001, max_depth=80.)['rmse']))
            if split == 'val':
                crf_ssi.append(float(evaluate_sample(crf, gt, align='ssi', min_depth=.001, max_depth=80.)['rmse']))
            valid = official_valid_mask(gt, .001, 80.)
            aligned = {name: np.clip(aligned_depth(x, gt, valid, 'ssi_log'), .001, 80.) for name, x in (('synth', raw), ('newcrf', crf_log))}
            cells[split].add(gt, valid, aligned)
        print(f'[scored] {split} {len(frames)} frames', flush=True)
    measured = dict(synth={k: float(np.mean(v)) for k, v in synth_rmse.items()}, newcrf_ssi_val=float(np.mean(crf_ssi)))
    control = dict(measured=measured, expected=dict(synth=SYNTH_CONTROL, newcrf_ssi_val=NEWCRF_SSI_VAL), tolerance_m=TOLERANCE_M,
                   passed=all(abs(measured['synth'][k]-SYNTH_CONTROL[k]) <= TOLERANCE_M for k in SYNTH_CONTROL)
                   and abs(measured['newcrf_ssi_val']-NEWCRF_SSI_VAL) <= TOLERANCE_M)
    print('[control]', json.dumps(control), flush=True)
    if not control['passed']:
        raise SystemExit('controls do not reproduce')
    result = dict(control=control, weights=str(path), weights_sha256=sha256_file(path), script_sha256=sha256_file(__file__),
                  splits={k: dict(near_0_20m=near_range(v), **v.table()) for k, v in cells.items()})
    for split, t in result['splits'].items():
        nr = t['near_0_20m']
        print(f'[{split}] near 0-20 m: synth {nr["rmse"]["synth"]:.3f}  newcrf {nr["rmse"]["newcrf"]:.3f}  ratio {nr["ratio"]:.3f}'
              f'  ({nr["points"]} pts)   pooled all: synth {t["pooled_rmse"]["synth"]:.3f}  newcrf {t["pooled_rmse"]["newcrf"]:.3f}')
        print('    rows    band     pts%  diff%  rmse_s  rmse_n  ratio  logbias_s  logbias_n')
        for r in t['cells']:
            if r['rmse_synth'] is None:
                continue
            print(f'    {r["rows"]:7s} {r["band"]:7s} {100*r["point_share"]:5.1f} {100*r["diff_share"]:6.1f} '
                  f'{r["rmse_synth"]:7.2f} {r["rmse_newcrf"]:7.2f} {r["rmse_synth"]/r["rmse_newcrf"]:6.2f} '
                  f'{r["log_bias_synth"]:+9.3f} {r["log_bias_newcrf"]:+9.3f}')
    (a.out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('[completed]', a.out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('runs', 'data-root', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
    run(p.parse_args())
