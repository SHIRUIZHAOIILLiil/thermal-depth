"""Which dense source keeps far depth through the frozen VAE? No training, no test frames.

Each source is calibrated on 80% of a frame's LiDAR, the same 80% is written back, and the
target goes through the frozen depth VAE. The other 20%, held out by 8x8 latent cell, scores it.
Final SSI-log alignment also fits ONLY on the 80%; legacy reproduction is a separate branch.
Training frames decide because that is where a target is used; validation frames are a check.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F

from lotus.utils.ms2_thermal_dataset import MS2ThermalDataset
from ms2_eval.io import load_ms2_gt, sha256_file
from ms2_eval.official_protocol import official_valid_mask
from tools import diagnose_depth_chain as chain
from tools import diagnose_iris_val_chain as val_chain
from tools.build_anythermal_pseudo_gt import calibrate_pseudo_depth

SOURCES = ('anythermal', 'bts', 'adabins', 'newcrf')
CANDIDATES = ('bts', 'adabins')  # NeWCRF is the comparison target, so it is a last resort.
CHECKPOINT = dict(bts='BTS', adabins='AdaBins', newcrf='NeWCRF')
SHA256 = dict(bts='5dccf48addeb454f069a566854fff70c85bfe1fd9f628a61563ad679fcd926dc',
              adabins='8260af411c379a5e01ed1868724ef54fad2b4768449eddb7e2d89f4309ae977e')
BANDS = (('0-40', 0., 40.), ('40-60', 40., 60.), ('60-80', 60., 80.))
QUANTILES = (.98, 1.)
SEED, FRAMES, HOLDOUT, CELL = 20261017, 96, .2, 8
BIAS_RATIO, NEAR_RMSE_M = .7, .02
# Prior chain (job 9174706), 60-80 m under ssi_log: calibrated pseudo and q98 target after VAE.
CONTROL = dict(source=-8.10, vae=-12.86)
CONTROL_TOLERANCE_M = 1.
HELDOUT_ALIGN = 'ssi_log_fit_only'


def holdout_mask(valid, sample_id, seed=SEED, fraction=HOLDOUT, cell=CELL):
    """Exclude direct LiDAR writes in selected cells; not a full VAE receptive-field holdout."""
    h, w = valid.shape
    rows, cols = np.indices((h, w))
    index = rows//cell*((w+cell-1)//cell) + cols//cell
    occupied = np.unique(index[valid])
    if not len(occupied):
        raise ValueError(f'{sample_id}: no valid LiDAR')
    rng = np.random.default_rng(chain.frame_seed(sample_id, seed))
    count = max(1, round(fraction*len(occupied)))
    return valid & np.isin(index, rng.choice(occupied, count, replace=False))


def score(pred, gt, mask):
    rows = {}
    for band, lo, hi in BANDS:
        selected = mask & (gt >= lo) & (gt < hi)
        if selected.any():
            residual = np.clip(pred[selected].astype(np.float64), .001, 80.) - gt[selected]
            rows[band] = dict(points=int(selected.sum()), bias_m=float(residual.mean()),
                              rmse_m=float(np.sqrt(np.mean(residual**2))))
    return rows


def frame_records(gt, valid, held, value, log_value, bounds):
    """Fit the affine log transform on fit points; apply it unchanged to held-out points."""
    records = []
    fit = valid & ~held
    if (held & ~valid).any() or not held.any() or fit.sum() < 2:
        raise ValueError('Invalid fit/held-out partition')
    aligned = chain.aligned_depth(log_value, gt, fit, 'ssi_log')
    target_scale = value if bounds is None else chain.inverse_log_unit(value, bounds)
    for mode, pred in ((HELDOUT_ALIGN, aligned), ('target_scale', target_scale)):
        for mask, chosen in (('heldout', held), ('fit', fit)):
            records.extend(dict(mode=mode, mask=mask, band=band, **row)
                           for band, row in score(pred, gt, chosen).items())
    return records


def aggregate(records):
    groups = {}
    for r in records:
        key = tuple(r[k] for k in ('split','source','q','stage','mode','mask','band'))
        groups.setdefault(key, []).append(r)
    return [dict(zip(('split','source','q','stage','mode','mask','band'), key), frames=len(rows),
                 points=sum(r['points'] for r in rows),
                 bias_m=float(np.mean([r['bias_m'] for r in rows])),
                 rmse_m=float(np.mean([r['rmse_m'] for r in rows])))
            for key, rows in sorted(groups.items(), key=lambda x: str(x[0]))]


def lookup(table, **want):
    found = [r for r in table if all(r[k] == v for k, v in want.items())]
    if len(found) != 1:
        raise ValueError(f'Expected one row for {want}, found {len(found)}')
    return found[0]


def control(table):
    common = dict(split='val', source='anythermal', mode='ssi_log', mask='control_all', band='60-80')
    measured = dict(source=lookup(table, q=None, stage='source', **common)['bias_m'],
                    vae=lookup(table, q=.98, stage='vae', **common)['bias_m'])
    passed = all(abs(measured[k]-CONTROL[k]) <= CONTROL_TOLERANCE_M+1e-12 for k in CONTROL)
    return dict(expected=CONTROL, measured=measured, tolerance_m=CONTROL_TOLERANCE_M, passed=passed)


def decide(table, control_passed):
    """Frozen before any result: q100 target after VAE, held-out LiDAR, training frames."""
    row = lambda source, band: lookup(table, split='train', source=source, q=1., stage='vae',
                                      mode=HELDOUT_ALIGN, mask='heldout', band=band)
    far, near = abs(row('anythermal', '60-80')['bias_m']), row('anythermal', '0-40')['rmse_m']
    if far <= 0:
        raise ValueError('Control far bias must be nonzero')
    sources = {}
    for source in ('bts', 'adabins', 'newcrf'):
        ratio = abs(row(source, '60-80')['bias_m'])/far
        near_change = row(source, '0-40')['rmse_m']-near
        sources[source] = dict(bias_ratio=ratio, near_rmse_change_m=near_change,
                               passed=ratio <= BIAS_RATIO+1e-12 and near_change <= NEAR_RMSE_M+1e-12)
    passed = [s for s in CANDIDATES if sources[s]['passed']]
    if not control_passed:
        status, chosen = 'control_failed', None
    elif passed:
        status, chosen = 'clean_source_passed', min(passed, key=lambda s: sources[s]['bias_ratio'])
    elif sources['newcrf']['passed']:
        status, chosen = 'only_newcrf_passed_needs_user_decision', None
    else:
        status, chosen = 'no_source_passed', None
    return dict(status=status, chosen=chosen, sources=sources, bias_ratio_limit=BIAS_RATIO,
                near_rmse_limit_m=NEAR_RMSE_M, split='train', quantile=1., mask='heldout', stage='vae',
                alignment=HELDOUT_ALIGN)


class TargetDataset(MS2ThermalDataset):
    """Reuse the training normalisation on a map that omits the held-out LiDAR."""
    dense = None

    def _completed_depth(self, row):
        if self.dense is None:
            raise ValueError('No dense target set')
        depth = torch.from_numpy(self.dense)[None]
        return depth, torch.zeros_like(depth), torch.zeros_like(depth)


def select_frames(contract, split, count, root, allowed, out):
    from tools.da2_caption_data import checked_manifest
    path = Path(contract[split]['path'])
    samples, info = checked_manifest(path, root, split, dict(train=75688, val=14249)[split])
    if info != contract[split]:
        raise ValueError(f'{split} manifest differs from the frozen recipe')
    rows = chain.select_rows(chain.read_jsonl(path), count, SEED, allowed)
    chosen = [samples[r['diagnostic_source_index']] for r in rows]
    if any(s.sample_id != r['id'] for s, r in zip(chosen, rows, strict=True)):
        raise ValueError('Subset order differs from the manifest')
    for r in rows:
        r['thermal_depth_path'] = r.get('thermal_depth_path') or r['depth_path']
    manifest = out/f'{split}_subset.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    return chosen, rows, manifest, info


def run_anythermal(rows, manifest, root, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(ROOT/'tools/run_ms2_anythermal_midas.py'), '--manifest', str(manifest),
                    '--ms2-root', str(root), '--output-dir', str(out), '--max-samples', '0'], check=True)
    if not val_chain.compatible_cache(out, rows):
        raise ValueError('AnyThermal predictions differ from the required inputs/recipe')
    return out/'raw_predictions'


@torch.no_grad()
def run_baseline(name, frames, a, out):
    from tools.run_ms2_supdepth_baselines import build_network, forward_depth, load_weights, preprocess_thermal
    path = a.ckpt_dir/f'MS2_MD_{CHECKPOINT[name]}_THR_ckpt.ckpt'
    digest = sha256_file(path)
    if name in SHA256 and digest != SHA256[name]:
        raise ValueError(f'{name} checkpoint differs from the downloaded file')
    network = build_network(name, a.bmsd_root)
    provenance = load_weights(network, path, strict=True)
    network = network.cuda().eval()
    out.mkdir(parents=True, exist_ok=True)
    for s in frames:
        depth = forward_depth(name, network, preprocess_thermal(Path(s.thermal_path)).cuda())
        gt = load_ms2_gt(s.thermal_gt_path, 256.)[1]
        prediction = depth[None] if depth.dim() == 3 else depth
        if prediction.shape[-2:] != gt.shape:
            prediction = F.interpolate(prediction, gt.shape, mode='bilinear', align_corners=False)
        np.save(out/f'{s.sample_id}.npy', prediction[0,0].float().cpu().numpy().astype(np.float32))
    del network
    torch.cuda.empty_cache()
    return dict(checkpoint=str(path), sha256=digest, tensors=provenance['tensors_loaded'])


def make_target(raw, gt, fit, config):
    """Only fit labels may affect source calibration, completion or subsequent q bounds."""
    pseudo = calibrate_pseudo_depth(raw, gt, fit, min_fit_pixels=config['min_fit_pixels'])[0].astype(np.float32)
    if pseudo.shape != gt.shape or not np.isfinite(pseudo).all():
        raise ValueError('Invalid calibrated source')
    return pseudo, np.clip(np.where(fit, gt, pseudo), .001, 80.).astype(np.float32)


def legacy_control_records(frames, manifest, predictions, config, a, vae):
    """Reproduce job 9174706 independently: ALL LiDAR calibrates, writes and scores.

    Same persisted pseudo dtype, real dataset, q98, two-map VAE batch (completed,
    pure pseudo), posterior mode and scale arithmetic as diagnose_iris_val_chain.
    These rows must never be used for the held-out source selection.
    """
    pseudo_dir = a.out/'legacy_control_pseudo'
    pseudo_dir.mkdir(exist_ok=False)
    for s in frames:
        gt = load_ms2_gt(s.thermal_gt_path,256.)[1]
        raw = np.load(predictions/f'{s.sample_id}.npy',allow_pickle=False)
        pseudo, _ = calibrate_pseudo_depth(raw,gt,official_valid_mask(gt,.001,80.),
                                           min_fit_pixels=config['min_fit_pixels'])
        np.save(pseudo_dir/f'{s.sample_id}.npy',pseudo.astype(config['dtype']))
    args = dict(norm_type='log_truncnorm',truncnorm_min=.02,truncnorm_max=.98)
    dataset = MS2ThermalDataset(manifest,a.data_root,pseudo_dir,**args)
    pure = MS2ThermalDataset(manifest,a.data_root,pseudo_dir,pure_pseudo_target=True,**args)
    device = next(vae.parameters()).device
    records = []
    for i,s in enumerate(frames):
        if dataset.rows[i]['id']!=s.sample_id:
            raise ValueError('Legacy control order differs')
        gt = load_ms2_gt(s.thermal_gt_path,256.)[1]
        valid = official_valid_mask(gt,.001,80.)
        target = torch.stack([dataset[i]['depth_values'],pure[i]['depth_values']]).to(device)
        scale = vae.config.scaling_factor
        decoded = vae.decode(vae.encode(target).latent_dist.mode()*scale/scale).sample
        unit = (decoded.mean(1)[0]/2+.5).float().cpu().numpy()
        pseudo = np.load(pseudo_dir/f'{s.sample_id}.npy',allow_pickle=False).astype(np.float32)
        for stage,q,log_value in (('source',None,np.log(np.maximum(pseudo,1e-6))),('vae',.98,unit)):
            pred = chain.aligned_depth(log_value,gt,valid,'ssi_log')
            records.extend(dict(split='val',id=s.sample_id,source='anythermal',q=q,stage=stage,
                mode='ssi_log',mask='control_all',band=band,**row)
                for band,row in score(pred,gt,valid).items())
    return records


def score_split(split, frames, manifest, predictions, config, a, vae):
    datasets = {q: TargetDataset(manifest, a.data_root, a.out, norm_type='log_truncnorm',
                                 truncnorm_min=.02, truncnorm_max=q) for q in QUANTILES}
    records = []
    for i, s in enumerate(frames):
        gt = load_ms2_gt(s.thermal_gt_path, 256.)[1]
        valid = official_valid_mask(gt, .001, 80.)
        held = holdout_mask(valid, s.sample_id)
        fit = valid & ~held
        for source in SOURCES:
            raw = np.load(predictions[source]/f'{s.sample_id}.npy', allow_pickle=False)
            pseudo, dense = make_target(raw,gt,fit,config)
            base = dict(split=split, id=s.sample_id, source=source)
            records.extend(dict(base, q=None, stage='source', **r) for r in
                           frame_records(gt, valid, held, pseudo, np.log(np.maximum(pseudo, 1e-6)), None))
            for q, dataset in datasets.items():
                dataset.dense = dense
                example = dataset[i]
                if dataset.rows[i]['id'] != s.sample_id:
                    raise ValueError('Dataset order differs')
                target = example['depth_values'][None].cuda()
                decoded = vae.decode(vae.encode(target).latent_dist.mode()).sample
                unit = (decoded.mean(1)[0]/2+.5).float().cpu().numpy()
                records.extend(dict(base, q=q, stage='vae', **r) for r in
                               frame_records(gt, valid, held, unit, unit, example['norm_bounds'].numpy()))
        if (i+1)%20 == 0:
            print(f'[scored] {split} {i+1}/{len(frames)}', flush=True)
    return records


def run(a):
    from diffusers import AutoencoderKL
    from tools.da2_caption_data import TRAIN, VAL
    from tools.train_iris_latent_release import parent_inputs
    if not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA')
    if a.frames != FRAMES:
        raise ValueError('Frozen decision/control require the original 96 frames per split')
    a.out.mkdir(parents=True, exist_ok=False)
    contract = parent_inputs(a.runs)['contract']
    config_path = Path(contract['pseudo']).parent/'build_config.json'
    if sha256_file(config_path) != contract['pseudo_config']:
        raise ValueError('Pseudo calibration recipe changed')
    config = json.loads(config_path.read_text())
    if (config['depth_scale'], config['gt_min_depth'], config['gt_max_depth']) != (256., .001, 80.):
        raise ValueError('Unexpected pseudo calibration depth range')
    splits = {split: select_frames(contract, split, a.frames, a.data_root, allowed, a.out)
              for split, allowed in (('train', TRAIN), ('val', VAL))}
    provenance = {}
    predictions = {}
    for split, (frames, rows, manifest, _) in splits.items():
        predictions[split] = dict(anythermal=run_anythermal(rows, manifest, a.data_root, a.out/'raw'/split/'anythermal'))
    for name in CHECKPOINT:
        for split, (frames, *_) in splits.items():
            folder = a.out/'raw'/split/name
            provenance[name] = run_baseline(name, frames, a, folder)
            predictions[split][name] = folder
    vae = AutoencoderKL.from_pretrained(contract['recipe']['lotus_model_path'], subfolder='vae',
        local_files_only=True, torch_dtype=torch.float32).to('cuda').eval().requires_grad_(False)
    records = []
    with torch.inference_mode():
        frames, _, manifest, _ = splits['val']
        records.extend(legacy_control_records(frames,manifest,predictions['val']['anythermal'],config,a,vae))
        for split, (frames, _, manifest, _) in splits.items():
            records.extend(score_split(split, frames, manifest, predictions[split], config, a, vae))
    with (a.out/'per_frame.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    table = aggregate(records)
    checked = control(table)
    decision = decide(table, checked['passed'])
    (a.out/'summary.json').write_text(json.dumps(dict(control=checked, decision=decision, table=table,
        checkpoints=provenance, seed=SEED, frames=a.frames, holdout=HOLDOUT, cell=CELL,
        protocol='fit_only_alignment_v2', pseudo_config_sha256=sha256_file(config_path),
        subsets={k:sha256_file(v[2]) for k,v in splits.items()},
        anythermal={k:sha256_file(predictions[k]['anythermal'].parent/'run_metadata.json') for k in splits},
        manifests={k:v[3] for k, v in splits.items()}, script_sha256=sha256_file(__file__)), indent=2, allow_nan=False))
    for r in table:
        if r['mode'] == HELDOUT_ALIGN and r['mask'] == 'heldout' and r['stage'] == 'vae' and r['q'] == 1.:
            print('[target]', json.dumps(r), flush=True)
    print('[control]', json.dumps(checked), flush=True)
    print('[decision]', json.dumps(decision), flush=True)
    print('[completed]', a.out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runs', 'out', 'data-root', 'bmsd-root', 'ckpt-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--frames', type=int, default=FRAMES)
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
    run(p.parse_args())
