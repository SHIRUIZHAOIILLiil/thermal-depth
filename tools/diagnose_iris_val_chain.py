"""Paired 60–80 m validation diagnostic on the original depth-chain subset."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ms2_eval.io import load_ms2_gt, sha256_file
from ms2_eval.official_protocol import official_valid_mask
from tools import diagnose_depth_chain as chain
from tools.build_anythermal_pseudo_gt import calibrate_pseudo_depth

PREVIOUS = {'pseudo':None, 'pseudo_only_unit':'pseudo', 'pseudo_only_vae':'pseudo_only_unit',
            'lidar_written':'pseudo', 'target_unit':'lidar_written', 'vae_mode':'target_unit', 'model4000':'vae_mode'}
STAGES = tuple(PREVIOUS)


def score_frame(gt, maps, bounds):
    """One mask for every stage. Fit SSI-log on the whole frame before binning."""
    valid = official_valid_mask(gt, .001, 80.)
    far = valid & (gt >= 60.)
    records = []
    for stage in STAGES:
        value = np.asarray(maps[stage], dtype=np.float32)
        if value.shape != gt.shape or not np.isfinite(value).all():
            raise ValueError(f'{stage}: invalid native-resolution map')
        metric = stage in ('pseudo', 'lidar_written')
        log_input = np.log(np.maximum(value, 1e-6)) if metric else value
        aligned = chain.aligned_depth(log_input, gt, valid, 'ssi_log')
        variants = [('ssi_log', aligned)]
        if stage != 'model4000':
            variants.append(('target_scale', value if metric else chain.inverse_log_unit(value, bounds[stage])))
        for mode, pred in variants:
            residual = np.clip(pred[far].astype(np.float64), .001, 80.) - gt[far]
            if residual.size:
                records.append(dict(stage=stage, mode=mode, points=residual.size,
                    bias_m=float(residual.mean()), rmse_m=float(np.sqrt(np.mean(residual**2))),
                    too_near_pct=float(100*np.mean(residual < 0))))
    return records


def aggregate(records):
    table = []
    for sequence in ['ALL', *sorted({r['sequence'] for r in records})]:
        for mode in ('ssi_log', 'target_scale'):
            biases = {}
            for stage in STAGES:
                group = [r for r in records if r['mode'] == mode and r['stage'] == stage
                         and sequence in ('ALL', r['sequence'])]
                if not group:
                    continue
                avg = {k: float(np.mean([r[k] for r in group])) for k in ('bias_m','rmse_m','too_near_pct')}
                before = PREVIOUS[stage]
                delta = None if before is None else avg['bias_m'] - biases[before]
                biases[stage] = avg['bias_m']
                table.append(dict(sequence=sequence, mode=mode, stage=stage, frames=len(group),
                                  points=sum(r['points'] for r in group), predecessor=before, delta_bias_m=delta, **avg))
    return table


def compatible_cache(folder, rows):
    meta = folder/'run_metadata.json'
    index = folder/'selected_manifest.jsonl'
    if not meta.is_file() or not index.is_file():
        return False
    info = json.loads(meta.read_text())
    source = info.get('model', {})
    prep = info.get('preprocessing', {})
    if ('Midas_anythermal' not in Path(source.get('checkpoint', '').replace('\\','/')).parts
            or source.get('missing_keys') != [] or source.get('unexpected_keys') != []
            or prep.get('raw_divisor') != 16384 or prep.get('normalize') != {'mean':.45,'std':.225}):
        return False
    index_rows = chain.read_jsonl(index)
    cached = {r['id']:r for r in index_rows}
    if len(cached) != len(index_rows):
        return False
    return all(r['id'] in cached and all(cached[r['id']].get(k) == r[k] for k in
               ('thermal_path','thermal_depth_path')) and (folder/'raw_predictions'/f"{r['id']}.npy").is_file()
               for r in rows)


def raw_source(a, rows, manifest):
    for folder in a.cache:
        if compatible_cache(folder, rows):
            return folder
    folder = a.out/'anythermal_subset'
    subprocess.run([sys.executable, str(ROOT/'tools/run_ms2_anythermal_midas.py'),
        '--manifest', str(manifest), '--ms2-root', str(a.data_root), '--output-dir', str(folder),
        '--max-samples', '0'], check=True)
    if not compatible_cache(folder, rows):
        raise ValueError('New AnyThermal predictions differ from the required inputs/recipe')
    return folder


def run(a):
    import torch
    from diffusers import AutoencoderKL
    from tools import test_iris_cross_region_pairs as selected
    from tools.da2_caption_data import VAL, checked_manifest
    from tools.train_route_suite import RouteModel
    from utils.ms2_thermal_dataset import MS2ThermalDataset

    a.out.mkdir(parents=True, exist_ok=False)
    base = a.runs/'outputs/lotus_line_v2'
    candidates = [p.parent for p in (base/'iris_far_bias_square_l050_frame_v1').glob('run_*/jobs.txt')
                  if 'frame train+val -> 9171487 (afterok:9171486)' in p.read_text().splitlines()]
    if len(candidates) != 1:
        raise ValueError(f'Expected exactly one output for job 9171487, found {candidates}')
    run_dir, = candidates
    settings = selected.read(run_dir/'contract.json')
    parent = base/'iris_far_pairs_l050_frame4k_v1/job_9163317'
    _, contract, source, start_conv, start = selected.prior.selection(parent)
    if (Path(settings['parent']) != parent or settings['selection_sha256'] != sha256_file(parent/'summary.json')
            or settings['start_conv_sha256'] != sha256_file(start_conv) or Path(settings['source']) != source):
        raise ValueError('Selected run does not match its frame parent')
    added, step, conv = selected.checkpoint(run_dir, start, start_conv, selected.read(parent/'summary.json'))
    if (start, added, step) != (2000, 2000, 4000):
        raise ValueError('Expected the selected far-bias 4000 checkpoint')
    path = Path(contract['val']['path'])
    samples, info = checked_manifest(path, a.data_root, 'val', 14249)
    if info != settings['val'] or info != contract['val'] or not info['sha256'].startswith('c26a5859'):
        raise ValueError('Validation manifest differs from training validation')
    rows = chain.select_rows(chain.read_jsonl(path), 96, 20261017, VAL)
    chosen = [samples[r['diagnostic_source_index']] for r in rows]  # keep full manifest noise indices
    if any(s.sample_id != r['id'] for s,r in zip(chosen, rows, strict=True)):
        raise ValueError('Subset order differs from validation')
    for r in rows:
        r['thermal_depth_path'] = r.get('thermal_depth_path') or r['depth_path']
    manifest = a.out/'val96.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    raw = raw_source(a, rows, manifest)
    config_path = Path(contract['pseudo']).parent/'build_config.json'
    if sha256_file(config_path) != contract['pseudo_config']:
        raise ValueError('Parent pseudo calibration recipe changed')
    config = selected.read(config_path)
    if (config['depth_scale'], config['gt_min_depth'], config['gt_max_depth']) != (256.,.001,80.):
        raise ValueError('Unexpected pseudo calibration depth range')
    pseudo_dir = a.out/'diagnostic_pseudo'
    pseudo_dir.mkdir()
    for s in chosen:
        gt = load_ms2_gt(s.thermal_gt_path, 256.)[1]
        pred = np.load(raw/'raw_predictions'/f'{s.sample_id}.npy', allow_pickle=False)
        pseudo, _ = calibrate_pseudo_depth(pred, gt, official_valid_mask(gt,.001,80.),
                                           min_fit_pixels=config['min_fit_pixels'])
        np.save(pseudo_dir/f'{s.sample_id}.npy', pseudo.astype(config['dtype']))
    dataset_args = dict(norm_type='log_truncnorm', truncnorm_min=.02, truncnorm_max=.98)
    dataset = MS2ThermalDataset(manifest, a.data_root, pseudo_dir, **dataset_args)
    pure = MS2ThermalDataset(manifest, a.data_root, pseudo_dir, pure_pseudo_target=True, **dataset_args)
    recipe = SimpleNamespace(**contract['recipe'])
    model = RouteModel(recipe, torch.device('cuda'), torch.float16)
    selected.pilot.load_selected(model, 'frame', source, conv, contract)
    vae = AutoencoderKL.from_pretrained(recipe.lotus_model_path, subfolder='vae', local_files_only=True,
                                       torch_dtype=torch.float32).to('cuda').eval().requires_grad_(False)
    records = []
    coverage = []
    with torch.inference_mode():
        predictions = selected.pilot.iris.predict_samples(model, chosen, 'frame', recipe)
        for i, (s, (ident, prediction, gt)) in enumerate(zip(chosen, predictions, strict=True)):
            if ident != s.sample_id or ident != dataset.rows[i]['id']:
                raise ValueError('Unpaired frame')
            ex = dataset[i]
            valid = official_valid_mask(gt,.001,80.)
            np.testing.assert_array_equal(ex['gt_valid_values'][0].numpy().astype(bool), valid)
            np.testing.assert_array_equal(ex['gt_depth_values'][0].numpy()[valid], gt[valid])
            pure_ex = pure[i]
            target = torch.stack([ex['depth_values'], pure_ex['depth_values']]).cuda()
            posterior = vae.encode(target).latent_dist
            scale = vae.config.scaling_factor
            decoded = vae.decode(posterior.mode() * scale / scale).sample
            pseudo = np.load(pseudo_dir/f'{ident}.npy', allow_pickle=False).astype(np.float32)
            dense, _, _ = dataset._completed_depth(dataset.rows[i])
            units = (target[:,0]/2+.5).cpu().numpy()
            decoded_units = (decoded.mean(1)/2+.5).cpu().numpy()
            values = dict(pseudo=pseudo, pseudo_only_unit=units[1], pseudo_only_vae=decoded_units[1],
                          lidar_written=dense[0].numpy(), target_unit=units[0], vae_mode=decoded_units[0],
                          model4000=prediction)
            bounds = {k:(pure_ex if k.startswith('pseudo_only') else ex)['norm_bounds'].numpy() for k in STAGES}
            result = score_frame(gt, values, bounds)
            coverage.append({'id':ident,'points':int((valid & (gt>=60)).sum())})
            records.extend(dict(id=ident, sequence=ident.rsplit('_',1)[0], **r) for r in result)
    if not records:
        raise ValueError('Subset contains no far LiDAR returns')
    with (a.out/'per_frame.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    table = aggregate(records)
    summary = dict(validation=info, subset_sha256=sha256_file(manifest), coverage=coverage,
                   checkpoint=str(conv), checkpoint_sha256=sha256_file(conv), pseudo_config_sha256=contract['pseudo_config'],
                   anythermal=str(raw), anythermal_metadata_sha256=sha256_file(raw/'run_metadata.json'),
                   vae=recipe.lotus_model_path, posterior='mode, float32',
                   aggregation='mean over frames with 60 <= GT < 80; SSI-log fit over whole frame',
                   table=table)
    (a.out/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    for r in table:
        if r['sequence'] == 'ALL':
            print('[stage]', json.dumps(r))
    print('[completed]', a.out/'summary.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('runs','out','data-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--cache', type=Path, action='append', default=[])
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
    run(p.parse_args())
