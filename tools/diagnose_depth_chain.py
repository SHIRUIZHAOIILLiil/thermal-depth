"""Read-only, fixed-frame log-target/VAE/checkpoint diagnostic. Never trains.

Run prepare -> targets -> models -> score as separate processes. Input manifests
must be from official train/val sequence lists; no test input is accepted.
All three arms use the same selected rows, captions and inference seed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ms2_eval.io import sha256_file
from ms2_eval.official_protocol import evaluate_sample, official_valid_mask
from tools.diagnose_error_sources import aligned_depth

ARMS = {'q098': .98, 'q0995': .995, 'q100': 1.0}
BINS = [(0, 5), (5, 20), (20, 40), (40, 60), (60, 80)]
SEQ = re.compile(r'\d{4}(?:-\d{2}){5}')
TRAIN_SEQUENCES = set('''2021-08-06-10-59-33 2021-08-06-11-37-46
2021-08-06-17-21-04 2021-08-06-17-44-55 2021-08-13-16-50-57
2021-08-13-17-06-04 2021-08-13-21-36-10 2021-08-13-22-36-41'''.split())


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def select_rows(rows, count, seed, allowed):
    if not 0 < count <= len(rows):
        raise ValueError(f'Requested {count} frames from {len(rows)}')
    ids = set()
    for row in rows:
        sid = row['id']
        if sid in ids or not re.fullmatch(r'\d{4}(?:-\d{2}){5}_\d+', sid):
            raise ValueError(f'Duplicate or unsafe ID: {sid}')
        ids.add(sid)
        fields = [sid, row['thermal_path'], row.get('thermal_depth_path') or row['depth_path']]
        seqs = [SEQ.findall(str(s)) for s in fields]
        if any(len(s) != 1 for s in seqs) or len({s[0] for s in seqs}) != 1:
            raise ValueError(f'Inconsistent sequence paths: {sid}')
        if seqs[0][0] not in allowed:
            raise ValueError(f'{sid} is outside the allowed split; refusing test/foreign sequences')
        if '/thr/' not in str(fields[2]).replace('\\', '/'):
            raise ValueError(f'Not thermal-view GT: {sid}')
        stems = [Path(str(path).replace('\\', '/')).stem for path in fields[1:]]
        if stems[0] != stems[1] or sid != f'{seqs[0][0]}_{stems[0]}':
            raise ValueError(f'Inconsistent frame ID/image/GT: {sid}')
        if not str(row.get('caption', '')).strip():
            raise ValueError(f'Missing caption: {sid}')
    picks = sorted(np.random.default_rng(seed).choice(len(rows), count, replace=False).tolist())
    return [dict(rows[i], diagnostic_source_index=i) for i in picks]


def frame_seed(sid, seed):
    return (seed + int(hashlib.sha256(sid.encode()).hexdigest()[:8], 16)) % (2**31)


def inverse_log_unit(y, bounds):
    """Trainer's un-clamped decoder unit y -> log metres -> metres."""
    lo, hi = map(float, bounds)
    return np.exp(np.clip(lo + np.asarray(y, np.float64) * (hi - lo + 1e-5), -9, 9))


def metrics(pred, gt, mask):
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite prediction')
    if not mask.any():
        return None
    g = gt[mask].astype(np.float64)
    p = np.clip(pred[mask].astype(np.float64), .001, 80)
    diff = p - g
    ratio = np.maximum(p/g, g/p)
    return dict(pixels=int(mask.sum()), abs_rel=float(np.mean(abs(diff)/g)),
                rmse=float(np.sqrt(np.mean(diff**2))), mse=float(np.mean(diff**2)),
                a1=float(np.mean(ratio < 1.25)), too_near=float(np.mean(p < g)),
                below_eval_range=float(np.mean(pred[mask] < .001)),
                above_eval_range=float(np.mean(pred[mask] > 80)),
                signed_log_error=float(np.mean(np.log(p/g))))


def score_stage(raw, gt, split, arm, stage, sid, inverse_bounds=None, log_input=False, aligned_only=False):
    """Fit once per frame on all GT; never refit per depth bin."""
    valid = official_valid_mask(gt, .001, 80)
    if not valid.any() or not np.isfinite(raw).all():
        raise ValueError(f'Invalid GT/prediction: {sid}')
    direct = inverse_log_unit(raw, inverse_bounds) if inverse_bounds is not None else raw
    fit_input = raw if log_input else np.log(np.maximum(raw, 1e-6))
    official = evaluate_sample(fit_input.astype(np.float32), gt, align='ssi_log')
    fitted = aligned_depth(fit_input.astype(np.float32), gt, valid, 'ssi_log')
    outputs = []
    variants = [('ssi_log', fitted)]
    if not aligned_only:
        variants.insert(0, ('known_target_scale' if inverse_bounds is not None else 'metric_no_fit', direct))
    for alignment, pred in variants:
        for label, mask in [('all', valid)] + [(f'{lo}-{hi}m', valid & (gt >= lo) & (gt < hi)) for lo, hi in BINS]:
            values = metrics(pred, gt, mask)
            if values is None:
                continue
            if label == 'all' and alignment == 'ssi_log':
                for name in ('abs_rel', 'rmse', 'a1'):
                    if not np.isclose(values[name], official[name], rtol=1e-7, atol=1e-9):
                        raise ValueError(f'Official evaluator mismatch: {name}')
            outputs.append(dict(split=split, arm=arm, stage=stage, id=sid,
                                alignment=alignment, bin=label,
                                alignment_scale=official['alignment_scale'] if alignment == 'ssi_log' else None,
                                alignment_shift=official['alignment_shift'] if alignment == 'ssi_log' else None, **values))
    return outputs


def write_tables(out, records):
    with (out / 'per_frame.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    grouped = defaultdict(list)
    for row in records:
        grouped[tuple(row[k] for k in ('split', 'arm', 'stage', 'alignment', 'bin'))].append(row)
    summary = []
    for key, rows in sorted(grouped.items()):
        result = dict(zip(('split', 'arm', 'stage', 'alignment', 'bin'), key))
        result.update(frames=len(rows), pixels=sum(r['pixels'] for r in rows))
        result.update({k: float(np.mean([r[k] for r in rows])) for k in
                       ('abs_rel', 'rmse', 'mse', 'a1', 'too_near', 'signed_log_error',
                        'below_eval_range', 'above_eval_range')})
        summary.append(result)
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (out / 'summary.json').write_text(json.dumps({'diagnostic_only': True,
        'aggregation': 'macro average over frames containing GT in bin; RMSE is mean per-frame bin RMSE',
        'warning': 'Known-target-scale uses target-derived bounds; not native metric inference. VAE roundtrip is not a hard model ceiling.',
        'rows': summary}, indent=2), encoding='utf-8')
    # Paired changes, not an additive causal decomposition of model errors.
    indexed = {(r['arm'], r['id'], r['stage'], r['alignment'], r['bin']): r
               for r in records if r['split'] == 'train'}
    deltas = []
    for row in records:
        if row['split'] != 'train' or row['alignment'] != 'known_target_scale':
            continue
        predecessor = {'unit': ('dense', 'metric_no_fit'), 'vae_mode': ('unit', 'known_target_scale'),
                       'vae_sample': ('unit', 'known_target_scale'), 'model': ('vae_mode', 'known_target_scale')}[row['stage']]
        before = indexed[(row['arm'], row['id'], *predecessor, row['bin'])]
        deltas.append(dict(arm=row['arm'], id=row['id'], bin=row['bin'],
                          transition=f'{predecessor[0]}->{row["stage"]}',
                          **{f'delta_{k}': row[k]-before[k] for k in ('abs_rel', 'rmse', 'mse', 'a1')}))
    if deltas:
        with (out/'stage_deltas.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(deltas[0]))
            writer.writeheader()
            writer.writerows(deltas)


def prepare(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError('Output directory must be empty; do not mix/reuse attempts')
    train = {s.strip().lstrip('_') for s in args.train_list.read_text().splitlines() if s.strip()}
    val = {s.strip().lstrip('_') for s in args.val_list.read_text().splitlines() if s.strip()}
    # Freeze the official split identities, not just a caller-supplied label.
    if train != TRAIN_SEQUENCES or val != {'2021-08-13-16-08-46', '2021-08-13-16-14-48', '2021-08-13-21-58-13'}:
        raise ValueError('Unexpected official train/val lists')
    checkpoints = {}
    for spec in args.checkpoint:
        arm, path = spec.split('=', 1)
        if arm not in ARMS or arm in checkpoints:
            raise ValueError(f'Unknown/duplicate arm: {arm}')
        path = Path(path).resolve()
        if path.name != 'step16000_weights.pt':
            raise ValueError('This diagnostic freezes all checkpoints at step16000')
        checkpoints[arm] = dict(path=str(path), sha256=sha256_file(path))
    if set(checkpoints) != set(ARMS):
        raise ValueError('Need q098, q0995, q100 checkpoints; no silent omission')
    config = dict(diagnostic_only=True, seed=args.seed, model_path=args.model_path,
                  checkpoints=checkpoints, source_hashes={}, selected_hashes={}, coverage={})
    from PIL import Image
    for split, source, count, allowed in [('train', args.train_manifest, args.train_frames, train),
                                          ('val', args.val_manifest, args.val_frames, val)]:
        rows = select_rows(read_jsonl(source), count, args.seed, allowed)
        coverage = defaultdict(int)
        for row in rows:
            row['split'] = split
            for field in ('thermal_path', 'thermal_depth_path'):
                value = row.get(field) or (row.get('depth_path') if field == 'thermal_depth_path' else None)
                path = args.data_root / value
                if not path.is_file():
                    raise FileNotFoundError(path)
                row[field] = str(path.resolve())
            gt = np.asarray(Image.open(row['thermal_depth_path']), dtype=np.float32)/256.
            valid = official_valid_mask(gt, .001, 80)
            if not valid.any():
                raise ValueError(f'No valid GT: {row["id"]}')
            for lo, hi in BINS:
                coverage[f'{lo}-{hi}m_pixels'] += int((valid & (gt >= lo) & (gt < hi)).sum())
            if split == 'train' and not (args.pseudo_dir / f'{row["id"]}.npy').is_file():
                raise FileNotFoundError(f'Missing pseudo: {row["id"]}')
            input_paths = [row['thermal_path'], row['thermal_depth_path']]
            if split == 'train':
                input_paths.append(str((args.pseudo_dir/f'{row["id"]}.npy').resolve()))
            row['diagnostic_input_sha256'] = {path: sha256_file(path) for path in input_paths}
        if coverage['60-80m_pixels'] == 0:
            raise ValueError(f'{split}: selected sample has no far GT; increase sample size')
        selected = out / f'{split}.jsonl'
        selected.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
        config['source_hashes'][split] = dict(path=str(source.resolve()), sha256=sha256_file(source))
        config['selected_hashes'][split] = sha256_file(selected)
        config['coverage'][split] = dict(coverage)
    config['pseudo_dir'] = str(args.pseudo_dir.resolve())
    config['environment'] = dict(thermal_stretch='percentile', text_padding='legacy',
        decode_clamp=False, condition_latent='mode', inference_steps=1, ensemble_size=1,
        random_flip=False, lower_quantile=.02, upper_quantiles=ARMS, checkpoint_step=16000)
    config['data_root'] = str(args.data_root.resolve())
    config['source_code_sha256'] = {name: sha256_file(ROOT / name) for name in
        ('tools/diagnose_depth_chain.py', 'tools/train_route_suite.py', 'lotus/utils/ms2_thermal_dataset.py')}
    (out / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(json.dumps(config, indent=2), flush=True)


def load_config(out):
    config = json.loads((out / 'config.json').read_text(encoding='utf-8'))
    for split, sha in config['selected_hashes'].items():
        if sha256_file(out / f'{split}.jsonl') != sha:
            raise ValueError('Frozen selection changed')
        for row in read_jsonl(out / f'{split}.jsonl'):
            for path, expected in row['diagnostic_input_sha256'].items():
                if sha256_file(path) != expected:
                    raise ValueError(f'Frozen input changed: {path}')
    for name, sha in config['source_code_sha256'].items():
        if sha256_file(ROOT / name) != sha:
            raise ValueError(f'Diagnostic code changed since prepare: {name}')
    return config


def targets(args, config):
    import torch
    from diffusers import AutoencoderKL
    spec = importlib.util.spec_from_file_location('chain_ms2', ROOT / 'lotus/utils/ms2_thermal_dataset.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    datasets = {arm: module.MS2ThermalDataset(args.out/'train.jsonl', config['data_root'],
        config['pseudo_dir'], norm_type='log_truncnorm', truncnorm_min=.02, truncnorm_max=q,
        transform=None, sky_mask_dir=None) for arm, q in ARMS.items()}
    vae = AutoencoderKL.from_pretrained(config['model_path'], subfolder='vae',
                                      local_files_only=True, torch_dtype=torch.float32).to('cuda').eval()
    vae.requires_grad_(False)
    digest = hashlib.sha256()
    for name, tensor in sorted(vae.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    vae_sha256 = digest.hexdigest()
    destination = args.out / 'targets'
    destination.mkdir(exist_ok=False)
    metadata = []
    for index, row in enumerate(read_jsonl(args.out/'train.jsonl')):
        pseudo_path = Path(config['pseudo_dir']) / f'{row["id"]}.npy'
        pseudo = np.load(pseudo_path, allow_pickle=False).astype(np.float32)
        if not np.isfinite(pseudo).all():
            raise ValueError(f'Nonfinite pseudo: {row["id"]}')
        for arm, dataset in datasets.items():
            ex = dataset[index]  # actual training target construction, no flip
            gt = ex['gt_depth_values'][0].numpy()
            dense, _, _ = dataset._completed_depth(dataset.rows[index])
            target = ex['depth_values'][None].cuda().float()
            bounds = ex['norm_bounds'].numpy()
            if pseudo.shape != gt.shape or not torch.isfinite(target).all():
                raise ValueError('Bad target shape/values')
            with torch.inference_mode():
                posterior = vae.encode(target).latent_dist
                # Same standard-normal draw for all three bounds of this frame.
                generator = torch.Generator(device='cuda').manual_seed(frame_seed(row['id'], config['seed']))
                eps = torch.randn(posterior.mean.shape, device='cuda', generator=generator)
                outputs = {}
                for mode, z in [('mode', posterior.mode()), ('sample', posterior.mean + posterior.std * eps)]:
                    scaled = z * vae.config.scaling_factor
                    decoded = vae.decode(scaled / vae.config.scaling_factor).sample
                    outputs['vae_'+mode] = (decoded.float().mean(1)[0]/2+.5).cpu().numpy()
            np.savez_compressed(destination / f'{arm}_{row["id"]}.npz', gt=gt,
                pseudo=pseudo, dense=dense[0].numpy(), unit=ex['depth_values'][0].numpy()/2+.5,
                bounds=bounds, **outputs)
            metadata.append(dict(arm=arm, id=row['id'], bounds=bounds.tolist(),
                vae_seed=frame_seed(row['id'], config['seed']), pseudo_sha256=sha256_file(pseudo_path),
                gt_sha256=sha256_file(row['thermal_depth_path']),
                thermal_min=float(ex['pixel_values'].min()), thermal_max=float(ex['pixel_values'].max()),
                thermal_std=float(ex['pixel_values'].std())))
        print(f'[targets] {index+1}/{len(datasets["q098"])} {row["id"]}', flush=True)
    (destination/'metadata.json').write_text(json.dumps(dict(precision='fp32', vae_state_sha256=vae_sha256,
        posterior='mode and one shared-epsilon sample; diagnostic, not exact fp16 training replay',
        pseudo='calibrated with same-frame LiDAR before overwrite; NOT independent ground truth',
        records=metadata), indent=2), encoding='utf-8')


def models(args, config):
    for arm, checkpoint in config['checkpoints'].items():
        if sha256_file(checkpoint['path']) != checkpoint['sha256']:
            raise ValueError(f'Checkpoint changed: {arm}')
        for split in ('train', 'val'):
            destination = args.out / 'models' / arm / split
            if destination.exists():
                raise ValueError(f'Refusing partial/stale inference directory: {destination}')
            command = [sys.executable, str(ROOT/'tools/train_route_suite.py'), '--route', 'b_thermal_unet',
                '--output-dir', str(destination), '--ms2-root', config['data_root'],
                '--train-manifest', str(args.out/'train.jsonl'), '--val-manifest', str(args.out/f'{split}.jsonl'),
                '--eval-checkpoint', checkpoint['path'], '--val-caption-mode', 'correct',
                '--align-mode', 'ssi_log', '--num-inference-steps', '1', '--condition-latent', 'mode',
                '--val-stride', '1', '--save-raw-pred', '--local-files-only', '--seed', str(config['seed']),
                '--lotus-model-path', config['model_path'], '--ensemble-size', '1']
            env = dict(os.environ, IRIS_THERMAL_STRETCH='percentile', IRIS_TEXT_PADDING='legacy', IRIS_DECODE_CLAMP='0')
            print('[inference-only]', ' '.join(command), flush=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)


def score(args, config):
    from PIL import Image
    records, comparisons = [], []
    for split in ('train', 'val'):
        for row in read_jsonl(args.out/f'{split}.jsonl'):
            gt = np.asarray(Image.open(row['thermal_depth_path']), dtype=np.float32)/256.
            valid = official_valid_mask(gt, .001, 80)
            for arm in ARMS:
                raw = np.load(args.out/'models'/arm/split/'raw_predictions'/f'{row["id"]}.npy', allow_pickle=False).squeeze()
                if raw.shape != gt.shape:
                    raise ValueError('Native model/GT resolution mismatch; do not resize sparse GT')
                if split == 'val':
                    # Only GT-aligned metric output: no target bounds from val.
                    records.extend(score_stage(raw, gt, split, arm, 'model', row['id'],
                        log_input=True, aligned_only=True))
                    continue
                with np.load(args.out/'targets'/f'{arm}_{row["id"]}.npz', allow_pickle=False) as data:
                    if not np.array_equal(data['gt'][valid], gt[valid]):
                        raise ValueError('GT changed between phases')
                    for stage in ('pseudo', 'dense'):
                        records.extend(score_stage(data[stage], gt, split, arm, stage, row['id']))
                    for stage in ('unit', 'vae_mode', 'vae_sample', 'model'):
                        value = raw if stage == 'model' else data[stage]
                        records.extend(score_stage(value, gt, split, arm, stage, row['id'],
                                                   inverse_bounds=data['bounds'], log_input=True))
                    for region, mask in [('lidar', valid), ('no_lidar', ~valid), ('all', np.ones_like(valid))]:
                        if mask.any():
                            comparisons.append(dict(arm=arm, id=row['id'], region=region,
                                model_vs_vae_mode_unit_mae=float(abs(raw[mask]-data['vae_mode'][mask]).mean()),
                                model_vs_target_unit_mae=float(abs(raw[mask]-data['unit'][mask]).mean())))
    destination = args.out / 'analysis'
    destination.mkdir(exist_ok=False)
    write_tables(destination, records)
    (destination/'model_vs_target_unit.json').write_text(json.dumps(comparisons, indent=2), encoding='utf-8')
    print(f'[done] {destination}; normalized MAE is NOT metres; train/val subsets are not main-table scores')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'targets', 'models', 'score'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--pseudo-dir', type=Path)
    parser.add_argument('--train-manifest', type=Path)
    parser.add_argument('--val-manifest', type=Path)
    parser.add_argument('--train-list', type=Path)
    parser.add_argument('--val-list', type=Path)
    parser.add_argument('--checkpoint', action='append', default=[])
    parser.add_argument('--train-frames', type=int, default=256)
    parser.add_argument('--val-frames', type=int, default=96)
    parser.add_argument('--seed', type=int, default=20261017)
    parser.add_argument('--model-path', default='jingheya/lotus-depth-g-v2-1-disparity')
    args = parser.parse_args()
    os.environ['IRIS_THERMAL_STRETCH'] = 'percentile'
    os.environ['IRIS_TEXT_PADDING'] = 'legacy'
    os.environ['IRIS_DECODE_CLAMP'] = '0'
    args.out = args.out.resolve()
    if args.phase == 'prepare':
        prepare(args)
    else:
        config = load_config(args.out)
        globals()[args.phase](args, config)


if __name__ == '__main__':
    main()
