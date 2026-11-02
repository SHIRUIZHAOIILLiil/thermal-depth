"""MS2 thermal NeWCRF: official checkpoint -> calibrated completion fine-tune."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ms2_eval.io import load_manifest, load_ms2_gt, sha256_file
from ms2_eval.official_protocol import OFFICIAL_METRICS, evaluate_sample
from tools.da2_caption_data import TRAIN, VAL, ImageStream, checked_manifest
from tools.run_ms2_supdepth_baselines import (
    build_network, forward_depth, load_weights, official_frames, preprocess_thermal,
)
from tools.sd21_pseudo_data import load_pseudo, prepare_inventory
from tools.test_da2_fixed import TESTS, validate_samples
from tools.train_da2_caption import json_write, seed_all

STEPS = (0, 500, 1000, 2000, 4000)
BATCH, ACCUM, LR = 2, 6, 1e-5
TRAIN_FILE = 'ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl'
VAL_FILE = 'ms2_val_official3_thermalcap_20260821.jsonl'
MODES = ('ssi', 'none', 'median')


def completed_target(pseudo, gt):
    # Same metric map as old Iris before its latent-space normalization.
    if pseudo.shape != gt.shape:
        raise ValueError('Pseudo/LiDAR geometry mismatch')
    real = np.isfinite(gt) & (gt > .001) & (gt < 80.)
    values = np.where(real, gt, pseudo)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite completion outside measured LiDAR')
    return np.clip(values, .001, 80.)


class Frames(Dataset):
    def __init__(self, samples, inventory=None):
        self.samples, self.inventory = samples, inventory

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        image = preprocess_thermal(Path(s.thermal_path))[0]
        if not torch.isfinite(image).all() or image.std() <= 0:
            raise ValueError(f'Invalid thermal contrast: {s.sample_id}')
        _, gt = load_ms2_gt(s.thermal_gt_path, 256.)
        if gt.shape != (256, 640):
            raise ValueError(f'Unexpected GT geometry: {s.sample_id}')
        if self.inventory is not None:
            path = Path(self.inventory['identity']['root']) / (s.sample_id + '.npy')
            pseudo, _ = load_pseudo(path, self.inventory['files'][s.sample_id])
            gt = completed_target(pseudo, gt)
            if torch.rand(()) < .5:
                return image.flip(-1), torch.from_numpy(gt).flip(-1), s.sample_id
        return image, torch.from_numpy(gt), s.sample_id


def predict(net, x, size):
    p = forward_depth('newcrf', net, x)
    if p.ndim != 3:
        raise ValueError(f'Unexpected depth shape {p.shape}')
    if p.shape[-2:] != size:
        p = F.interpolate(p[:, None], size=size, mode='bilinear', align_corners=False)[:, 0]
    return p


def train_mode(net):
    # NeWCRF's PSP head uses BN; microbatch 2 statistics would overwrite the
    # official running stats. Freeze them, keep the affine parameters trainable.
    net.train()
    for m in net.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def bn_stats(net):
    return {n: b.detach().clone() for n, b in net.named_buffers() if n.endswith(('running_mean', 'running_var'))}


def silog(p, target):
    if p.shape != target.shape or not torch.isfinite(p).all() or (p <= 0).any():
        raise ValueError('Invalid metric depth for SiLog')
    # Mirror the old completion's pixel mask: clipped near endpoint is valid,
    # upper endpoint is excluded. Do not turn an 80 m clipping into a measurement.
    valid = torch.isfinite(target) & (target >= .001) & (target < 80.)
    if not valid.any():
        raise ValueError('No valid completion pixels')
    e = p[valid].float().log() - target[valid].float().log()
    variance = e.square().mean() - .85 * e.mean().square()
    return 10 * (variance.clamp_min(1e-12).sqrt())


@torch.no_grad()
def evaluate(net, samples, workers, csv_path=None, subset=None):
    net.eval()
    rows = []
    loader = DataLoader(Frames(samples), batch_size=BATCH, num_workers=workers)
    device = next(net.parameters()).device
    for image, gt, ids in loader:
        pred = predict(net, image.to(device), gt.shape[-2:]).float().cpu().numpy()
        for ident, p, g in zip(ids, pred, gt.numpy()):
            row = {'id': ident}
            for mode in MODES:
                m = evaluate_sample(p, g, align=mode, min_depth=.001, max_depth=80.)
                row.update({f'{mode}_{k}': m[k] for k in OFFICIAL_METRICS})
            rows.append(row)
    def mean(group):
        return {'frames': len(group), 'metrics': {
            mode: {k: float(np.mean([r[f'{mode}_{k}'] for r in group])) for k in OFFICIAL_METRICS}
            for mode in MODES}}
    result = mean(rows)
    if subset is not None:
        common = [r for r in rows if r['id'] in subset]
        if not common:
            raise ValueError('No official test IDs in manifest')
        result['official_common'] = mean(common)
        result['official_common']['crawl_count'] = len(subset)
        result['official_common']['missing_ids'] = sorted(subset - {r['id'] for r in common})
    if csv_path:
        with csv_path.open('w', encoding='utf8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    return result


def selection(curve):
    def rank(step):
        return curve[step]['metrics']['ssi']['abs_rel'], step
    return min(curve, key=rank), min((s for s in curve if s), key=rank)


def pipeline_contract(a):
    upstream_dir = a.bmsd_root / 'models/network/newcrf'
    sources = sorted(upstream_dir.rglob('*.py'))
    if not sources:
        raise FileNotFoundError(upstream_dir)
    return dict(source=str(a.checkpoint.resolve()), source_sha256=sha256_file(a.checkpoint),
                upstream={str(p.relative_to(upstream_dir)): sha256_file(p) for p in sources},
                runner_sha256=sha256_file(Path(__file__)),
                benchmark_sha256=sha256_file(ROOT/'tools/run_ms2_supdepth_baselines.py'))


def official_model(a):
    net = build_network('newcrf', a.bmsd_root)
    load_weights(net, a.checkpoint, strict=True)
    return net.cuda().eval()


def fit(a):
    smoke = a.stage == 'smoke'
    train, ti = checked_manifest(a.manifest_dir/TRAIN_FILE, a.data_root, 'train', 75688)
    val, vi = checked_manifest(a.manifest_dir/VAL_FILE, a.data_root, 'val', 14249)
    train_sequences = {s.sample_id.rsplit('_', 1)[0] for s in train}
    official_train = {s.strip().lstrip('_') for s in (a.data_root/'train_list.txt').read_text().splitlines() if s.strip()}
    if train_sequences != TRAIN or official_train != TRAIN or {s.sample_id.rsplit('_', 1)[0] for s in val} != VAL:
        raise ValueError('Train/val sequences differ from frozen MS2 protocol')
    if not vi['sha256'].startswith('c26a5859'):
        raise ValueError('Unexpected validation manifest')
    index_path = a.out/'inventory.json'
    if not smoke and not index_path.is_file():
        raise ValueError('Run smoke first')
    inventory = prepare_inventory(train, ti['sha256'], a.pseudo_dir, index_path)
    contract = dict(**pipeline_contract(a), train=ti, val=vi, inventory_sha256=sha256_file(index_path),
                    seed=43, lr=LR, batch=BATCH, accumulation=ACCUM, schedule=list(STEPS),
                    augmentation='paired horizontal flip p=.5 (training only)',
                    batchnorm='running statistics frozen (eval mode); affine parameters trained',
                    loss='dense metric SiLog (lambda .85, factor 10); LiDAR writeback, clipped [.001,80], mask [.001,80)',
                    selection='stride4 val SSI AbsRel, tie -> earliest, including step 0')
    proof = a.out/'smoke.json'
    if not smoke and json.loads(proof.read_text()) != contract:
        raise ValueError('Smoke inputs or implementation changed')
    run = a.out/('smoke' if smoke else 'train')
    run.mkdir(exist_ok=False)
    json_write(run/'contract.json', contract)
    seed_all(43)
    net = official_model(a)
    official_bn = bn_stats(net)
    if smoke and not official_bn:
        raise ValueError('Smoke: expected BN running statistics in the NeWCRF head')
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=.01)
    steps = (0, 2) if smoke else STEPS
    val_curve = val[:8] if smoke else val[::4]
    curve = {0: evaluate(net, val_curve, a.workers)}
    train_loader = DataLoader(Frames(train, inventory), batch_size=BATCH, num_workers=a.workers,
                              sampler=ImageStream(len(train), 43, 0, steps[-1]*ACCUM*BATCH))
    best = None
    for j, (image, target, ids) in enumerate(train_loader, 1):
        train_mode(net)
        pred = predict(net, image.cuda(), target.shape[-2:])
        loss = silog(pred, target.cuda())
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite SiLog: {ids}')
        (loss/ACCUM).backward()
        if j % ACCUM:
            continue
        grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        if smoke and not float(grad) > 0:
            raise ValueError('Smoke: no gradient')
        opt.step()
        opt.zero_grad(set_to_none=True)
        step = j//ACCUM
        if step == 1 or step % 100 == 0:
            print(f'[train] step={step} loss={float(loss):.6f} grad={float(grad):.6f}', flush=True)
        if step in steps:
            curve[step] = evaluate(net, val_curve, a.workers)
            _, trained = selection(curve)
            if trained != best:
                tmp = run/'best_trained.tmp'
                torch.save({'step': step, 'model': net.state_dict()}, tmp)
                os.replace(tmp, run/'best_trained.pt')
                best = trained
            json_write(run/'curve.json', curve)
            print('[val]', step, curve[step]['metrics']['ssi'], flush=True)
    if any(not torch.equal(v, official_bn[n]) for n, v in bn_stats(net).items()):
        raise ValueError('BN running statistics changed during fine-tuning')
    selected, trained = selection(curve)
    state = torch.load(run/'best_trained.pt', map_location='cpu', weights_only=True)
    if state['step'] != trained:
        raise ValueError('Invalid trained checkpoint')
    if selected == 0:
        load_weights(net, a.checkpoint)
    else:
        net.load_state_dict(state['model'])
    full = evaluate(net, val[:8] if smoke else val, a.workers, run/'selected_val.csv')
    json_write(run/'summary.json', dict(selected_step=selected, best_trained_step=trained,
               trained_sha256=sha256_file(run/'best_trained.pt'), curve=curve, selected_val=full))
    if smoke:
        json_write(proof, contract)
    print('[selected]', selected, 'best_nonzero', trained, full['metrics']['ssi'], flush=True)


def test(a):
    run = a.out/'train'
    summary = json.loads((run/'summary.json').read_text())
    contract = json.loads((run/'contract.json').read_text())
    if any(contract[k] != v for k,v in pipeline_contract(a).items()):
        raise ValueError('Test inputs differ from training')
    path = run/'best_trained.pt'
    if sha256_file(path) != summary['trained_sha256'] or summary['selected_val']['frames'] != 14249:
        raise ValueError('Checkpoint or full validation mismatch')
    curve = {int(k): v for k,v in summary['curve'].items()}
    if tuple(sorted(curve)) != STEPS or selection(curve) != (summary['selected_step'], summary['best_trained_step']):
        raise ValueError('Unverified validation selection')
    suffix, prefix, count, seqs = TESTS[a.condition]
    samples, info = load_manifest(a.manifest_dir/f'ms2_test_{suffix}_thermalcap_20260821.jsonl',
                                  a.data_root, allow_legacy_depth_path=True)
    if not info['sha256'].startswith(prefix) or len(samples) != count:
        raise ValueError('Unexpected test manifest')
    ids = validate_samples(samples, seqs)
    subset = {s['id'] for s in official_frames(a.data_root, 'test_'+a.condition, 10)}
    expected = {'day': (2331,2332), 'night': (2292,2292), 'rain': (2503,2503)}
    if (len(ids & subset), len(subset)) != expected[a.condition]:
        raise ValueError('Official test subset differs from previous Iris runs')
    out = a.out/'test'/a.condition
    out.mkdir(parents=True, exist_ok=True)
    net = official_model(a)
    results = {}
    for step in (0, summary['best_trained_step']):
        if step:
            state = torch.load(path, map_location='cpu', weights_only=True)
            if state['step'] != step:
                raise ValueError('Wrong nonzero checkpoint')
            net.load_state_dict(state['model'])
            del state
        results[step] = evaluate(net, samples, a.workers, out/f'step{step}_frames.csv', subset)
        print('[test-official]', a.condition, step, results[step]['official_common']['metrics']['ssi'], flush=True)
    json_write(out/'summary.json', dict(manifest=info, selected_step=summary['selected_step'],
               validation_sha256=sha256_file(run/'summary.json'), results=results))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['smoke', 'train', 'test'])
    for flag in ('data-root', 'manifest-dir', 'bmsd-root', 'checkpoint', 'out'):
        p.add_argument('--'+flag, type=Path, required=True)
    p.add_argument('--pseudo-dir', type=Path)
    p.add_argument('--condition', choices=tuple(TESTS))
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    if (a.stage == 'test' and not a.condition) or (a.stage != 'test' and not a.pseudo_dir):
        p.error('Test needs --condition; training needs --pseudo-dir')
    a.out.mkdir(parents=True, exist_ok=True)
    if a.stage == 'test':
        test(a)
    else:
        fit(a)
