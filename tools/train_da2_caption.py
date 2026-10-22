"""Sparse-LiDAR DA V2 experiment. Separate from the historical relative-depth baseline.

Fresh smoke/train runs refuse existing outputs. Resume restores the complete state.
No test manifests or test modes are accepted. All pilot evaluation is full validation.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.da2_caption import CaptionDepth, masked_log_l1
from tools.da2_caption_data import checked_manifest, ThermalLidar, ImageStream, caption_hash
from ms2_eval.io import sha256_file
from ms2_eval.official_protocol import evaluate_sample, OFFICIAL_METRICS

TOTAL_STEPS = 20000
MICROBATCH = 4
ACCUMULATION = 3


def json_write(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def make_optimizer(model):
    encoder = list(model.core.pretrained.parameters())
    encoder_ids = {id(p) for p in encoder}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
    opt = torch.optim.AdamW([dict(params=encoder, lr=5e-6), dict(params=other, lr=5e-5)],
                           betas=(.9, .999), weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: max(0., 1 - step / TOTAL_STEPS) ** .9)
    return opt, scheduler


def save_checkpoint(path, model, optimizer, scheduler, step, contract):
    state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                 step=step, image_cursor=step * MICROBATCH * ACCUMULATION,
                 rng=rng_state(), amp=dict(dtype='bfloat16', scaler=None), contract=contract)
    tmp = Path(str(path) + '.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer, scheduler, contract, restore_random=True):
    # Only trust checkpoints produced locally by this experiment, never arbitrary downloads.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['contract'] != contract:
        raise ValueError('Checkpoint contract differs: data/assets/code/config changed; refusing resume')
    if state['image_cursor'] != state['step'] * MICROBATCH * ACCUMULATION:
        raise ValueError('Invalid consumed-image cursor')
    model.load_state_dict(state['model'], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
    if restore_random:
        restore_rng(state['rng'])
    return state['step']


def token_cache(samples, tokenizer, assets, manifest_hash, create):
    signature = hashlib.sha256((caption_hash(samples) + assets['text_fingerprint'] + manifest_hash).encode()).hexdigest()
    path = Path(assets['root']) / f'tokens_{signature}.pt'
    if path.exists():
        result = torch.load(path, map_location='cpu', weights_only=True)
        if result['signature'] != signature or len(result['input_ids']) != len(samples):
            raise ValueError('Caption cache metadata mismatch')
        if list(result['input_ids'].shape) != [len(samples), 77] or not result['attention_mask'].any(dim=1).all():
            raise ValueError('Caption cache shape/mask invalid')
        return result
    if not create:
        raise FileNotFoundError(f'Run smoke first to create token cache: {path}')
    ids, masks, truncated = [], [], 0
    for start in range(0, len(samples), 256):
        texts = [s.caption for s in samples[start:start + 256]]
        lengths = tokenizer(texts, truncation=False, padding=False)['input_ids']
        truncated += sum(len(x) > 77 for x in lengths)
        encoded = tokenizer(texts, padding='max_length', max_length=77, truncation=True, return_tensors='pt')
        ids.append(encoded['input_ids']); masks.append(encoded['attention_mask'].bool())
    result = dict(input_ids=torch.cat(ids), attention_mask=torch.cat(masks), signature=signature,
                  truncated=truncated, samples=len(samples))
    tmp = Path(str(path) + '.tmp')
    torch.save(result, tmp); os.replace(tmp, path)
    print(f'[tokens] {path.name}: truncated={truncated}/{len(samples)}', flush=True)
    return result


def verify_assets(path):
    assets = json.loads(Path(path).read_text(encoding='utf-8'))
    for file, digest in assets['hashes'].items():
        if sha256_file(file) != digest:
            raise ValueError(f'Asset checksum mismatch: {file}')
    repo = Path(assets['da2_repo'])
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if commit != assets['da2_commit'] or dirty.strip():
        raise ValueError('Official DA V2 checkout changed since asset preparation')
    return assets


def build_model(assets, arm, seed):
    sys.path.insert(0, str(Path(assets['da2_repo']) / 'metric_depth'))
    from depth_anything_v2.dpt import DepthAnythingV2
    # Refuse an accidentally imported relative-depth implementation.
    import depth_anything_v2.dpt as module
    if Path(module.__file__).resolve().parent != (Path(assets['da2_repo']) / 'metric_depth/depth_anything_v2').resolve():
        raise ValueError('Wrong DA V2 implementation imported (must be metric_depth)')
    seed_all(seed)
    core = DepthAnythingV2(encoder='vitb', features=128, out_channels=[96, 192, 384, 768], max_depth=80.)
    core.load_state_dict(torch.load(assets['weights'], map_location='cpu', weights_only=True), strict=True)
    text = None
    if arm == 'caption':
        from transformers import CLIPTextModel
        text = CLIPTextModel.from_pretrained(assets['text_path'], subfolder='text_encoder', local_files_only=True)
        if text.config.hidden_size != 1024:
            raise ValueError('Expected audited SD2.1 text width 1024')
    model = CaptionDepth(core, text, adapter_seed=seed)
    # Backbone/adapter construction is not allowed to change the training RNG stream.
    seed_all(seed + 100000)
    return model


def predict(model, batch, device, amp):
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        return model(batch['image'].to(device),
                     batch.get('token_ids', None).to(device) if 'token_ids' in batch else None,
                     batch.get('attention_mask', None).to(device) if 'attention_mask' in batch else None)


def evaluate(model, dataset, device, amp, out, step):
    saved_rng, was_training = rng_state(), model.training
    model.eval()
    sums = {mode: {k: 0. for k in OFFICIAL_METRICS} for mode in ('ssi', 'none')}
    sequence_sums = {}
    bin_sums = {}
    try:
        with (out / f'val_step{step:06d}_frames.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['id', 'sequence', 'alignment', *OFFICIAL_METRICS])
            writer.writeheader()
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                generator=torch.Generator().manual_seed(0))
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    p = predict(model, batch, device, amp)[0].float().cpu().numpy()
                    g = batch['gt'][0].numpy(); seq = batch['sequence'][0]
                    for mode in sums:
                        metrics = evaluate_sample(p, g, align=mode, min_depth=.001, max_depth=80.)
                        row = {k: float(metrics[k]) for k in OFFICIAL_METRICS}
                        writer.writerow(dict(id=batch['id'][0], sequence=seq, alignment=mode, **row))
                        bucket = sequence_sums.setdefault((mode, seq), dict(frames=0, **{k: 0. for k in OFFICIAL_METRICS}))
                        bucket['frames'] += 1
                        for k in row:
                            sums[mode][k] += row[k]; bucket[k] += row[k]
                        aligned = p.astype(np.float64)
                        if mode == 'ssi':
                            aligned = aligned * metrics['alignment_scale'] + metrics['alignment_shift']
                        aligned = np.clip(aligned, .001, 80.)
                        for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 80)):
                            mask = np.isfinite(g) & (g > .001) & (g >= lo) & (g < hi)
                            if not mask.any():
                                continue
                            error = aligned[mask] - g[mask]
                            stats = dict(abs_rel=float(np.mean(np.abs(error) / g[mask])),
                                         rmse=float(np.sqrt(np.mean(error ** 2))),
                                         a1=float(np.mean(np.maximum(aligned[mask] / g[mask], g[mask] / aligned[mask]) < 1.25)))
                            bin_bucket = bin_sums.setdefault((mode, f'{lo}-{hi}m'), dict(frames=0, abs_rel=0., rmse=0., a1=0.))
                            bin_bucket['frames'] += 1
                            for k, v in stats.items(): bin_bucket[k] += v
                    if (i + 1) % 2000 == 0:
                        print(f'[val] {i+1}/{len(dataset)}', flush=True)
        report = dict(step=step, samples=len(dataset), metrics={m: {k: v / len(dataset) for k, v in d.items()} for m, d in sums.items()},
                      sequences={m: {seq: {k: (v if k == 'frames' else v / d['frames']) for k, v in d.items()}
                                     for (mode, seq), d in sequence_sums.items() if mode == m} for m in sums})
        report['depth_bins'] = {m: {band: {k: (v if k == 'frames' else v / d['frames']) for k, v in d.items()}
                                    for (mode, band), d in bin_sums.items() if mode == m} for m in sums}
        json_write(out / f'val_step{step:06d}.json', report)
        print('[val-result] ' + json.dumps(report), flush=True)
    finally:
        model.train(was_training); restore_rng(saved_rng)


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError('Actual training requires CUDA; use CPU unit tests locally')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('Frozen recipe requires BF16-capable GPU')
    device = torch.device('cuda')
    assets = verify_assets(args.assets)
    train, train_info = checked_manifest(args.train_manifest, args.data_root, 'train', 75688)
    val, val_info = checked_manifest(args.val_manifest, args.data_root, 'val', 14249)
    if {s.sample_id for s in train} & {s.sample_id for s in val}:
        raise ValueError('Train/val overlap')
    files = ['models/da2_caption.py', 'tools/da2_caption_data.py', 'tools/train_da2_caption.py',
             'models/anythermal_encoder.py', 'ms2_eval/official_protocol.py', 'ms2_eval/io.py']
    contract = dict(arm=args.arm, seed=args.seed, total_steps=TOTAL_STEPS, batch=MICROBATCH, accumulation=ACCUMULATION,
                    train=train_info, val=val_info, assets=sha256_file(args.assets),
                    caption_source=args.caption_source, loss='per-frame masked log-L1', alignment='ssi',
                    code={f: sha256_file(ROOT / f) for f in files},
                    software=dict(torch=torch.__version__, numpy=np.__version__), amp='bf16',
                    caption_train_hash=caption_hash(train), caption_val_hash=caption_hash(val))
    if args.mode == 'train':
        if args.smoke_contract is None:
            raise ValueError('Pilot requires --smoke-contract from the successful paired smoke')
        previous = json.loads(args.smoke_contract.read_text(encoding='utf-8'))
        if {k: v for k, v in previous.items() if k != 'arm'} != {k: v for k, v in contract.items() if k != 'arm'}:
            raise ValueError('Current code/assets/data differ from smoke contract; run a fresh smoke')
    out = args.output.resolve()
    if args.resume:
        if not (out / 'last.pt').is_file():
            raise FileNotFoundError('resume requires output/last.pt')
    elif out.exists():
        raise FileExistsError(f'Refusing existing fresh output: {out}')
    else:
        out.mkdir(parents=True)
    if args.caption_source != 'thermal':
        print('[provenance] RGB-derived captions: this is not thermal-only inference', flush=True)
    tokens = (None, None)
    if args.arm == 'caption':
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained(assets['text_path'], subfolder='tokenizer', local_files_only=True)
        tokens = (token_cache(train, tokenizer, assets, train_info['sha256'], args.mode == 'smoke'),
                  token_cache(val, tokenizer, assets, val_info['sha256'], args.mode == 'smoke'))
    training = ThermalLidar(train, tokens[0]); validation = ThermalLidar(val, tokens[1])
    # Train-only preprocessing sanity audit; no use of test baseline ranges.
    audit = []
    for i in np.linspace(0, len(training)-1, 8, dtype=int):
        sample = training[int(i)]
        valid = torch.isfinite(sample['gt']) & (sample['gt'] > .001) & (sample['gt'] < 80)
        audit.append(dict(id=sample['id'], shape=list(sample['gt'].shape),
                          thermal_min_max_std=sample['thermal_stats'].tolist(), lidar_pixels=int(valid.sum())))
    json_write(out / 'input_audit.json', audit)
    model = build_model(assets, args.arm, args.seed).to(device)
    optimizer, scheduler = make_optimizer(model)
    step = load_checkpoint(out / 'last.pt', model, optimizer, scheduler, contract) if args.resume else 0
    if args.mode == 'smoke' and args.resume:
        raise ValueError('Smoke must be fresh')
    stop = 32 if args.mode == 'smoke' else args.stop_step
    if not step <= stop <= TOTAL_STEPS:
        raise ValueError(f'Need checkpoint step {step} <= stop {stop} <= 20000')
    json_write(out / 'contract.json', contract)
    # A time limit may interrupt validation after its checkpoint was saved.
    # Repair only missing validation outputs, then restore the latest training state.
    if args.resume and args.mode == 'train':
        for milestone in (6000, 8000):
            if step >= milestone and not (out / f'val_step{milestone:06d}.json').exists():
                snapshot = out / f'step{milestone:06d}.pt'
                if not snapshot.exists() and step == milestone:
                    snapshot = out / 'last.pt'
                load_checkpoint(snapshot, model, None, None, contract, False)
                evaluate(model, validation, device, True, out, milestone)
        load_checkpoint(out / 'last.pt', model, optimizer, scheduler, contract)
    # Exact zero-gate sanity BEFORE training, FP32; do not compare autocast rounding paths.
    first = next(iter(DataLoader(training, batch_size=1, generator=torch.Generator().manual_seed(0))))
    model.eval()
    with torch.no_grad():
        p = predict(model, first, device, False)
        if args.arm == 'caption' and step == 0:
            image = first['image'].to(device); h, w = image.shape[-2:]
            padded = torch.nn.functional.pad(image, (0, (-w)%14, 0, (-h)%14), mode='reflect')
            base = model.core(padded)[..., :h, :w]
            torch.testing.assert_close(p, base, rtol=0., atol=0.)
            print('[gate] zero adapters reproduce metric DA V2 exactly', flush=True)
        if p.shape != first['gt'].shape or not torch.isfinite(p).all():
            raise ValueError('Invalid initial prediction')
    print(f'[config] {args.arm}, fresh={not args.resume}, steps={step}->{stop}/20000, batch4 x accumulate3; NO pseudo GT', flush=True)
    print(f'[parameters] trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}', flush=True)
    model.train()
    stream = ImageStream(len(training), args.seed, step * MICROBATCH * ACCUMULATION,
                         (stop - step) * MICROBATCH * ACCUMULATION)
    loader = DataLoader(training, batch_size=MICROBATCH, sampler=stream, num_workers=args.workers,
                        pin_memory=True, generator=torch.Generator().manual_seed(args.seed + 1))
    iterator = iter(loader)
    # Loader/prediction sanity must not perturb saved training RNG when resuming.
    if args.resume:
        state = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        restore_rng(state['rng']); del state
    start_time, start_step = time.monotonic(), step
    with (out / 'train.jsonl').open('a', encoding='utf-8') as log:
        while step < stop:
            optimizer.zero_grad(set_to_none=True)
            total, empty_frames, ids = 0., 0, []
            for _ in range(ACCUMULATION):
                batch = next(iterator); ids.extend(batch['id'])
                prediction = predict(model, batch, device, True)
                loss, empty = masked_log_l1(prediction, batch['gt'].to(device))
                if not torch.isfinite(loss):
                    raise ValueError(f'Nonfinite loss on {ids}')
                (loss / ACCUMULATION).backward()
                total += float(loss.detach()) / ACCUMULATION; empty_frames += empty
            checks = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if not checks or not torch.stack(checks).all():
                raise ValueError('Missing or nonfinite gradients')
            if model.text_encoder is not None and any(p.grad is not None for p in model.text_encoder.parameters()):
                raise ValueError('Frozen text encoder unexpectedly received gradients')
            if args.mode == 'smoke' and args.arm == 'caption' and step == 0:
                if not any(a.alpha.grad is not None and a.alpha.grad.abs() > 0 for a in model.adapters):
                    raise ValueError('Caption gate receives no gradient')
            optimizer.step(); scheduler.step(); step += 1
            record = dict(step=step, loss=total, empty_gt_frames=empty_frames,
                          image_cursor=step * MICROBATCH * ACCUMULATION,
                          lr=[g['lr'] for g in optimizer.param_groups],
                          alpha=[float(a.alpha.detach()) for a in model.adapters],
                          data_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest())
            log.write(json.dumps(record) + '\n'); log.flush()
            if step % 20 == 0 or step == stop:
                print('[train] ' + json.dumps(record), flush=True)
            if step % 1000 == 0 or step == stop:
                save_checkpoint(out / 'last.pt', model, optimizer, scheduler, step, contract)
            if args.mode != 'smoke' and step % 2000 == 0:
                save_checkpoint(out / f'step{step:06d}.pt', model, optimizer, scheduler, step, contract)
            if args.mode != 'smoke' and step in (6000, 8000):
                evaluate(model, validation, device, True, out, step)
    elapsed = time.monotonic() - start_time
    report = dict(mode=args.mode, arm=args.arm, step=step, elapsed_seconds=elapsed,
                  seconds_per_update_including_eval=elapsed / (step-start_step) if step > start_step else None,
                  max_gpu_allocated_gb=torch.cuda.max_memory_allocated() / 2**30)
    if args.mode == 'smoke':
        report['projected_8000_train_hours_excluding_val'] = elapsed / (step-start_step) * 8000 / 3600
        if args.arm == 'caption' and not any(abs(float(a.alpha.detach())) > 0 for a in model.adapters):
            raise ValueError('Caption adapters never activated')
    json_write(out / 'completed.json', report)
    print('[completed] ' + json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['smoke', 'train'], required=True)
    p.add_argument('--arm', choices=['no-caption', 'caption'], required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--train-manifest', type=Path, required=True)
    p.add_argument('--val-manifest', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--caption-source', choices=['thermal', 'rgb'], required=True)
    p.add_argument('--stop-step', type=int, choices=[8000, 20000], default=8000)
    p.add_argument('--seed', type=int, choices=[43, 44], default=43)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke-contract', type=Path)
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()
