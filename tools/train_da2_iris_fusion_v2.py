"""Matched Iris weight/text arms, plus selected-caption-model permutation validation."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from models.da2_iris_fusion import FusionDepth, IrisFeatures
from models.da2_caption import masked_log_l1
from tools.da2_caption_data import ThermalLidar, ImageStream, checked_manifest, caption_hash
from tools.train_da2_caption import (build_model, verify_assets, token_cache, json_write,
                                    seed_all, rng_state, restore_rng)
from tools.da2_continuation import validate_state, check_source, select_step
from tools.train_sd21_dual_pair import prepare_assets, verify_assets as verify_iris_assets
from ms2_eval.io import sha256_file
from ms2_eval.official_protocol import evaluate_sample, OFFICIAL_METRICS

STEPS, BATCH, ACCUM, SMOKE_STEPS = 20000, 4, 3, 32
MILESTONES = tuple(range(2000, STEPS+1, 2000))
EXPERIMENT = 'da2_iris_fusion_matched_pair_v2'
IRIS_RUNS = dict(frame='sd2_cap_pct_logtn_s43', empty='sd2_nocap_pct_logtn_s43')
PERMUTATION_SEED = 43
CODE = ('models/da2_iris_fusion.py', 'tools/train_da2_iris_fusion_v2.py',
        'tests/test_da2_iris_fusion_v2.py', 'slurm/da2_iris_fusion_v2.sbatch', 'slurm/submit_da2_iris_fusion_v2.sh',
        'models/da2_caption.py', 'tools/da2_caption_data.py', 'tools/train_da2_caption.py',
        'tools/da2_continuation.py', 'models/anythermal_encoder.py', 'ms2_eval/io.py',
        'ms2_eval/official_protocol.py', 'tools/train_sd21_dual_pair.py', 'models/sd21_dual.py',
        'models/sd21_tadp.py', 'models/da2_silog.py', 'tools/sd21_dual_data.py', 'tools/sd21_pseudo_data.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def tensor_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def validate_iris_payload(payload, arm):
    mode = {'frame': 'correct', 'empty': 'empty'}[arm]
    if (payload.get('route') != 'b_thermal_unet' or payload.get('global_step') != 18000
            or payload.get('caption_mode') != mode or set(payload.get('state_dicts', {})) != {'unet'}):
        raise ValueError(f'{arm}: requires converted {mode}-trained Iris-G step18000')


def load_iris(path, checkpoint, arm):
    from diffusers import UNet2DConditionModel, AutoencoderKL
    from transformers import CLIPTextModel
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    validate_iris_payload(payload, arm)
    config = UNet2DConditionModel.load_config(path, subfolder='unet', local_files_only=True)
    config.update(in_channels=8, class_embed_type='projection', projection_class_embeddings_input_dim=4)
    unet = UNet2DConditionModel.from_config(config)
    unet.load_state_dict(payload['state_dicts']['unet'], strict=True)
    if unet.config.cross_attention_dim != 1024: raise ValueError('Expected SD2.1 text width')
    del payload
    vae = AutoencoderKL.from_pretrained(path, subfolder='vae', local_files_only=True, use_safetensors=True)
    text = CLIPTextModel.from_pretrained(path, subfolder='text_encoder', local_files_only=True)
    return IrisFeatures(unet, vae, text)


def matched_sources(cap_path, empty_path):
    sources = {}
    for arm, path in (('frame', cap_path), ('empty', empty_path)):
        if (path.name != 'step18000_weights.pt' or path.parent.name != 'converted'
                or path.parent.parent.name != IRIS_RUNS[arm]):
            raise ValueError(f'{arm}: requires {IRIS_RUNS[arm]}/converted/step18000_weights.pt')
        sources[arm] = dict(path=str(path.resolve()), sha256=sha256_file(path), run=IRIS_RUNS[arm],
                            step=18000, trained_prompt='correct' if arm=='frame' else 'empty')
    if sources['frame']['sha256'] == sources['empty']['sha256']:
        raise ValueError('Matched arms must not use the same Iris checkpoint')
    return sources


def permutation_tokens(samples, tokens, seed=PERMUTATION_SEED):
    """Uniform random permutation conditioned on no self-pairs; no cyclic shift.

    Caption/token duplicates remain and are explicitly reported, not silently
    removed (which would change the corpus marginal distribution).
    """
    n = len(samples)
    if n < 2 or len({s.sample_id for s in samples}) != n:
        raise ValueError('Permutation needs >=2 unique validation frames')
    if tokens['input_ids'].shape != (n,77) or tokens['attention_mask'].shape != (n,77):
        raise ValueError('Permutation expects full validation token rows')
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)): break
    else: raise RuntimeError('Could not construct a derangement')
    order = torch.from_numpy(perm.copy())
    result = {key: tokens[key][order].clone() for key in ('input_ids','attention_mask')}
    pairs = [[s.sample_id, samples[int(j)].sample_id] for s,j in zip(samples,perm)]
    info = dict(seed=seed, algorithm='numpy.default_rng.permutation-rejection-no-self-v1', samples=n,
                self_pairs=0, identical_text_count=sum(s.caption==samples[int(j)].caption for s,j in zip(samples,perm)),
                identical_token_count=int((tokens['input_ids']==result['input_ids']).all(1).sum()),
                source_tokens_sha256=digest(tokens['input_ids'].contiguous().numpy().tobytes()),
                permuted_tokens_sha256=digest(result['input_ids'].contiguous().numpy().tobytes()),
                mapping_sha256=digest(json.dumps(pairs,ensure_ascii=False,separators=(',',':')).encode()))
    return result, info, pairs


def content_validation(model, validation, permuted, out, step, correct, info):
    """Only after checkpoint selection. Same image IDs/noise/weights, replace text."""
    if correct['step'] != step or correct['samples'] != len(validation):
        raise ValueError('Content comparison must use the same checkpoint/full frame set')
    before = tensor_hash(model.fusions)
    print('[condition] selected-caption: permuted validation start', flush=True)
    changed = FusionData(validation.samples, permuted)
    result = evaluate(model, changed, out/'permuted_selected', step)
    if tensor_hash(model.fusions) != before: raise ValueError('Content evaluation modified model')
    deltas = {m: {k: result['metrics'][m][k]-correct['metrics'][m][k] for k in OFFICIAL_METRICS}
              for m in ('ssi','none')}
    report = dict(step=step, fusion_state_sha256=before, permutation=info, correct=correct,
                  permuted=result, delta_permuted_minus_correct=deltas,
                  note='Post-selection diagnostic; never used for checkpoint selection. Positive error delta favors correct.')
    json_write(out/'caption_content.json', report)
    print('[caption-content] '+json.dumps(dict(step=step,permutation=info,deltas=deltas)), flush=True)
    return report


class FusionData(ThermalLidar):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        if item['image'].shape != (3, 256, 640):
            raise ValueError('Expected native MS2 256x640, no resizing of sparse GT')
        # Invert only the known normalization, quantize back to original uint8.
        # Both branches therefore consume exactly the same percentile-converted image.
        mean = torch.tensor([.485, .456, .406])[:, None, None]
        std = torch.tensor([.229, .224, .225])[:, None, None]
        gray = ((item['image'] * std + mean)*255).round().clamp(0, 255)
        item['iris_image'] = gray / 127.5 - 1.
        return item


def make_tokens(samples, tokenizer, assets, info, arm, cache, create):
    if arm == 'frame':
        return token_cache(samples, tokenizer, dict(root=str(cache), text_fingerprint=assets['fingerprint']), info['sha256'], create)
    if arm != 'empty':
        raise ValueError('Only frame/empty arms')
    e = tokenizer([''], padding='max_length', max_length=77, truncation=True, return_tensors='pt')
    return {k: e[k].expand(len(samples), -1) for k in ('input_ids', 'attention_mask')}


def optimizer_for(model):
    params = list(model.fusions.parameters())
    if {id(p) for p in params} != {id(p) for p in model.parameters() if p.requires_grad}:
        raise ValueError('Only fusion parameters may train')
    opt = torch.optim.AdamW(params, lr=5e-5, betas=(.9, .999), eps=1e-8, weight_decay=.01)
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0., 1-s/STEPS)**.9)


def save(path, model, opt, sched, step, contract, resumable=True):
    state = dict(fusions=model.fusions.state_dict(), step=step, image_cursor=step*BATCH*ACCUM, contract=contract)
    if resumable:
        state.update(optimizer=opt.state_dict(), scheduler=sched.state_dict(), rng=rng_state())
    tmp = Path(str(path)+'.tmp'); torch.save(state, tmp); os.replace(tmp, path)


def restore(path, model, contract, opt=None, sched=None):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['contract'] != contract or state['image_cursor'] != state['step']*BATCH*ACCUM:
        raise ValueError('Checkpoint contract/cursor mismatch')
    model.fusions.load_state_dict(state['fusions'], strict=True)
    if opt is not None:
        if state['scheduler']['last_epoch'] != state['step']:
            raise ValueError('Scheduler cursor mismatch')
        opt.load_state_dict(state['optimizer']); sched.load_state_dict(state['scheduler']); restore_rng(state['rng'])
    return state['step']


def repair_log(path, step):
    rows = [json.loads(s) for s in path.read_text().splitlines()][:step]
    if [r['step'] for r in rows] != list(range(1, step+1)):
        raise ValueError('Resume log gap')
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(''.join(json.dumps(r)+'\n' for r in rows)); os.replace(tmp, path)


def pair_smoke(directory):
    contracts, logs, completed = [], [], []
    for arm in ('frame', 'empty'):
        d = directory/arm
        c, done = read(d/'contract.json'), read(d/'completed.json')
        if c['arm'] != arm or done['mode'] != 'smoke' or done['step'] != SMOKE_STEPS or not done['smoke_checks_passed']:
            raise ValueError('Incomplete successful paired smoke')
        if c.get('experiment') != EXPERIMENT:
            raise ValueError('Old fusion smoke is not valid for matched v2')
        source = c['iris_sources'][arm]
        if source['run'] != IRIS_RUNS[arm] or source['trained_prompt'] != ('correct' if arm=='frame' else 'empty'):
            raise ValueError('Iris weight/text pairing is wrong')
        if c['iris_sources']['frame']['sha256'] == c['iris_sources']['empty']['sha256']:
            raise ValueError('The two old Iris checkpoints must differ')
        contracts.append({k: v for k, v in c.items() if k not in ('arm','initial_iris_sha256')})
        completed.append(done)
        rows = [json.loads(s) for s in (d/'train.jsonl').read_text().splitlines()]
        if [r['step'] for r in rows] != list(range(1, SMOKE_STEPS+1)):
            raise ValueError('Incomplete smoke log')
        logs.append(rows)
    if contracts[0] != contracts[1]:
        raise ValueError('Paired recipes/initialization differ')
    if read(directory/'frame/contract.json')['initial_iris_sha256'] == read(directory/'empty/contract.json')['initial_iris_sha256']:
        raise ValueError('Matched Iris parameter states unexpectedly identical')
    for key in ('initial_prediction_sha256', 'initial_baseline_sha256'):
        if completed[0][key] != completed[1][key]:
            raise ValueError('Initial paired predictions differ')
    for a, b in zip(*logs):
        for key in ('data_ids_sha256', 'input_sha256', 'noise_sha256'):
            if not a[key] or a[key] != b[key]:
                raise ValueError(f'Paired {key} differs')
    print('[paired-smoke] PASS: matched distinct Iris weights; same DA2/fusion/images/noise; exact initial B0; frozen weights unchanged', flush=True)


def forward_batch(model, batch, trace=False, amp=True):
    device = next(model.parameters()).device
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        return model(batch['image'].to(device), batch['iris_image'].to(device),
                     batch['token_ids'].to(device), batch['attention_mask'].to(device), batch['id'], trace=trace)


def evaluate(model, dataset, out, step):
    out.mkdir(parents=True, exist_ok=True)
    state, was_training = rng_state(), model.training
    model.eval(); sums = {m: dict.fromkeys(OFFICIAL_METRICS, 0.) for m in ('ssi', 'none')}; sequences = {}
    try:
        loader = DataLoader(dataset, batch_size=1, num_workers=0, generator=torch.Generator().manual_seed(0))
        with torch.no_grad(), (out/f'val_step{step:06d}_frames.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['id', 'sequence', 'alignment', *OFFICIAL_METRICS]); writer.writeheader()
            for i, batch in enumerate(loader):
                p = forward_batch(model, batch)[0].float().cpu().numpy(); gt = batch['gt'][0].numpy()
                if p.shape != gt.shape or not np.isfinite(p).all():
                    raise ValueError('Invalid prediction')
                seq = batch['sequence'][0]
                for mode in sums:
                    metrics = evaluate_sample(p, gt, align=mode, min_depth=.001, max_depth=80.)
                    row = {k: float(metrics[k]) for k in OFFICIAL_METRICS}
                    if not all(np.isfinite(v) for v in row.values()): raise ValueError('Nonfinite metric')
                    bucket = sequences.setdefault((mode, seq), dict(frames=0, **dict.fromkeys(OFFICIAL_METRICS, 0.)))
                    bucket['frames'] += 1
                    for k, v in row.items(): sums[mode][k] += v; bucket[k] += v
                    writer.writerow(dict(id=batch['id'][0], sequence=seq, alignment=mode, **row))
                if (i+1) % 2000 == 0: print(f'[val] {i+1}/{len(dataset)}', flush=True)
        result = dict(step=step, samples=len(dataset), metrics={m: {k: v/len(dataset) for k, v in d.items()} for m, d in sums.items()},
                      sequences={m: {seq: {k: (v if k == 'frames' else v/d['frames']) for k, v in d.items()}
                                     for (mode, seq), d in sequences.items() if mode == m} for m in sums})
        json_write(out/f'val_step{step:06d}.json', result)
        print('[val-result] '+json.dumps(result), flush=True)
        return result
    finally:
        model.train(was_training); restore_rng(state)


def run(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Requires BF16 GPU; run unit tests on CPU separately')
    assets = verify_assets(args.assets)
    iris_assets = prepare_assets(args.iris_assets) if args.mode == 'smoke' else verify_iris_assets(args.iris_assets)
    sources = matched_sources(args.iris_caption_checkpoint, args.iris_empty_checkpoint)
    state = torch.load(args.base_checkpoint, map_location='cpu', weights_only=False)
    original = state['contract']; validate_state(state, original, 18000); check_source(original)
    if original['arm'] != 'no-caption' or original['seed'] != 43 or sha256_file(args.assets) != original['assets']:
        raise ValueError('Wrong DA2 B0/assets')
    train, ti = checked_manifest(args.train_manifest, args.data_root, 'train', 75688)
    val, vi = checked_manifest(args.val_manifest, args.data_root, 'val', 14249)
    if (ti != original['train'] or vi != original['val'] or caption_hash(train) != original['caption_train_hash']
            or caption_hash(val) != original['caption_val_hash']):
        raise ValueError('Data/caption provenance differs from B0')
    seed_all(43)
    base = build_model(assets, 'no-caption', 43); base.load_state_dict(state['model'], strict=True); del state
    iris = load_iris(iris_assets['path'], sources[args.arm]['path'], args.arm)
    model = FusionDepth(base.core, iris)
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(iris_assets['path'], subfolder='tokenizer', local_files_only=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    tokens = [make_tokens(s, tokenizer, iris_assets, info, args.arm, args.cache_dir, args.mode=='smoke')
              for s, info in ((train, ti), (val, vi))]
    full_val_tokens = make_tokens(val, tokenizer, iris_assets, vi, 'frame', args.cache_dir, args.mode=='smoke')
    permuted, permutation, mapping = permutation_tokens(val, full_val_tokens)
    contract = dict(experiment=EXPERIMENT, arm=args.arm, seed=43,
        base_path=str(args.base_checkpoint.resolve()), base_sha256=sha256_file(args.base_checkpoint),
        iris_sources=sources, permutation=permutation,
        assets_sha256=sha256_file(args.assets), iris_assets_sha256=sha256_file(args.iris_assets),
        train=ti, val=vi, train_caption_hash=caption_hash(train), val_caption_hash=caption_hash(val),
        steps=STEPS, batch=BATCH, accumulation=ACCUM, loss='per_frame_masked_log_l1', lr=5e-5,
        weight_decay=.01, clip_grad=1., precision='FP32 frozen weights; BF16 U-Net/DA2/fusion; FP32 VAE/CLIP/loss',
        padding=77, explicit_text_mask=False, t=999, image_latent='mode', noise='sha256(frame_id,43),fixed',
        initial_fusion_sha256=tensor_hash(model.fusions), initial_da2_sha256=tensor_hash(model.core),
        initial_iris_sha256=tensor_hash(model.iris),
        code={f: sha256_file(ROOT/f) for f in CODE},
        software={p: importlib.metadata.version(p) for p in ('torch','numpy','diffusers','transformers')})
    if args.mode == 'train':
        if args.smoke_output is None: raise ValueError('Require successful paired smoke')
        pair_smoke(args.smoke_output)
        if read(args.smoke_output/args.arm/'contract.json') != contract: raise ValueError('Recipe changed after smoke')
    out = args.output
    if args.resume:
        if args.mode != 'train' or not (out/'last.pt').is_file() or read(out/'contract.json') != contract:
            raise ValueError('Explicit resume requires matching formal checkpoint')
    else:
        out.mkdir(parents=True, exist_ok=False); json_write(out/'contract.json', contract)
    json_write(out/'permutation.json', dict(**permutation, pairs=mapping))
    training, validation = FusionData(train, tokens[0]), FusionData(val, tokens[1])
    curve_data = Subset(validation, list(range(0, len(validation), 4)))
    print('[config] '+json.dumps(contract), flush=True)
    model.to('cuda'); opt, sched = optimizer_for(model)
    step = restore(out/'last.pt', model, contract, opt, sched) if args.resume else 0
    stop = SMOKE_STEPS if args.mode == 'smoke' else STEPS
    if not 0 <= step <= stop: raise ValueError('Invalid resume cursor')
    initial_pred_hash = initial_base_hash = None
    if step == 0:
        batch = next(iter(DataLoader(Subset(training, [0,1,2,3]), batch_size=4)))
        print('[thermal] '+str(batch['thermal_stats'].tolist()), flush=True)
        model.eval()
        with torch.no_grad():
            for amp in (False, True):
                prediction = forward_batch(model, batch, amp=amp)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                    expected = model.baseline(batch['image'].cuda())
                torch.testing.assert_close(prediction, expected, rtol=0, atol=0)
                if not torch.isfinite(prediction).all() or prediction.shape != batch['gt'].shape:
                    raise ValueError('Invalid initial depth')
            initial_pred_hash = digest(prediction.float().cpu().numpy().tobytes())
            initial_base_hash = digest(expected.float().cpu().numpy().tobytes())
        print('[reproduction] PASS: exact B0 output on four frames, FP32 and BF16', flush=True)
    if args.resume:
        for s in MILESTONES:
            if s <= step and not (out/'curve'/f'val_step{s:06d}.json').exists():
                restore(out/f'step{s:06d}.pt', model, contract); evaluate(model, curve_data, out/'curve', s)
        restore(out/'last.pt', model, contract, opt, sched); repair_log(out/'train.jsonl', step)
    loader = DataLoader(training, batch_size=BATCH,
        sampler=ImageStream(len(training), 43, step*BATCH*ACCUM, (stop-step)*BATCH*ACCUM),
        num_workers=args.workers, pin_memory=True, generator=torch.Generator().manual_seed(44))
    iterator = iter(loader)
    if not args.resume: seed_all(100043)
    model.train(); started = time.monotonic(); start_step = step; project_grad = [False, False]
    with (out/'train.jsonl').open('a') as log:
        while step < stop:
            opt.zero_grad(set_to_none=True); losses=0.; empty=0; ids=[]; noises=[]; input_hash=hashlib.sha256()
            for _ in range(ACCUM):
                batch = next(iterator); ids.extend(batch['id'])
                prediction = forward_batch(model, batch, trace=args.mode=='smoke')
                if not torch.isfinite(prediction).all(): raise ValueError('Nonfinite depth')
                loss, nempty = masked_log_l1(prediction, batch['gt'].cuda())
                (loss/ACCUM).backward(); losses+=float(loss.detach())/ACCUM; empty+=nempty
                if args.mode == 'smoke':
                    noises.append(model.iris.last_noise_hash)
                    for key in ('image','iris_image','gt'): input_hash.update(batch[key].contiguous().numpy().tobytes())
            if any(p.grad is not None for m in (model.core, model.iris) for p in m.parameters()):
                raise ValueError('Frozen module received gradients')
            for i, fusion in enumerate(model.fusions):
                g = fusion.project[0].weight.grad
                project_grad[i] |= g is not None and bool(g.abs().sum()>0)
            norm = torch.nn.utils.clip_grad_norm_(model.fusions.parameters(), 1., error_if_nonfinite=True)
            if not any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.fusions.parameters()):
                raise ValueError('No fusion gradient')
            opt.step(); sched.step(); step+=1
            row = dict(step=step, image_cursor=step*BATCH*ACCUM, loss=losses, empty_gt_frames=empty,
                       grad_norm=float(norm), lr=opt.param_groups[0]['lr'], data_ids_sha256=digest('\n'.join(ids).encode()),
                       input_sha256=input_hash.hexdigest() if args.mode=='smoke' else None,
                       noise_sha256=digest('\n'.join(noises).encode()) if args.mode=='smoke' else None)
            log.write(json.dumps(row)+'\n'); log.flush()
            if step%20==0 or step==stop: print('[train] '+json.dumps(row), flush=True)
            if args.mode == 'train':
                if step in MILESTONES: save(out/f'step{step:06d}.pt', model, opt, sched, step, contract, False)
                if step%1000==0: save(out/'last.pt', model, opt, sched, step, contract)
                if step in MILESTONES: evaluate(model, curve_data, out/'curve', step)
    frozen_ok = (tensor_hash(model.core)==contract['initial_da2_sha256'] and tensor_hash(model.iris)==contract['initial_iris_sha256'])
    if not frozen_ok: raise ValueError('Frozen weights changed')
    smoke_ok = None
    if args.mode == 'smoke':
        smoke_ok = all(project_grad) and tensor_hash(model.fusions)!=contract['initial_fusion_sha256']
        if not smoke_ok: raise ValueError('Fusion projection gradient/update gate failed')
        evaluate(model, Subset(validation, np.linspace(0, len(validation)-1, 8, dtype=int).tolist()), out/'validation', step)
        if args.arm == 'frame':
            evaluate(model, Subset(FusionData(val,permuted), np.linspace(0,len(val)-1,8,dtype=int).tolist()), out/'permutation_smoke', step)
    else:
        fixed = evaluate(model, validation, out/'fixed20000', STEPS)
        curve = {s: read(out/'curve'/f'val_step{s:06d}.json') for s in MILESTONES}
        chosen = select_step(curve)
        selected = fixed
        if chosen != STEPS:
            restore(out/f'step{chosen:06d}.pt', model, contract)
            selected = evaluate(model, validation, out/'selected_full', chosen)
        content = None
        if args.arm == 'frame':
            content = content_validation(model, validation, permuted, out, chosen, selected, permutation)
            content['checkpoint_sha256'] = sha256_file(out/f'step{chosen:06d}.pt')
            json_write(out/'caption_content.json', content)
        json_write(out/'summary.json', dict(selected_step=chosen, curve=curve, fixed20000=fixed,
                                            selected_full=selected, caption_content=content, contract=contract))
        print(f'[selected] step={chosen}; validation only', flush=True)
    elapsed = time.monotonic()-started
    json_write(out/'completed.json', dict(mode=args.mode, arm=args.arm, step=step, frozen_unchanged=frozen_ok,
        smoke_checks_passed=smoke_ok, initial_prediction_sha256=initial_pred_hash, initial_baseline_sha256=initial_base_hash,
        seconds_including_validation=elapsed, seconds_per_update=elapsed/max(1,step-start_step),
        max_gpu_gb=torch.cuda.max_memory_allocated()/2**30))
    print(f'[completed] {out}; no test evaluation', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['smoke','train'], required=True)
    p.add_argument('--arm', choices=['frame','empty'], required=True)
    for name in ('assets','iris-assets','base-checkpoint','iris-caption-checkpoint','iris-empty-checkpoint','train-manifest','val-manifest','data-root','cache-dir','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--smoke-output', type=Path); p.add_argument('--resume', action='store_true')
    p.add_argument('--workers', type=int, default=4)
    run(p.parse_args())
