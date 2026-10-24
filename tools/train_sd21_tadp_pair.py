"""Independent SD2.1 direct-depth pair: frame caption versus tokenized empty.

No pseudo GT, no teacher, no old Lotus checkpoint, no test-set mode.
"""
import argparse
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
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from models.sd21_tadp import load_sd21
from models.da2_silog import masked_silog
from models.anythermal_encoder import AnyThermalEncoder
from tools.da2_caption_data import checked_manifest, ImageStream, caption_hash
from tools.train_da2_caption import (json_write, seed_all, rng_state, restore_rng, token_cache, evaluate)
from ms2_eval.io import load_ms2_gt, sha256_file

STEPS = 20000
BATCH, ACCUM = 4, 3
MILESTONES = tuple(range(2000, STEPS+1, 2000))
SD21_UNET_SHA = '6dfae3e5f7d459b50f4b0850ead945972c75bb0e1897628933e169eb43974214'
CODE = ('models/sd21_tadp.py', 'tools/train_sd21_tadp_pair.py', 'models/da2_silog.py',
        'tools/da2_caption_data.py', 'tools/train_da2_caption.py', 'models/anythermal_encoder.py',
        'models/da2_caption.py', 'models/da2_decoder.py', 'ms2_eval/official_protocol.py', 'ms2_eval/io.py',
        'slurm/sd21_tadp_pair.sbatch', 'slurm/submit_sd21_tadp_pair.sh', 'tests/test_sd21_tadp_pair.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def tensor_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_assets(descriptor):
    assets = read(descriptor)
    for name, digest in assets['hashes'].items():
        if sha256_file(Path(assets['path']) / name) != digest:
            raise ValueError(f'SD2 asset changed: {name}')
    if assets['hashes']['unet/diffusion_pytorch_model.safetensors'] != SD21_UNET_SHA:
        raise ValueError('Not the audited original SD2.1-base U-Net')
    return assets


def prepare_assets(descriptor):
    """Only resolve existing HF cache; never download on GPU or change old assets."""
    if descriptor.exists(): return verify_assets(descriptor)
    from huggingface_hub import snapshot_download
    path = Path(snapshot_download('Manojb/stable-diffusion-2-1-base', local_files_only=True))
    hashes = {}
    for folder in ('unet', 'vae', 'text_encoder', 'tokenizer'):
        files = sorted((path/folder).glob('*'))
        if not files: raise FileNotFoundError(f'Missing cached {folder}')
        for file in files:
            if file.is_file(): hashes[file.relative_to(path).as_posix()] = sha256_file(file)
    for name in ('unet/diffusion_pytorch_model.safetensors', 'vae/diffusion_pytorch_model.safetensors',
                 'unet/config.json', 'vae/config.json', 'text_encoder/config.json',
                 'tokenizer/vocab.json', 'tokenizer/merges.txt'):
        if name not in hashes: raise FileNotFoundError(f'Missing cached asset: {name}')
    if hashes['unet/diffusion_pytorch_model.safetensors'] != SD21_UNET_SHA:
        raise ValueError('Original SD2.1-base U-Net hash mismatch')
    c = read(path/'text_encoder/config.json')
    if c['hidden_size'] != 1024 or c['max_position_embeddings'] != 77:
        raise ValueError('Wrong text encoder config')
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    json_write(descriptor, dict(path=str(path), hashes=hashes, fingerprint=fingerprint,
                               repo='Manojb/stable-diffusion-2-1-base', snapshot=path.name))
    return verify_assets(descriptor)


def make_tokens(samples, tokenizer, assets, manifest_hash, arm, cache, create):
    if arm == 'frame':
        return token_cache(samples, tokenizer, dict(root=str(cache), text_fingerprint=assets['fingerprint']),
                           manifest_hash, create)
    if arm != 'empty': raise ValueError('Only frame/empty arms')
    t = tokenizer([''], padding='max_length', max_length=77, truncation=True, return_tensors='pt')
    return dict(input_ids=t['input_ids'].expand(len(samples), -1),
                attention_mask=t['attention_mask'].bool().expand(len(samples), -1))


class ThermalDirect(Dataset):
    def __init__(self, samples, tokens):
        self.samples, self.tokens = samples, tokens
        os.environ['IRIS_THERMAL_STRETCH'] = 'percentile'

    def __len__(self): return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        with Image.open(s.thermal_path) as im: raw = np.asarray(im)
        if raw.dtype != np.uint16 or raw.ndim != 2: raise ValueError('Expected uint16 left thermal')
        gray = AnyThermalEncoder._array_to_uint8(raw)
        if gray.shape != (256, 640) or gray.std() < 1e-3:
            raise ValueError(f'Unexpected/saturated thermal: {s.sample_id}, {gray.shape}')
        x = np.repeat(gray[None], 3, axis=0).astype(np.float32)/127.5-1.
        _, gt = load_ms2_gt(s.thermal_gt_path, 256.)
        if gt.shape != gray.shape: raise ValueError('Image/GT shape mismatch; do not interpolate sparse GT')
        return dict(image=torch.from_numpy(x), gt=torch.from_numpy(gt), id=s.sample_id,
                    sequence=s.sample_id.rsplit('_', 1)[0],
                    thermal_stats=torch.tensor([gray.min(), gray.max(), gray.std()], dtype=torch.float32),
                    token_ids=self.tokens['input_ids'][index], attention_mask=self.tokens['attention_mask'][index])


def lr_factor(step):
    # LambdaLR initializes step=0: first update is nonzero, peak at update 500.
    if step < 500: return (step+1)/500.
    return max(0., 1-(step-500)/(STEPS-500))**.9


def optimizer_for(model):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        base_lr = 5e-6 if name.startswith('unet.') else 5e-4
        wd = 0. if p.ndim == 1 or name.endswith('.bias') else .1
        groups.setdefault((base_lr, wd), []).append(p)
    opt = torch.optim.AdamW([dict(params=ps, lr=lr, weight_decay=wd) for (lr, wd), ps in groups.items()],
                            betas=(.9, .999))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)


def save(path, model, opt, sched, step, contract, resumable):
    state = dict(unet=model.unet.state_dict(), head=model.head.state_dict(), step=step,
                 image_cursor=step*BATCH*ACCUM, contract=contract)
    if resumable:
        state.update(optimizer=opt.state_dict(), scheduler=sched.state_dict(), rng=rng_state())
    tmp = Path(str(path)+'.tmp')
    torch.save(state, tmp); os.replace(tmp, path)


def restore(path, model, contract, opt=None, sched=None):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['contract'] != contract or state['image_cursor'] != state['step']*BATCH*ACCUM:
        raise ValueError('Checkpoint contract/cursor mismatch')
    model.unet.load_state_dict(state['unet'], strict=True)
    model.head.load_state_dict(state['head'], strict=True)
    if opt is not None:
        if state['scheduler']['last_epoch'] != state['step']: raise ValueError('Scheduler cursor mismatch')
        opt.load_state_dict(state['optimizer']); sched.load_state_dict(state['scheduler'])
        restore_rng(state['rng'])
    return state['step']


def repair_log(path, step):
    rows = []
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if len(rows) == step: break
            row = json.loads(line)
            if row['step'] != len(rows)+1: raise ValueError('Training log gap')
            rows.append(row)
    if len(rows) != step: raise ValueError('Training log shorter than checkpoint')
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
    os.replace(tmp, path)


def pair_smoke(directory):
    ca, cb = [read(directory/a/'contract.json') for a in ('frame', 'empty')]
    if ca['arm'] != 'frame' or cb['arm'] != 'empty': raise ValueError('Wrong paired arms')
    if {k:v for k,v in ca.items() if k != 'arm'} != {k:v for k,v in cb.items() if k != 'arm'}:
        raise ValueError('Smoke initialization/config/data/code mismatch')
    rows = []
    for arm in ('frame', 'empty'):
        d = directory/arm
        done = read(d/'completed.json')
        if done['step'] != 32 or done['mode'] != 'smoke' or not done['smoke_checks_passed']:
            raise ValueError('Incomplete/failed paired smoke')
        r = [json.loads(x) for x in (d/'train.jsonl').read_text().splitlines()]
        if [x['step'] for x in r] != list(range(1, 33)): raise ValueError('Incomplete smoke log')
        rows.append(r)
    if any(a['data_ids_sha256'] != b['data_ids_sha256'] for a,b in zip(*rows)):
        raise ValueError('Paired image order differs')
    print('[paired-smoke] PASS: same initialization/data/config; both forward/backward gates passed', flush=True)


def selected_step(curve):
    return min(curve, key=lambda step: (curve[step]['metrics']['ssi']['abs_rel'], int(step)))


def eval_at(model, dataset, directory, step):
    directory.mkdir(parents=True, exist_ok=True)
    evaluate(model, dataset, torch.device('cuda'), True, directory, step)
    return read(directory/f'val_step{step:06d}.json')


def run(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Requires BF16 GPU; CPU unit tests are separate')
    assets = prepare_assets(args.assets) if args.mode == 'smoke' else verify_assets(args.assets)
    train, ti = checked_manifest(args.train_manifest, args.data_root, 'train', 75688)
    val, vi = checked_manifest(args.val_manifest, args.data_root, 'val', 14249)
    if {s.sample_id for s in train} & {s.sample_id for s in val}: raise ValueError('Train/val overlap')
    seed_all(43)
    model = load_sd21(assets['path'])
    initial = dict(unet=tensor_hash(model.unet), head=tensor_hash(model.head))
    contract = dict(experiment='sd21_tadp_pair_v1', arm=args.arm, seed=43, assets=assets,
        initial=initial, train=ti, val=vi, train_captions=caption_hash(train), val_captions=caption_hash(val),
        steps=STEPS, batch=BATCH, accumulation=ACCUM, unet_lr=5e-6, head_lr=5e-4, weight_decay=.1,
        warmup=500, decay='poly_0.9_to_zero', clip_grad=1., precision='bf16; VAE/text/loss float32',
        loss='SiLog lambda=.5 pooled per microbatch; finite .001<GT<80',
        image='uint16 percentile; native 256x640; [-1,1]; no augmentation',
        target='thermal-view real LiDAR only; no pseudo GT, no GT VAE encoding',
        text='native SD2 frozen; 77 tokens; no explicit padding mask; empty string not zero embedding',
        latent='VAE posterior mode * scaling_factor', timestep=1,
        feature_taps='up0.res1; up1.attn1; up2.attn1; up3.attn2 (before upsample)',
        selection='ssi AbsRel min at 2k:2k:20k stride4, ties earlier; fixed20k separate',
        evaluation='Bridge official .001<GT<80; ssi and none; macro; not unified-v1',
        code={f:sha256_file(ROOT/f) for f in CODE},
        software={p:importlib.metadata.version(p) for p in ('torch','numpy','diffusers','transformers')})
    out = args.output
    if args.mode == 'train':
        if args.smoke_output is None: raise ValueError('Formal training requires paired smoke')
        pair_smoke(args.smoke_output)
        if read(args.smoke_output/args.arm/'contract.json') != contract:
            raise ValueError('Recipe/assets/software changed since smoke')
    if args.resume:
        if args.mode != 'train' or read(out/'contract.json') != contract:
            raise ValueError('Resume only an identical formal run')
    else:
        out.mkdir(parents=True, exist_ok=False)
        json_write(out/'contract.json', contract)
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(assets['path'], subfolder='tokenizer', local_files_only=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    ts = [make_tokens(s, tokenizer, assets, i['sha256'], args.arm, args.cache_dir, args.mode=='smoke')
          for s,i in ((train,ti),(val,vi))]
    training, validation = ThermalDirect(train, ts[0]), ThermalDirect(val, ts[1])
    curve_data = Subset(validation, list(range(0, len(validation), 4)))
    audit = []
    for index in np.linspace(0, len(training)-1, 8, dtype=int):
        sample = training[int(index)]
        valid = torch.isfinite(sample['gt']) & (sample['gt']>.001) & (sample['gt']<80)
        audit.append(dict(id=sample['id'], thermal_stats=sample['thermal_stats'].tolist(),
                          lidar_pixels=int(valid.sum()), shape=list(sample['gt'].shape)))
    json_write(out/'data_audit.json', audit)
    print('[config] '+json.dumps(contract), flush=True)
    print('[thermal] '+json.dumps(audit), flush=True)
    model.to('cuda'); opt, sched = optimizer_for(model)
    step = restore(out/'last.pt', model, contract, opt, sched) if args.resume else 0
    stop = 32 if args.mode == 'smoke' else STEPS
    if not 0 <= step <= stop: raise ValueError('Invalid checkpoint step')
    frozen = (tensor_hash(model.vae), tensor_hash(model.text_encoder)) if args.mode == 'smoke' else None
    # Recover interrupted validation before advancing training. All milestone weights are immutable.
    if args.resume:
        for s in MILESTONES:
            if s <= step and not (out/'curve'/f'val_step{s:06d}.json').exists():
                restore(out/f'step{s:06d}.pt', model, contract)
                eval_at(model, curve_data, out/'curve', s)
        restore(out/'last.pt', model, contract, opt, sched)
        repair_log(out/'train.jsonl', step)
    loader = DataLoader(training, batch_size=BATCH,
        sampler=ImageStream(len(training), 43, step*BATCH*ACCUM, (stop-step)*BATCH*ACCUM),
        num_workers=args.workers, pin_memory=True, generator=torch.Generator().manual_seed(44))
    iterator = iter(loader)
    if not args.resume: seed_all(100043)
    model.train(); started = time.monotonic(); start_step = step; attention_grad = False
    attention_params = [p for name,p in model.unet.named_parameters() if 'attn2.to_k.weight' in name]
    if not attention_params: raise ValueError('No pretrained text cross-attention parameters found')
    with (out/'train.jsonl').open('a', encoding='utf-8') as log:
        while step < stop:
            opt.zero_grad(set_to_none=True); loss_sum=0.; empty=0; ids=[]
            for _ in range(ACCUM):
                batch = next(iterator); ids.extend(batch['id'])
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    pred = model(batch['image'].cuda(), batch['token_ids'].cuda(), batch['attention_mask'].cuda())
                    if not torch.isfinite(pred).all(): raise ValueError('Nonfinite prediction')
                    loss, nempty = masked_silog(pred, batch['gt'].cuda())
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                (loss/ACCUM).backward(); loss_sum += float(loss.detach())/ACCUM; empty += nempty
            if any(p.grad is not None for m in (model.vae,model.text_encoder) for p in m.parameters()):
                raise ValueError('Frozen encoder received gradients')
            attention_grad |= any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in attention_params)
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                  1., error_if_nonfinite=True)
            opt.step(); sched.step(); step += 1
            record = dict(step=step, image_cursor=step*BATCH*ACCUM, loss=loss_sum, empty_gt_frames=empty,
                lr=[g['lr'] for g in opt.param_groups], grad_norm=float(norm),
                data_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest())
            log.write(json.dumps(record)+'\n'); log.flush()
            if step%20 == 0 or step == stop: print('[train] '+json.dumps(record), flush=True)
            if args.mode == 'train':
                if step in MILESTONES:
                    save(out/f'step{step:06d}.pt', model, opt, sched, step, contract, False)
                if step%1000 == 0:
                    save(out/'last.pt', model, opt, sched, step, contract, True)
                if step in MILESTONES:
                    eval_at(model, curve_data, out/'curve', step)
    elapsed_train = time.monotonic()-started
    smoke_passed = None
    if args.mode == 'smoke':
        if (frozen != (tensor_hash(model.vae), tensor_hash(model.text_encoder)) or not attention_grad or
                initial['unet'] == tensor_hash(model.unet) or initial['head'] == tensor_hash(model.head)):
            raise ValueError('Freeze/update/cross-attention gradient smoke gate failed')
        eval_at(model, Subset(validation, np.linspace(0,len(validation)-1,8,dtype=int).tolist()), out/'validation', step)
        smoke_passed = True
    else:
        # Report fixed-step and selected-step results separately; never use test data.
        fixed = eval_at(model, validation, out/'fixed20000', STEPS)
        curve = {s:read(out/'curve'/f'val_step{s:06d}.json') for s in MILESTONES}
        chosen = selected_step(curve)
        if chosen == STEPS:
            selected = fixed
        else:
            restore(out/f'step{chosen:06d}.pt', model, contract)
            selected = eval_at(model, validation, out/'selected_full', chosen)
        json_write(out/'summary.json', dict(selected_step=chosen, curve=curve, fixed20000=fixed,
                                            selected_full=selected, contract=contract))
        print(f'[selected] step={chosen}; validation-only; no test evaluation', flush=True)
    json_write(out/'completed.json', dict(mode=args.mode, step=step, arm=args.arm,
        smoke_checks_passed=smoke_passed, seconds_train_including_curve=elapsed_train,
        seconds_per_update=elapsed_train/max(1,step-start_step),
        max_gpu_gb=torch.cuda.max_memory_allocated()/2**30))
    print(f'[completed] {out}; no test data used', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['smoke','train'], required=True)
    p.add_argument('--arm', choices=['frame','empty'], required=True)
    for name in ('assets','train-manifest','val-manifest','data-root','output','cache-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--smoke-output', type=Path)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--workers', type=int, default=4)
    run(p.parse_args())
