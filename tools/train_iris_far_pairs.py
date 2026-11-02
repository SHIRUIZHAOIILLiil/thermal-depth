"""Short convolution-only continuation from the matched lambda=.05 Iris 16k weights."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT/'tools'))
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tools import iris_far_stretch as old
from tools.da2_caption_data import ImageStream, checked_manifest
from tools.iris_far_pair_loss import local_pairs, signed_gap_loss, trainable_convs
from tools.train_da2_caption import json_write, seed_all
from ms2_eval.io import sha256_file

STEPS = (0,500,1000,2000)
BATCH, ACCUM, LR = 4, 3, 3e-6
PAIR_WEIGHT = .01  # per metre of signed pair-gap error; one fixed coefficient.


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def fingerprint(items):
    h = hashlib.sha256()
    for name,p in items:
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def restore_params(params, file):
    state = torch.load(file, map_location='cpu', weights_only=True)
    if set(state) != set(params):
        raise ValueError('Changed trainable scope')
    with torch.no_grad():
        for name,p in params.items():
            p.copy_(state[name])


def select(curve):
    zero = curve[0]
    eligible = [s for s in curve if s and all(old.gates(zero, curve[s]).values())]
    return min(eligible, key=lambda s:(curve[s]['rmse'], s)) if eligible else 0


class TrainingComponents:
    def __init__(self, source):
        from diffusers import AutoencoderKL, DDPMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer
        opts = dict(local_files_only=True)
        self.tokenizer = CLIPTokenizer.from_pretrained(source, subfolder='tokenizer', **opts)
        self.text = CLIPTextModel.from_pretrained(source, subfolder='text_encoder', **opts).requires_grad_(False).cuda().half().eval()
        vae = AutoencoderKL.from_pretrained(source, subfolder='vae', **opts).requires_grad_(False)
        self.decoder = copy.deepcopy(vae)
        self.decoder.encoder = None
        self.decoder = self.decoder.cuda().train()
        self.decoder.enable_gradient_checkpointing()
        vae.decoder = None
        self.vae = vae.cuda().half().eval()
        self.scheduler = DDPMScheduler.from_pretrained(source, subfolder='scheduler', **opts)
        self.scheduler.register_to_config(prediction_type='sample')

    def loss(self, net, batch, micro, trace):
        from train_iris_ms2_g import decode_image_depth, _ssi_log_aligned_depth
        image = batch['pixel_values'].cuda()
        depth = batch['depth_values'].cuda()
        with torch.no_grad():
            def encode(x):
                return self.vae.encode(x.half()).latent_dist.sample() * self.vae.config.scaling_factor
            rgb = encode(torch.cat([image,image]))
            target = encode(torch.cat([depth,image]))
            noise = torch.randn_like(target)
            ts = torch.full((len(target),), 999, device='cuda', dtype=torch.long)
            inputs = torch.cat([rgb, self.scheduler.add_noise(target, noise, ts)], dim=1)
            prompts = list(batch['text_descriptions']) + ['']*len(image)
            ids = self.tokenizer(prompts, padding=True, truncation=True, max_length=self.tokenizer.model_max_length,
                                 return_tensors='pt').input_ids.cuda()
            text = self.text(ids, return_dict=False)[0]
            task = torch.eye(2, device='cuda').repeat_interleave(len(image),dim=0)
            labels = torch.cat([task.sin(),task.cos()],dim=1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = net(inputs, ts, text, class_labels=labels, return_dict=False)[0]
        anno, recon = pred.float().chunk(2)
        keep = (batch['valid_mask_values']+batch['sky_mask_values']).cuda().bool()
        keep = (~F.max_pool2d((~keep).float(), 8, 8).bool()).expand_as(anno)
        if not keep.any():
            raise ValueError('No valid depth latent cells')
        latent = F.mse_loss(anno[keep], target[:len(image)].float()[keep])
        reconstruction = F.mse_loss(recon,target[len(image):].float())
        gt = batch['gt_depth_values'].cuda()
        valid = batch['gt_valid_values'].cuda().bool()
        raw, metric, lo, hi = decode_image_depth(self.decoder, anno, batch['norm_bounds'].cuda(), 'log_truncnorm')
        if not (torch.isfinite(lo).all() and torch.isfinite(hi).all() and torch.isfinite(raw).all()):
            raise ValueError('Invalid frame normalization/decoded prediction')
        image_l1 = (metric[valid]-gt[valid]).abs().mean() if valid.any() else anno.new_zeros(())
        aligned, usable, _, _ = _ssi_log_aligned_depth(raw, gt, valid)
        pairs = []
        for i, path in enumerate(batch['image_pathes']):
            indices = local_pairs(batch['gt_depth_values'][i,0].numpy(), 43 + micro*BATCH + i)
            pairs.append(indices)
            trace.update(path.encode())
            trace.update(indices.tobytes())
        gap, count = signed_gap_loss(aligned, gt, pairs, usable)
        total = latent + reconstruction + .05*image_l1 + PAIR_WEIGHT*gap
        return total, gap, count


def run(a):
    from huggingface_hub import snapshot_download
    from tools.train_route_suite import RouteModel
    from utils.ms2_thermal_dataset import MS2ThermalDataset, MS2ThermalTransform, collate_fn_ms2
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Requires a BF16 CUDA GPU')
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_TEXT_PADDING='legacy', IRIS_DECODE_CLAMP='0')
    smoke = a.mode=='smoke'
    folder = a.run/a.mode/a.arm
    folder.mkdir(parents=True, exist_ok=False)
    train, ti = checked_manifest(a.manifests/old.TRAIN_FILE,a.data_root,'train',75688)
    val, vi = checked_manifest(a.manifests/old.VAL_FILE,a.data_root,'val',14249)
    if not vi['sha256'].startswith('c26a5859'):
        raise ValueError('Wrong validation split')
    pseudo_config = a.pseudo_dir.parent/'build_config.json'
    meta = read(pseudo_config)
    if meta.get('depth_scale') != 256. or meta.get('stored') != 'calibrated depth in metres, no range handling applied':
        raise ValueError('Expected calibrated AnyThermal depth in metres')
    ds = MS2ThermalDataset(a.manifests/old.TRAIN_FILE,a.data_root,a.pseudo_dir,
                           transform=MS2ThermalTransform(random_flip=True),norm_type='log_truncnorm',
                           truncnorm_min=.02,truncnorm_max=.98,use_captions=a.arm=='frame')
    if [s.sample_id for s in train] != [r['id'] for r in ds.rows]:
        raise ValueError('Train loader/manifest ID mismatch')
    source = old.checkpoint(a.checkpoints,a.arm)
    sd = snapshot_download('Manojb/stable-diffusion-2-1-base',local_files_only=True)
    lotus = snapshot_download('jingheya/lotus-depth-g-v2-1-disparity',local_files_only=True)
    recipe = SimpleNamespace(route='b_thermal_unet',backbone='g',lotus_model_path=lotus,local_files_only=True,
                             gt_decode_fp32=True,condition_latent='mode',timestep=999,num_inference_steps=1,seed=20260703)
    model = RouteModel(recipe,torch.device('cuda'),torch.float16)
    old.load_arm(model,source,a.arm)
    net = model.unet
    params = trainable_convs(net)
    # Leave the init untouched; save only the small set of trained tensors.
    contract = dict(arm=a.arm,source=str(source),source_sha256=sha256_file(source),sd=sd,recipe=vars(recipe),
                    train=ti,val=vi,pseudo=str(a.pseudo_dir.resolve()),pseudo_config=sha256_file(pseudo_config),
                    trainable=list(params),pair_weight=PAIR_WEIGHT,lr=LR,seed=43,batch=BATCH,accum=ACCUM,steps=STEPS,
                    code={p:sha256_file(ROOT/p) for p in ('tools/train_iris_far_pairs.py','tools/iris_far_pair_loss.py',
                      'tools/iris_far_stretch.py','lotus/train_iris_ms2_g.py','lotus/utils/ms2_thermal_dataset.py')})
    json_write(folder/'contract.json',contract)
    # Compare the serialized form (tuples become lists).
    if not smoke:
        passed = read(a.run/'smoke_pass.json')
        if passed[a.arm] != sha256_file(folder/'contract.json'):
            raise ValueError('Recipe differs from passing paired smoke')
    frozen = lambda: fingerprint((n,p) for n,p in net.named_parameters() if not p.requires_grad)
    before = frozen() if smoke else None
    initial = fingerprint(params.items())
    train_parts = TrainingComponents(sd)
    net.train()
    net.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(params.values(), lr=LR, weight_decay=.01)
    seed_all(43)
    nsteps = 2 if smoke else STEPS[-1]
    loader = DataLoader(ds,batch_size=BATCH,collate_fn=collate_fn_ms2,num_workers=0,
                        sampler=ImageStream(len(ds),43,0,nsteps*BATCH*ACCUM))
    trace = hashlib.sha256()
    n_pairs, checked_grad = 0, False
    stream = iter(loader)
    for micro in range(len(loader)):
        # Flip and latent/noise draws are paired, independent of previous model computation.
        torch.manual_seed(43+micro)
        batch = next(stream)
        total, gap, count = train_parts.loss(net,batch,micro,trace)
        n_pairs += count
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite continuation loss')
        if smoke and count and not checked_grad:
            grad = torch.autograd.grad(gap, list(params.values()),retain_graph=True,allow_unused=True)
            checked_grad = any(g is not None and bool(torch.isfinite(g).all() and g.abs().max()>0) for g in grad)
        (total/ACCUM).backward()
        if (micro+1)%ACCUM:
            continue
        step = (micro+1)//ACCUM
        norm = torch.nn.utils.clip_grad_norm_(list(params.values()),1.,error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step%100==0 or step==1:
            print(f'[train] {a.arm} new_step={step} loss={float(total):.5f} pairs_seen={n_pairs} grad={float(norm):.5f}',flush=True)
        if step in STEPS or smoke and step==nsteps:
            torch.save({n:p.detach().cpu() for n,p in params.items()}, folder/f'conv_{step}.pt')
    if not n_pairs or fingerprint(params.items())==initial:
        raise ValueError('Missing pairs or trainable convolutions did not update')
    if smoke and (before != frozen() or not checked_grad):
        raise ValueError('Frozen U-Net changed or pair loss did not backpropagate')
    del train_parts, optimizer, total, gap
    torch.cuda.empty_cache()
    net.disable_gradient_checkpointing()
    net.eval()
    if smoke:
        restore_params(params,folder/f'conv_{nsteps}.pt')
        old.measure(val[:4], old.predict_samples(model,val[:4],a.arm,recipe),[0.])
        json_write(folder/'completed.json',dict(trace=trace.hexdigest(),pairs=n_pairs))
        return

    def score(step,samples):
        if step == 0:
            old.load_arm(model,source,a.arm)
        else:
            restore_params(params,folder/f'conv_{step}.pt')
        return old.measure(samples,old.predict_samples(model,samples,a.arm,recipe),[0.])[0.]
    curve = {s:score(s,val[::4]) for s in STEPS}
    provisional = select(curve)
    full = {s:score(s,val) for s in sorted({0,provisional,STEPS[-1]})}
    if a.arm=='frame' and (abs(full[0]['abs_rel']-.07936)>.00005 or abs(full[0]['rmse']-2.9457)>.002
                           or abs(full[0]['a1']-.94227)>.00005):
        raise ValueError('Frame step zero does not reproduce historical validation')
    selected = provisional if all(old.gates(full[0],full[provisional]).values()) else 0
    json_write(folder/'summary.json',dict(curve=curve,full=full,provisional=provisional,selected=selected,
               trace=trace.hexdigest(),pairs=n_pairs))
    print('[selected]',a.arm,selected,full[selected],flush=True)


def pair(a):
    smoke = a.mode=='check'
    phase = 'smoke' if smoke else 'train'
    paths = {k:a.run/phase/k for k in old.RUNS}
    contracts = {k:read(p/'contract.json') for k,p in paths.items()}
    exclude = {'arm','source','source_sha256'}
    common = [{k:v for k,v in c.items() if k not in exclude} for c in contracts.values()]
    if common[0]!=common[1] or contracts['frame']['source_sha256']==contracts['empty']['source_sha256']:
        raise ValueError('Arms must share protocol and have distinct matched weights')
    rows = {k:read(p/('completed.json' if smoke else 'summary.json')) for k,p in paths.items()}
    f,e = rows['frame'], rows['empty']
    if f['trace']!=e['trace'] or f['pairs']!=e['pairs']:
        raise ValueError('Frame order/pairs differ between arms')
    if smoke:
        json_write(a.run/'smoke_pass.json',{k:sha256_file(p/'contract.json') for k,p in paths.items()})
        print('[paired-smoke] PASS',flush=True)
    else:
        diffs = {s:{k:f['full'][s][k]-e['full'][s][k] for k in old.OFFICIAL_METRICS} for s in ('0','2000')}
        matched_curve = {s:{k:f['curve'][s][k]-e['curve'][s][k] for k in old.OFFICIAL_METRICS} for s in f['curve']}
        json_write(a.run/'paired.json',dict(fixed_frame_minus_empty=diffs, stride4_frame_minus_empty=matched_curve,
                   selected={k:r['selected'] for k,r in rows.items()}))
        print('[fixed-frame-minus-empty]',diffs,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['smoke','train','check','report'])
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--arm',choices=tuple(old.RUNS))
    for k in ('manifests','data-root','pseudo-dir'):
        p.add_argument('--'+k,type=Path)
    p.add_argument('--checkpoints',nargs='+',type=Path)
    a=p.parse_args()
    if a.mode in ('smoke','train'):
        if not all((a.arm,a.manifests,a.data_root,a.pseudo_dir,a.checkpoints)):
            p.error('Missing training inputs')
        run(a)
    else:
        pair(a)
