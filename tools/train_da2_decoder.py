"""Text-conditioned DPT training; frozen vision/text encoders, paired frame/fixed arms."""
import argparse
import hashlib
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
from models.da2_decoder import DecoderDepth
from models.da2_caption import masked_log_l1
from tools.da2_caption_data import ThermalLidar, ImageStream, checked_manifest, caption_hash
from tools.train_da2_caption import (build_model, verify_assets, token_cache, evaluate,
    json_write, seed_all, rng_state, restore_rng)
from tools.da2_continuation import check_source, validate_state, read
from ms2_eval.io import sha256_file

FIXED_TEXT = 'A thermal image of a scene.'
MILESTONES = (10000, 20000)


def tensor_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def optimizer_for(model):
    opt = torch.optim.AdamW([
        dict(params=model.core.depth_head.parameters(), lr=1e-5),
        dict(params=model.fusions.parameters(), lr=5e-5)], betas=(.9, .999), weight_decay=.01)
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0., 1-s/20000)**.9)


def save(path, model, opt, sched, step, contract):
    state = dict(decoder=model.core.depth_head.state_dict(), fusions=model.fusions.state_dict(),
                 optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                 step=step, image_cursor=step*12, rng=rng_state(), amp='bf16', contract=contract)
    tmp = Path(str(path)+'.tmp'); torch.save(state, tmp); os.replace(tmp, path)


def restore(path, model, opt, sched, contract, random_state=True):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['contract'] != contract or state['image_cursor'] != state['step']*12 or state['amp'] != 'bf16':
        raise ValueError('Checkpoint contract/cursor/precision mismatch')
    if state['scheduler']['last_epoch'] != state['step']: raise ValueError('Scheduler cursor mismatch')
    model.core.depth_head.load_state_dict(state['decoder'], strict=True)
    model.fusions.load_state_dict(state['fusions'], strict=True)
    if opt is not None:
        opt.load_state_dict(state['optimizer']); sched.load_state_dict(state['scheduler'])
    if random_state: restore_rng(state['rng'])
    return state['step']


def make_tokens(samples, tokenizer, assets, info, arm, create):
    if arm == 'frame': return token_cache(samples, tokenizer, assets, info['sha256'], create)
    encoded = tokenizer([FIXED_TEXT], padding='max_length', max_length=77, truncation=True, return_tensors='pt')
    return dict(input_ids=encoded['input_ids'].expand(len(samples), -1),
                attention_mask=encoded['attention_mask'].bool().expand(len(samples), -1))


def pair_smoke(directory):
    a, b = (directory / arm for arm in ('frame', 'fixed'))
    ca, cb = read(a/'contract.json'), read(b/'contract.json')
    if ca['arm'] != 'frame' or cb['arm'] != 'fixed': raise ValueError('Wrong smoke arms')
    if {k:v for k,v in ca.items() if k!='arm'} != {k:v for k,v in cb.items() if k!='arm'}:
        raise ValueError('Paired smoke contracts differ')
    logs = []
    for d in (a, b):
        done = read(d/'completed.json')
        if done['step'] != 32 or done['mode'] != 'smoke' or not done['frozen_unchanged']:
            raise ValueError('Incomplete smoke')
        logs.append([json.loads(line) for line in (d/'train.jsonl').read_text().splitlines()])
    if [r['step'] for r in logs[0]] != list(range(1,33)) or [r['step'] for r in logs[1]] != list(range(1,33)):
        raise ValueError('Smoke step log incomplete')
    if any(x['data_ids_sha256'] != y['data_ids_sha256'] for x,y in zip(*logs)):
        raise ValueError('Smoke image order differs')
    print('[paired-smoke] PASS: same initialization/config/data; both GPU checks passed', flush=True)


class BaseOnly(torch.nn.Module):
    def __init__(self, core): super().__init__(); self.core = core
    def forward(self, image, token_ids=None, attention_mask=None):
        h,w = image.shape[-2:]; pad=(0,(-w)%14,0,(-h)%14)
        x = torch.nn.functional.pad(image, pad, mode='reflect') if any(pad) else image
        return self.core(x)[...,:h,:w].float()


def run(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('Requires BF16 GPU')
    assets = verify_assets(args.assets)
    state = torch.load(args.base_checkpoint, map_location='cpu', weights_only=False)
    original = state['contract']
    validate_state(state, original, 18000); check_source(original)
    if original['arm'] != 'no-caption' or original['seed'] != 43: raise ValueError('Wrong frozen baseline')
    if sha256_file(args.assets) != original['assets']: raise ValueError('Assets differ from baseline')
    train, ti = checked_manifest(args.train_manifest, args.data_root, 'train', 75688)
    val, vi = checked_manifest(args.val_manifest, args.data_root, 'val', 14249)
    if ti != original['train'] or vi != original['val']: raise ValueError('Data differs from baseline')
    if caption_hash(train) != original['caption_train_hash'] or caption_hash(val) != original['caption_val_hash']:
        raise ValueError('Caption provenance changed')
    seed_all(43)
    baseline = build_model(assets, 'no-caption', 43)
    baseline.load_state_dict(state['model'], strict=True); del state
    from transformers import CLIPTextModel, CLIPTokenizer
    text = CLIPTextModel.from_pretrained(assets['text_path'], subfolder='text_encoder', local_files_only=True)
    if text.config.hidden_size != 1024: raise ValueError('Wrong text encoder width')
    model = DecoderDepth(baseline.core, text)
    files = ('models/da2_decoder.py','tools/train_da2_decoder.py','models/da2_caption.py','tools/train_da2_caption.py',
             'tools/da2_caption_data.py','tools/da2_continuation.py','models/anythermal_encoder.py',
             'ms2_eval/official_protocol.py','ms2_eval/io.py')
    contract = dict(arm=args.arm, seed=43, base_path=str(args.base_checkpoint.resolve()),
        base_sha256=sha256_file(args.base_checkpoint), assets_sha256=sha256_file(args.assets),
        train=ti, val=vi, train_caption_hash=caption_hash(train), val_caption_hash=caption_hash(val),
        fixed_text=FIXED_TEXT, total_steps=20000, batch=4, accumulation=3, decoder_lr=1e-5, fusion_lr=5e-5,
        weight_decay=.01, clip_grad=1., fusion_width=256, amp='bf16',
        initial_head_sha256=tensor_hash(model.fusions), initial_decoder_sha256=tensor_hash(model.core.depth_head), code={f:sha256_file(ROOT/f) for f in files},
        software=dict(torch=torch.__version__, numpy=np.__version__))
    if args.mode == 'train':
        if args.smoke_output is None: raise ValueError('Require successful paired smoke')
        pair_smoke(args.smoke_output)
        if read(args.smoke_output/args.arm/'contract.json') != contract: raise ValueError('Recipe changed since smoke')
    out = args.output
    if args.resume:
        if args.mode != 'train' or not (out/'last.pt').is_file(): raise ValueError('Resume requires formal last.pt')
    else: out.mkdir(parents=True, exist_ok=False)
    json_write(out/'contract.json', contract)
    cache = args.cache_dir; cache.mkdir(parents=True, exist_ok=True)
    tokenizer = CLIPTokenizer.from_pretrained(assets['text_path'], subfolder='tokenizer', local_files_only=True)
    tok = [make_tokens(s, tokenizer, {**assets,'root':str(cache)}, info, args.arm, args.mode=='smoke')
           for s,info in ((train,ti),(val,vi))]
    training, validation = ThermalLidar(train,tok[0]), ThermalLidar(val,tok[1])
    model.to('cuda'); opt, sched = optimizer_for(model)
    step = restore(out/'last.pt',model,opt,sched,contract) if args.resume else 0
    stop = 32 if args.mode=='smoke' else 20000
    if not 0 <= step <= stop: raise ValueError('Invalid resume step')
    seed_all(100043)
    first = training[0]
    model.eval()
    with torch.no_grad():
        x=first['image'][None].cuda(); ids=first['token_ids'][None].cuda(); mask=first['attention_mask'][None].cuda()
        p=model(x,ids,mask)
        if step==0: torch.testing.assert_close(p,BaseOnly(model.core)(x),rtol=0,atol=0)
        if p.shape != first['gt'][None].shape or not torch.isfinite(p).all(): raise ValueError('Shape/finite smoke failed')
    frozen_before = (tensor_hash(model.core.pretrained),tensor_hash(model.text_encoder)) if args.mode=='smoke' else None
    initial_head = tensor_hash(model.fusions)
    initial_decoder = tensor_hash(model.core.depth_head)
    print('[config] '+json.dumps(contract),flush=True)
    if args.mode=='train':
        if args.resume:
            for milestone in MILESTONES:
                if milestone <= step and not (out/f'val_step{milestone:06d}.json').exists():
                    restore(out/f'step{milestone:06d}.pt',model,None,None,contract,False)
                    evaluate(model,validation,torch.device('cuda'),True,out,milestone)
    loader=DataLoader(training,batch_size=4,sampler=ImageStream(len(training),43,step*12,(stop-step)*12),
        num_workers=args.workers,pin_memory=True,generator=torch.Generator().manual_seed(44))
    iterator=iter(loader)
    if args.resume: restore(out/'last.pt',model,opt,sched,contract)
    else: seed_all(100043)
    model.train(); started=time.monotonic(); start_step=step; attention_grad_seen=False
    with (out/'train.jsonl').open('a',encoding='utf-8') as log:
        while step < stop:
            opt.zero_grad(set_to_none=True); loss_sum=0.; empty=0; ids=[]
            for _ in range(3):
                batch=next(iterator); ids.extend(batch['id'])
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    pred=model(batch['image'].cuda(),batch['token_ids'].cuda(),batch['attention_mask'].cuda())
                    loss,nempty=masked_log_l1(pred,batch['gt'].cuda())
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                (loss/3).backward(); loss_sum+=float(loss.detach())/3; empty+=nempty
            if any(p.grad is not None for m in (model.core.pretrained,model.text_encoder) for p in m.parameters()):
                raise ValueError('Frozen module has gradients')
            grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads): raise ValueError('Invalid gradients')
            attention_grad_seen |= any(f.attention.in_proj_weight.grad is not None and
                bool(f.attention.in_proj_weight.grad.abs().sum()>0) for f in model.fusions)
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            opt.step(); sched.step(); step+=1
            record=dict(step=step,loss=loss_sum,image_cursor=step*12,empty_gt_frames=empty,
                lr=[g['lr'] for g in opt.param_groups],grad_norm=float(norm),
                data_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest())
            log.write(json.dumps(record)+'\n'); log.flush()
            if step%20==0 or step==stop: print('[train] '+json.dumps(record),flush=True)
            if step in MILESTONES: save(out/f'step{step:06d}.pt',model,opt,sched,step,contract)
            if step%1000==0 or step==stop: save(out/'last.pt',model,opt,sched,step,contract)
            if args.mode=='train' and step in MILESTONES:
                evaluate(model,validation,torch.device('cuda'),True,out,step)
    unchanged=None
    if args.mode=='smoke':
        unchanged=frozen_before==(tensor_hash(model.core.pretrained),tensor_hash(model.text_encoder))
        if (not unchanged or not attention_grad_seen or initial_head==tensor_hash(model.fusions)
                or initial_decoder==tensor_hash(model.core.depth_head)):
            raise ValueError('Frozen/fusion/decoder/attention-gradient smoke gate failed')
        evaluate(model,Subset(validation,np.linspace(0,len(validation)-1,8,dtype=int).tolist()),
                 torch.device('cuda'),True,out,32)
    elapsed=time.monotonic()-started
    json_write(out/'completed.json',dict(mode=args.mode,arm=args.arm,step=step,frozen_unchanged=unchanged,
        seconds=elapsed,seconds_per_update=elapsed/max(1,step-start_step),max_gpu_gb=torch.cuda.max_memory_allocated()/2**30))
    print(f'[completed] {args.arm} {step} updates; {elapsed/60:.1f} min',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['smoke','train'],required=True)
    p.add_argument('--arm',choices=['frame','fixed'],required=True)
    for name in ('assets','base-checkpoint','train-manifest','val-manifest','data-root','output','cache-dir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--smoke-output',type=Path); p.add_argument('--resume',action='store_true')
    p.add_argument('--workers',type=int,default=4)
    run(p.parse_args())


