"""Paired 1k continuation from caption far-bias 4k; only latent masking differs."""
import argparse
import gc
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import train_iris_far_bias_square as bias
from tools import test_iris_cross_region_pairs as parent_test
from tools.iris_latent_release import ARMS, START, STEPS, BANDS, release_cells, latent_loss, select, gates
from tools.train_iris_far_bias_square import pair, torch, F, BIAS_WEIGHT, PAIR_WEIGHT, local_pairs, signed_gap_loss, far_bias_square
from tools.train_iris_far_pairs import BATCH, ACCUM, LR
from tools.diagnose_error_sources import aligned_depth, official_valid_mask
from ms2_eval.official_protocol import evaluate_sample, OFFICIAL_METRICS
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = ('tools/train_iris_latent_release.py','tools/iris_latent_release.py',
        'tools/train_iris_far_bias_square.py','tools/test_iris_latent_release.py',
        'tools/test_iris_cross_region_pairs.py','tools/test_iris_far_pairs.py',
        'lotus/utils/ms2_thermal_dataset.py','ms2_eval/official_protocol.py')


class Components(bias.BiasComponents):
    def __init__(self, sd, arm, audit=False):
        super().__init__(sd)
        self.arm, self.audit = arm, audit
        self.removed = self.original = 0

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
        released = release_cells(batch['gt_depth_values'].cuda(), batch['gt_valid_values'].cuda(),
                                 batch['norm_bounds'].cuda())
        self.removed += int((released & keep[:, :1]).sum().item())
        self.original += int(keep[:, :1].sum().item())
        latent = latent_loss(anno, target[:len(image)], keep, released, self.arm)
        if self.audit:
            for value in (image, target, noise, ids, text, keep, released):
                trace.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
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
        bias, frames = far_bias_square(aligned, gt, valid, usable)
        return total + BIAS_WEIGHT * bias, bias, frames, count


def parent_inputs(runs):
    base = runs/'outputs/lotus_line_v2'
    matches = [p.parent for p in (base/'iris_far_bias_square_l050_frame_v1').glob('run_*/jobs.txt')
               if 'frame train+val -> 9171487 (afterok:9171486)' in p.read_text().splitlines()]
    if len(matches) != 1:
        raise ValueError(f'Expected one completed bias-square job 9171487, got {matches}')
    parent = matches[0]
    settings = pair.read(parent/'contract.json')
    ancestor = base/'iris_far_pairs_l050_frame4k_v1/job_9163317'
    paired, contract, source, start_conv, start = parent_test.prior.selection(ancestor)
    if (Path(settings['parent']) != ancestor or settings['selection_sha256'] != pair.sha256_file(ancestor/'summary.json')
            or settings['start_conv_sha256'] != pair.sha256_file(start_conv)
            or Path(settings['source']) != source or settings['trainable'] != contract['trainable']
            or settings['runner_sha256'] != pair.sha256_file(bias.__file__)):
        raise ValueError('Bias-square parent lineage changed')
    added, step, conv = parent_test.checkpoint(parent,start,start_conv,pair.read(ancestor/'summary.json'))
    if (start,added,step) != (2000,2000,START):
        raise ValueError('Expected the selected caption bias-square 4000 checkpoint')
    return dict(parent=parent, ancestor=ancestor, paired=paired, contract=contract, source=source, conv=conv,
                baseline=pair.read(parent/'summary.json')['full_by_added_steps']['2000'],
                original=pair.read(ancestor/'summary.json')['original'])


def contract_for(inputs):
    c = inputs['contract']
    return dict(parent=str(inputs['parent']), parent_summary_sha256=pair.sha256_file(inputs['parent']/'summary.json'),
                parent_contract_sha256=pair.sha256_file(inputs['parent']/'contract.json'),
                start=START, checkpoint=str(inputs['conv']), checkpoint_sha256=pair.sha256_file(inputs['conv']),
                source=str(inputs['source']), source_sha256=c['source_sha256'], train=c['train'], val=c['val'],
                trainable=c['trainable'], optimizer='fresh AdamW', lr=LR, weight_decay=.01,
                seed=43, batch=BATCH, accumulation=ACCUM, image_cursor=START*BATCH*ACCUM,
                checkpoints=list(STEPS), latent_denominator='original valid elements',
                mask='valid LiDAR 60<=GT<80 and log(GT)>frame upper bound; maxpool8, no dilation',
                gates=dict(rmse_gain=.02, absrel_tol=.0005, a1_tol=.001, band_absrel_tol=.0005,
                           references='both 4k start and original l050 for headline; 4k start for near bands'),
                losses=dict(latent=1.,reconstruction=1.,image_l1=.05,pairs=PAIR_WEIGHT,far_bias=BIAS_WEIGHT),
                code={name:pair.sha256_file(ROOT/name) for name in CODE})


def measure(model, recipe, samples):
    totals = np.zeros(len(OFFICIAL_METRICS), np.float64)
    bands = {f'{lo}-{hi}':dict(total=0.,frames=0) for lo,hi in BANDS}
    for i,(ident,raw,gt) in enumerate(pair.old.predict_samples(model,samples,'frame',recipe)):
        if ident != samples[i].sample_id:
            raise ValueError('Validation sample order changed')
        m = evaluate_sample(raw,gt,align='ssi_log',min_depth=.001,max_depth=80.)
        values = np.asarray([m[k] for k in OFFICIAL_METRICS])
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite validation metric')
        totals += values
        valid = official_valid_mask(gt,.001,80.)
        pred = np.clip(aligned_depth(raw,gt,valid,'ssi_log'),.001,80.)
        for lo,hi in BANDS:
            mask = valid & (gt>=lo) & (gt<hi)
            if mask.any():
                b = bands[f'{lo}-{hi}']
                b['total'] += float(np.mean(np.abs(pred[mask]-gt[mask])/gt[mask]))
                b['frames'] += 1
        if (i+1)%2000 == 0:
            print('[validation-progress]',i+1,flush=True)
    if any(not b['frames'] for b in bands.values()):
        raise ValueError('Missing near-band validation support')
    return dict(metrics=dict(zip(OFFICIAL_METRICS,totals/len(samples))),
                bands={k:dict(abs_rel=b['total']/b['frames'],frames=b['frames']) for k,b in bands.items()})


def train(model, params, dataset, collate, sd, out, arm, smoke):
    net = model.unet
    components = Components(sd,arm,audit=True)
    net.train()
    net.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(params.values(),lr=LR,weight_decay=.01)
    pair.seed_all(43)
    budget = 4 if smoke else 1000
    sampler = pair.ImageStream(len(dataset),43,START*BATCH*ACCUM,budget*BATCH*ACCUM)
    batches = iter(pair.DataLoader(dataset,batch_size=BATCH,collate_fn=collate,num_workers=0,sampler=sampler))
    trace = hashlib.sha256()
    initial = pair.fingerprint(params.items())
    frozen = [(k,p) for k,p in net.named_parameters() if not p.requires_grad]
    before = pair.fingerprint(frozen)
    total_pairs = supported = 0
    for micro in range(START*ACCUM,(START+budget)*ACCUM):
        batch = next(batches)
        torch.manual_seed(43+micro)
        if smoke and arm == 'control' and micro == START*ACCUM:
            # Compare the normal entry point against the historical loss on exactly the same draws.
            with torch.no_grad():
                reference = bias.BiasComponents.loss(components,net,batch,micro,hashlib.sha256())[0]
            torch.manual_seed(43+micro)
        loss, far_bias, frames, count = components.loss(net,batch,micro,trace)
        if smoke and arm == 'control' and micro == START*ACCUM:
            torch.testing.assert_close(loss,reference,rtol=1e-6,atol=1e-6)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        supported += frames
        total_pairs += count
        (loss/ACCUM).backward()
        if (micro+1)%ACCUM:
            continue
        step = (micro+1)//ACCUM
        norm = torch.nn.utils.clip_grad_norm_(params.values(),1.,error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if smoke or step%50 == 0:
            print('[train]',arm,step,'loss',loss.item(),'grad',float(norm),
                  'eligible_removed_fraction',components.removed/max(components.original,1),flush=True)
        if not smoke and step in STEPS:
            torch.save({k:p.detach().cpu() for k,p in params.items()},out/f'conv_{step}.pt')
    net.disable_gradient_checkpointing()
    net.eval()
    if (not components.removed or not total_pairs or not supported
            or before != pair.fingerprint(frozen) or initial == pair.fingerprint(params.items())):
        raise ValueError('Missing supervision, unchanged trainable tensors or modified frozen tensors')
    return dict(trace=trace.hexdigest(),initial=initial,removed=components.removed,
                original=components.original,pairs=total_pairs,far_frames=supported,
                updates=budget,final_image_cursor=(START+budget)*BATCH*ACCUM)


def verify_pair(root):
    smoke = pair.read(root/'smoke_pass.json')
    reports = [pair.read(root/arm/'summary.json') for arm in ARMS]
    for arm,r in zip(ARMS,reports):
        if (r['arm'] != arm or r['contract_sha256'] != pair.sha256_file(root/'contract.json')
                or r['baseline'] != smoke['baseline']):
            raise ValueError('Training arms differ from shared smoke contract')
    if reports[0]['training'] != reports[1]['training']:
        raise ValueError('Training data, stochastic draws, supervision or initial weights differ')
    if reports[0]['training']['initial'] != smoke['training']['initial']:
        raise ValueError('Formal training started from different parameters than smoke')
    return reports


def selected_weights(root,arm,inputs):
    summary = pair.read(root/arm/'summary.json')
    curve = {int(k):v for k,v in summary['curve'].items()}
    smoke = pair.read(root/'smoke_pass.json')
    if summary['baseline'] != smoke['baseline'] or curve[START] != smoke['baseline']:
        raise ValueError('Starting validation scores changed')
    step = select(curve,inputs['original'])
    conv = inputs['conv'] if step == START else root/arm/f'conv_{step}.pt'
    if (summary['selected'] != step or Path(summary['selected_conv']) != conv
            or summary['selected_sha256'] != pair.sha256_file(conv)):
        raise ValueError('Checkpoint disagrees with frozen validation rule')
    return step,conv


def run(a):
    from tools.train_route_suite import RouteModel
    from utils.ms2_thermal_dataset import MS2ThermalDataset, MS2ThermalTransform, collate_fn_ms2
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Requires BF16 CUDA GPU')
    inputs = parent_inputs(a.runs)
    contract = contract_for(inputs)
    c = inputs['contract']
    pseudo = Path(c['pseudo'])
    if pair.sha256_file(pseudo.parent/'build_config.json') != c['pseudo_config']:
        raise ValueError('Pseudo target configuration changed')
    train_rows,ti = pair.checked_manifest(Path(c['train']['path']),a.data_root,'train',75688)
    val,vi = pair.checked_manifest(Path(c['val']['path']),a.data_root,'val',14249)
    if ti != c['train'] or vi != c['val']:
        raise ValueError('Parent manifests changed')
    dataset = MS2ThermalDataset(Path(ti['path']),a.data_root,pseudo,
        transform=MS2ThermalTransform(random_flip=True),norm_type='log_truncnorm',
        truncnorm_min=.02,truncnorm_max=.98,use_captions=True)
    if [s.sample_id for s in train_rows] != [r['id'] for r in dataset.rows]:
        raise ValueError('Training sample order differs')
    recipe = SimpleNamespace(**c['recipe'])
    smoke = a.mode=='smoke'
    if smoke:
        if (a.run/'contract.json').exists():
            raise FileExistsError('Smoke output already exists')
        pair.json_write(a.run/'contract.json',contract)
    elif (pair.read(a.run/'contract.json') != contract
          or pair.read(a.run/'smoke_pass.json')['contract_sha256'] != pair.sha256_file(a.run/'contract.json')):
        raise ValueError('Training configuration differs from paired smoke')
    reports = {}
    for arm in ARMS if smoke else (a.arm,):
        folder = a.run/('smoke_'+arm if smoke else arm)
        folder.mkdir(parents=True,exist_ok=False)
        model = RouteModel(recipe,torch.device('cuda'),torch.float16)
        parent_test.pilot.load_selected(model,'frame',inputs['source'],inputs['conv'],c)
        params = pair.trainable_convs(model.unet)
        if list(params) != c['trainable']:
            raise ValueError('Trainable convolution scope differs')
        if smoke and arm=='control':
            baseline = measure(model,recipe,val)
            for k,tol in [('rmse',.002),('abs_rel',.00005),('a1',.00005)]:
                if abs(baseline['metrics'][k]-inputs['baseline'][k])>tol:
                    raise ValueError(f'Start does not reproduce {k}')
            baseline['metrics'] = inputs['baseline']
        if not smoke:
            baseline = pair.read(a.run/'smoke_pass.json')['baseline']
        report = train(model,params,dataset,collate_fn_ms2,c['sd'],folder,arm,smoke)
        reports[arm] = report
        if not smoke:
            if report['initial'] != pair.read(a.run/'smoke_pass.json')['training']['initial']:
                raise ValueError('Formal initial weights differ from paired smoke')
            curve = {START:baseline}
            for step in STEPS:
                pair.restore_params(params,folder/f'conv_{step}.pt')
                curve[step] = measure(model,recipe,val)
                print('[validation]',arm,step,curve[step],flush=True)
            step = select(curve,inputs['original'])
            conv = inputs['conv'] if step == START else folder/f'conv_{step}.pt'
            pair.json_write(folder/'summary.json',dict(arm=arm,curve=curve,baseline=baseline,
                selected=step,selected_conv=str(conv),selected_sha256=pair.sha256_file(conv),
                training=report,contract_sha256=pair.sha256_file(a.run/'contract.json'),
                gates={s:gates(curve[s],baseline,inputs['original']) for s in STEPS}))
            print('[selected]',arm,step,curve[step]['metrics'],flush=True)
        del params,model
        gc.collect()
        torch.cuda.empty_cache()
    if smoke:
        if reports['control'] != reports['masked']:
            raise ValueError('Paired smoke differs in data/random draws/initial weights')
        pair.json_write(a.run/'smoke_pass.json',dict(contract_sha256=pair.sha256_file(a.run/'contract.json'),
                                                   baseline=baseline,training=reports['control']))
        print('[paired-smoke] PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['smoke','train'])
    p.add_argument('--arm',choices=ARMS,default='control')
    for name in ('runs','run','data-root'):
        p.add_argument('--'+name,type=Path,required=True)
    os.environ.update(IRIS_THERMAL_STRETCH='percentile',IRIS_TEXT_PADDING='legacy',IRIS_DECODE_CLAMP='0')
    run(p.parse_args())
