"""Train only the original VAE depth decoder on deployment-path predicted latents."""
import argparse
import dataclasses
import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from ms2_eval.io import load_ms2_gt
from ms2_eval.official_protocol import OFFICIAL_METRICS, evaluate_sample
from tools import train_iris_latent_release as parent
from tools import train_iris_far_bias_square as bias
from tools import train_iris_far_pairs as pair
from tools.iris_decoder_only import InferenceInput, predict_frozen, make_decoder, decode_unit, select
from tools.iris_decoder_only import STEPS, LR, BATCH, ACCUM, SEED
from tools.iris_tmax1_selection import POLICY, read_reference

CODE = ('tools/train_iris_decoder_only.py', 'tools/iris_decoder_only.py',
        'tools/train_route_suite.py', 'tools/train_ms2_joint_gt_v3.py',
        'tools/train_iris_latent_release.py', 'tools/train_iris_far_pairs.py',
        'tools/test_iris_far_pairs.py', 'tools/iris_far_stretch.py',
        'tools/train_iris_far_bias_square.py', 'tools/iris_far_pair_loss.py',
        'tools/iris_tmax1_selection.py', 'lotus/train_iris_ms2_g.py',
        'lotus/utils/ms2_thermal_dataset.py', 'ms2_eval/official_protocol.py')


def frozen_parameters(model, decoder):
    for label, module in (('unet',model.unet), ('input_vae',model.lotus.vae),
                          ('text',model.lotus.text_encoder), ('old_depth_vae',model.gt_vae),
                          ('new_depth_vae',decoder)):
        for key, value in module.named_parameters():
            if not value.requires_grad:
                yield label+'.'+key, value


def loss_for_batch(decoder, latent, examples, indices):
    from train_iris_ms2_g import decode_image_depth, _ssi_log_aligned_depth
    device = latent.device
    gt = torch.stack([x['gt_depth_values'] for x in examples]).to(device)
    valid = torch.stack([x['gt_valid_values'] for x in examples]).to(device).bool()
    bounds = torch.stack([x['norm_bounds'] for x in examples]).to(device)
    if not valid.any():
        raise ValueError('Batch has no real LiDAR supervision')
    raw, metric, lo, hi = decode_image_depth(decoder, latent.detach(), bounds, 'log_truncnorm')
    if not all(torch.isfinite(x).all() for x in (raw, metric, lo, hi)):
        raise ValueError('Nonfinite decoder output/bounds')
    l1 = (metric[valid]-gt[valid]).abs().mean()
    aligned, usable, _, _ = _ssi_log_aligned_depth(raw,gt,valid)
    pairs = [bias.local_pairs(x['gt_depth_values'][0].numpy(), SEED+int(idx))
             for x,idx in zip(examples,indices,strict=True)]
    gap, count = bias.signed_gap_loss(aligned,gt,pairs,usable)
    far, frames = bias.far_bias_square(aligned,gt,valid,usable)
    total = .05*l1 + bias.PAIR_WEIGHT*gap + bias.BIAS_WEIGHT*far
    return total, dict(image_l1=float(l1.detach()), pair_loss=float(gap.detach()),
                       far_bias_m2=float(far.detach()), pairs=count, far_frames=frames)


def validate(model, decoder, samples, recipe, verify_initial=False):
    totals = np.zeros(len(OFFICIAL_METRICS), np.float64)
    near = {name:[] for name in ('0-20','20-40','40-60','60-80')}
    worst = 0.
    decoder.eval()
    for i,s in enumerate(samples):
        original, latent = predict_frozen(model, InferenceInput.from_sample(s), recipe)
        with torch.no_grad():
            raw = decode_unit(decoder,latent)[0,0]
        if verify_initial:
            reference = original.squeeze()
            torch.testing.assert_close(raw,reference,rtol=1e-6,atol=1e-6)
            worst = max(worst,float((raw-reference).abs().max()))
        # GT is loaded only after inference and decoding are complete.
        raw = raw.cpu().numpy()
        _,gt = load_ms2_gt(s.thermal_gt_path,256.)
        metrics = evaluate_sample(raw,gt,align='ssi_log',min_depth=.001,max_depth=80.)
        vals = np.array([metrics[k] for k in OFFICIAL_METRICS])
        if not np.isfinite(vals).all():
            raise ValueError('Nonfinite validation metrics')
        totals += vals
        valid = parent.official_valid_mask(gt,.001,80.)
        pred = np.clip(parent.aligned_depth(raw,gt,valid,'ssi_log'),.001,80.)
        for lo,hi in ((0,20),(20,40),(40,60),(60,80)):
            mask = valid & (gt>=lo) & (gt<hi)
            if mask.any():
                near[f'{lo}-{hi}'].append(float(np.mean(np.abs(pred[mask]-gt[mask])/gt[mask])))
        if (i+1)%2000==0:
            print('[validation-progress]',i+1,flush=True)
    return dict(metrics=dict(zip(OFFICIAL_METRICS,totals/len(samples))), samples=len(samples),
                bands={k:dict(abs_rel=float(np.mean(v)) if v else None,frames=len(v)) for k,v in near.items()},
                initial_max_abs_output_error=worst if verify_initial else None)


def train(model, decoder, params, dataset, samples, recipe, folder, smoke):
    budget = 4 if smoke else STEPS[-1]
    optimizer = torch.optim.AdamW(params.values(),lr=LR,weight_decay=.01)
    before = pair.fingerprint(frozen_parameters(model,decoder))
    initial = pair.fingerprint(params.items())
    # A fresh decoder training stage; retain the existing 4k continuation data cursor.
    stream = iter(pair.ImageStream(len(samples),SEED,4000*BATCH*ACCUM,budget*BATCH*ACCUM))
    trace = hashlib.sha256()
    supported = total_pairs = 0
    max_grad = 0.
    decoder.train()
    decoder.enable_gradient_checkpointing()
    for update in range(1,budget+1):
        for micro in range(ACCUM):
            indices = [next(stream) for _ in range(BATCH)]
            latent_list = []
            # Each forward is batch-one, preserving the exact deployed prompt/noise behavior.
            for idx in indices:
                s = samples[idx]
                _, z = predict_frozen(model,InferenceInput.from_sample(s),recipe)
                latent_list.append(z)
                trace.update(s.sample_id.encode())
                trace.update(z.cpu().contiguous().view(torch.uint8).numpy().tobytes())
            latent = torch.cat(latent_list)
            # Dataset GT/targets never reach predict_frozen. No flip/crop mismatch.
            examples = [dataset[idx] for idx in indices]
            total, stats = loss_for_batch(decoder,latent,examples,indices)
            if not torch.isfinite(total):
                raise ValueError('Nonfinite training loss')
            supported += stats['far_frames']
            total_pairs += stats['pairs']
            (total/ACCUM).backward()
        norm = torch.nn.utils.clip_grad_norm_(params.values(),1.,error_if_nonfinite=True)
        max_grad = max(max_grad,float(norm))
        if any(p.grad is not None for _,p in frozen_parameters(model,decoder)):
            raise ValueError('Gradient entered a frozen module')
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if smoke or update%50==0:
            print('[train]',update,dict(loss=float(total.detach()),grad=float(norm),**stats),flush=True)
        if not smoke and update in STEPS[1:]:
            torch.save({k:p.detach().cpu() for k,p in params.items()},folder/f'decoder_{update}.pt')
    decoder.disable_gradient_checkpointing()
    decoder.eval()
    if before != pair.fingerprint(frozen_parameters(model,decoder)):
        raise ValueError('Frozen module weights changed')
    if initial == pair.fingerprint(params.items()) or max_grad <= 0 or not supported or not total_pairs:
        raise ValueError('Missing decoder update or real far/pair supervision')
    return dict(initial=initial, frozen_sha256=before, frozen_unchanged=True,
                trace=trace.hexdigest(), max_grad=max_grad, far_frames=supported, pairs=total_pairs,
                updates=budget, final=pair.fingerprint(params.items()))


def run(a):
    from tools.train_route_suite import RouteModel
    from utils.ms2_thermal_dataset import MS2ThermalDataset
    if not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA')
    inputs = parent.parent_inputs(a.runs)
    c = inputs['contract']
    train_samples, ti = pair.checked_manifest(Path(c['train']['path']),a.data_root,'train',75688)
    val, vi = pair.checked_manifest(Path(c['val']['path']),a.data_root,'val',14249)
    if ti!=c['train'] or vi!=c['val']:
        raise ValueError('Parent manifests changed')
    pseudo = Path(c['pseudo'])
    if pair.sha256_file(pseudo.parent/'build_config.json') != c['pseudo_config']:
        raise ValueError('Parent pseudo recipe changed')
    archive = a.runs/'outputs/lotus_line_v2/iris_pair_rmse_val_v1/job_9167854'
    reference = read_reference(archive,pair.read(archive/'summary.json'),vi,[s.sample_id for s in val])
    recipe = SimpleNamespace(**c['recipe'])
    if (recipe.route!='b_thermal_unet' or not recipe.gt_decode_fp32
            or recipe.condition_latent!='mode' or recipe.num_inference_steps!=1):
        raise ValueError('Unexpected parent deployment recipe')
    dataset = MS2ThermalDataset(Path(ti['path']),a.data_root,pseudo,
        norm_type='log_truncnorm',truncnorm_min=.02,truncnorm_max=.98,use_captions=True)
    if [s.sample_id for s in train_samples] != [r['id'] for r in dataset.rows]:
        raise ValueError('Training order mismatch')
    pair.seed_all(SEED)
    model = RouteModel(recipe,torch.device('cuda'),torch.float16)
    parent.parent_test.pilot.load_selected(model,'frame',inputs['source'],inputs['conv'],c)
    decoder, params = make_decoder(model)
    settings = dict(parent=str(inputs['parent']),parent_summary_sha256=pair.sha256_file(inputs['parent']/'summary.json'),
        source=str(inputs['source']),source_sha256=pair.sha256_file(inputs['source']),
        conv=str(inputs['conv']),conv_sha256=pair.sha256_file(inputs['conv']),
        recipe=c['recipe'],train=ti,val=vi,pseudo=str(pseudo),pseudo_config_sha256=c['pseudo_config'],
        trainable=list(params),lr=LR,batch=BATCH,accumulation=ACCUM,budget=STEPS[-1],steps=STEPS,
        seed=SEED,image_cursor=4000*BATCH*ACCUM,weight_decay=.01,augmentation='none: deployment input',
        pixel_loss_weight=.05,pair_weight=bias.PAIR_WEIGHT,bias_weight=bias.BIAS_WEIGHT,
        latent_loss='constant with frozen U-Net; omitted',reconstruction_loss='unchanged frozen task; omitted',
        prediction_inputs='thermal + frame caption + original manifest index; target_latent=None',
        policy=POLICY,newcrf=reference,code={f:pair.sha256_file(ROOT/f) for f in CODE})
    # JSON normalisation makes tuple/list comparisons stable across processes.
    import json
    settings = json.loads(json.dumps(settings))
    smoke = a.mode=='smoke'
    folder = a.run/('smoke' if smoke else 'train')
    folder.mkdir(parents=True,exist_ok=False)
    if smoke:
        if (a.run/'contract.json').exists():
            raise FileExistsError('Run already has a contract')
        pair.json_write(a.run/'contract.json',settings)
        baseline = validate(model,decoder,val,recipe,verify_initial=True)
        for k,tol in (('rmse',.002),('abs_rel',.00005),('a1',.00005)):
            if abs(baseline['metrics'][k]-inputs['baseline'][k])>tol:
                raise ValueError(f'Parent 4000 validation mismatch: {k}')
        print('[reproduced] caption 4000',baseline,flush=True)
        # The actual GPU inference result must survive changing ONLY the GT path.
        sample = train_samples[0]
        poisoned = dataclasses.replace(sample,thermal_gt_path='/GT_MUST_NOT_BE_READ/missing.png')
        raw1,z1 = predict_frozen(model,InferenceInput.from_sample(sample),recipe)
        raw2,z2 = predict_frozen(model,InferenceInput.from_sample(poisoned),recipe)
        torch.testing.assert_close(z1,z2,rtol=0,atol=0)
        torch.testing.assert_close(raw1,raw2,rtol=0,atol=0)
        print('[gt-isolation] PASS; nonexistent GT path leaves deployed prediction identical',flush=True)
        report = train(model,decoder,params,dataset,train_samples,recipe,folder,True)
        pair.json_write(a.run/'smoke_pass.json',dict(contract_sha256=pair.sha256_file(a.run/'contract.json'),
            baseline=baseline,training=report,gt_isolation=True))
        print('[smoke] PASS',report,flush=True)
        return
    passed = pair.read(a.run/'smoke_pass.json')
    if settings!=pair.read(a.run/'contract.json') or passed['contract_sha256']!=pair.sha256_file(a.run/'contract.json'):
        raise ValueError('Formal configuration differs from smoke')
    if pair.fingerprint(params.items()) != passed['training']['initial']:
        raise ValueError('Formal decoder differs from smoke start')
    report = train(model,decoder,params,dataset,train_samples,recipe,folder,False)
    curve = {0:passed['baseline']}
    for step in STEPS[1:]:
        pair.restore_params(params,folder/f'decoder_{step}.pt')
        curve[step] = validate(model,decoder,val,recipe)
        pair.json_write(folder/f'val_step{step:06d}.json',curve[step])
        print('[validation]',step,curve[step],flush=True)
    chosen, decision = select({s:v['metrics'] for s,v in curve.items()},reference['metrics'])
    path = folder/f'decoder_{chosen}.pt' if chosen else None
    pair.json_write(folder/'summary.json',dict(selected=chosen,curve=curve,decision=decision,
        selected_decoder=str(path) if path else None,selected_sha256=pair.sha256_file(path) if path else None,
        parent_step=4000,training=report,contract_sha256=pair.sha256_file(a.run/'contract.json')))
    print('[decision]',decision,flush=True)
    print('[selected] decoder updates',chosen,curve[chosen]['metrics'],flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['smoke','train'])
    for name in ('runs','run','data-root'):
        p.add_argument('--'+name,type=Path,required=True)
    os.environ.update(IRIS_THERMAL_STRETCH='percentile',IRIS_TEXT_PADDING='legacy',IRIS_DECODE_CLAMP='0')
    run(p.parse_args())
