"""Matched q98 decoder continuation vs last-two-up-block convolution joint tuning."""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import train_iris_decoder_only as old
from tools import eval_iris_decoder_only as prior
from tools.train_iris_tmax1_decoder import compare_metrics

pair, torch = old.pair, old.torch
ARMS = ('decoder', 'joint')
STEPS = (0, 500, 1000)
UNET_LR, DECODER_LR, LOSS_SCALE = 1e-6, 1e-5, 128.
CURSOR = 60000  # previous decoder used [48000,60000)
CODE = tuple(dict.fromkeys((*old.CODE, 'tools/eval_iris_decoder_only.py',
    'tools/train_iris_tmax1_decoder.py', 'tools/train_iris_decoder_joint.py',
    'tests/test_iris_decoder_joint.py', 'slurm/iris_decoder_joint.sbatch',
    'slurm/submit_iris_decoder_joint.sh')))


def configure(model, decoder, arm):
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    for module in (model.unet, model.lotus.vae, model.lotus.text_encoder, model.gt_vae, decoder):
        module.requires_grad_(False).eval()
        for p in module.parameters():
            p.grad = None
    decoder.decoder.requires_grad_(True)
    dp = {k:p for k,p in decoder.named_parameters() if p.requires_grad}
    up = pair.trainable_convs(model.unet) if arm == 'joint' else {}
    if not dp or any(not k.startswith('decoder.') for k in dp):
        raise ValueError('Wrong decoder scope')
    actual = {k for k,p in model.unet.named_parameters() if p.requires_grad}
    if actual != set(up):
        raise ValueError('Unexpected U-Net trainable parameter')
    return dp, up


def predict_latent(model, item, recipe, joint, trace=None):
    """Use the SAME deployed forward, allowing gradients only in the joint arm."""
    from tools.train_route_suite import load_input_tensor
    if type(item) is not old.InferenceInput or not item.caption.strip():
        raise ValueError('Expected GT-free captioned input')
    row = dict(id=item.sample_id,image_path=item.image_path,manifest_index=item.manifest_index)
    with torch.no_grad():
        image, _ = load_input_tensor(row,model.modality,recipe)
        prompt = model.encode_prompt(item.caption)
    if trace is not None:
        trace.update(json.dumps([item.sample_id,item.manifest_index,item.caption],ensure_ascii=True).encode())
        # Hash input pixels/text rather than GPU-dependent text activations.
        trace.update(str(recipe.seed+item.manifest_index).encode())
        for x in (image,):
            trace.update(x.detach().float().cpu().numpy().tobytes())
    with torch.set_grad_enabled(joint):
        raw, latent = model.predict_disparity(row,image,prompt,
            target_latent=None,return_latent=True,noise_offset=0)
        del raw  # unused original decoder graph must not survive the forward
    return latent if joint else latent.detach()


def loss_for_batch(decoder, latent, examples, indices):
    # Same three numerical loss primitives/weights as old.loss_for_batch.
    # The intentional difference is the absence of latent.detach().
    from train_iris_ms2_g import decode_image_depth, _ssi_log_aligned_depth
    gt = torch.stack([x['gt_depth_values'] for x in examples]).to(latent.device)
    valid = torch.stack([x['gt_valid_values'] for x in examples]).to(latent.device).bool()
    bounds = torch.stack([x['norm_bounds'] for x in examples]).to(latent.device)
    if not valid.any():
        raise ValueError('Batch has no real LiDAR supervision')
    raw, metric, lo, hi = decode_image_depth(decoder,latent,bounds,'log_truncnorm')
    if not all(torch.isfinite(x).all() for x in (raw,metric,lo,hi)):
        raise ValueError('Nonfinite depth/bounds')
    l1 = (metric[valid]-gt[valid]).abs().mean()
    aligned, usable, _, _ = _ssi_log_aligned_depth(raw,gt,valid)
    pairs = [old.bias.local_pairs(x['gt_depth_values'][0].numpy(),old.SEED+int(idx))
             for x,idx in zip(examples,indices,strict=True)]
    gap, count = old.bias.signed_gap_loss(aligned,gt,pairs,usable)
    far, frames = old.bias.far_bias_square(aligned,gt,valid,usable)
    total = .05*l1 + old.bias.PAIR_WEIGHT*gap + old.bias.BIAS_WEIGHT*far
    return total, dict(image_l1=float(l1.detach()),pair_loss=float(gap.detach()),
                       far_bias_m2=float(far.detach()),pairs=count,far_frames=frames)


class Optimizer:
    """FP32 masters avoid rounding away 1e-6 updates to native FP16 convolutions.

    Forward weights/dtypes remain the deployed ones. Accumulated scaled model
    gradients are copied/unscaled once per update; masters retain small updates.
    Decoder and U-Net are clipped separately so adding U-Net does not change the
    decoder's clipping threshold. Both arms use identical decoder numerics.
    """
    def __init__(self, dp, up):
        self.dp, self.up = dp, up
        self.master = {k:torch.nn.Parameter(p.detach().float().clone()) for k,p in up.items()}
        groups = [dict(params=list(dp.values()),lr=DECODER_LR)]
        if up:
            groups.append(dict(params=list(self.master.values()),lr=UNET_LR))
        self.optimizer = torch.optim.AdamW(groups,weight_decay=.01)

    def step(self):
        for p in self.dp.values():
            if p.grad is not None:
                p.grad.div_(LOSS_SCALE)
        for k,p in self.up.items():
            self.master[k].grad = None if p.grad is None else p.grad.detach().float()/LOSS_SCALE
        norms = {}
        for name, params in (('decoder',self.dp),('unet',self.master)):
            norms[name] = float(torch.nn.utils.clip_grad_norm_(list(params.values()),1.,
                error_if_nonfinite=True)) if params else 0.
        self.optimizer.step()
        with torch.no_grad():
            for k,p in self.up.items():
                p.copy_(self.master[k])
        self.optimizer.zero_grad(set_to_none=True)
        for p in self.up.values():
            p.grad = None
        return norms


def cpu_state(params):
    return {k:p.detach().cpu().clone() for k,p in params.items()}


def restore(params, state):
    if set(params) != set(state):
        raise ValueError('Checkpoint parameter scope differs')
    with torch.no_grad():
        for k,p in params.items():
            p.copy_(state[k])


def initial_state(model, dp):
    return dict(unet=old.fingerprint(model.unet.named_parameters()),decoder=old.fingerprint(dp.items()))


def train(model, decoder, dp, up, dataset, samples, recipe, folder, arm, smoke):
    budget = 4 if smoke else 1000
    optimizer = Optimizer(dp,up)
    before = old.fingerprint(old.frozen_parameters(model,decoder))
    initial = initial_state(model,dp)
    trace = hashlib.sha256()
    stream = iter(pair.ImageStream(len(samples),old.SEED,CURSOR,budget*old.BATCH*old.ACCUM))
    max_grads = dict(decoder=0.,unet=0.)
    supported = count = 0
    decoder.train()
    decoder.enable_gradient_checkpointing()
    # U-Net stays eval: changing scope must not introduce dropout/BN randomness.
    for update in range(1,budget+1):
        for micro in range(old.ACCUM):
            indices = [next(stream) for _ in range(old.BATCH)]
            latents = [predict_latent(model,old.InferenceInput.from_sample(samples[idx]),recipe,
                                     arm=='joint',trace) for idx in indices]
            latent = torch.cat(latents)
            if latent.requires_grad != (arm=='joint'):
                raise ValueError('Incorrect latent gradient boundary')
            examples = [dataset[idx] for idx in indices]  # only now load GT
            total, stats = loss_for_batch(decoder,latent,examples,indices)
            if not torch.isfinite(total):
                raise ValueError('Nonfinite loss')
            (total*(LOSS_SCALE/old.ACCUM)).backward()
            supported += stats['far_frames']
            count += stats['pairs']
            del latents, latent, examples
        if any(p.grad is not None for _,p in old.frozen_parameters(model,decoder)):
            raise ValueError('Gradient entered a frozen parameter')
        norms = optimizer.step()
        for k in max_grads:
            max_grads[k] = max(max_grads[k],norms[k])
        if smoke or update%50==0:
            print('[train]',arm,update,dict(loss=float(total.detach()),grad=norms,**stats),flush=True)
        if not smoke and update in STEPS[1:]:
            torch.save(dict(arm=arm,added_steps=update,decoder=cpu_state(dp),unet=cpu_state(up),
                unet_master=cpu_state(optimizer.master),optimizer=optimizer.optimizer.state_dict()),
                folder/f'weights_{update}.pt')
    decoder.disable_gradient_checkpointing()
    decoder.eval()
    final = initial_state(model,dp)
    if old.fingerprint(old.frozen_parameters(model,decoder)) != before:
        raise ValueError('Frozen parameters changed')
    if initial['decoder']==final['decoder'] or max_grads['decoder']<=0 or not supported or not count:
        raise ValueError('Decoder update or real supervision missing')
    changed = initial['unet'] != final['unet']
    if changed != (arm=='joint') or (arm=='joint' and max_grads['unet']<=0):
        raise ValueError('U-Net did not follow the requested scope')
    return dict(initial=initial,final=final,frozen_unchanged=True,input_trace=trace.hexdigest(),
                max_grad=max_grads,far_frames=supported,pairs=count,updates=budget)


def source_info(run):
    contract, expected, step, weights = prior.selection(run)
    if step!=1000 or contract['budget']!=1000 or contract['image_cursor']!=48000:
        raise ValueError('Expected the historical q98 decoder1000')
    reproduction = run/'train/test_selected_v1/reproduced.json'
    r = pair.read(reproduction)
    if r['step']!=1000 or r['decoder_sha256']!=pair.sha256_file(weights):
        raise ValueError('Wrong deployed validation source')
    prior.check_reproduction(r['measured'],expected)
    return contract, expected, weights, dict(run=str(run),contract_sha256=pair.sha256_file(run/'contract.json'),
        summary_sha256=pair.sha256_file(run/'train/summary.json'),decoder_sha256=pair.sha256_file(weights),
        reproduction_sha256=pair.sha256_file(reproduction))


def report(run):
    rows = {arm:pair.read(run/arm/'summary.json') for arm in ARMS}
    contract = pair.read(run/'contract.json')
    digest = pair.sha256_file(run/'contract.json')
    if any(r['contract_sha256']!=digest for r in rows.values()):
        raise ValueError('Mixed run contracts')
    if rows['decoder']['training']['input_trace']!=rows['joint']['training']['input_trace']:
        raise ValueError('Arm image/text/order traces differ')
    reference = contract['newcrf']['metrics']
    comparisons = {}
    for label in ('fixed1000','selected'):
        metrics = {}
        for arm,r in rows.items():
            step = 1000 if label=='fixed1000' else r['selected']
            metrics[arm] = r['curve'][str(step)]['metrics']
        comparisons[label] = compare_metrics(metrics['joint'],metrics['decoder'],reference)
        print('[joint-minus-decoder]',label,comparisons[label],flush=True)
    pair.json_write(run/'comparison.json',dict(contract_sha256=digest,comparisons=comparisons,
        note='Both arms use frame caption; this tests training scope, not caption causality. No test evaluation.'))


def run(a):
    if a.mode=='report':
        return report(a.run)
    from lotus.utils.ms2_thermal_dataset import MS2ThermalDataset
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    source, expected, weights, provenance = source_info(a.source_run)
    pair.seed_all(old.SEED)
    model, recipe, c = prior.build(a,source,weights)
    decoder, _ = old.make_decoder(model)
    samples, ti = pair.checked_manifest(Path(c['train']['path']),a.data_root,'train',75688)
    val, vi = pair.checked_manifest(Path(c['val']['path']),a.data_root,'val',14249)
    if ti!=source['train'] or vi!=source['val'] or c['recipe']!=source['recipe']:
        raise ValueError('Parent data/recipe differs')
    if pair.sha256_file(Path(c['pseudo']).parent/'build_config.json')!=source['pseudo_config_sha256']:
        raise ValueError('Pseudo calibration differs')
    archive = a.runs/'outputs/lotus_line_v2/iris_pair_rmse_val_v1/job_9167854'
    reference = old.read_reference(archive,pair.read(archive/'summary.json'),vi,[s.sample_id for s in val])
    if reference!=source['newcrf']:
        raise ValueError('NeWCRF reference differs')
    if (recipe.route!='b_thermal_unet' or recipe.condition_latent!='mode'
            or recipe.num_inference_steps!=1 or not recipe.gt_decode_fp32):
        raise ValueError('Unexpected inference recipe')
    dataset = MS2ThermalDataset(Path(ti['path']),a.data_root,Path(c['pseudo']),
        norm_type='log_truncnorm',truncnorm_min=.02,truncnorm_max=.98,use_captions=True)
    if [s.sample_id for s in samples]!=[r['id'] for r in dataset.rows]:
        raise ValueError('Training sample order differs')
    dp, up = configure(model,decoder,'joint')
    start_d, start_u = cpu_state(dp), cpu_state(up)
    initial = initial_state(model,dp)
    settings = dict(experiment='iris_decoder_joint_q98_v1',source=provenance,initial=initial,
        recipe=c['recipe'],train=ti,val=vi,pseudo=c['pseudo'],pseudo_config_sha256=source['pseudo_config_sha256'],
        arms=list(ARMS),decoder_trainable=list(dp),joint_unet_trainable=list(up),
        decoder_lr=DECODER_LR,unet_lr=UNET_LR,weight_decay=.01,clip_per_group=1.,loss_scale=LOSS_SCALE,
        steps=list(STEPS),budget=1000,batch=old.BATCH,accumulation=old.ACCUM,seed=old.SEED,image_cursor=CURSOR,
        losses=dict(image_l1=.05,pairs=old.bias.PAIR_WEIGHT,far_bias=old.bias.BIAS_WEIGHT),
        quantiles=[.02,.98],augmentation='none',optimizer='fresh AdamW, FP32 U-Net masters, native forward dtype',
        policy=old.POLICY,newcrf=reference,code={f:pair.sha256_file(ROOT/f) for f in CODE})
    settings = json.loads(json.dumps(settings))
    if a.mode=='smoke':
        folder = a.run/'smoke'
        folder.mkdir(parents=True,exist_ok=False)
        pair.json_write(a.run/'contract.json',settings)
        baseline = old.validate(model,decoder,val,recipe,verify_initial=True)
        prior.check_reproduction(baseline['metrics'],expected)
        print('[reproduced] q98 decoder1000',baseline,flush=True)
        item = old.InferenceInput.from_sample(samples[0])
        poison = old.InferenceInput.from_sample(dataclasses.replace(samples[0],thermal_gt_path='/NO_GT/missing.png'))
        with torch.no_grad():
            _, z0 = old.predict_frozen(model,item,recipe)
        reports = {}
        for arm in ARMS:
            restore(dp,start_d)
            restore(up,start_u)
            dp_arm, up_arm = configure(model,decoder,arm)
            pair.seed_all(old.SEED)
            for x in (item,poison):
                z = predict_latent(model,x,recipe,arm=='joint')
                torch.testing.assert_close(z.detach(),z0,rtol=0,atol=0)
                if z.requires_grad!=(arm=='joint'):
                    raise ValueError('Initial latent graph missing/leaking')
                del z
            reports[arm] = train(model,decoder,dp_arm,up_arm,dataset,samples,recipe,folder,arm,True)
            if reports[arm]['initial']!=initial:
                raise ValueError('Smoke arms did not start identically')
            print('[smoke]',arm,'PASS',reports[arm],flush=True)
        if reports['decoder']['input_trace']!=reports['joint']['input_trace']:
            raise ValueError('Smoke arm inputs differ')
        pair.json_write(a.run/'smoke_pass.json',dict(baseline=baseline,arms=reports,
            gt_isolation=True,contract_sha256=pair.sha256_file(a.run/'contract.json')))
        print('[paired-smoke] PASS',flush=True)
        return
    if a.arm not in ARMS:
        raise ValueError('Training needs an arm')
    passed = pair.read(a.run/'smoke_pass.json')
    digest = pair.sha256_file(a.run/'contract.json')
    if settings!=pair.read(a.run/'contract.json') or passed['contract_sha256']!=digest:
        raise ValueError('Configuration/initial weights changed after smoke')
    dp, up = configure(model,decoder,a.arm)
    folder = a.run/a.arm
    folder.mkdir(parents=True,exist_ok=False)
    pair.seed_all(old.SEED)
    training = train(model,decoder,dp,up,dataset,samples,recipe,folder,a.arm,False)
    curve = {0:passed['baseline']}
    for step in STEPS[1:]:
        state = torch.load(folder/f'weights_{step}.pt',map_location='cpu',weights_only=True)
        if state['arm']!=a.arm or state['added_steps']!=step:
            raise ValueError('Checkpoint arm/step mismatch')
        restore(dp,state['decoder'])
        restore(up,state['unet'])
        curve[step] = old.validate(model,decoder,val,recipe)
        pair.json_write(folder/f'val_step{step:06d}.json',curve[step])
        print('[validation]',a.arm,step,curve[step],flush=True)
    chosen, decision = old.select({s:r['metrics'] for s,r in curve.items()},reference['metrics'])
    selected = folder/f'weights_{chosen}.pt' if chosen else None
    pair.json_write(folder/'summary.json',dict(arm=a.arm,parent_decoder_step=1000,selected=chosen,
        selected_weights=str(selected) if selected else None,selected_sha256=pair.sha256_file(selected) if selected else None,
        contract_sha256=digest,training=training,curve=curve,decision=decision))
    print('[selected]',a.arm,'added_steps',chosen,decision['status'],curve[chosen]['metrics'],flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('smoke','train','report'))
    p.add_argument('--arm',choices=ARMS)
    p.add_argument('--run',type=Path,required=True)
    for name in ('runs','source-run','data-root'):
        p.add_argument('--'+name,type=Path)
    os.environ.update(IRIS_THERMAL_STRETCH='percentile',IRIS_TEXT_PADDING='legacy',IRIS_DECODE_CLAMP='0')
    run(p.parse_args())
