"""Train a frame-caption local-pair model with one additional far-band bias loss."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from torch.nn import functional as F
from tools import diagnose_pair_rmse as diag
from tools import extend_iris_far_pairs as ext
from tools import train_iris_far_pairs as pair
from tools import test_iris_cross_region_pairs as test_runner
from tools.train_iris_far_pairs import ACCUM, BATCH, LR, PAIR_WEIGHT, local_pairs, signed_gap_loss, torch

BIAS_WEIGHT = .0005
MIN_PIXELS = 50


def far_bias_square(aligned, gt, valid, usable):
    """Mean of squared frame means in metres, with equal weight per supported frame."""
    mask = valid.bool() & (gt >= 60.) & (gt < 80.)
    count = mask.flatten(1).sum(1)
    keep = (count >= MIN_PIXELS) & torch.as_tensor(usable, device=count.device, dtype=torch.bool)
    error = torch.where(mask, aligned.float().clamp(.001, 80.) - gt, 0.)
    mean = error.flatten(1).sum(1) / count.clamp_min(1)
    num = int(keep.sum().item())
    return mean[keep].square().sum() / max(num, 1), num


class BiasComponents(pair.TrainingComponents):
    # Keep the historical, hash-pinned local-pair trainer intact.
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
        bias, frames = far_bias_square(aligned, gt, valid, usable)
        return total + BIAS_WEIGHT * bias, bias, frames, count


def train(net, params, dataset, collate, sd, folder, smoke):
    components = BiasComponents(sd)
    net.train()
    net.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(params.values(), lr=LR, weight_decay=.01)
    pair.seed_all(43)
    start = ext.START
    stop = start + (4 if smoke else 2000)
    sampler = pair.ImageStream(len(dataset), 43, start*BATCH*ACCUM, (stop-start)*BATCH*ACCUM)
    batches = iter(pair.DataLoader(dataset, batch_size=BATCH, collate_fn=collate, num_workers=0, sampler=sampler))
    trace = hashlib.sha256()
    supported = pairs = 0
    bias_grad = 0.
    for micro in range(start*ACCUM, stop*ACCUM):
        torch.manual_seed(43+micro)
        loss, bias, n, num_pairs = components.loss(net, next(batches), micro, trace)
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite loss at microbatch {micro}')
        pairs += num_pairs
        supported += n
        if smoke and n and bias_grad == 0.:
            grads = torch.autograd.grad(bias, list(params.values()), retain_graph=True, allow_unused=True)
            bias_grad = max(float(g.detach().abs().max()) for g in grads if g is not None)
        (loss/ACCUM).backward()
        step, remainder = divmod(micro+1, ACCUM)
        if remainder:
            continue
        norm = torch.nn.utils.clip_grad_norm_(params.values(), 1., error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if smoke or step % 100 == 0:
            print(f'[train] step={step} total={loss.item():.6f} last_micro_bias_m2={bias.item():.4f} '
                  f'far_frames={supported} pairs={pairs} grad={float(norm):.5f}', flush=True)
        if not smoke and step in ext.STEPS:
            torch.save({k: p.detach().cpu() for k,p in params.items()}, folder/f'conv_{step}.pt')
    if not supported or not pairs or smoke and not (math.isfinite(bias_grad) and bias_grad > 0):
        raise ValueError('Missing supervision or far bias gradient')
    net.disable_gradient_checkpointing()
    net.eval()
    return dict(far_frames=supported, pairs=pairs, trace=trace.hexdigest(), smoke_bias_grad=bias_grad if smoke else None)


def validate(model, recipe, val, archive):
    """Report the same 40–60/60–80 attribution as job 9167854 during normal validation."""
    rows = []
    def predictions():
        for ident, raw, gt in pair.old.predict_samples(model, val, 'frame', recipe):
            rows.append(diag.moments(raw, gt, 'ssi_log'))
            yield ident, raw, gt
    metrics = pair.old.measure(val, predictions(), [0.])[0.]
    current = tuple(map(np.asarray, zip(*rows)))
    ids = [s.sample_id for s in val]
    with np.load(archive/'newcrf_moments.npz', allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], ids)
        ref = tuple(z[k] for k in ('counts','signed','squared','official_rmse'))
    diag.close(current[3].mean(), metrics['rmse'])
    seqs = np.array([s.rsplit('_',1)[0] for s in ids])
    return metrics, {r['depth']:r for r in diag.breakdown(current, ref, seqs)
                     if r['sequence']=='ALL' and r['depth'] in ('40-60m','60-80m')}


def run(a):
    from tools.train_route_suite import RouteModel
    from utils.ms2_thermal_dataset import MS2ThermalDataset, MS2ThermalTransform, collate_fn_ms2
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Requires a BF16 CUDA GPU')
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_TEXT_PADDING='legacy', IRIS_DECODE_CLAMP='0')
    _, c, source, conv, start = test_runner.prior.selection(a.parent)
    if start != ext.START:
        raise ValueError('Expected the selected frame local-pair 2000 checkpoint')
    parent_result = pair.read(a.parent/'summary.json')
    ref = parent_result['full'][str(start)]
    archive = pair.read(a.archive/'summary.json')
    digest = pair.sha256_file(conv)
    if archive['sha256']['local'] != digest or archive['validation'] != c['val'] or archive['align']['newcrf'] != 'ssi':
        raise ValueError('Attribution source does not match the training parent')
    pseudo = Path(c['pseudo'])
    if pair.sha256_file(pseudo.parent/'build_config.json') != c['pseudo_config']:
        raise ValueError('Pseudo-label configuration differs from parent')
    train_rows, ti = pair.checked_manifest(Path(c['train']['path']), a.data_root, 'train', 75688)
    val, vi = pair.checked_manifest(Path(c['val']['path']), a.data_root, 'val', 14249)
    if ti != c['train'] or vi != c['val']:
        raise ValueError('Parent manifests changed')
    dataset = MS2ThermalDataset(Path(ti['path']), a.data_root, pseudo,
        transform=MS2ThermalTransform(random_flip=True), norm_type='log_truncnorm',
        truncnorm_min=.02, truncnorm_max=.98, use_captions=True)
    if [s.sample_id for s in train_rows] != [r['id'] for r in dataset.rows]:
        raise ValueError('Training order differs from original')
    smoke = a.mode == 'smoke'
    folder = a.run/'smoke' if smoke else a.run
    if not smoke and (folder/'contract.json').exists():
        raise FileExistsError('Training directory already used')
    folder.mkdir(parents=True, exist_ok=not smoke)
    recipe = SimpleNamespace(**c['recipe'])
    model = RouteModel(recipe, torch.device('cuda'), torch.float16)
    pair.old.load_arm(model, source, 'frame')
    params = pair.trainable_convs(model.unet)
    if list(params) != c['trainable']:
        raise ValueError('Trainable convolutions differ')
    pair.restore_params(params, conv)
    contract = dict(parent=str(a.parent), start=start, selection_sha256=pair.sha256_file(a.parent/'summary.json'),
        source=str(source), start_conv=str(conv), start_conv_sha256=digest, val=vi, train=ti,
        trainable=list(params), lr=LR, pair_weight=PAIR_WEIGHT, bias_weight=BIAS_WEIGHT,
        far_m=[60,80], min_pixels=MIN_PIXELS, initial_image_cursor=start*BATCH*ACCUM, gain_m=ext.GAIN,
        optimizer='fresh AdamW', budget=2000, runner_sha256=pair.sha256_file(__file__),
        reference_stats_sha256=pair.sha256_file(a.archive/'newcrf_moments.npz'))
    if not smoke and pair.read(a.run/'smoke'/'passed.json')['contract'] != contract:
        raise ValueError('Configuration differs from smoke')
    pair.json_write(folder/'contract.json', contract)
    if smoke:
        frozen = [(k,p) for k,p in model.unet.named_parameters() if not p.requires_grad]
        before, initial = pair.fingerprint(frozen), pair.fingerprint(params.items())
        report = train(model.unet, params, dataset, collate_fn_ms2, c['sd'], folder, True)
        if before != pair.fingerprint(frozen) or initial == pair.fingerprint(params.items()):
            raise ValueError('Smoke update scope mismatch')
        pair.json_write(folder/'passed.json', dict(contract=contract, **report))
        print('[smoke] PASS', report, flush=True)
        return
    metrics, parts = validate(model, recipe, val, a.archive)
    for k,tol in [('rmse',.002), ('abs_rel',.00005), ('a1',.00005)]:
        if abs(metrics[k]-ref[k]) > tol:
            raise ValueError(f'Start does not reproduce {k}')
    baseline_part = next(r for r in archive['attribution'] if (r['a'],r['b'],r['sequence'],r['depth'])
                         == ('local','newcrf','ALL','60-80m'))['common_bias_gap_m']
    if abs(parts['60-80m']['common_bias_gap_m'] - baseline_part) > .002:
        raise ValueError('Far bias decomposition did not reproduce')
    full, mechanism = {start:ref}, {start:parts}
    report = train(model.unet, params, dataset, collate_fn_ms2, c['sd'], folder, False)
    torch.cuda.empty_cache()
    for step in ext.STEPS:
        pair.restore_params(params, folder/f'conv_{step}.pt')
        full[step], mechanism[step] = validate(model, recipe, val, a.archive)
        print('[validation]', step, full[step], flush=True)
        print('[mechanism]', step, mechanism[step], flush=True)
    selected = ext.select(full, parent_result['original'])
    weight = conv if selected == start else folder/f'conv_{selected}.pt'
    pair.json_write(folder/'summary.json', dict(start=start, selected=selected, added_steps=selected-start,
        full_by_added_steps={s-start:m for s,m in full.items()}, selected_conv=str(weight),
        selected_sha256=pair.sha256_file(weight), mechanism_by_step=mechanism, train=report))
    print('[selected] frame', selected, full[selected], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['smoke','train'])
    for key in ('run','parent','data-root','archive'):
        p.add_argument('--'+key, type=Path, required=True)
    run(p.parse_args())
