"""Continue the validation-selected frame model using spatially distant LiDAR pairs."""
import argparse
import hashlib
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from torch.nn import functional as F
from tools import train_iris_far_pairs as pair
from tools.train_iris_far_pairs import ACCUM, BATCH, LR, PAIR_WEIGHT, old, signed_gap_loss, torch
from tools.test_iris_far_pairs_frame_4k import selection

BUDGET = 2000
OFFSETS = (1000, BUDGET)


def cross_region_pairs(gt, seed):
    """Sample distinct mid/far pairs at least 64 native pixels apart on either axis."""
    width = gt.shape[1]
    mid = np.flatnonzero((gt >= 10.) & (gt < 30.))
    far = np.flatnonzero((gt >= 40.) & (gt < 80.))
    if not len(mid) or not len(far):
        return np.empty((0, 2), dtype=np.int64)
    rng = np.random.default_rng(seed)
    a = rng.choice(mid, 2048)
    b = rng.choice(far, 2048)
    separate = np.maximum(abs(a // width - b // width), abs(a % width - b % width)) >= 64
    pairs = np.unique(np.column_stack((a[separate], b[separate])), axis=0)
    if len(pairs) > 256:
        pairs = pairs[rng.choice(len(pairs), 256, replace=False)]
    return pairs


class CrossRegionComponents(pair.TrainingComponents):
    # The parent training source is fingerprinted in previously completed contracts.
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
            indices = cross_region_pairs(batch['gt_depth_values'][i,0].numpy(), 43 + micro*BATCH + i)
            pairs.append(indices)
            trace.update(path.encode())
            trace.update(indices.tobytes())
        gap, count = signed_gap_loss(aligned, gt, pairs, usable)
        total = latent + reconstruction + .05*image_l1 + PAIR_WEIGHT*gap
        return total, gap, count


def train(net, params, dataset, collate, sd, out, start):
    components = CrossRegionComponents(sd)
    net.train()
    net.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(params.values(), lr=LR, weight_decay=.01)
    pair.seed_all(43)
    batches = iter(pair.DataLoader(dataset, batch_size=BATCH, collate_fn=collate,
                                   num_workers=0, sampler=pair.ImageStream(len(dataset), 43, start * BATCH * ACCUM, BUDGET * BATCH * ACCUM)))
    trace = hashlib.sha256()
    count = 0
    for micro in range(start * ACCUM, (start + BUDGET) * ACCUM):
        torch.manual_seed(43 + micro)
        loss, _, n = components.loss(net, next(batches), micro, trace)
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite loss at microbatch {micro}')
        count += n
        (loss / ACCUM).backward()
        if (micro + 1) % ACCUM:
            continue
        step = (micro + 1) // ACCUM
        norm = torch.nn.utils.clip_grad_norm_(params.values(), 1., error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 100 == 0:
            print(f'[train] step={step} loss={loss.item():.5f} grad={norm.item():.5f} pairs={count}', flush=True)
        if step - start in OFFSETS:
            if count == 0:
                raise ValueError('No usable laser pairs')
            torch.save({k: p.detach().cpu() for k, p in params.items()}, out/f'conv_{step}.pt')
    net.disable_gradient_checkpointing()
    net.eval()


def run(a):
    from tools.train_route_suite import RouteModel
    from utils.ms2_thermal_dataset import MS2ThermalDataset, MS2ThermalTransform, collate_fn_ms2
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Requires a BF16 CUDA GPU')
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_TEXT_PADDING='legacy', IRIS_DECODE_CLAMP='0')
    _, c, source, start_file, start = selection(a.parent)
    selection_path = a.parent/'summary.json'
    previous = pair.read(selection_path)
    pseudo = Path(c['pseudo'])
    if pair.sha256_file(pseudo.parent/'build_config.json') != c['pseudo_config']:
        raise ValueError('Parent calibrated pseudo-label configuration changed')
    train_rows, ti = pair.checked_manifest(Path(c['train']['path']), a.data_root, 'train', 75688)
    val, vi = pair.checked_manifest(Path(c['val']['path']), a.data_root, 'val', 14249)
    if ti != c['train'] or vi != c['val']:
        raise ValueError('Parent manifests changed')
    ds = MS2ThermalDataset(Path(ti['path']), a.data_root, pseudo,
                           transform=MS2ThermalTransform(random_flip=True), norm_type='log_truncnorm',
                           truncnorm_min=.02, truncnorm_max=.98, use_captions=True)
    if [s.sample_id for s in train_rows] != [row['id'] for row in ds.rows]:
        raise ValueError('Training sample order changed')
    a.out.mkdir(parents=True, exist_ok=False)
    recipe = SimpleNamespace(**c['recipe'])
    model = RouteModel(recipe, torch.device('cuda'), torch.float16)
    old.load_arm(model, source, 'frame')
    params = pair.trainable_convs(model.unet)
    if list(params) != c['trainable']:
        raise ValueError('Convolution scope changed')
    pair.restore_params(params, start_file)
    def score():
        predictions = old.predict_samples(model, val, 'frame', recipe)
        return old.measure(val, predictions, [0.])[0.]


    reference = previous['full'][str(start)]
    reproduced = score()
    for k in ('rmse', 'abs_rel', 'a1'):
        if not math.isclose(reproduced[k], reference[k], rel_tol=0.,
                            abs_tol=.002 if k == 'rmse' else .00005):
            raise ValueError(f'Selected parent does not reproduce {k}')
    pair.json_write(a.out/'contract.json', dict(parent=str(a.parent), start=start,
        selection_sha256=pair.sha256_file(selection_path), source=str(source), start_conv=str(start_file),
        start_conv_sha256=pair.sha256_file(start_file), train=ti, val=vi, lr=LR, pair_weight=PAIR_WEIGHT,
        initial_image_cursor=start*BATCH*ACCUM, optimizer='fresh AdamW', budget=BUDGET,
        mid_m=[10,30], far_m=[40,80], min_separation_pixels=64, gain_m=.02))
    print('[start] frame', start, start_file, reproduced, flush=True)
    train(model.unet, params, ds, collate_fn_ms2, c['sd'], a.out, start)
    torch.cuda.empty_cache()
    full = {0: reference}
    for offset in OFFSETS:
        pair.restore_params(params, a.out/f'conv_{start + offset}.pt')
        full[offset] = score()
        print('[validation] added_steps', offset, full[offset], flush=True)
    original = previous['original']
    eligible = [off for off in OFFSETS if full[off]['rmse'] <= reference['rmse'] - .02
                and full[off]['abs_rel'] <= original['abs_rel'] + old.ABS_TOL
                and full[off]['a1'] >= original['a1'] - old.A1_TOL]
    best = min(eligible, key=lambda s: (full[s]['rmse'], s), default=0)
    selected = a.out/f'conv_{start + best}.pt' if best else start_file
    pair.json_write(a.out/'summary.json', dict(start=start, full_by_added_steps=full,
        selected=start+best, added_steps=best, selected_conv=str(selected), selected_sha256=pair.sha256_file(selected)))
    print('[selected] frame', start + best, 'added_steps', best, full[best], flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('parent', 'data-root', 'out'):
        parser.add_argument('--'+flag, type=Path, required=True)
    run(parser.parse_args())
