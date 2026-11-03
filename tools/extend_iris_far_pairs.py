"""Extend only the validated frame arm from 2k local-pair updates to 4k."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import train_iris_far_pairs as pair
from tools.train_iris_far_pairs import ACCUM, BATCH, LR, old, torch

START, STEPS = 2000, (3000, 4000)
GAIN = .02


def select(full, original):
    base = full[START]
    eligible = [s for s in STEPS if full[s]['rmse'] <= base['rmse'] - GAIN
                and full[s]['abs_rel'] <= original['abs_rel'] + old.ABS_TOL
                and full[s]['a1'] >= original['a1'] - old.A1_TOL]
    return min(eligible, key=lambda s: (full[s]['rmse'], s), default=START)


def parent_inputs(parent):
    folder = parent/'train'/'frame'
    contract = pair.read(folder/'contract.json')
    summary = pair.read(folder/'summary.json')
    if (contract['arm'] != 'frame' or summary['selected'] != START
            or pair.read(parent/'paired.json')['selected']['frame'] != START
            or pair.sha256_file(folder/'contract.json') != pair.read(parent/'smoke_pass.json')['frame']):
        raise ValueError('Expected the completed, smoke-checked frame arm selected at 2000')
    for name, expected in contract['code'].items():
        actual = pair.sha256_file(pair.ROOT/name)
        # bbea99bf5 only converted the three NumPy gate booleans to Python bool.
        if (name == 'tools/iris_far_stretch.py'
                and expected == '4312bc091e09ebce7a0e8610328b686946ddd24bb84dcc55ac7fa82633e8c335'
                and actual == 'd53705b8a3308dc4d0e5e261ed7ad7dff73e65ffa072c9ffb36265a6f8982e7a'):
            continue
        if actual != expected:
            raise ValueError(f'Parent training code differs: {name}')
    source = Path(contract['source'])
    if (source.parts[-3:] != (old.RUNS['frame'], 'converted', 'step16000_weights.pt')
            or pair.sha256_file(source) != contract['source_sha256']):
        raise ValueError('Original 16k frame weights differ')
    return contract, summary['full'], source, folder/'conv_2000.pt'


def stream(size):
    return pair.ImageStream(size, 43, START * BATCH * ACCUM, (STEPS[-1]-START) * BATCH * ACCUM)


def train(net, params, dataset, collate, sd, out):
    components = pair.TrainingComponents(sd)
    net.train()
    net.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(params.values(), lr=LR, weight_decay=.01)
    pair.seed_all(43)
    batches = iter(pair.DataLoader(dataset, batch_size=BATCH, collate_fn=collate,
                                   num_workers=0, sampler=stream(len(dataset))))
    trace = hashlib.sha256()
    count = 0
    for micro in range(START * ACCUM, STEPS[-1] * ACCUM):
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
        if step in STEPS:
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
    c, previous, source, start_file = parent_inputs(a.parent)
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

    full = {START: score()}
    for k in ('rmse', 'abs_rel', 'a1'):
        if not math.isclose(full[START][k], previous[str(START)][k], rel_tol=0.,
                            abs_tol=.002 if k == 'rmse' else .00005):
            raise ValueError(f'Starting weights did not reproduce validation {k}')
    # Use the frozen parent full-validation row as the 2 cm reference.
    reproduced = full[START]
    full[START] = previous[str(START)]
    paths = {START: start_file, **{s: a.out/f'conv_{s}.pt' for s in STEPS}}
    pair.json_write(a.out/'contract.json', dict(
        parent=str(a.parent), parent_weights=str(start_file), parent_sha256=pair.sha256_file(start_file),
        original_weights=str(source), original_sha256=c['source_sha256'], val=vi, train=ti,
        initial_image_cursor=START*BATCH*ACCUM, optimizer='fresh AdamW', lr=LR,
        gain_m=GAIN, original_absrel_tol=old.ABS_TOL, original_a1_tol=old.A1_TOL))
    print('[reproduced] frame 2000', reproduced, flush=True)
    train(model.unet, params, ds, collate_fn_ms2, c['sd'], a.out)
    torch.cuda.empty_cache()
    for s in STEPS:
        pair.restore_params(params, paths[s])
        full[s] = score()
        print('[validation] frame', s, full[s], flush=True)
    selected = select(full, previous['0'])
    pair.json_write(a.out/'summary.json', dict(full=full, original=previous['0'], reproduced=reproduced,
        selected=selected, selected_conv=str(paths[selected]), selected_sha256=pair.sha256_file(paths[selected])))
    print('[selected] frame', selected, full[selected], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('parent', 'data-root', 'out'):
        parser.add_argument('--'+flag, type=Path, required=True)
    run(parser.parse_args())
