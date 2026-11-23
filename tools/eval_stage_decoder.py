"""Score a decoder-stage run (train_iris_tmax1_decoder.run) on test, once, after reproducing validation.

For the ablation's last rows (docs/SD2_POSTHOC_V1.md): runs from train_iris_synth_decoder.py, i.e. a frozen
U-Net (step 16000 or a 12000-20000 average) plus the adapted depth decoder.
    --stage start     the frozen U-Net with the original VAE decoder (decoder updates 0): the "+ averaging" row
    --stage selected  the decoder step chosen on validation by the frozen rule: the final row
Identity is checked by sha256 of the U-Net payload and the decoder file, and the deployed path must reproduce
the run's own full-validation number before any test frame is read. Unlike eval_iris_decoder_only.py this
does not require the training code to be byte-identical: the weights and the reproduction are the check.
"""
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import eval_iris_decoder_only as prior
from tools.iris_decoder_only import select

pair, torch, pilot = prior.pair, prior.torch, prior.pilot


def selection(run, stage):
    """(contract, expected val metrics, decoder updates, decoder path or None)."""
    contract = pair.read(run/'contract.json')
    summary = pair.read(run/'train'/'summary.json')
    if summary['contract_sha256'] != pair.sha256_file(run/'contract.json'):
        raise ValueError('Summary belongs to a different contract')
    curve = {int(step): row['metrics'] for step, row in summary['curve'].items()}
    if stage == 'start':
        return contract, curve[0], 0, None
    chosen = select(curve, summary['decision']['reference'])[0]
    if chosen != summary['selected']:
        raise ValueError('Selected step differs from the frozen validation rule')
    if not chosen:
        raise ValueError('Validation retained the start; test it with --stage start')
    path = run/'train'/f'decoder_{chosen}.pt'
    if pair.sha256_file(path) != summary['selected_sha256']:
        raise ValueError('Selected decoder file differs from validation')
    return contract, curve[chosen], chosen, path


def load_unet(model, contract):
    source = Path(contract['source'])
    if pair.sha256_file(source) != contract['source_sha256']:
        raise ValueError(f'U-Net weights changed since training: {source}')
    payload = torch.load(source, map_location='cpu', weights_only=False)
    if 'averaged_steps' in payload:
        if (payload.get('route') != 'b_thermal_unet' or payload.get('caption_mode') != 'correct'
                or set(payload.get('state_dicts', {})) != {'unet'}):
            raise ValueError(f'Unexpected averaged payload: {source}')
        model.load_state_dicts(payload['state_dicts'], str(source))
        model.unet.requires_grad_(False)
        model.set_train(False)
    else:
        pair.old.load_arm(model, source, 'frame')
    return source


def build(contract, decoder):
    from tools.train_route_suite import RouteModel
    recipe = SimpleNamespace(**contract['recipe'])
    model = RouteModel(recipe, torch.device('cuda'), torch.float16)
    load_unet(model, contract)
    if decoder is not None:
        prior.install_decoder(model, decoder)
    return model, recipe


def folder(run, stage):
    return run/'train'/f'test_{stage}_v1'


def reproduce(a):
    contract, expected, step, decoder = selection(a.run, a.stage)
    out = folder(a.run, a.stage)
    out.mkdir(parents=True, exist_ok=True)
    model, recipe = build(contract, decoder)
    samples, info = pair.checked_manifest(Path(contract['val']['path']), a.data_root, 'val', 14249)
    if info != contract['val']:
        raise ValueError('Validation manifest differs from the run')
    tmp = out/'val_frames.tmp'
    measured = pilot.score(samples, pilot.iris.predict_samples(model, samples, 'frame', recipe),
                           {s.sample_id for s in samples}, tmp)['full']
    prior.check_reproduction(measured, expected)
    os.replace(tmp, out/'val_frames.csv')
    pair.json_write(out/'reproduced.json', dict(stage=a.stage, step=step, unet_sha256=contract['source_sha256'],
        decoder_sha256=pair.sha256_file(decoder) if decoder else None, measured=measured, expected=expected))
    print('[reproduced]', a.stage, step, measured, flush=True)


def test(a):
    contract, _, step, decoder = selection(a.run, a.stage)
    digest = pair.sha256_file(decoder) if decoder else None
    base = folder(a.run, a.stage)
    done = pair.read(base/'reproduced.json')
    if done['decoder_sha256'] != digest or done['unet_sha256'] != contract['source_sha256']:
        raise ValueError('These weights were not reproduced through the deployed path')
    suffix, prefix, n, seqs = pilot.TESTS[a.condition]
    samples, manifest = pilot.load_manifest(a.manifests/f'ms2_test_{suffix}_thermalcap_20260821.jsonl',
                                            a.data_root, allow_legacy_depth_path=True)
    if len(samples) != n or not manifest['sha256'].startswith(prefix):
        raise ValueError('Incorrect test manifest')
    ids = pilot.validate_samples(samples, seqs)
    crawl = {r['id'] for r in pilot.official_frames(a.data_root, 'test_'+a.condition, 10)}
    wanted, missing = ids & crawl, sorted(crawl-ids)
    if (len(wanted), len(crawl)) != pilot.COMMON[a.condition]:
        raise ValueError('Official fold differs')
    provenance = dict(manifest=manifest, condition=a.condition, alignment='ssi_log', valid_gt_m=[.001, 80.],
                      stage=a.stage, step=step, unet_sha256=contract['source_sha256'], decoder_sha256=digest,
                      contract_sha256=pair.sha256_file(a.run/'contract.json'), runner_sha256=pair.sha256_file(__file__))
    out = base/a.condition
    out.mkdir(parents=True, exist_ok=True)
    if (out/'summary.json').is_file():
        if pair.read(out/'summary.json')['provenance'] != provenance:
            raise ValueError('A different test run already exists')
        print('[completed-test] already scored:', out/'summary.json', flush=True)
        return
    model, recipe = build(contract, decoder)
    tmp = out/'frame_frames.tmp'
    results = pilot.score(samples, pilot.iris.predict_samples(model, samples, 'frame', recipe), wanted, tmp)
    os.replace(tmp, out/'frame_frames.csv')
    pair.json_write(out/'summary.json', dict(provenance=provenance, frames=n, common_frames=len(wanted),
                                             missing_official_ids=missing, results={'frame': results}))
    for group, label in (('full', 'test-full'), ('official_common', 'test-official')):
        print(f'[{label}] {a.stage} {a.condition} decoder={step}', results[group], flush=True)
    print('[completed-test]', out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('reproduce', 'test'))
    for name in ('run', 'data-root', 'manifests'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--stage', choices=('start', 'selected'), required=True)
    p.add_argument('--condition', choices=('day', 'night', 'rain'))
    args = p.parse_args()
    if (args.mode == 'test') != (args.condition is not None):
        p.error('--condition is required for test and only for test')
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_TEXT_PADDING='legacy', IRIS_DECODE_CLAMP='0')
    if not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA')
    (reproduce if args.mode == 'reproduce' else test)(args)
