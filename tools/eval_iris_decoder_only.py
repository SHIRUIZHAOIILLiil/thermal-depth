"""Reproduce the selected VAE decoder through the deployed path, then score it on test once."""
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.iris_decoder_only import make_decoder, select
from tools.train_iris_decoder_only import pair, parent, torch

pilot = parent.parent_test.pilot
TOLERANCE = (('abs_rel', .00005), ('rmse', .002), ('a1', .00005))


def selection(run):
    """Accept only the decoder chosen on validation by the frozen rule."""
    contract = pair.read(run/'contract.json')
    summary = pair.read(run/'train'/'summary.json')
    for name, digest in contract['code'].items():
        if pair.sha256_file(ROOT/name) != digest:
            raise ValueError(f'Training code changed: {name}')
    if summary['contract_sha256'] != pair.sha256_file(run/'contract.json'):
        raise ValueError('Summary belongs to a different contract')
    curve = {int(step): row['metrics'] for step, row in summary['curve'].items()}
    chosen = select(curve, summary['decision']['reference'])[0]
    if chosen != summary['selected']:
        raise ValueError('Selected step differs from the frozen validation rule')
    if not chosen:
        raise ValueError('Validation retained the start; nothing new to test')
    path = run/'train'/f'decoder_{chosen}.pt'
    if pair.sha256_file(path) != summary['selected_sha256']:
        raise ValueError('Selected decoder file differs from validation')
    return contract, curve[chosen], chosen, path


def check_reproduction(measured, expected):
    for key, tolerance in TOLERANCE:
        if abs(measured[key]-expected[key]) > tolerance:
            raise ValueError(f'Deployed path does not reproduce {key}: {measured[key]} vs {expected[key]}')


def install_decoder(model, path):
    """predict_disparity decodes with model.gt_vae, so the selected weights go there."""
    decoder, params = make_decoder(model)
    pair.restore_params(params, path)
    model.gt_vae = decoder.requires_grad_(False).eval()
    return model


def build(a, contract, path):
    from tools.train_route_suite import RouteModel
    inputs = parent.parent_inputs(a.runs)
    c = inputs['contract']
    if (contract['conv_sha256'] != pair.sha256_file(inputs['conv'])
            or contract['source_sha256'] != pair.sha256_file(inputs['source'])):
        raise ValueError('Parent weights differ from the decoder contract')
    recipe = SimpleNamespace(**c['recipe'])
    model = RouteModel(recipe, torch.device('cuda'), torch.float16)
    pilot.load_selected(model, 'frame', inputs['source'], inputs['conv'], c)
    return install_decoder(model, path), recipe, c


def reproduce(a):
    contract, expected, step, path = selection(a.run)
    out = a.run/'train'/'test_selected_v1'
    out.mkdir(parents=True, exist_ok=True)
    model, recipe, c = build(a, contract, path)
    samples, _ = pair.checked_manifest(Path(c['val']['path']), a.data_root, 'val', 14249)
    tmp = out/'val_frames.tmp'
    measured = pilot.score(samples, pilot.iris.predict_samples(model, samples, 'frame', recipe),
                           {s.sample_id for s in samples}, tmp)['full']
    check_reproduction(measured, expected)
    os.replace(tmp, out/'val_frames.csv')
    pair.json_write(out/'reproduced.json', dict(step=step, decoder_sha256=pair.sha256_file(path),
                                                measured=measured, expected=expected))
    print('[reproduced]', step, measured, flush=True)


def test(a):
    contract, _, step, path = selection(a.run)
    digest = pair.sha256_file(path)
    folder = a.run/'train'/'test_selected_v1'
    if pair.read(folder/'reproduced.json')['decoder_sha256'] != digest:
        raise ValueError('Decoder was not reproduced through the deployed path')
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
                      step=step, decoder_sha256=digest, contract_sha256=pair.sha256_file(a.run/'contract.json'),
                      selection_sha256=pair.sha256_file(a.run/'train'/'summary.json'),
                      runner_sha256=pair.sha256_file(__file__), code_sha256=pair.sha256_file(pilot.__file__))
    out = folder/a.condition
    out.mkdir(parents=True, exist_ok=True)
    if (out/'summary.json').is_file():
        if pair.read(out/'summary.json')['provenance'] != provenance:
            raise ValueError('A different test run already exists')
        print('[completed-test] already scored:', out/'summary.json', flush=True)
        return
    model, recipe, _ = build(a, contract, path)
    tmp = out/'frame_frames.tmp'
    results = pilot.score(samples, pilot.iris.predict_samples(model, samples, 'frame', recipe), wanted, tmp)
    os.replace(tmp, out/'frame_frames.csv')
    pair.json_write(out/'summary.json', dict(provenance=provenance, frames=n, common_frames=len(wanted),
                                             missing_official_ids=missing, results={'frame': results}))
    for group, label in (('full', 'test-full'), ('official_common', 'test-official')):
        print(f'[{label}] frame {a.condition} decoder={step}', results[group], flush=True)
    print('[completed-test]', out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('reproduce', 'test'))
    for name in ('runs', 'run', 'data-root', 'manifests'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--condition', choices=('day', 'night', 'rain'))
    args = p.parse_args()
    if (args.mode == 'test') != (args.condition is not None):
        p.error('--condition is required for test and only for test')
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_TEXT_PADDING='legacy', IRIS_DECODE_CLAMP='0')
    if not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA')
    (reproduce if args.mode == 'reproduce' else test)(args)
