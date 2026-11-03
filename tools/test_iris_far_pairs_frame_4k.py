"""Test the frame-only continuation at the frozen full-validation selection."""
import argparse
import csv
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import extend_iris_far_pairs as ext
from tools import test_iris_far_pairs as pilot
from ms2_eval.io import sha256_file

read = ext.pair.read


def selection(run):
    settings = read(run/'contract.json')
    parent = Path(settings['parent'])
    contract, previous, source, start = ext.parent_inputs(parent)
    summary = read(run/'summary.json')
    full = {int(k): v for k, v in summary['full'].items()}
    if (set(full) != {ext.START, *ext.STEPS} or full[ext.START] != previous[str(ext.START)]
            or summary['original'] != previous['0'] or settings['parent_sha256'] != sha256_file(start)
            or settings['original_sha256'] != contract['source_sha256']):
        raise ValueError('Continuation does not match the validated parent')
    step = ext.select(full, previous['0'])
    conv = start if step == ext.START else run/f'conv_{step}.pt'
    if (summary['selected'] != step or Path(summary['selected_conv']) != conv
            or summary['selected_sha256'] != sha256_file(conv)):
        raise ValueError('Selected checkpoint differs from the frozen validation decision')
    return parent, contract, source, conv, step


def reuse(summary_path, samples, wanted, missing, protocol, arm_record):
    previous = read(summary_path)
    p = previous['provenance']
    if (any(p[k] != v for k, v in protocol.items()) or p['arms']['frame'] != arm_record
            or previous['frames'] != len(samples) or previous['common_frames'] != len(wanted)
            or previous['missing_official_ids'] != missing):
        raise ValueError('Previous test checkpoint, protocol or fold differs')
    metrics = pilot.OFFICIAL_METRICS
    totals = pilot.np.zeros((2, len(metrics)), dtype='float64')
    with (summary_path.parent/'frame_frames.csv').open(encoding='utf8', newline='') as f:
        rows = csv.reader(f)
        if next(rows) != ['id', *metrics]:
            raise ValueError('Unexpected cached CSV columns')
        for sample, (ident, *values) in zip(samples, rows, strict=True):
            if ident != sample.sample_id:
                raise ValueError('Cached frame IDs differ')
            row = pilot.np.asarray(values, dtype='float64')
            if len(row) != len(metrics) or not pilot.np.isfinite(row).all():
                raise ValueError('Invalid cached metric row')
            totals[0] += row
            if ident in wanted:
                totals[1] += row
    result = previous['results']['frame']
    for group, total, count in zip(('full', 'official_common'), totals, (len(samples), len(wanted))):
        if not pilot.np.allclose(total/count, [result[group][k] for k in metrics], rtol=1e-12, atol=1e-12):
            raise ValueError('Cached metrics differ from per-frame scores')
    return result


def run(a):
    parent, contract, source, conv, step = selection(a.run)
    suffix, prefix, n, sequences = pilot.TESTS[a.condition]
    samples, manifest = pilot.load_manifest(a.manifests/f'ms2_test_{suffix}_thermalcap_20260821.jsonl',
                                            a.data_root, allow_legacy_depth_path=True)
    if len(samples) != n or not manifest['sha256'].startswith(prefix):
        raise ValueError('Wrong test manifest')
    ids = pilot.validate_samples(samples, sequences)
    crawl = {r['id'] for r in pilot.official_frames(a.data_root, 'test_'+a.condition, 10)}
    wanted = ids & crawl
    if (len(wanted), len(crawl)) != pilot.COMMON[a.condition]:
        raise ValueError('Wrong official fold')
    missing = sorted(crawl - ids)
    folder = parent/'train'/'frame'
    record = dict(source=str(source), source_sha256=contract['source_sha256'], step=step,
                  conv_sha256=sha256_file(conv), contract_sha256=sha256_file(folder/'contract.json'),
                  validation_sha256=sha256_file(folder/'summary.json'))
    protocol = dict(manifest=manifest, condition=a.condition, alignment='ssi_log', valid_gt_m=[.001, 80.],
                    code_sha256=sha256_file(pilot.__file__), paired_sha256=sha256_file(parent/'paired.json'))
    out = a.run/'test_selected_v1'/a.condition
    out.mkdir(parents=True, exist_ok=True)
    provenance = dict(**protocol, frame=record, selection_sha256=sha256_file(a.run/'summary.json'),
                      runner_sha256=sha256_file(__file__))
    if step == ext.START:
        previous = parent/'test_selected_v1'/a.condition/'summary.json'
        result = reuse(previous, samples, wanted, missing, protocol, record)
        provenance.update(reused_from=str(previous), reused_sha256=sha256_file(previous))
        frames = previous.parent/'frame_frames.csv'
    else:
        from tools.train_route_suite import RouteModel
        os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
        recipe = SimpleNamespace(**contract['recipe'])
        model = RouteModel(recipe, pilot.torch.device('cuda'), pilot.torch.float16)
        pilot.load_selected(model, 'frame', source, conv, contract)
        tmp = out/'frame_frames.tmp'
        result = pilot.score(samples, pilot.iris.predict_samples(model, samples, 'frame', recipe), wanted, tmp)
        frames = out/'frame_frames.csv'
        os.replace(tmp, frames)
    pilot.json_write(out/'summary.json', dict(provenance=provenance, frames=n, common_frames=len(wanted),
                     missing_official_ids=missing, frame_csv=str(frames), results={'frame': result}))
    for group, label in [('full', 'test-full'), ('official_common', 'test-official')]:
        print(f'[{label}] frame {a.condition} step={step} reused={step == ext.START}', result[group], flush=True)
    print('[completed-test]', out/'summary.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('run', 'data-root', 'manifests'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--condition', choices=tuple(pilot.TESTS), required=True)
    run(parser.parse_args())
