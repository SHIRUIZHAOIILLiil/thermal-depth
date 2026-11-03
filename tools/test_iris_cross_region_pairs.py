"""Evaluate the validation-selected cross-region model or verify its parent's test rows."""
import argparse
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import test_iris_far_pairs_frame_4k as prior
from tools.train_iris_cross_region_pairs import OFFSETS
from tools.test_iris_far_pairs_frame_4k import pilot, read, sha256_file


def checkpoint(run, start, start_conv, parent_summary):
    summary = read(run/'summary.json')
    full = summary['full_by_added_steps']
    baseline = parent_summary['full'][str(start)]
    if (summary['start'] != start or full['0'] != baseline
            or set(full) != {str(s) for s in (0, *OFFSETS)}):
        raise ValueError('Incomplete or mismatched validation curve')
    original = parent_summary['original']
    eligible = [s for s in OFFSETS if full[str(s)]['rmse'] <= baseline['rmse'] - .02
                and full[str(s)]['abs_rel'] <= original['abs_rel'] + pilot.iris.ABS_TOL
                and full[str(s)]['a1'] >= original['a1'] - pilot.iris.A1_TOL]
    added = min(eligible, key=lambda s: (full[str(s)]['rmse'], s), default=0)
    step = start + added
    conv = run/f'conv_{step}.pt' if added else start_conv
    if (summary['added_steps'] != added or summary['selected'] != step
            or Path(summary['selected_conv']) != conv or sha256_file(conv) != summary['selected_sha256']):
        raise ValueError('Selected cross-region weights disagree with validation')
    return added, step, conv


def verify_cache(path, frames, samples, wanted, missing, expected):
    cache = read(path)
    if (any(cache['provenance'].get(k) != v for k, v in expected.items())
            or cache['frames'] != len(samples) or cache['common_frames'] != len(wanted)
            or cache['missing_official_ids'] != missing or Path(cache['frame_csv']) != frames):
        raise ValueError('Previous test weights, fold, protocol or frame CSV differ')
    keys = pilot.OFFICIAL_METRICS
    totals = pilot.np.zeros((2, len(keys)))
    with frames.open(encoding='utf8', newline='') as f:
        rows = csv.reader(f)
        if next(rows) != ['id', *keys]:
            raise ValueError('Unexpected CSV columns')
        for sample, (ident, *values) in zip(samples, rows, strict=True):
            row = pilot.np.asarray(values, dtype='float64')
            if ident != sample.sample_id or row.shape != (len(keys),) or not pilot.np.isfinite(row).all():
                raise ValueError('Invalid cached frame')
            totals[0] += row
            if ident in wanted:
                totals[1] += row
    result = cache['results']['frame']
    for group, total, count in zip(('full', 'official_common'), totals, (len(samples), len(wanted))):
        if not pilot.np.allclose(total/count, [result[group][k] for k in keys], rtol=1e-12, atol=1e-12):
            raise ValueError('Previous test summary differs from frame scores')
    return result


def run(a):
    settings = read(a.run/'contract.json')
    parent = Path(settings['parent'])
    if parent != a.parent:
        raise ValueError('Cross-region training used a different parent job')
    paired, contract, source, start_conv, start = prior.selection(parent)
    parent_hash = sha256_file(parent/'summary.json')
    if (settings['start'] != start or settings['selection_sha256'] != parent_hash
            or settings['start_conv_sha256'] != sha256_file(start_conv)
            or Path(settings['start_conv']) != start_conv or Path(settings['source']) != source):
        raise ValueError('Cross-region parent checkpoint differs')
    added, step, conv = checkpoint(a.run, start, start_conv, read(parent/'summary.json'))
    suffix, prefix, n, seqs = pilot.TESTS[a.condition]
    samples, manifest = pilot.load_manifest(a.manifests/f'ms2_test_{suffix}_thermalcap_20260821.jsonl',
                                            a.data_root, allow_legacy_depth_path=True)
    if len(samples) != n or not manifest['sha256'].startswith(prefix):
        raise ValueError('Incorrect test manifest')
    ids = pilot.validate_samples(samples, seqs)
    crawl = {r['id'] for r in pilot.official_frames(a.data_root, 'test_'+a.condition, 10)}
    wanted = ids & crawl
    missing = sorted(crawl - ids)
    if (len(wanted), len(crawl)) != pilot.COMMON[a.condition]:
        raise ValueError('Incorrect official subset')
    protocol = dict(manifest=manifest, condition=a.condition, alignment='ssi_log', valid_gt_m=[.001, 80.],
                    code_sha256=sha256_file(pilot.__file__), paired_sha256=sha256_file(paired/'paired.json'))
    folder = paired/'train'/'frame'
    record = dict(source=str(source), source_sha256=contract['source_sha256'], step=step,
                  conv_sha256=sha256_file(conv), contract_sha256=sha256_file(folder/'contract.json'),
                  validation_sha256=sha256_file(folder/'summary.json'))
    provenance = dict(**protocol, frame=record, selection_sha256=sha256_file(a.run/'summary.json'),
                      parent_selection_sha256=parent_hash, runner_sha256=sha256_file(__file__))
    out = a.run/'test_selected_v1'/a.condition
    out.mkdir(parents=True, exist_ok=True)
    if added == 0:
        cached = parent/'test_selected_v1'/a.condition/'summary.json'
        csv_run = paired if start == prior.ext.START else parent
        frames = csv_run/'test_selected_v1'/a.condition/'frame_frames.csv'
        expected = dict(**protocol, frame=record, selection_sha256=parent_hash, runner_sha256=sha256_file(prior.__file__))
        result = verify_cache(cached, frames, samples, wanted, missing, expected)
        provenance.update(reused_from=str(cached), reused_sha256=sha256_file(cached))
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
    pilot.json_write(out/'summary.json', dict(provenance=provenance, selected=step, added_steps=added,
                     frames=n, common_frames=len(wanted), missing_official_ids=missing,
                     frame_csv=str(frames), results={'frame': result}))
    for group, label in [('full', 'test-full'), ('official_common', 'test-official')]:
        print(f'[{label}] frame {a.condition} step={step} added={added} reused={added == 0}', result[group])
    print('[completed-test]', out/'summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('run', 'parent', 'data-root', 'manifests'):
        p.add_argument('--'+k, type=Path, required=True)
    p.add_argument('--condition', choices=tuple(pilot.TESTS), required=True)
    run(p.parse_args())
