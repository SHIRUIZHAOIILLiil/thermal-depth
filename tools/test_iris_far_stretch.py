"""Test a validation-adopted far stretch using the parent's frozen recipe."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from ms2_eval.io import load_manifest, sha256_file
from tools import iris_far_stretch as stretch
from tools.run_ms2_supdepth_baselines import official_frames
from tools.test_da2_fixed import TESTS, validate_samples
from tools.train_da2_caption import json_write

COMMON = {'day': (2331, 2332), 'night': (2292, 2292), 'rain': (2503, 2503)}


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def selection(parent):
    contract, summary, betas = (read(parent/name) for name in
                                ('contract.json', 'summary.json', 'frozen_betas.json'))
    if (contract['runner_sha256'] != sha256_file(stretch.__file__)
            or summary['validation_frames'] != 14249 or set(betas) != set(stretch.RUNS)):
        raise ValueError('Parent implementation, validation count or arms differ')
    for arm,beta in betas.items():
        fit = read(parent/f'{arm}_fit.json')
        row = summary['results'][arm]
        rules = stretch.gates(row['zero'], row['corrected'])
        if (beta not in stretch.GRID or beta != fit['beta'] or beta != row['beta']
                or beta != stretch.choose_beta({float(k):v for k,v in fit['train_scores'].items()})
                or row['gates'] != rules or row['pass_self'] != all(rules.values())):
            raise ValueError(f'Inconsistent frozen fit or validation for {arm}')
    adopted = summary['results']['frame']['pass_self']
    if adopted != summary['adopt_frame_stretch']:
        raise ValueError('Inconsistent validation decision')
    return adopted, betas, contract


def official_fold(path, wanted, betas):
    scores = {b:[] for b in betas}
    seen = {b:set() for b in scores}
    with path.open(encoding='utf8', newline='') as f:
        for row in csv.DictReader(f):
            ident, b = row['id'], float(row['beta'])
            if ident in wanted:
                if ident in seen[b]:
                    raise ValueError(f'Duplicate official row: {ident}')
                seen[b].add(ident)
                scores[b].append([float(row[k]) for k in stretch.OFFICIAL_METRICS])
    if not wanted or any(s != wanted for s in seen.values()):
        raise ValueError('Incomplete official common subset')
    return {b:dict(zip(stretch.OFFICIAL_METRICS, np.mean(rows, axis=0))) for b,rows in scores.items()}


def run(a):
    adopt, betas, contract = selection(a.parent)
    out = a.parent/'test_selected_v1'/a.condition
    out.mkdir(parents=True, exist_ok=True)
    if not adopt:
        json_write(out/'skipped.json', dict(reason='frame failed validation gate',
                   validation_sha256=sha256_file(a.parent/'summary.json')))
        print('[skipped-test]', a.condition, 'validation gate failed; no test data read')
        return
    import torch
    from tools.train_route_suite import RouteModel
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
    suffix, prefix, count, seqs = TESTS[a.condition]
    samples, info = load_manifest(a.manifests/f'ms2_test_{suffix}_thermalcap_20260821.jsonl',
                                  a.data_root, allow_legacy_depth_path=True)
    if not info['sha256'].startswith(prefix) or len(samples) != count:
        raise ValueError('Wrong test manifest')
    ids = validate_samples(samples, seqs)
    crawl = {r['id'] for r in official_frames(a.data_root, 'test_'+a.condition, 10)}
    wanted = crawl & ids
    if (len(wanted),len(crawl)) != COMMON[a.condition]:
        raise ValueError('Unexpected official subset')
    recipe = SimpleNamespace(**contract['recipe'])
    sources = {}
    for arm,run_name in stretch.RUNS.items():
        source = contract['checkpoints'][arm]
        path = Path(source['path'])
        if (path.parts[-3:] != (run_name, 'converted', 'step16000_weights.pt')
                or sha256_file(path) != source['sha256']):
            raise ValueError(f'{arm} checkpoint differs from parent')
        sources[arm] = path
    provenance = dict(condition=a.condition, manifest=info, frozen_betas=betas,
                      contract_sha256=sha256_file(a.parent/'contract.json'),
                      validation_sha256=sha256_file(a.parent/'summary.json'),
                      script_sha256=sha256_file(Path(__file__)), alignment='ssi_log')
    if (out/'summary.json').exists():
        if read(out/'summary.json')['provenance'] != provenance:
            raise ValueError('Existing test results have different provenance')
        print('[completed-test] already complete:', out/'summary.json')
        return
    model = RouteModel(recipe, torch.device('cuda'), torch.float16)
    results = {}
    for arm,path in sources.items():
        stretch.load_arm(model, path, arm)
        beta = betas[arm]
        evaluated = sorted({0., beta})
        tmp = out/f'{arm}_frames.tmp'
        with tmp.open('w', encoding='utf8', newline='') as f:
            full = stretch.measure(samples, stretch.predict_samples(model, samples, arm, recipe), evaluated, f)
        common = official_fold(tmp, wanted, evaluated)
        os.replace(tmp, out/f'{arm}_frames.csv')
        results[arm] = {name:{'zero':group[0.], 'corrected':group[beta]}
                        for name,group in (('full',full), ('official_common',common))}
        for stage,m in results[arm]['official_common'].items():
            print('[test-official]', a.condition, arm, stage, len(wanted), m, flush=True)
    differences = {group:{stage:{k:results['frame'][group][stage][k]-results['empty'][group][stage][k]
                                 for k in stretch.OFFICIAL_METRICS} for stage in ('zero','corrected')}
                   for group in ('full','official_common')}
    json_write(out/'summary.json', dict(provenance=provenance, frames=len(ids), common_frames=len(wanted),
               missing_official_ids=sorted(crawl-ids), results=results, frame_minus_empty=differences))
    print('[completed-test]', out/'summary.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent','manifests','data-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--condition', choices=tuple(TESTS), required=True)
    run(p.parse_args())
