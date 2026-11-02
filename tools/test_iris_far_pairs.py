"""Test each far-pair arm once at its frozen validation-selected step."""
import argparse
import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from ms2_eval.io import load_manifest, sha256_file
from ms2_eval.official_protocol import OFFICIAL_METRICS, evaluate_sample
from tools import iris_far_stretch as iris
from tools import train_iris_far_pairs as train
from tools.run_ms2_supdepth_baselines import official_frames
from tools.test_da2_fixed import TESTS, validate_samples
from tools.test_iris_far_stretch import COMMON
from tools.train_da2_caption import json_write


def selected_step(summary, paired_step):
    curve = {int(s): m for s,m in summary['curve'].items()}
    full = {int(s): m for s,m in summary['full'].items()}
    provisional = train.select(curve)
    if set(curve) != set(train.STEPS) or set(full) != {0, provisional, train.STEPS[-1]}:
        raise ValueError('Incomplete validation results')
    step = provisional if all(iris.gates(full[0],full[provisional]).values()) else 0
    if (summary['provisional'] != provisional or summary['selected'] != step or paired_step != step):
        raise ValueError('Selection differs from the completed paired validation')
    return step


def selected_weights(run, arm, paired):
    folder = run/'train'/arm
    contract = train.read(folder/'contract.json')
    contract_hash = sha256_file(folder/'contract.json')
    if (contract['arm'] != arm or contract_hash != train.read(run/'smoke_pass.json')[arm]
            or contract['val']['sample_count'] != 14249 or not contract['val']['sha256'].startswith('c26a5859')):
        raise ValueError('Wrong training contract or validation manifest')
    for name,digest in contract['code'].items():
        if sha256_file(train.ROOT/name) != digest:
            raise ValueError(f'Training code changed: {name}')
    step = selected_step(train.read(folder/'summary.json'), paired['selected'][arm])
    source = Path(contract['source'])
    if (source.parts[-3:] != (iris.RUNS[arm], 'converted', 'step16000_weights.pt')
            or sha256_file(source) != contract['source_sha256']):
        raise ValueError(f'Wrong {arm} parent')
    conv = folder/f'conv_{step}.pt' if step else None
    provenance = dict(step=step, source=str(source), source_sha256=contract['source_sha256'],
                      conv_sha256=sha256_file(conv) if conv else None, contract_sha256=contract_hash,
                      validation_sha256=sha256_file(folder/'summary.json'))
    return contract, source, conv, provenance


def load_selected(model, arm, source, conv, contract):
    iris.load_arm(model,source,arm)
    if conv is not None:
        params = train.trainable_convs(model.unet)
        if list(params) != contract['trainable']:
            raise ValueError('Convolution names differ from training')
        train.restore_params(params,conv)
        model.unet.requires_grad_(False)
    model.set_train(False)


def score(samples, predictions, wanted, csv_path):
    totals = np.zeros((2,len(OFFICIAL_METRICS)),np.float64)
    seen = set()
    with csv_path.open('w',encoding='utf8',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(('id',*OFFICIAL_METRICS))
        for i,(sample,(ident,raw,gt)) in enumerate(zip(samples,predictions,strict=True),1):
            if ident != sample.sample_id:
                raise ValueError('Prediction/manifest order mismatch')
            m = evaluate_sample(raw,gt,align='ssi_log',min_depth=.001,max_depth=80.)
            values = [m[k] for k in OFFICIAL_METRICS]
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite test metrics: {ident}')
            totals[0] += values
            if ident in wanted:
                if ident in seen:
                    raise ValueError(f'Duplicate official ID: {ident}')
                seen.add(ident)
                totals[1] += values
            writer.writerow((ident,*values))
            if i % 1000 == 0:
                print(f'[test] {i}/{len(samples)}', flush=True)
    if not wanted or seen != wanted:
        raise ValueError('Incomplete official common subset')
    return {k:dict(zip(OFFICIAL_METRICS, t/n)) for k,t,n in
            zip(('full','official_common'),totals,(len(samples),len(wanted)))}


def run(a):
    from tools.train_route_suite import RouteModel
    paired = train.read(a.run/'paired.json')
    states = {arm:selected_weights(a.run,arm,paired) for arm in iris.RUNS}
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
    suffix,prefix,count,seqs = TESTS[a.condition]
    samples,manifest = load_manifest(a.manifests/f'ms2_test_{suffix}_thermalcap_20260821.jsonl',
                                     a.data_root,allow_legacy_depth_path=True)
    if not manifest['sha256'].startswith(prefix) or len(samples) != count:
        raise ValueError('Wrong test manifest')
    ids = validate_samples(samples,seqs)
    crawl = {r['id'] for r in official_frames(a.data_root,'test_'+a.condition,10)}
    wanted = ids & crawl
    if (len(wanted),len(crawl)) != COMMON[a.condition]:
        raise ValueError('Official fold differs from previous Iris/NeWCRF runs')
    provenance = dict(manifest=manifest, condition=a.condition, arms={k:s[3] for k,s in states.items()},
                      paired_sha256=sha256_file(a.run/'paired.json'), code_sha256=sha256_file(Path(__file__)),
                      alignment='ssi_log', valid_gt_m=[.001,80.])
    out = a.run/'test_selected_v1'/a.condition
    out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').is_file():
        if train.read(out/'summary.json')['provenance'] != provenance:
            raise ValueError('Different test run already exists')
        print('[completed-test] already scored:',out/'summary.json')
        return
    recipe = SimpleNamespace(**states['frame'][0]['recipe'])
    if vars(recipe) != states['empty'][0]['recipe']:
        raise ValueError('Test arms must share the same inference recipe')
    model = RouteModel(recipe, torch.device('cuda'), torch.float16)
    results = {}
    for arm,(contract,source,conv,record) in states.items():
        load_selected(model,arm,source,conv,contract)
        tmp = out/f'{arm}_frames.tmp'
        result = score(samples,iris.predict_samples(model,samples,arm,recipe),wanted,tmp)
        os.replace(tmp,out/f'{arm}_frames.csv')
        results[arm] = result
        for group,label in [('full','test-full'),('official_common','test-official')]:
            print(f'[{label}]',arm,a.condition,'new_step',record['step'],result[group],flush=True)
    diff = {g:{k:results['frame'][g][k]-results['empty'][g][k] for k in OFFICIAL_METRICS}
            for g in ('full','official_common')}
    json_write(out/'summary.json',dict(provenance=provenance,frames=len(samples),common_frames=len(wanted),
               missing_official_ids=sorted(crawl-ids),results=results,frame_minus_empty=diff))
    print('[frame-minus-empty]',a.condition,diff['official_common'],flush=True)
    print('[completed-test]',out/'summary.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','manifests','data-root'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--condition',choices=tuple(TESTS),required=True)
    run(p.parse_args())
