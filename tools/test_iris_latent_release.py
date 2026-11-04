"""Validation-selected day/night/rain evaluation for the paired latent-mask experiment."""
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import train_iris_latent_release as train
from tools.train_iris_latent_release import pair, parent_test, ARMS, START


def run(a):
    inputs = train.parent_inputs(a.runs)
    if pair.read(a.run/'contract.json') != train.contract_for(inputs):
        raise ValueError('Test configuration differs from training')
    train.verify_pair(a.run)
    step,conv = train.selected_weights(a.run,a.arm,inputs)
    pilot = parent_test.pilot
    suffix,prefix,n,seqs = pilot.TESTS[a.condition]
    samples,manifest = pilot.load_manifest(a.manifests/f'ms2_test_{suffix}_thermalcap_20260821.jsonl',
                                           a.data_root,allow_legacy_depth_path=True)
    if len(samples)!=n or not manifest['sha256'].startswith(prefix):
        raise ValueError('Incorrect test manifest')
    ids = pilot.validate_samples(samples,seqs)
    crawl = {r['id'] for r in pilot.official_frames(a.data_root,'test_'+a.condition,10)}
    wanted,missing = ids & crawl,sorted(crawl-ids)
    if (len(wanted),len(crawl)) != pilot.COMMON[a.condition]:
        raise ValueError('Official fold differs')
    protocol = dict(manifest=manifest,condition=a.condition,alignment='ssi_log',valid_gt_m=[.001,80.],
                    code_sha256=pair.sha256_file(pilot.__file__),
                    paired_sha256=pair.sha256_file(inputs['paired']/'paired.json'))
    original_folder = inputs['paired']/'train'/'frame'
    record = dict(source=str(inputs['source']),source_sha256=inputs['contract']['source_sha256'],
                  step=step,conv_sha256=pair.sha256_file(conv),
                  contract_sha256=pair.sha256_file(original_folder/'contract.json'),
                  validation_sha256=pair.sha256_file(original_folder/'summary.json'))
    expected_parent = dict(**protocol,frame=record,
        selection_sha256=pair.sha256_file(inputs['parent']/'summary.json'),
        parent_selection_sha256=pair.sha256_file(inputs['ancestor']/'summary.json'),
        runner_sha256=pair.sha256_file(parent_test.__file__))
    provenance = dict(**protocol,frame=record,arm=a.arm,
        selection_sha256=pair.sha256_file(a.run/a.arm/'summary.json'),
        experiment_contract_sha256=pair.sha256_file(a.run/'contract.json'),
        runner_sha256=pair.sha256_file(__file__))
    out = a.run/a.arm/'test_selected_v1'/a.condition
    out.mkdir(parents=True,exist_ok=True)
    cache = out/'summary.json'
    if cache.is_file():
        previous = pair.read(cache)
        results = parent_test.verify_cache(cache,Path(previous['frame_csv']),samples,wanted,missing,provenance)
        reused = previous['reused']
        frames = Path(previous['frame_csv'])
        provenance = previous['provenance']
    elif step==START:
        cache = inputs['parent']/'test_selected_v1'/a.condition/'summary.json'
        frames = cache.parent/'frame_frames.csv'
        results = parent_test.verify_cache(cache,frames,samples,wanted,missing,expected_parent)
        provenance.update(reused_from=str(cache),reused_sha256=pair.sha256_file(cache))
        reused = True
    else:
        from tools.train_route_suite import RouteModel
        recipe = SimpleNamespace(**inputs['contract']['recipe'])
        model = RouteModel(recipe,pilot.torch.device('cuda'),pilot.torch.float16)
        pilot.load_selected(model,'frame',inputs['source'],conv,inputs['contract'])
        tmp = out/'frame_frames.tmp'
        results = pilot.score(samples,pilot.iris.predict_samples(model,samples,'frame',recipe),wanted,tmp)
        frames = out/'frame_frames.csv'
        os.replace(tmp,frames)
        reused = False
    pair.json_write(out/'summary.json',dict(provenance=provenance,selected=step,reused=reused,
        frames=n,common_frames=len(wanted),missing_official_ids=missing,frame_csv=str(frames),
        results={'frame':results}))
    for group,label in [('full','test-full'),('official_common','test-official')]:
        print(f'[{label}] {a.arm} {a.condition} step={step} reused={reused}',results[group],flush=True)
    print('[completed-test]',out/'summary.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('runs','run','data-root','manifests'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--condition',choices=('day','night','rain'),required=True)
    os.environ.update(IRIS_THERMAL_STRETCH='percentile',IRIS_TEXT_PADDING='legacy',IRIS_DECODE_CLAMP='0')
    run(p.parse_args())
