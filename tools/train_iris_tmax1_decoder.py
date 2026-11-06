"""q02/q100 caption 16k -> frozen U-Net -> original depth decoder adaptation.

Reuses the completed decoder-only training/validation implementation unchanged.
No U-Net continuation, no q98 decoder transplant, no test split.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import train_iris_decoder_only as old
from tools import train_iris_tmax1_bias as tmax
from tools import eval_iris_decoder_only as prior_eval
from tools.iris_tmax1_selection import comparison, checked_metrics

pair, torch = old.pair, old.torch
LOWER, UPPER = .02, 1.
CODE = tuple(dict.fromkeys((*old.CODE, 'tools/train_iris_tmax1_decoder.py',
    'tools/train_iris_tmax1_bias.py', 'tools/eval_iris_decoder_only.py',
    'tests/test_iris_tmax1_decoder.py', 'slurm/iris_tmax1_decoder.sbatch',
    'slurm/submit_iris_tmax1_decoder.sh')))


def make_dataset(contract, data_root):
    from lotus.utils.ms2_thermal_dataset import MS2ThermalDataset
    return MS2ThermalDataset(Path(contract['train']['path']), data_root, Path(contract['pseudo']),
        norm_type='log_truncnorm', truncnorm_min=LOWER, truncnorm_max=UPPER, use_captions=True)


def check_comparator_contract(contract, current, reference):
    """Equal decoder-stage budget/protocol, not equal upstream training history."""
    required = dict(lr=old.LR, batch=old.BATCH, accumulation=old.ACCUM, budget=1000,
        steps=list(old.STEPS), seed=old.SEED, image_cursor=4000*old.BATCH*old.ACCUM,
        weight_decay=.01, augmentation='none: deployment input', pixel_loss_weight=.05,
        pair_weight=old.bias.PAIR_WEIGHT, bias_weight=old.bias.BIAS_WEIGHT)
    if any(contract.get(k) != v for k,v in required.items()):
        raise ValueError('Historical decoder training budget/settings differ')
    for k in ('train','val','recipe','pseudo'):
        if contract[k] != current[k]:
            raise ValueError(f'Historical comparison {k} differs')
    if (contract['pseudo_config_sha256'] != current['pseudo_config']
            or contract['newcrf'] != reference):
        raise ValueError('Historical pseudo/reference provenance differs')


def read_comparator(run, current, reference):
    contract, expected, chosen, path = prior_eval.selection(run)
    check_comparator_contract(contract, current, reference)
    summary = pair.read(run/'train/summary.json')
    reproduced_path = run/'train/test_selected_v1/reproduced.json'
    reproduced = pair.read(reproduced_path)  # validation only; never read test scene results
    if reproduced['step'] != chosen or reproduced['decoder_sha256'] != pair.sha256_file(path):
        raise ValueError('Historical deployed validation has different weights')
    prior_eval.check_reproduction(reproduced['measured'], expected)
    if any(v['samples'] != current['val']['sample_count'] for v in summary['curve'].values()):
        raise ValueError('Historical validation frame count differs')
    return dict(run=str(run), selected=chosen, curve=summary['curve'],
        contract_sha256=pair.sha256_file(run/'contract.json'),
        summary_sha256=pair.sha256_file(run/'train/summary.json'),
        reproduction_sha256=pair.sha256_file(reproduced_path), decoder_sha256=pair.sha256_file(path),
        note='Performance comparison only: upstream q98/lambda=.05 + 4k differs from q100/lambda=.1 16k.')


def compare_metrics(candidate, incumbent, reference):
    a, b = checked_metrics(candidate), checked_metrics(incumbent)
    delta = {k:a[k]-b[k] for k in a}
    better = a['abs_rel']<=b['abs_rel'] and a['rmse']<=b['rmse'] and a['a1']>=b['a1']
    worse = a['abs_rel']>=b['abs_rel'] and a['rmse']>=b['rmse'] and a['a1']<=b['a1']
    status = ('equal_primary3' if better and worse else 'candidate_dominates_primary3' if better
              else 'incumbent_dominates_primary3' if worse else 'tradeoff_primary3')
    return dict(candidate=candidate, incumbent=incumbent, candidate_minus_incumbent=delta,
        status=status, candidate_vs_newcrf=comparison(candidate,reference),
        incumbent_vs_newcrf=comparison(incumbent,reference))


def run(a):
    from tools.train_route_suite import RouteModel
    if not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA')
    # Reuse the frozen data/inference recipe, NOT this q98 parent's U-Net or decoder.
    inputs = old.parent.parent_inputs(a.runs)
    c = inputs['contract']
    source = tmax.check_source(a.source)
    samples, ti = pair.checked_manifest(Path(c['train']['path']),a.data_root,'train',75688)
    val, vi = pair.checked_manifest(Path(c['val']['path']),a.data_root,'val',14249)
    if ti!=c['train'] or vi!=c['val']:
        raise ValueError('Frozen manifests changed')
    if pair.sha256_file(Path(c['pseudo']).parent/'build_config.json') != c['pseudo_config']:
        raise ValueError('Calibrated pseudo recipe changed')
    archive = a.runs/'outputs/lotus_line_v2/iris_pair_rmse_val_v1/job_9167854'
    reference = old.read_reference(archive,pair.read(archive/'summary.json'),vi,[s.sample_id for s in val])
    incumbent = read_comparator(a.comparison_run,c,reference)
    recipe = SimpleNamespace(**c['recipe'])
    if (recipe.route!='b_thermal_unet' or not recipe.gt_decode_fp32
            or recipe.condition_latent!='mode' or recipe.num_inference_steps!=1):
        raise ValueError('Unexpected deployed inference recipe')
    dataset = make_dataset(c,a.data_root)
    if [s.sample_id for s in samples] != [r['id'] for r in dataset.rows]:
        raise ValueError('Dataset/manifest order differs')
    pair.seed_all(old.SEED)
    model = RouteModel(recipe,torch.device('cuda'),torch.float16)
    pair.old.load_arm(model,source,'frame')
    decoder, params = old.make_decoder(model)  # original VAE; never load the q98 adapted decoder
    settings = dict(experiment='iris_tmax1_decoder_v1', source=str(source),
        source_sha256=pair.sha256_file(source), source_step=16000,
        source_training=dict(image_l1=.1,quantiles=[LOWER,UPPER]), decoder_initialization='original fp32 VAE',
        recipe=c['recipe'], train=ti, val=vi, pseudo=c['pseudo'], pseudo_config_sha256=c['pseudo_config'],
        quantiles=[LOWER,UPPER], trainable=list(params), lr=old.LR, batch=old.BATCH,
        accumulation=old.ACCUM, budget=1000, steps=list(old.STEPS), seed=old.SEED,
        image_cursor=4000*old.BATCH*old.ACCUM, weight_decay=.01, augmentation='none: deployment input',
        image_cursor_reason='match the prior decoder-stage sample order; not a resumed tmax optimizer',
        pixel_loss_weight=.05, pair_weight=old.bias.PAIR_WEIGHT, bias_weight=old.bias.BIAS_WEIGHT,
        prediction_inputs='thermal + frame caption + manifest index; target_latent=None',
        policy=old.POLICY,newcrf=reference,incumbent=incumbent,
        code={p:pair.sha256_file(ROOT/p) for p in CODE})
    settings = json.loads(json.dumps(settings))
    smoke = a.mode=='smoke'
    folder = a.run/('smoke' if smoke else 'train')
    folder.mkdir(parents=True,exist_ok=False)
    if smoke:
        if (a.run/'contract.json').exists():
            raise FileExistsError('Run contract already exists')
        pair.json_write(a.run/'contract.json',settings)
        baseline = old.validate(model,decoder,val,recipe,verify_initial=True)
        tmax.require_start(baseline['metrics'])
        print('[reproduced] tmax1 original 16000',baseline,flush=True)
        original = samples[0]
        poisoned = dataclasses.replace(original,thermal_gt_path='/GT_MUST_NOT_BE_READ/missing.png')
        raw,z = old.predict_frozen(model,old.InferenceInput.from_sample(original),recipe)
        raw2,z2 = old.predict_frozen(model,old.InferenceInput.from_sample(poisoned),recipe)
        torch.testing.assert_close(raw,raw2,rtol=0,atol=0)
        torch.testing.assert_close(z,z2,rtol=0,atol=0)
        print('[gt-isolation] PASS',flush=True)
        report = old.train(model,decoder,params,dataset,samples,recipe,folder,True)
        pair.json_write(a.run/'smoke_pass.json',dict(baseline=baseline,training=report,gt_isolation=True,
            contract_sha256=pair.sha256_file(a.run/'contract.json')))
        print('[smoke] PASS',report,flush=True)
        return
    passed = pair.read(a.run/'smoke_pass.json')
    if settings!=pair.read(a.run/'contract.json') or passed['contract_sha256']!=pair.sha256_file(a.run/'contract.json'):
        raise ValueError('Formal configuration differs from smoke')
    if old.fingerprint(params.items()) != passed['training']['initial']:
        raise ValueError('Decoder must restart from the original weights, not smoke updates')
    report = old.train(model,decoder,params,dataset,samples,recipe,folder,False)
    curve = {0:passed['baseline']}
    for step in old.STEPS[1:]:
        pair.restore_params(params,folder/f'decoder_{step}.pt')
        curve[step] = old.validate(model,decoder,val,recipe)
        pair.json_write(folder/f'val_step{step:06d}.json',curve[step])
        print('[validation]',step,curve[step],flush=True)
    chosen, decision = old.select({s:v['metrics'] for s,v in curve.items()},reference['metrics'])
    prior = incumbent['curve']
    comparisons = dict(
        fixed1000=compare_metrics(curve[1000]['metrics'],prior['1000']['metrics'],reference['metrics']),
        selected=compare_metrics(curve[chosen]['metrics'],prior[str(incumbent['selected'])]['metrics'],reference['metrics']))
    path = folder/f'decoder_{chosen}.pt' if chosen else None
    pair.json_write(folder/'summary.json',dict(selected=chosen,curve=curve,decision=decision,
        source_step=16000,selected_decoder=str(path) if path else None,
        selected_sha256=pair.sha256_file(path) if path else None,training=report,
        contract_sha256=pair.sha256_file(a.run/'contract.json'),comparisons=comparisons,
        comparison_note=incumbent['note']))
    print('[selected] decoder updates',chosen,curve[chosen]['metrics'],flush=True)
    for name,row in comparisons.items():
        print('[vs-q98-decoder]',name,row,flush=True)
    print('[completed]',folder/'summary.json','validation only; no test evaluation',flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('smoke','train'))
    for name in ('runs','run','data-root','source','comparison-run'):
        p.add_argument('--'+name,type=Path,required=True)
    os.environ.update(IRIS_THERMAL_STRETCH='percentile',IRIS_TEXT_PADDING='legacy',IRIS_DECODE_CLAMP='0')
    run(p.parse_args())
