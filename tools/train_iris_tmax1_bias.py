"""Continue the existing tmax1 frame weights with local pairs and the far-band bias loss.

Starts from lambda=.1/q100 16k, then uses lambda=.05 plus local-pair and far-bias
losses. This is a combined continuation, not a single-variable q98 ablation.
Only decoder-side convolutions change. Validation only; no test evaluation.
"""
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import extend_iris_far_pairs as ext
from tools import test_iris_cross_region_pairs as lineage
from tools import train_iris_far_bias_square as bias
from tools.train_iris_far_bias_square import BIAS_WEIGHT, MIN_PIXELS, PAIR_WEIGHT, pair, torch
from tools.train_iris_far_pairs import ACCUM, BATCH, LR
from tools.iris_tmax1_selection import POLICY, checked_metrics, read_reference, select

SOURCE_RUN = 'sd2_cap_pct_logtn_img_l0100_tmax1_s43'
UPPER_QUANTILE = 1.
EXPECTED = dict(abs_rel=.08137, rmse=2.7571, a1=.93937)


def check_source(path):
    if tuple(Path(path).parts[-3:]) != (SOURCE_RUN, 'converted', 'step16000_weights.pt'):
        raise ValueError(f'Expected the tmax1 step 16000 caption weights, got {path}')
    return Path(path)


def require_start(metrics):
    checked_metrics(metrics)
    for key, tolerance in (('rmse', .002), ('abs_rel', .00005), ('a1', .00005)):
        if abs(metrics[key] - EXPECTED[key]) > tolerance:
            raise ValueError(f'tmax1 start does not reproduce {key}: {metrics[key]} vs {EXPECTED[key]}')


def run(a):
    from tools.train_route_suite import RouteModel
    from utils.ms2_thermal_dataset import MS2ThermalDataset, MS2ThermalTransform, collate_fn_ms2
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Requires a BF16 CUDA GPU')
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_TEXT_PADDING='legacy', IRIS_DECODE_CLAMP='0')
    source = check_source(a.source)
    _, c, _, _, _ = lineage.prior.selection(a.recipe_run)
    pseudo = Path(c['pseudo'])
    if pair.sha256_file(pseudo.parent/'build_config.json') != c['pseudo_config']:
        raise ValueError('Pseudo-label configuration differs from the recipe run')
    archive = pair.read(a.archive/'summary.json')
    if archive['validation'] != c['val'] or archive['align']['newcrf'] != 'ssi':
        raise ValueError('Attribution reference does not match the validation recipe')
    train_rows, ti = pair.checked_manifest(Path(c['train']['path']), a.data_root, 'train', 75688)
    val, vi = pair.checked_manifest(Path(c['val']['path']), a.data_root, 'val', 14249)
    if ti != c['train'] or vi != c['val']:
        raise ValueError('Recipe manifests changed')
    reference = read_reference(a.archive, archive, vi, [s.sample_id for s in val])
    print('[newcrf-reference]', reference, flush=True)
    dataset = MS2ThermalDataset(Path(ti['path']), a.data_root, pseudo,
        transform=MS2ThermalTransform(random_flip=True), norm_type='log_truncnorm',
        truncnorm_min=.02, truncnorm_max=UPPER_QUANTILE, use_captions=True)
    if [s.sample_id for s in train_rows] != [r['id'] for r in dataset.rows]:
        raise ValueError('Training order differs from original')
    smoke = a.mode == 'smoke'
    folder = a.run/'smoke' if smoke else a.run
    if not smoke and (folder/'contract.json').exists():
        raise FileExistsError('Training directory already used')
    folder.mkdir(parents=True, exist_ok=not smoke)
    recipe = SimpleNamespace(**c['recipe'])
    model = RouteModel(recipe, torch.device('cuda'), torch.float16)
    pair.old.load_arm(model, source, 'frame')
    params = pair.trainable_convs(model.unet)
    if list(params) != c['trainable']:
        raise ValueError('Trainable convolutions differ')
    contract = dict(source=str(source), source_sha256=pair.sha256_file(source), recipe_run=str(a.recipe_run),
        val=vi, train=ti, trainable=list(params), lr=LR, pair_weight=PAIR_WEIGHT, bias_weight=BIAS_WEIGHT,
        far_m=[60, 80], min_pixels=MIN_PIXELS, upper_quantile=UPPER_QUANTILE, image_l1=.05,
        initial_image_cursor=ext.START*BATCH*ACCUM, gain_m=ext.GAIN, optimizer='fresh AdamW', budget=2000,
        selection_policy=POLICY, newcrf_reference=reference,
        selection_sha256=pair.sha256_file(Path(__file__).with_name('iris_tmax1_selection.py')),
        runner_sha256=pair.sha256_file(__file__), loss_sha256=pair.sha256_file(bias.__file__),
        reference_stats_sha256=pair.sha256_file(a.archive/'newcrf_moments.npz'))
    if not smoke and pair.read(a.run/'smoke'/'passed.json')['contract'] != contract:
        raise ValueError('Configuration differs from smoke')
    pair.json_write(folder/'contract.json', contract)
    if smoke:
        frozen = [(k, p) for k, p in model.unet.named_parameters() if not p.requires_grad]
        before, initial = pair.fingerprint(frozen), pair.fingerprint(params.items())
        report = bias.train(model.unet, params, dataset, collate_fn_ms2, c['sd'], folder, True)
        if before != pair.fingerprint(frozen) or initial == pair.fingerprint(params.items()):
            raise ValueError('Smoke update scope mismatch')
        pair.json_write(folder/'passed.json', dict(contract=contract, **report))
        print('[smoke] PASS', report, flush=True)
        return
    metrics, parts = bias.validate(model, recipe, val, a.archive)
    require_start(metrics)
    print('[validation]', ext.START, metrics, flush=True)
    full, mechanism = {ext.START: metrics}, {ext.START: parts}
    report = bias.train(model.unet, params, dataset, collate_fn_ms2, c['sd'], folder, False)
    torch.cuda.empty_cache()
    for step in ext.STEPS:
        pair.restore_params(params, folder/f'conv_{step}.pt')
        full[step], mechanism[step] = bias.validate(model, recipe, val, a.archive)
        print('[validation]', step, full[step], flush=True)
        print('[mechanism]', step, mechanism[step], flush=True)
    selected, decision = select(full, reference['metrics'])
    pair.json_write(folder/'summary.json', dict(selected=selected, added_steps=selected-ext.START,
        full_by_added_steps={s-ext.START: m for s, m in full.items()}, mechanism_by_step=mechanism, train=report,
        decision=decision, newcrf_reference=reference,
        selected_conv=None if selected == ext.START else str(folder/f'conv_{selected}.pt')))
    for step, row in decision['candidates'].items():
        print('[candidate-vs-newcrf]', step, row, flush=True)
    print('[decision]', decision['status'], flush=True)
    print('[selected] tmax1 frame', selected, full[selected], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['smoke', 'train'])
    for key in ('run', 'source', 'recipe-run', 'data-root', 'archive'):
        p.add_argument('--'+key, type=Path, required=True)
    run(p.parse_args())
