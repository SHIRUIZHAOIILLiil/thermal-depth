"""Matched DA2-Iris fusion test at each arm's validation-selected step.

Pilot official_protocol, the same as the DA2 and SD2.1 tests: .001<GT<80, frame mean,
ssi and none separate. Not the final-paper unified protocol. The step comes from the
arm's validation summary; test results never feed back into any selection.
Each arm runs under its own matched Iris weights and text; no permuted-text test.
Full test frames are scored, then the official every-tenth subset is folded from
the same per-frame rows.
"""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from ms2_eval.io import load_manifest, sha256_file
from models.da2_iris_fusion import FusionDepth
from tools.da2_caption_data import caption_hash
from tools.da2_continuation import validate_state, check_source
from tools.run_ms2_supdepth_baselines import official_frames
from tools.test_da2_fixed import TESTS, validate_samples
from tools.test_sd21_tadp import ENVIRONMENTS, METRICS, official_subset
from tools.train_da2_caption import build_model, verify_assets, json_write
from tools.train_sd21_dual_pair import verify_assets as verify_iris_assets
from tools.train_da2_iris_fusion_v2 import (EXPERIMENT, FusionData, evaluate, load_iris, make_tokens,
                                            read, restore)


def check_selection(arm, contract, summary):
    if contract['arm'] != arm or contract['experiment'] != EXPERIMENT:
        raise ValueError('Wrong arm/experiment')
    step = summary['selected_step']
    full = summary['selected_full']
    if full['step'] != step or full['samples'] != 14249: raise ValueError('Missing full validation of the selected step')
    if summary['contract'] != contract: raise ValueError('Summary does not match the run contract')
    return step


def run(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('Requires BF16 CUDA')
    run_dir = args.base / f'{args.arm}_s43'
    contract = read(run_dir / 'contract.json')
    selection = run_dir / 'summary.json'
    step = check_selection(args.arm, contract, read(selection))
    assets = verify_assets(args.assets)
    if sha256_file(args.assets) != contract['assets_sha256']: raise ValueError('DA2 assets changed since training')
    iris_assets = verify_iris_assets(args.base / 'iris_assets.json')
    if sha256_file(args.base / 'iris_assets.json') != contract['iris_assets_sha256']:
        raise ValueError('Iris assets changed since training')
    base_path, source = Path(contract['base_path']), contract['iris_sources'][args.arm]
    if sha256_file(base_path) != contract['base_sha256']: raise ValueError('DA2 B0 changed since training')
    if sha256_file(source['path']) != source['sha256']: raise ValueError('Iris weights changed since training')
    checkpoint = run_dir / f'step{step:06d}.pt'
    bundles, manifests, seen = {}, {}, set()
    for condition, (suffix, prefix, count, sequences) in TESTS.items():
        path = args.manifest_dir / f'ms2_test_{suffix}_thermalcap_20260821.jsonl'
        samples, info = load_manifest(path, args.data_root, allow_legacy_depth_path=True)
        if len(samples) != count or not info['sha256'].startswith(prefix): raise ValueError('Test count/hash mismatch')
        ids = validate_samples(samples, sequences)
        if ids & seen: raise ValueError('Test conditions overlap')
        seen |= ids
        bundles[condition] = (samples, info)
        manifests[condition] = dict(**info, caption_sha256=caption_hash(samples))
    out = args.base / 'test_selected_20261030' / args.arm
    provenance = dict(arm=args.arm, step=step, checkpoint_sha256=sha256_file(checkpoint),
        selection_sha256=sha256_file(selection), contract_sha256=sha256_file(run_dir / 'contract.json'),
        iris_source=source, manifests=manifests,
        protocol='pilot official_protocol; .001<GT<80; frame mean; ssi/none separate',
        code_sha256=sha256_file(Path(__file__)))
    if out.exists():
        if read(out / 'provenance.json') != provenance: raise ValueError('Test provenance changed')
    else:
        out.mkdir(parents=True)
        json_write(out / 'provenance.json', provenance)
    state = torch.load(base_path, map_location='cpu', weights_only=False)
    original = state['contract']; validate_state(state, original, 18000); check_source(original)
    base = build_model(assets, 'no-caption', 43); base.load_state_dict(state['model'], strict=True); del state
    model = FusionDepth(base.core, load_iris(iris_assets['path'], Path(source['path']), args.arm))
    if restore(checkpoint, model, contract) != step: raise ValueError('Checkpoint step mismatch')
    model.to('cuda').eval()
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(iris_assets['path'], subfolder='tokenizer', local_files_only=True)
    cache = out / 'tokens'; cache.mkdir(exist_ok=True)
    results = {}
    for condition, (samples, info) in bundles.items():
        folder = out / condition
        result_path = folder / f'val_step{step:06d}.json'
        if not result_path.exists():
            tokens = make_tokens(samples, tokenizer, iris_assets, info, args.arm, cache, True)
            print(f'[test] {args.arm} step={step} {condition} frames={len(samples)}', flush=True)
            evaluate(model, FusionData(samples, tokens), folder, step)
        result = read(result_path)
        if result['step'] != step or result['samples'] != len(samples): raise ValueError('Invalid cached result')
        for mode in ('ssi', 'none'):
            print('[test-full]', args.arm, condition, mode, result['samples'],
                  {k: round(result['metrics'][mode][k], 5) for k in METRICS}, flush=True)
        try:
            wanted = {f['id'] for f in official_frames(args.data_root, ENVIRONMENTS[condition], 10)}
            result['official_ssi'] = official_subset(folder / f'val_step{step:06d}_frames.csv', wanted)
            o = result['official_ssi']
            print('[test-official]', args.arm, condition, f"{o['frames']}/{o['official']}",
                  {k: round(v, 5) for k, v in o['metrics'].items()}, flush=True)
        except SystemExit as error:
            # The full-frame result stands; only the fold is skipped when split lists are absent.
            print(f'[test-official] {args.arm} {condition} skipped: {error}', flush=True)
        results[condition] = result
    json_write(out / 'summary.json', dict(arm=args.arm, step=step, manifests=manifests, results=results))
    print(f'[completed-test] {out}/summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['frame', 'empty'], required=True)
    for name in ('base', 'assets', 'manifest-dir', 'data-root'): p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
