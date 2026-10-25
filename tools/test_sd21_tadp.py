"""Frozen SD2.1 direct-depth test at each arm's validation-selected step.

Pilot official_protocol, the same as the DA2 frozen test: .001<GT<80, frame mean,
ssi and none separate. Not the final-paper unified protocol. The step comes from the
arm's validation summary; test results never feed back into any selection.
Full test frames are scored, then the official every-tenth subset is folded from
the same per-frame rows so the NeWCRF comparison needs no second inference pass.
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from ms2_eval.io import load_manifest, sha256_file
from models.sd21_tadp import load_sd21
from tools.da2_caption_data import caption_hash
from tools.run_ms2_supdepth_baselines import official_frames
from tools.test_da2_fixed import TESTS, validate_samples
from tools.train_da2_caption import evaluate, json_write
from tools.train_sd21_tadp_pair import ThermalDirect, make_tokens, read, verify_assets

ENVIRONMENTS = {'day': 'test_day', 'night': 'test_night', 'rain': 'test_rain'}
# Same SD21Depth model and checkpoint layout; only the training loss differs.
EXPERIMENTS = ('sd21_tadp_pair_v1', 'sd21_tadp_pseudo_pair_v1')
METRICS = ('abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3')


def check_selection(arm, contract, summary, experiment='sd21_tadp_pair_v1'):
    if contract['arm'] != arm or contract['experiment'] != experiment:
        raise ValueError('Wrong arm/experiment')
    step = summary['selected_step']
    full = summary['selected_full']
    if full['step'] != step or full['samples'] != 14249: raise ValueError('Missing full validation of the selected step')
    return step


def official_subset(frames_csv, wanted):
    rows = [r for r in csv.DictReader(open(frames_csv, encoding='utf-8')) if r['alignment'] == 'ssi']
    subset = [r for r in rows if r['id'] in wanted]
    if not subset: raise ValueError('No official frames in result')
    return dict(frames=len(subset), official=len(wanted),
                metrics={k: statistics.fmean(float(r[k]) for r in subset) for k in METRICS})


def run(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('Requires BF16 CUDA')
    run_dir = args.base / f'{args.arm}_s43'
    contract = read(run_dir / 'contract.json')
    selection = run_dir / 'summary.json'
    step = check_selection(args.arm, contract, read(selection), args.experiment)
    assets = verify_assets(args.assets)
    if assets != contract['assets']: raise ValueError('Assets changed since training')
    checkpoint = run_dir / f'step{step:06d}.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['step'] != step or state['contract'] != contract: raise ValueError('Checkpoint does not match the run')
    bundles, manifests, seen = {}, {}, set()
    for condition, (suffix, prefix, count, sequences) in TESTS.items():
        path = args.manifest_dir / f'ms2_test_{suffix}_thermalcap_20260821.jsonl'
        samples, info = load_manifest(path, args.data_root, allow_legacy_depth_path=True)
        if len(samples) != count or not info['sha256'].startswith(prefix): raise ValueError('Test count/hash mismatch')
        ids = validate_samples(samples, sequences)
        if ids & seen: raise ValueError('Test conditions overlap')
        seen |= ids
        bundles[condition] = samples
        manifests[condition] = dict(**info, caption_sha256=caption_hash(samples))
    out = args.base / 'test_selected_20261025' / args.arm
    provenance = dict(arm=args.arm, step=step, checkpoint_sha256=sha256_file(checkpoint),
        selection_sha256=sha256_file(selection), contract_sha256=sha256_file(run_dir / 'contract.json'),
        manifests=manifests, protocol='pilot official_protocol; .001<GT<80; frame mean; ssi/none separate',
        code_sha256=sha256_file(Path(__file__)))
    if out.exists():
        if read(out / 'provenance.json') != provenance: raise ValueError('Test provenance changed')
    else:
        out.mkdir(parents=True)
        json_write(out / 'provenance.json', provenance)
    model = load_sd21(assets['path'])
    model.unet.load_state_dict(state['unet'], strict=True)
    model.head.load_state_dict(state['head'], strict=True)
    del state
    model.to('cuda').eval()
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(assets['path'], subfolder='tokenizer', local_files_only=True)
    cache = out / 'tokens'; cache.mkdir(exist_ok=True)
    results = {}
    for condition, samples in bundles.items():
        folder = out / condition; folder.mkdir(exist_ok=True)
        result_path = folder / f'val_step{step:06d}.json'
        if not result_path.exists():
            tokens = make_tokens(samples, tokenizer, assets, manifests[condition]['sha256'], args.arm, cache, True)
            print(f'[test] {args.arm} step={step} {condition} frames={len(samples)}', flush=True)
            evaluate(model, ThermalDirect(samples, tokens), torch.device('cuda'), True, folder, step)
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
    p.add_argument('--experiment', choices=EXPERIMENTS, default=EXPERIMENTS[0])
    for name in ('base', 'assets', 'manifest-dir', 'data-root'): p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
