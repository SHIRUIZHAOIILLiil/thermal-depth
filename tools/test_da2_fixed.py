"""Frozen DA2 tests. Pilot official_protocol, not final-paper unified protocol."""
import argparse
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from ms2_eval.io import load_manifest, sha256_file
from tools.da2_caption_data import TRAIN, VAL, ThermalLidar, caption_hash
from tools.da2_continuation import read, check_source, validate_state
from tools.train_da2_caption import build_model, evaluate, json_write, token_cache, verify_assets

FIXED = {'no-caption': 18000, 'caption': 20000}
TESTS = {
 'day': ('day3_common', 'f50d4506', 23311, {'2021-08-06-11-23-45', '2021-08-13-15-46-56', '2021-08-13-16-31-10'}),
 'night': ('night3', '273310de', 22916, {'2021-08-13-21-18-04', '2021-08-13-22-03-03', '2021-08-13-22-16-02'}),
 'rain': ('rainy3', '5a63743f', 25023, {'2021-08-06-16-19-00', '2021-08-06-16-45-28', '2021-08-06-16-59-13'}),
}

def validate_samples(samples, allowed):
    ids, sequences = set(), set()
    if allowed & (TRAIN | VAL): raise ValueError('Test overlaps train/val')
    for s in samples:
        if not re.fullmatch(r'\d{4}(?:-\d{2}){5}_\d+', s.sample_id) or s.sample_id in ids:
            raise ValueError('Invalid or duplicate ID')
        ids.add(s.sample_id)
        seq, frame = s.sample_id.rsplit('_', 1)
        sequences.add(seq)
        if seq not in allowed or s.sequence not in (seq, 'unknown'): raise ValueError('Wrong sequence')
        if s.split not in ('unknown', 'test'): raise ValueError('Non-test split')
        for value in (s.thermal_path, s.thermal_gt_path):
            if re.findall(r'\d{4}(?:-\d{2}){5}', value) != [seq] or Path(value).stem != frame:
                raise ValueError('ID/path disagreement')
            if not Path(value).is_file(): raise FileNotFoundError(value)
        if '/thr/' not in s.thermal_gt_path.replace(chr(92), '/'): raise ValueError('Require thermal GT')
        if not s.caption or not s.caption.strip(): raise ValueError('Missing caption')
    if sequences != allowed: raise ValueError('Missing sequence')
    return ids

def check_selection(arm, summary):
    if summary['arm'] != arm or summary['selected_step'] != FIXED[arm]: raise ValueError('Wrong selected step')
    if summary['selected_full']['step'] != FIXED[arm] or summary['selected_full']['samples'] != 14249:
        raise ValueError('Missing full validation')

def run(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('Requires BF16 CUDA')
    arm, step = args.arm, FIXED[args.arm]
    run_dir = args.base / f'{arm}_s43'
    contract = read(run_dir / 'contract.json')
    if contract['arm'] != arm or contract['seed'] != 43: raise ValueError('Wrong arm/seed')
    check_source(contract)
    selection = run_dir / 'validation_20000/summary.json'
    check_selection(arm, read(selection))
    assets = verify_assets(args.assets)
    if sha256_file(args.assets) != contract['assets']: raise ValueError('Assets changed')
    checkpoint = run_dir / f'step{step:06d}.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    validate_state(state, contract, step)
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
    out = args.base / 'test_fixed_20261023' / arm
    provenance = dict(arm=arm, step=step, checkpoint_sha256=sha256_file(checkpoint),
        selection_sha256=sha256_file(selection), contract=contract, manifests=manifests,
        protocol='pilot official_protocol; .001<GT<80; frame mean; ssi/none separate',
        code_sha256=sha256_file(Path(__file__)))
    if out.exists():
        if read(out / 'provenance.json') != provenance: raise ValueError('Test provenance changed')
    else:
        out.mkdir(parents=True)
        json_write(out / 'provenance.json', provenance)
    model = build_model(assets, arm, contract['seed'])
    model.load_state_dict(state['model'], strict=True)
    del state
    model.requires_grad_(False).to('cuda').eval()
    tokenizer = None
    if arm == 'caption':
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained(assets['text_path'], subfolder='tokenizer', local_files_only=True)
    cache_dir = out / 'tokens'; cache_dir.mkdir(exist_ok=True)
    results = {}
    for condition, samples in bundles.items():
        folder = out / condition; folder.mkdir(exist_ok=True)
        result_path = folder / f'val_step{step:06d}.json'
        if not result_path.exists():
            tokens = None if tokenizer is None else token_cache(samples, tokenizer, {**assets, 'root': str(cache_dir)},
                manifests[condition]['sha256'], create=True)
            print(f'[test] {arm} step={step} {condition} frames={len(samples)}', flush=True)
            evaluate(model, ThermalLidar(samples, tokens), torch.device('cuda'), True, folder, step)
        result = read(result_path)
        if result['step'] != step or result['samples'] != len(samples): raise ValueError('Invalid cached result')
        results[condition] = result
        for mode in ('ssi', 'none'):
            print('[test-result]', arm, condition, mode, result['samples'], result['metrics'][mode], flush=True)
    json_write(out / 'summary.json', dict(arm=arm, step=step, manifests=manifests, results=results))
    print(f'[completed-test] {out}/summary.json', flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=list(FIXED), required=True)
    for name in ('base', 'assets', 'manifest-dir', 'data-root'): p.add_argument('--'+name, type=Path, required=True)
    run(p.parse_args())
