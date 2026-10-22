"""Validation-only caption interventions on ONE trusted local step-8000 checkpoint.

Uses the pilot's official_protocol evaluator, not the final-paper unified protocol.
Never creates an optimizer, updates weights, exports predictions, or reads test data.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from ms2_eval.io import sha256_file
from tools.da2_caption_data import ThermalLidar, checked_manifest, caption_hash
from tools.train_da2_caption import (build_model, evaluate, json_write, token_cache,
                                    verify_assets, seed_all)


def derangement(size, seed):
    """Uniform rejection sampling of permutations with no self matches; local RNG."""
    if size < 2:
        raise ValueError('Need at least two frames to shuffle')
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        order = rng.permutation(size)
        if np.all(order != np.arange(size)):
            return order
    raise RuntimeError('Failed to draw derangement')


class CaptionIntervention(Dataset):
    def __init__(self, base, tokens, order):
        self.base, self.tokens, self.order = base, tokens, np.asarray(order)
        if sorted(self.order.tolist()) != list(range(len(base))):
            raise ValueError('Caption mapping must be a permutation of all validation frames')

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = dict(self.base[index])
        source = int(self.order[index])
        item['token_ids'] = self.tokens['input_ids'][source]
        item['attention_mask'] = self.tokens['attention_mask'][source]
        return item


class AdapterOff(nn.Module):
    """Same checkpoint's visual core; no mutation of its text modules or gates."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, token_ids=None, attention_mask=None):
        h, w = image.shape[-2:]
        pad = (0, (-w) % 14, 0, (-h) % 14)
        x = torch.nn.functional.pad(image, pad, mode='reflect') if any(pad) else image
        return self.model.core(x)[..., :h, :w]


def validate_checkpoint(state, contract):
    if state.get('contract') != contract or contract.get('arm') != 'caption':
        raise ValueError('Need caption checkpoint matching contract.json')
    if state.get('step') != 8000 or state.get('image_cursor') != 96000:
        raise ValueError('Only step 8000 / image_cursor 96000 is authorized')


def check_reproduction(reference, actual, atol=1e-6):
    if any(r.get('step') != 8000 or r.get('samples') != 14249 for r in (reference, actual)):
        raise ValueError('Reproduction requires full step-8000 validation')
    differences = {}
    for mode in ('ssi', 'none'):
        groups = [('overall', reference['metrics'][mode], actual['metrics'][mode])]
        if reference['sequences'][mode].keys() != actual['sequences'][mode].keys():
            raise ValueError('Validation sequences changed')
        groups += [(s, v, actual['sequences'][mode][s]) for s, v in reference['sequences'][mode].items()]
        for name, a, b in groups:
            if a.keys() != b.keys():
                raise ValueError('Metric fields changed')
            for key, value in a.items():
                delta = abs(float(b[key]) - float(value))
                differences[f'{mode}/{name}/{key}'] = delta
                if not np.isfinite(delta) or delta > atol:
                    raise ValueError(f'Reproduction failed: {mode}/{name}/{key} delta={delta}; stop, do not relax automatically')
    return differences


def paired_summary(paths):
    rows = {}
    for condition, path in paths.items():
        with path.open(newline='', encoding='utf-8') as handle:
            data = list(csv.DictReader(handle))
        keys = [(r['id'], r['sequence'], r['alignment']) for r in data]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate per-frame metric keys')
        rows[condition] = dict(zip(keys, data))
    correct = rows['correct']
    output = {}
    for condition in ('shuffled', 'adapter_off'):
        if correct.keys() != rows[condition].keys():
            raise ValueError('Conditions scored different frames')
        output[condition] = {}
        for mode in ('ssi', 'none'):
            groups = ['overall'] + sorted({k[1] for k in correct})
            output[condition][mode] = {}
            for group in groups:
                keys = [k for k in correct if k[2] == mode and (group == 'overall' or k[1] == group)]
                stats = {}
                for metric in ('abs_rel', 'rmse', 'a1'):
                    diff = np.array([float(rows[condition][k][metric]) - float(correct[k][metric]) for k in keys])
                    if not len(diff) or not np.isfinite(diff).all():
                        raise ValueError('Invalid paired metrics')
                    stats[metric] = dict(mean_delta=float(diff.mean()), median_delta=float(np.median(diff)),
                                        correct_better_fraction=float(np.mean(diff < 0 if metric == 'a1' else diff > 0)))
                output[condition][mode][group] = dict(frames=len(keys), metrics=stats)
    return dict(delta_definition='intervention minus correct; positive error delta favors correct, negative a1 delta favors correct',
                note='Descriptive paired results; frames are temporally correlated. Not a significance test.', comparisons=output)


def run(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Requires BF16-capable CUDA GPU, same precision as pilot')
    started = time.monotonic()
    contract = json.loads((args.run_dir / 'contract.json').read_text(encoding='utf-8'))
    checkpoint = args.run_dir / 'step008000.pt'
    # Trusted user-owned training output only (contains optimizer/RNG Python objects).
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    validate_checkpoint(state, contract)
    assets = verify_assets(args.assets)
    if sha256_file(args.assets) != contract['assets']:
        raise ValueError('Assets differ from training contract')
    for filename, digest in contract['code'].items():
        if sha256_file(ROOT / filename) != digest:
            raise ValueError(f'Training/evaluation source changed: {filename}')
    if contract['software'] != dict(torch=torch.__version__, numpy=np.__version__) or contract['amp'] != 'bf16':
        raise ValueError('Software/precision differs from pilot')
    samples, info = checked_manifest(args.val_manifest, args.data_root, 'val', 14249)
    if info != contract['val'] or caption_hash(samples) != contract['caption_val_hash']:
        raise ValueError('Validation manifest/captions changed')
    reference_path = args.run_dir / 'val_step008000.json'
    reference = json.loads(reference_path.read_text(encoding='utf-8'))
    from transformers import CLIPTokenizer
    tokenizer = CLIPTokenizer.from_pretrained(assets['text_path'], subfolder='tokenizer', local_files_only=True)
    tokens = token_cache(samples, tokenizer, assets, info['sha256'], create=False)
    order = derangement(len(samples), args.shuffle_seed)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'shuffle_mapping.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['frame_id', 'caption_source_id', 'same_caption', 'same_token_input'])
        identical_text = identical_tokens = 0
        for i, j in enumerate(order):
            same_text = samples[i].caption == samples[int(j)].caption
            same_tokens = bool(torch.equal(tokens['input_ids'][i], tokens['input_ids'][j]) and
                               torch.equal(tokens['attention_mask'][i], tokens['attention_mask'][j]))
            identical_text += same_text; identical_tokens += same_tokens
            writer.writerow([samples[i].sample_id, samples[int(j)].sample_id, same_text, same_tokens])
    if identical_tokens == len(samples):
        raise ValueError('Shuffle leaves every text input unchanged')
    metadata = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256_file(checkpoint),
                    reference_sha256=sha256_file(reference_path), step=8000, samples=len(samples),
                    contract=contract, shuffle_seed=args.shuffle_seed, shuffle_scope='global validation derangement',
                    identical_text_count=identical_text, identical_token_count=identical_tokens,
                    diagnostic_code={f: sha256_file(ROOT / f) for f in ('tools/eval_da2_caption_ablation.py',)},
                    gpu=torch.cuda.get_device_name(), reproduction_absolute_tolerance=1e-6)
    json_write(args.output / 'metadata.json', metadata)
    print('[shuffle] ' + json.dumps({k: metadata[k] for k in ('shuffle_seed', 'identical_text_count', 'identical_token_count')}), flush=True)
    seed_all(contract['seed'])
    model = build_model(assets, 'caption', contract['seed'])
    model.load_state_dict(state['model'], strict=True)
    del state
    model.requires_grad_(False).to('cuda').eval()
    base = ThermalLidar(samples)
    reports, paths, timings = {}, {}, {}
    for condition, dataset, predictor in (
            ('correct', CaptionIntervention(base, tokens, np.arange(len(samples))), model),
            ('shuffled', CaptionIntervention(base, tokens, order), model),
            ('adapter_off', base, AdapterOff(model).eval())):
        out = args.output / condition
        out.mkdir()
        torch.cuda.synchronize()
        begin = time.monotonic()
        print(f'[condition] {condition}: start {len(dataset)} frames', flush=True)
        evaluate(predictor, dataset, torch.device('cuda'), True, out, 8000)
        torch.cuda.synchronize()
        timings[condition] = time.monotonic() - begin
        reports[condition] = json.loads((out / 'val_step008000.json').read_text())
        paths[condition] = out / 'val_step008000_frames.csv'
        if condition == 'correct':
            diffs = check_reproduction(reference, reports[condition])
            json_write(args.output / 'reproduction.json', dict(passed=True, absolute_differences=diffs))
            print(f'[reproduction] PASS; first pass {timings[condition]/60:.1f} min; remaining roughly <= twice this', flush=True)
    summary = dict(step=8000, samples=len(samples), conditions=reports, paired=paired_summary(paths),
                   condition_seconds=timings, elapsed_seconds=time.monotonic()-started)
    json_write(args.output / 'summary.json', summary)
    for condition, report in reports.items():
        for mode, m in report['metrics'].items():
            print(f"[result] {condition:12s} {mode:4s} AbsRel={m['abs_rel']:.8f} RMSE={m['rmse']:.6f} a1={m['a1']:.8f}", flush=True)
    print(f'[completed] {args.output}/summary.json; no training performed', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--val-manifest', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shuffle-seed', type=int, default=43)
    run(p.parse_args())
