"""Independent validation after full-budget training; original trainer untouched."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from ms2_eval.io import sha256_file
from tools.da2_caption_data import ThermalLidar, checked_manifest, caption_hash
from tools.train_da2_caption import build_model, evaluate, json_write, token_cache, verify_assets
from tools.da2_continuation import STEPS, read, validate_state, check_source, select_step


def run(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Requires BF16 CUDA GPU')
    directory = args.run_dir
    contract = read(directory / 'contract.json')
    check_source(contract)
    done = read(directory / 'completed.json')
    if done['step'] != 20000 or done['mode'] != 'train':
        raise ValueError('Require successfully completed 20000-step training')
    assets = verify_assets(args.assets)
    if sha256_file(args.assets) != contract['assets']:
        raise ValueError('Assets changed')
    samples, info = checked_manifest(args.val_manifest, args.data_root, 'val', 14249)
    if info != contract['val'] or caption_hash(samples) != contract['caption_val_hash']:
        raise ValueError('Validation data/captions changed')
    anchor = read(directory / 'continuation_anchor.json')
    if anchor != dict(contract_sha256=sha256_file(directory / 'contract.json'),
                      checkpoint8000_sha256=sha256_file(directory / 'step008000.pt'),
                      validation8000_sha256=sha256_file(directory / 'val_step008000.json')):
        raise ValueError('Pilot snapshot changed during continuation')
    tokens = None
    if contract['arm'] == 'caption':
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained(assets['text_path'], subfolder='tokenizer', local_files_only=True)
        tokens = token_cache(samples, tokenizer, assets, info['sha256'], create=False)
    hashes = {str(s): sha256_file(directory / f'step{s:06d}.pt') for s in STEPS}
    provenance = dict(contract=contract, checkpoint_hashes=hashes, stride=4,
                      selection='ssi abs_rel; ties earlier', curve_steps=list(STEPS),
                      validation_sha256=info['sha256'],
                      code={f: sha256_file(ROOT / f) for f in ('tools/eval_da2_continuation.py', 'tools/da2_continuation.py')})
    out = directory / 'validation_20000'
    if out.exists():
        if read(out / 'provenance.json') != provenance:
            raise ValueError('Existing validation belongs to different data/code/checkpoints')
    else:
        out.mkdir()
        json_write(out / 'provenance.json', provenance)
    dataset = ThermalLidar(samples, tokens)
    # Manifest-order stride 4; same exact frame indices for all steps and both arms.
    subset = torch.utils.data.Subset(dataset, list(range(0, len(dataset), 4)))
    model = build_model(assets, contract['arm'], contract['seed']).to('cuda').eval()
    model.requires_grad_(False)

    def score(step, data, folder):
        folder.mkdir(exist_ok=True)
        result_path = folder / f'val_step{step:06d}.json'
        if result_path.exists():
            result = read(result_path)
            if result['step'] != step or result['samples'] != len(data):
                raise ValueError('Invalid cached evaluation')
            return result
        state = torch.load(directory / f'step{step:06d}.pt', map_location='cpu', weights_only=False)
        validate_state(state, contract, step)
        model.load_state_dict(state['model'], strict=True)
        del state
        print(f'[validation] arm={contract["arm"]} step={step} frames={len(data)}', flush=True)
        evaluate(model, data, torch.device('cuda'), True, folder, step)
        return read(result_path)

    curve = {step: score(step, subset, out / 'curve_stride4') for step in STEPS}
    selected = select_step(curve)
    json_write(out / 'selection.json', dict(step=selected, rule=provenance['selection'],
                                           curve={s: r['metrics']['ssi']['abs_rel'] for s, r in curve.items()}))
    fixed = score(20000, dataset, out / 'full')
    best = fixed if selected == 20000 else score(selected, dataset, out / 'full')
    json_write(out / 'summary.json', dict(arm=contract['arm'], selected_step=selected,
                 validation_sha256=info['sha256'], curve_steps=list(STEPS), stride=4,
                 selection=provenance['selection'], curve=curve, fixed20000=fixed, selected_full=best))
    print(f'[completed-validation] {contract["arm"]}: selected step {selected}; {out}/summary.json', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir', 'assets', 'val-manifest', 'data-root'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
