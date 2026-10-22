"""Read-only continuation guards and paired final report; original trainer unchanged."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from ms2_eval.io import sha256_file
from tools.train_da2_caption import json_write

STEPS = tuple(range(2000, 20001, 2000))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_state(state, contract, step=None):
    current = state.get('step', -1)
    if state.get('contract') != contract or contract.get('arm') not in ('caption', 'no-caption'):
        raise ValueError('Checkpoint contract/arm mismatch')
    if step is not None and current != step:
        raise ValueError(f'Expected step {step}, found {current}')
    if not 0 < current <= 20000 or state.get('image_cursor') != current * 12:
        raise ValueError('Invalid checkpoint step/image cursor')
    for key in ('model', 'optimizer', 'scheduler', 'rng', 'amp'):
        if key not in state:
            raise ValueError(f'Incomplete training state: {key}')
    if state['scheduler'].get('last_epoch') != current:
        raise ValueError('Scheduler position differs from optimizer step')


def check_source(contract):
    for filename, digest in contract['code'].items():
        if sha256_file(ROOT / filename) != digest:
            raise ValueError(f'Original source changed: {filename}')
    if contract['software'] != dict(torch=torch.__version__, numpy=np.__version__) or contract['amp'] != 'bf16':
        raise ValueError('Software or precision differs from original run')


def preflight(run_dir, arm, smoke_output):
    from tools.check_da2_caption_pair import check
    check(smoke_output / 'no-caption', smoke_output / 'caption', smoke=True)
    contract = read(run_dir / 'contract.json')
    if contract['arm'] != arm or contract['seed'] != 43:
        raise ValueError('Wrong continuation arm/seed')
    smoke = read(smoke_output / 'no-caption/contract.json')
    if {k: v for k, v in contract.items() if k != 'arm'} != {k: v for k, v in smoke.items() if k != 'arm'}:
        raise ValueError('Run differs from paired smoke contract')
    check_source(contract)
    for step in (2000, 4000, 6000, 8000):
        if not (run_dir / f'step{step:06d}.pt').is_file():
            raise FileNotFoundError(f'Missing curve checkpoint {step}')
    for step in (6000, 8000):
        report = read(run_dir / f'val_step{step:06d}.json')
        if report['step'] != step or report['samples'] != 14249:
            raise ValueError('Missing full pilot validation')
    state = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
    validate_state(state, contract)
    if state['step'] < 8000:
        raise ValueError('Must resume an existing >=8000-step run, never start fresh')
    step = state['step']; del state
    anchor = run_dir / 'continuation_anchor.json'
    frozen = dict(contract_sha256=sha256_file(run_dir / 'contract.json'),
                  checkpoint8000_sha256=sha256_file(run_dir / 'step008000.pt'),
                  validation8000_sha256=sha256_file(run_dir / 'val_step008000.json'))
    if anchor.exists():
        if read(anchor) != frozen:
            raise ValueError('Frozen pilot files changed after continuation began')
    else:
        json_write(anchor, frozen)
    print(f'[continuation] {arm}: resume {step}->20000; original 8000 snapshot preserved', flush=True)


def select_step(curve):
    if set(curve) != set(STEPS):
        raise ValueError('Need all ten predeclared curve points')
    scores = {s: float(curve[s]['metrics']['ssi']['abs_rel']) for s in STEPS}
    if not all(np.isfinite(v) for v in scores.values()):
        raise ValueError('Nonfinite selection score')
    return min(STEPS, key=lambda s: (scores[s], s))


def audit_logs(a, b):
    contracts = [read(p / 'contract.json') for p in (a, b)]
    if [c['arm'] for c in contracts] != ['no-caption', 'caption']:
        raise ValueError('Expected A=no-caption and B=caption')
    if {k: v for k, v in contracts[0].items() if k != 'arm'} != {k: v for k, v in contracts[1].items() if k != 'arm'}:
        raise ValueError('Training contracts differ beyond arm')
    logs = []
    for p in (a, b):
        done = read(p / 'completed.json')
        if done['step'] != 20000 or done['mode'] != 'train':
            raise ValueError('Training not complete at 20000')
        rows = {r['step']: r for r in map(json.loads, (p / 'train.jsonl').read_text().splitlines())}
        if set(range(1, 20001)) - rows.keys():
            raise ValueError('Incomplete 20000-step training log')
        if any(rows[s]['image_cursor'] != s*12 for s in range(1, 20001)):
            raise ValueError('Wrong consumed image cursor')
        logs.append(rows)
    if any(logs[0][s]['data_ids_sha256'] != logs[1][s]['data_ids_sha256'] for s in range(1, 20001)):
        raise ValueError('Paired training image order differs')


def report_pair(base):
    a, b = (base / f'{arm}_s43' for arm in ('no-caption', 'caption'))
    audit_logs(a, b)
    reports = [read(p / 'validation_20000/summary.json') for p in (a, b)]
    if reports[0]['validation_sha256'] != reports[1]['validation_sha256']:
        raise ValueError('Validation manifests differ')
    for r in reports:
        if r['curve_steps'] != list(STEPS) or r['stride'] != 4 or r['selection'] != 'ssi abs_rel; ties earlier':
            raise ValueError('Unexpected selection protocol')
    for kind in ('fixed20000', 'selected_full'):
        print(f'=== {kind} (do not mix fixed-step and selected comparisons) ===')
        for arm, r in zip(('no-caption', 'caption'), reports):
            result = r[kind]
            if result['samples'] != 14249:
                raise ValueError('Expected full validation')
            for mode in ('ssi', 'none'):
                m = result['metrics'][mode]
                print(f"{arm:10s} step={result['step']:5d} {mode:4s} AbsRel={m['abs_rel']:.8f} RMSE={m['rmse']:.6f} a1={m['a1']:.8f}")
    print('[pair] 20000 updates checked, same image order; no test evaluation or automatic decision')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    p = commands.add_parser('preflight')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--arm', choices=['no-caption', 'caption'], required=True)
    p.add_argument('--smoke-output', type=Path, required=True)
    p = commands.add_parser('report')
    p.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'preflight': preflight(args.run_dir, args.arm, args.smoke_output)
    else: report_pair(args.base)
