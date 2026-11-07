"""Unattended steps around the far-target retrain: choose, infer, report.

choose and report use only the standard library so they can run without the conda
environment. Neither choose, infer, report nor gate reads a test split.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CANDIDATES = ('bts', 'adabins')
PROTOCOL = 'fit_only_alignment_v2'
STOP = 3
TRAIN_FRAMES = 75688
SHAPE = (256, 640)
KEYS = ('abs_rel', 'rmse', 'a1')
# Frozen in docs/FAR_TARGET_RETRAIN_PIPELINE_V1.md before any retrain result exists.
GATE = dict(rmse_gain_m=.10, absrel_increase=.002, a1_decrease=.003)


def chosen_source(report):
    """Return the source only for the one status that may start a retrain."""
    decision, control = report['decision'], report['control']
    if report.get('protocol') != PROTOCOL:
        raise ValueError(f"Unexpected diagnostic protocol {report.get('protocol')}")
    if decision['status'] != 'clean_source_passed':
        return None
    source = decision['chosen']
    if source not in CANDIDATES or control['passed'] is not True or decision['sources'][source]['passed'] is not True:
        raise ValueError('Diagnostic decision contradicts its own control or source rows')
    return source


def choose(a):
    report = json.loads(a.summary.read_text())
    source = chosen_source(report)
    decision = report['decision']
    print('[choose]', json.dumps(dict(status=decision['status'], chosen=source, control=report['control'],
                                      sources=decision['sources'])), flush=True)
    if source is None:
        print('[choose] no automatic retrain for this status', flush=True)
        return STOP
    temporary = a.out.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(source=source, summary=str(a.summary), status=decision['status']), indent=2))
    os.replace(temporary, a.out)
    return 0


def save_atomic(path, value):
    import numpy as np
    temporary = path.with_name(f'.{path.stem}.tmp.npy')
    np.save(temporary, value)
    os.replace(temporary, path)


def pending(samples, out):
    return [s for s in samples if not (out/f'{s.sample_id}.npy').is_file()]


def infer(a):
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
    import numpy as np
    import torch
    from torch.nn import functional as F
    from ms2_eval.io import load_ms2_gt, sha256_file
    from tools.da2_caption_data import checked_manifest
    from tools.diagnose_far_target_source import CHECKPOINT, SHA256
    from tools.run_ms2_supdepth_baselines import build_network, forward_depth, load_weights, preprocess_thermal
    from tools.train_iris_latent_release import parent_inputs

    if not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA')
    name = json.loads(a.choice.read_text())['source']
    if name not in CANDIDATES:
        raise ValueError(f'{name} is not an automatic source')
    contract = parent_inputs(a.runs)['contract']['train']
    samples, info = checked_manifest(contract['path'], a.data_root, 'train', TRAIN_FRAMES)
    if info != contract or sha256_file(a.manifest) != info['sha256']:
        raise ValueError('Training manifest differs from the frozen recipe')
    if load_ms2_gt(samples[0].thermal_gt_path, 256.)[1].shape != SHAPE:
        raise ValueError('Unexpected GT shape')
    path = a.ckpt_dir/f'MS2_MD_{CHECKPOINT[name]}_THR_ckpt.ckpt'
    if sha256_file(path) != SHA256[name]:
        raise ValueError(f'{name} checkpoint differs from the diagnosed file')
    a.out.mkdir(parents=True, exist_ok=True)
    todo = pending(samples, a.out)
    print(f'[infer] {name}: {len(todo)} of {len(samples)} frames left', flush=True)
    network = build_network(name, a.bmsd_root)
    load_weights(network, path, strict=True)
    network = network.cuda().eval()
    with torch.inference_mode():
        for i, s in enumerate(todo, 1):
            depth = forward_depth(name, network, preprocess_thermal(Path(s.thermal_path)).cuda())
            prediction = depth[None] if depth.dim() == 3 else depth
            if prediction.shape[-2:] != SHAPE:
                prediction = F.interpolate(prediction, SHAPE, mode='bilinear', align_corners=False)
            value = prediction[0, 0].float().cpu().numpy().astype(np.float32)
            if not np.isfinite(value).all():
                raise ValueError(f'{s.sample_id}: nonfinite prediction')
            save_atomic(a.out/f'{s.sample_id}.npy', value)
            if i % 2000 == 0:
                print(f'[infer] {i}/{len(todo)}', flush=True)
    if pending(samples, a.out):
        raise ValueError('Raw predictions are incomplete')
    print(f'[infer] complete -> {a.out}', flush=True)
    return 0


def metrics(path):
    raw = json.loads(Path(path).read_text())
    values = {k: float(raw[k]) for k in KEYS}
    if not all(math.isfinite(x) for x in values.values()):
        raise ValueError(f'Nonfinite metrics in {path}')
    return values


def retrain_gate(candidate, start):
    """Caption arm against the q100 AnyThermal arm at the same 16k step."""
    gain = start['rmse']-candidate['rmse']
    gates = dict(rmse=gain >= GATE['rmse_gain_m']-1e-12,
                 abs_rel=candidate['abs_rel'] <= start['abs_rel']+GATE['absrel_increase']+1e-12,
                 a1=candidate['a1'] >= start['a1']-GATE['a1_decrease']-1e-12)
    return dict(passed=all(gates.values()), gates=gates, rmse_gain_m=gain, absrel_change=candidate['abs_rel']-start['abs_rel'],
                a1_change=candidate['a1']-start['a1'], limits=GATE)


def line(label, value):
    return f"{label:<10} AbsRel {value['abs_rel']*100:.3f}  RMSE {value['rmse']:.4f}  d1 {value['a1']*100:.2f}"


def report(a):
    if a.choice.is_file():
        print('[source]', json.loads(a.choice.read_text())['source'])
    else:
        print('[source] none chosen; read the diagnostic decision below')
        if a.summary.is_file():
            source = json.loads(a.summary.read_text())
            print('[decision]', json.dumps(source['decision']))
            print('[control]', json.dumps(source['control']))
        else:
            print(f'[decision] missing {a.summary}')
    start = metrics(a.reference)
    print(line('tmax1', start))
    found = {}
    for label, path in (('caption', a.caption), ('empty', a.empty)):
        if path.is_file():
            found[label] = metrics(path)
            print(line(label, found[label]))
        else:
            print(f'{label:<10} no 16k validation result')
    if 'caption' not in found:
        print('[gate] not reached')
        return 0
    result = retrain_gate(found['caption'], start)
    print('[gate]', 'PASSED' if result['passed'] else 'FAILED', json.dumps(result))
    return 0


def gate(a):
    """Exit status decides whether the test jobs run; validation numbers only."""
    result = retrain_gate(metrics(a.caption), metrics(a.reference))
    print('[gate]', 'PASSED' if result['passed'] else 'FAILED', json.dumps(result), flush=True)
    return 0 if result['passed'] else STOP


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    item = commands.add_parser('choose')
    item.add_argument('--summary', type=Path, required=True)
    item.add_argument('--out', type=Path, required=True)
    item.set_defaults(run=choose)
    item = commands.add_parser('infer')
    for name in ('runs', 'choice', 'manifest', 'data-root', 'bmsd-root', 'ckpt-dir', 'out'):
        item.add_argument('--'+name, type=Path, required=True)
    item.set_defaults(run=infer)
    item = commands.add_parser('report')
    for name in ('choice', 'summary', 'reference', 'caption', 'empty'):
        item.add_argument('--'+name, type=Path, required=True)
    item.set_defaults(run=report)
    item = commands.add_parser('gate')
    for name in ('reference', 'caption'):
        item.add_argument('--'+name, type=Path, required=True)
    item.set_defaults(run=gate)
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse()
    raise SystemExit(args.run(args))
