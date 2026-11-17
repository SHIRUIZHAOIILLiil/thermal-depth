"""Where along each validation sequence does the synth arm trail NeWCRF, and what do those frames show?

Diagnostic only (not a result figure). Per sequence: the per-frame RMSE gap (synth - NeWCRF, both
ssi_log) in frame order with a 101-frame rolling mean, the three worst and the best non-overlapping
200-frame stretches, and thermal thumbnails (1-99 percentile stretch, display only) from them.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import val_gap_breakdown as breakdown

WINDOW = 200
ROLL = 101


def stretches(gap, count, worst=True):
    """Non-overlapping WINDOW-frame stretches with the highest (or lowest) mean gap."""
    means = np.convolve(gap, np.ones(WINDOW)/WINDOW, mode='valid')
    order = np.argsort(means)[::-1] if worst else np.argsort(means)
    picked = []
    for start in order:
        if all(abs(start-p) >= WINDOW for p in picked):
            picked.append(int(start))
        if len(picked) == count:
            break
    return [(p, float(means[p])) for p in picked]


def thumb(path):
    with Image.open(path) as im:
        x = np.asarray(im).astype(np.float64)
    lo, hi = np.percentile(x, (1, 99))
    return np.clip((x-lo)/max(hi-lo, 1e-6), 0, 1)


def run(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tools.da2_caption_data import checked_manifest  # imports torch

    before = json.loads((a.previous/'summary.json').read_text())
    if not before['validation']['sha256'].startswith('c26a5859'):
        raise ValueError('Expected the frozen official validation manifest')
    samples, _ = checked_manifest(Path(before['validation']['path']), a.data_root, 'val', 14249)
    synth = breakdown.read(a.runs/'eval'/f'{breakdown.TAGS["synth"]}_val_{breakdown.VAL_FP}_stride1_correct_s16000'/
                           'eval_eval_affine_invariant_log_space_per_sample.csv')
    newcrf = breakdown.read(a.newcrf, 'ssi_log_')
    a.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for seq in sorted({s.sample_id.rsplit('_', 1)[0] for s in samples}):
        frames = sorted((s for s in samples if s.sample_id.startswith(seq+'_')), key=lambda s: int(s.sample_id.rsplit('_', 1)[1]))
        gap = np.array([synth[s.sample_id]['rmse']-newcrf[s.sample_id]['rmse'] for s in frames])
        worst, best = stretches(gap, 3), stretches(gap, 1, worst=False)
        report[seq] = dict(frames=len(frames), mean_gap=float(gap.mean()),
                           worst=[dict(first=frames[p].sample_id, last=frames[p+WINDOW-1].sample_id, mean_gap=m) for p, m in worst],
                           best=[dict(first=frames[p].sample_id, last=frames[p+WINDOW-1].sample_id, mean_gap=m) for p, m in best])
        print(seq, json.dumps(report[seq]), flush=True)
        picks = [(f'worst{i+1} {m:+.2f} m', p+WINDOW//2) for i, (p, m) in enumerate(worst)] + \
                [(f'best {m:+.2f} m', p+WINDOW//2) for p, m in best]
        fig = plt.figure(figsize=(16, 6))
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(gap, lw=.3, color='0.7')
        ax.plot(np.arange(ROLL//2, len(gap)-ROLL//2), np.convolve(gap, np.ones(ROLL)/ROLL, mode='valid'), lw=1.2, color='C3')
        ax.axhline(0, color='k', lw=.6)
        for label, c in picks:
            ax.axvspan(c-WINDOW//2, c+WINDOW//2, color='C0' if label.startswith('best') else 'C1', alpha=.2)
        ax.set_title(f'{seq}: per-frame RMSE gap synth - NeWCRF (ssi_log), mean {gap.mean():+.3f} m')
        ax.set_xlabel('frame order')
        for k, (label, c) in enumerate(picks):
            for j, off in enumerate((-60, 60)):
                axi = fig.add_subplot(2, 8, 9+2*k+j)
                f = frames[min(max(c+off, 0), len(frames)-1)]
                axi.imshow(thumb(f.thermal_path), cmap='gray')
                axi.set_title(f'{label}\n{f.sample_id.rsplit("_", 1)[1]}', fontsize=7)
                axi.axis('off')
        fig.tight_layout()
        fig.savefig(a.out/f'{seq}.png', dpi=110)
        plt.close(fig)
    (a.out/'stretches.json').write_text(json.dumps(report, indent=2))
    print('[completed]', a.out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('runs', 'newcrf', 'previous', 'data-root', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    run(p.parse_args())
