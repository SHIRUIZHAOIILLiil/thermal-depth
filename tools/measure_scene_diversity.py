"""How many genuinely different scenes are in the training set?

Counting sequences and frame strides answers a question nobody asked: it
describes how the data was recorded, not how much the picture changes while
driving. A seventeen-minute drive can cross several districts, and two frames
a hundred apart may share nothing.

So measure it on the pixels. Each frame is reduced to a 32x16 signature under
the same per-image min-max the depth model's encoder applies, and the distance
between frames is tracked as a function of how far apart in time they are. The
lag at which that distance stops growing is how long it takes the view to
become unrelated to itself -- and total frames divided by that lag is the
number of effectively independent looks the training set contains.
"""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

MANIFEST = Path("E:/project/captioning/outputs/abs_manifests/ms2_train_day2seq_clip75_20260728.jsonl")
SIG_W, SIG_H = 32, 16


def fix(p):
    return p.replace("/mnt/e/", "E:/")


def signature(path):
    a = np.asarray(Image.open(path).resize((SIG_W, SIG_H), Image.BILINEAR), np.float32)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / max(hi - lo, 1e-6)


rows = [json.loads(l) for l in MANIFEST.open(encoding="utf-8") if l.strip()]
seqs = {}
for r in rows:
    seqs.setdefault(r["id"].rsplit("_", 1)[0], []).append(r)

started = time.time()
sigs = {}
for name, group in seqs.items():
    group.sort(key=lambda r: r["id"])
    arr = np.stack([signature(fix(r["thermal_path"])) for r in group]).reshape(len(group), -1)
    sigs[name] = arr
    print(f"{name}: {len(group)} 帧读完，{time.time()-started:.0f}s", flush=True)

rng = np.random.default_rng(0)
allsig = np.concatenate(list(sigs.values()))
i = rng.integers(0, len(allsig), 200000)
j = rng.integers(0, len(allsig), 200000)
baseline = float(np.sqrt(((allsig[i] - allsig[j]) ** 2).mean(1)).mean())
print(f"\n随机两帧的基准距离 = {baseline:.4f}（'完全不相关'的水平）\n")

lags = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 800, 1200, 2000]
print(f"{'lag(帧)':>8}{'距离':>10}{'占基准':>9}")
reach = {}
for lag in lags:
    vals = []
    for arr in sigs.values():
        if len(arr) > lag:
            vals.append(np.sqrt(((arr[:-lag] - arr[lag:]) ** 2).mean(1)))
    d = float(np.concatenate(vals).mean())
    reach[lag] = d / baseline
    print(f"{lag:>8}{d:>10.4f}{d/baseline:>8.0%}")

for target in (0.80, 0.90, 0.95):
    hit = next((l for l in lags if reach[l] >= target), None)
    if hit:
        n = sum(len(a) for a in sigs.values())
        print(f"\n达到基准的 {target:.0%} 需要相隔 {hit} 帧 -> 有效独立场景数 ≈ {n // hit}")
