"""Print the RGB and thermal caption for the same frame, beside the frame itself.

Iterating on a caption prompt means reading captions, and reading them one set
at a time hides the only thing that matters: whether the two descriptions of
one frame disagree about where things are. Paired output makes a disagreement
about foreground/midground/background visible at a glance, which no aggregate
score reports.

Two traps this avoids. Frames are joined on `id`, not on the numeric suffix --
every sequence restarts at 000000, so bare frame numbers collide across
sequences. And the two manifests store paths differently (the RGB one absolute
POSIX, the thermal one relative to the MS2 root), so both forms are resolved
rather than one being assumed.

    python tools/compare_caption_sets.py [帧数] [偏移]

Writes caption_diff_frames.png: the frames rendered through the same full-range
min-max the captioner and the depth model both see, numbered to match.
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

A = Path("E:/project/captioning/outputs/abs_manifests/ms2_train_day2seq_clip75_20260728.jsonl")
B = Path("E:/project/captioning/outputs/abs_manifests/ms2_train_day2seq_thermalcap_v2_20260815.jsonl")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
OFFSET = int(sys.argv[2]) if len(sys.argv) > 2 else 0

load = lambda p: {r["id"]: r for r in (json.loads(l) for l in p.open(encoding="utf-8") if l.strip())}
rgb, thr = load(A), load(B)
shared = sorted(set(rgb) & set(thr))
print(f"RGB {len(rgb)} 条 / 热像 {len(thr)} 条 / 同 id {len(shared)} 条\n")

step = len(shared) // (N + 1)
picked = [shared[(OFFSET + i + 1) * step % len(shared)] for i in range(N)]

tiles = []
for k, fid in enumerate(picked, 1):
    r, t = rgb[fid], thr[fid]
    print(f"[{k}] {fid}")
    print(f"    RGB  ({len(r['caption'].split()):>2} 词, {r.get('caption_prompt_version')}):")
    print(f"         {r['caption']}")
    print(f"    热像 ({len(t['caption'].split()):>2} 词, {t.get('caption_prompt_version')}):")
    print(f"         {t['caption']}\n")

    # the thermal manifest stores paths relative to the MS2 root while the RGB
    # one stores absolute POSIX paths -- resolve both rather than assume either
    raw = r["thermal_path"]
    path = Path(raw.replace("/mnt/e/", "E:/")) if raw.startswith("/") else Path("E:/dataset/ms2") / raw
    a = np.asarray(Image.open(path), np.float32)
    lo, hi = a.min(), a.max()
    g = ((a - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
    tile = Image.fromarray(g, "L").convert("RGB").resize((480, 192), Image.BILINEAR)
    d = ImageDraw.Draw(tile)
    d.rectangle([0, 0, 30, 20], fill=(0, 0, 0))
    d.text((10, 5), str(k), fill=(255, 255, 255))
    tiles.append(tile)

cols = 2
sheet = Image.new("RGB", (480 * cols, 192 * ((len(tiles) + cols - 1) // cols)), (255, 255, 255))
for i, tile in enumerate(tiles):
    sheet.paste(tile, ((i % cols) * 480, (i // cols) * 192))
sheet.save("caption_diff_frames.png")
print(f"-> caption_diff_frames.png {sheet.size}")
