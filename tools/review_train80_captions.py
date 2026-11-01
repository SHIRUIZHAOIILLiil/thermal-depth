"""Build a blind, offline caption review from the frozen official training manifest."""
import argparse
import base64
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import random
import re
import sys

import numpy as np

MANIFEST = 'ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl'
SEQUENCES = tuple('''2021-08-06-10-59-33 2021-08-06-11-37-46 2021-08-06-17-21-04
2021-08-06-17-44-55 2021-08-13-16-50-57 2021-08-13-17-06-04
2021-08-13-21-36-10 2021-08-13-22-36-41'''.split())
SEED = 43
PROMPT = ('Describe this thermal image in one concise English sentence for monocular depth estimation. '
          'Mention only objects or structures you can clearly locate, and where they are in the image. '
          'Describe relative depth or occlusion only when it is clearly visible. Omit uncertain details. '
          'A short description of a single visible structure is enough; do not fill in missing objects, '
          'distances or relationships. Do not infer time of day, weather, lighting or temperature.')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_bytes(data)
    temp.replace(path)


def write_json(path, value):
    atomic(path, (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode())


def sample_train(path, root, expected_sha):
    if path.name != MANIFEST or sha(path) != expected_sha:
        raise ValueError('Expected the exact train manifest used by the local-gate experiment')
    groups = {seq:{} for seq in SEQUENCES}
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row.get('id') or row['sample_id']
            seq, frame = sid.rsplit('_', 1)
            if (seq not in groups or not re.fullmatch(r'\d+', frame) or row.get('sequence', seq) != seq
                    or row.get('split', 'train') != 'train' or not isinstance(row.get('caption'), str)
                    or not row['caption'].strip()):
                raise ValueError(f'Unexpected train record {sid}')
            image = Path(row.get('thermal_path') or row.get('thermal') or row['thr_path'])
            if image.parts[-5:] != ('sync_data', '_'+seq, 'thr', 'img_left', frame+'.png'):
                raise ValueError(f'Unexpected thermal image for {sid}: {image}')
            index = int(frame)
            if index in groups[seq]:
                raise ValueError(f'Duplicate frame {sid}')
            groups[seq][index] = (sid, root/image, row['caption'])
    if sum(map(len, groups.values())) != 75688:
        raise ValueError('Expected 75,688 training frames')
    return choose(groups)


def choose(groups):
    picks = []
    for seq in SEQUENCES:
        ids = sorted(groups[seq])
        if len(ids) < 40:
            raise ValueError(f'Too few frames in {seq}')
        rng = random.Random(f'train80-sample-{SEED}-{seq}')
        chosen = []
        for j in range(10):
            lo, hi = j*len(ids)//10, (j+1)*len(ids)//10
            margin = (hi-lo)//4
            chosen.append(ids[rng.randrange(lo+margin, hi-margin)])
        if any(b-a < 2 for a,b in zip(chosen, chosen[1:])):
            raise ValueError(f'Adjacent frames in {seq}')
        # Independent, balanced randomization of A/B within each sequence.
        slots = list('AAAAABBBBB')
        random.Random(f'train80-sides-{SEED}-{seq}').shuffle(slots)
        picks.extend((*groups[seq][i], slot) for i,slot in zip(chosen, slots))
    random.Random(f'train80-order-{SEED}').shuffle(picks)
    return picks


def render(loader, path):
    from PIL import Image
    with Image.open(path) as im:
        raw = np.asarray(im)
        if raw.ndim != 2 or raw.dtype.kind not in 'iu' or raw.dtype.itemsize < 2:
            raise ValueError(f'Expected a high-bit-depth thermal PNG: {path}')
    image = loader(path, thermal_render='grayscale')
    pixels = np.asarray(image)
    if (pixels.dtype != np.uint8 or pixels.shape != (*raw.shape, 3) or
            not np.array_equal(pixels[...,0], pixels[...,1]) or
            not np.array_equal(pixels[...,0], pixels[...,2]) or pixels[...,0].std() < 1e-3):
        raise ValueError(f'Invalid captioner grayscale rendering: {path}')
    return image, [int(pixels.min()), int(pixels.max()), round(float(pixels[...,0].std()), 5)]


def review(out, rows):
    cards = []
    for rid, png, a, b in rows:
        encoded = base64.b64encode(png).decode('ascii')
        cols = []
        for side, caption in [('A', a), ('B', b)]:
            cols.append(f'<td><b>{side}</b><p class="caption">{html.escape(caption)}</p>'
                        f'<label>有明确错述？<select data-field="{side}_error"><option></option>'
                        '<option>有</option><option>未发现</option><option>无法确认</option></select></label>'
                        f'<label>有可定位且有深度价值的信息？<select data-field="{side}_useful"><option></option>'
                        '<option>有</option><option>没有</option><option>无法确认</option></select></label></td>')
        cards.append(f'<tr data-id="{rid}"><td><b>{rid}</b><br><a download="{rid}.png" title="点击下载原尺寸灰度预览">'
                     f'<img src="data:image/png;base64,{encoded}" alt="{rid}"></a></td>'+''.join(cols)+
                     '<td><label>总体偏好<select data-field="preference"><option></option><option>A</option>'
                     '<option>B</option><option>相当</option><option>两句都差</option><option>无法确认</option></select></label>'
                     '<label>备注<textarea data-field="note" rows="3"></textarea></label></td></tr>')
    template = Path(__file__).with_name('train80_caption_review.html').read_text(encoding='utf-8')
    review_id = hashlib.sha256(repr([(r,a,b) for r,_,a,b in rows]).encode()).hexdigest()[:16]
    atomic(out/'review.html', template.replace('<!-- REVIEW_ROWS -->','\n'.join(cards)).replace('__REVIEW_ID__',review_id).encode())


def build(args):
    # Load the same external generator used by caption_ms2.sbatch and bench_caption.py.
    script = (args.caption_repo/'scripts/generate_captions.py').resolve(strict=True)
    sys.path.insert(0, str(script.parent))
    import generate_captions as gen
    if Path(gen.__file__).resolve() != script:
        raise ValueError('Unexpected captioner module')
    parent = json.loads(args.train_contract.read_text())
    expected_sha = parent['train']['sha256']
    picks = sample_train(args.manifest, args.data_root, expected_sha)
    private = args.out/'private'
    private.mkdir(parents=True, exist_ok=True)
    cv = private/'recipe.json'
    recipe = dict(seed=SEED, prompt=PROMPT, train_sha256=expected_sha, generator_sha256=sha(script),
                  generator=gen.InternVLCaptioner.default_model_id, snapshot=str(args.snapshot.resolve(strict=True)),
                  render='grayscale', max_new_tokens=160,
                  frames=[dict(id=s, thermal_sha256=sha(p), slot=slot) for s,p,_,slot in picks])
    if cv.exists() and json.loads(cv.read_text(encoding='utf-8')) != recipe:
        raise ValueError('Existing review is for a different recipe or different inputs')
    write_json(cv, recipe)
    model, rows, key = None, [], []
    for j, (sid, path, old, slot) in enumerate(picks, 1):
        rid = f'R{j:03d}'
        image, stats = render(gen.load_image, path)
        cache = private/(sid+'.json')
        if cache.exists():
            saved = json.loads(cache.read_text(encoding='utf-8'))
            if saved['id'] != sid or saved['render_stats'] != stats:
                raise ValueError(f'Wrong cached frame: {sid}')
            new = saved['caption']
        else:
            if model is None:
                model = gen.InternVLCaptioner(model_id=str(args.snapshot), device='cuda', dtype='auto', max_new_tokens=160)
                model.load()
            new = model.generate(image, prompt=PROMPT, prompt_type='thermal_grounded_train80_v1')
            if not isinstance(new,str) or not new.strip():
                raise ValueError(f'Empty generator output: {sid}')
            write_json(cache, dict(id=sid, render_stats=stats, caption=new))
        if not new.strip():
            raise ValueError(f'Empty cached caption: {sid}')
        png = io.BytesIO()
        image.getchannel('R').save(png, format='PNG')  # Identical grayscale pixels, native resolution.
        a,b = (new,old) if slot=='A' else (old,new)
        rows.append((rid,png.getvalue(),a,b))
        key.append((rid,sid,slot))
        if j % 10 == 0:
            print(f'[caption] {j}/80', flush=True)
    buf = io.StringIO(newline='')
    writer = csv.writer(buf)
    writer.writerow(('review_id','train_frame_id','new_caption_side'))
    writer.writerows(key)
    atomic(private/'answer_key.csv', buf.getvalue().encode('utf-8-sig'))
    review(args.out, rows)
    print(f'[completed] 80 train frames; open {args.out}/review.html', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('manifest','train-contract','data-root','caption-repo','snapshot','out'):
        parser.add_argument('--'+arg, type=Path, required=True)
    build(parser.parse_args())
