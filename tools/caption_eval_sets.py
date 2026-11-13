"""Iris's captions for its five zero-shot depth test sets, keyed the way eval_rng.py looks them up.

Iris's `internVL/scripts/lotus_inference/inference_{nyu,kitti,eth3d,scannet,diode}.py` read the
first column of each Marigold filename list, open that member ("./" + path) of the set's tar and
caption it with the same InternVL3-8B, prompt and generation settings as the training sets
(`tools/caption_synth.py`, checked against those scripts by the tests). `lotus/eval_rng.py` then
looks the caption up by the loader's `rgb_relative_path`, which is that same first column
(`evaluation/dataset_depth/base_depth_dataset.py::_get_data_path`), under the config `name`.

As in caption_synth.py, a missing or failed caption makes finalize fail instead of becoming an
empty prompt, and eval is meant to run with --strict_text_mode.

  check     every listed RGB image is a member of its tar (after download)
  shard     caption rows index, index+count, ... across all five sets (resumable)
  finalize  write <set>_depth_descriptions.json, the file names eval-depth-g.sh passes
"""
import argparse
import io
import json
import os
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import caption_synth as cap

SPLITS = ROOT/'lotus'/'datasets'/'eval'/'depth'/'data_split'
# config name (eval_rng.py key) -> (tar under <data>/depth, filename list under data_split, output stem)
SETS = {
    'nyu_v2': ('nyuv2/nyu_labeled_extracted.tar', 'nyu/labeled/filename_list_test.txt', 'nyu'),
    'kitti': ('kitti/kitti_eigen_split_test.tar', 'kitti/eigen_test_files_with_gt.txt', 'kitti'),
    'eth3d': ('eth3d/eth3d.tar', 'eth3d/eth3d_filename_list.txt', 'eth3d'),
    'scannet': ('scannet/scannet_val_sampled_800_1.tar', 'scannet/scannet_val_sampled_list_800_1.txt', 'scannet'),
    'diode': ('diode/diode_val.tar', 'diode/diode_val_all_filename_list.txt', 'diode'),
}
SEED = 20261113


def listed_images(split_file):
    """First column of each line, sorted and unique, as Iris's find_all_images + dict keys."""
    out = set()
    for line in open(split_file, encoding='utf-8'):
        parts = line.strip().split()
        if parts:
            out.add(parts[0])
    return sorted(out)


def all_rows(splits=SPLITS):
    return [(name, key) for name, (_, lst, _) in SETS.items() for key in listed_images(splits/lst)]


def member(tar, key):
    """Iris opens "./" + path; accept the bare path too."""
    for name in ('./' + key, key):
        try:
            return tar.getmember(name)
        except KeyError:
            continue
    raise KeyError(key)


def image_bytes(tar, key):
    return tar.extractfile(member(tar, key)).read()


def run_check(a):
    total = 0
    for name, (tar_rel, lst, _) in SETS.items():
        keys = listed_images(a.splits/lst)
        with tarfile.open(a.data/'depth'/tar_rel) as tar:
            names = set(tar.getnames())
        missing = [k for k in keys if './' + k not in names and k not in names]
        if missing:
            raise ValueError(f'{name}: {len(missing)}/{len(keys)} listed images not in {tar_rel}, e.g. {missing[:3]}')
        print(f'[check] {name}: {len(keys)} images present in {tar_rel}', flush=True)
        total += len(keys)
    print(f'[check] {total} images in all', flush=True)


def run_shard(a):
    import importlib.util
    import torch
    from transformers import AutoModel, AutoTokenizer
    rows = all_rows(a.splits)[a.index::a.count]
    path = cap.shard_path(a.out, a.index, a.count)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {k for k, r in cap.read_shard(path).items() if r['status'] == 'ok'}
    todo = [r for r in rows if r not in done]
    print(f'[caption] shard {a.index}/{a.count}: {len(rows)} images, {len(todo)} to caption', flush=True)
    if not todo:
        return
    torch.manual_seed(SEED + a.index)
    flash = importlib.util.find_spec('flash_attn') is not None
    model = AutoModel.from_pretrained(cap.MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                      use_flash_attn=flash, trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(cap.MODEL, trust_remote_code=True, use_fast=False)
    tars, bad = {}, 0
    with open(path, 'a', encoding='utf-8') as f:
        for i, (name, key) in enumerate(todo, 1):
            try:
                if name not in tars:
                    tars[name] = tarfile.open(a.data/'depth'/SETS[name][0])
                pixel_values = cap.load_image(io.BytesIO(image_bytes(tars[name], key)), max_num=12)
                text = model.chat(tokenizer, pixel_values.to(torch.bfloat16).cuda(),
                                  f'<image>\n{cap.PROMPT}', dict(cap.GENERATION)).strip()
                rec = dict(kind=name, key=key, description=text, status='ok' if text else 'empty')
            except Exception as e:
                rec = dict(kind=name, key=key, description='', status=f'error: {e}')
            bad += rec['status'] != 'ok'
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            f.flush()
            if i % 100 == 0:
                print(f'[caption] {i}/{len(todo)} (failed {bad})', flush=True)
    for t in tars.values():
        t.close()
    if bad > 0.02 * len(todo):
        sys.exit(f'{bad}/{len(todo)} captions failed: environment problem, not individual images')
    print(f'[caption] shard {a.index} done, failed {bad}', flush=True)


def run_finalize(a):
    rows = all_rows(a.splits)
    merged = {}
    for path in sorted((a.out/'shards').glob(f'captions_*of{a.count:03d}.jsonl')):
        for k, r in cap.read_shard(path).items():
            if r['status'] == 'ok':
                merged[k] = r['description']
    missing = [r for r in rows if not merged.get(r)]
    if missing:
        raise ValueError(f'{len(missing)}/{len(rows)} images have no caption, e.g. {missing[:3]}')
    extra = set(merged) - set(rows)
    if extra:
        raise ValueError(f'{len(extra)} captions for images no filename list names, e.g. {sorted(extra)[:3]}')
    for name, (_, _, stem) in SETS.items():
        table = {key: {'description': merged[(n, key)]} for n, key in rows if n == name}
        target = a.out/f'{stem}_depth_descriptions.json'
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp, target)
        print(f'[finalize] {name}: {len(table)} captions -> {target}', flush=True)
    (a.out/'caption_config.json').write_text(json.dumps(dict(
        model=cap.MODEL, prompt=cap.PROMPT, generation=cap.GENERATION, seed=SEED, shards=a.count,
        data=str(a.data), splits=str(a.splits)), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=('check', 'shard', 'finalize'))
    p.add_argument('--data', type=Path, required=True, help='BASE_TEST_DATA_DIR (holds depth/<set>/*.tar)')
    p.add_argument('--splits', type=Path, default=SPLITS)
    p.add_argument('--out', type=Path)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--count', type=int, default=1)
    a = p.parse_args()
    if a.mode != 'check' and a.out is None:
        sys.exit('--out is required')
    if not 0 <= a.index < a.count:
        sys.exit(f'Shard {a.index} of {a.count}')
    {'check': run_check, 'shard': run_shard, 'finalize': run_finalize}[a.mode](a)
