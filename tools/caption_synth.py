"""Iris's captions for Hypersim and Virtual KITTI 2, keyed the way the training loaders look them up.

Model, prompt, image tiling and generation settings are copied from Iris's
`internVL/scripts/lotus_inference/inference_{hypersim,vkitti}.py` (InternVL3-8B, bf16,
max_new_tokens=77, do_sample=True). Iris did not release its caption files, so they are regenerated.

Two departures, both so that a frame can never silently train with an empty prompt:
- The loaders look a caption up by the exact image path string they build
  (`hypersim_dataset.get_hypersim_dataset_depth_normal`, `VKITTIDataset.__init__`) and fall back
  to "" when it is absent. Keys here are built the same way from the same data roots, and
  finalize refuses unless every image the loaders will see has a non-empty caption.
- Iris's script skipped frames that raised; here they are recorded and finalize fails.
Sampling is seeded per shard (Iris's was unseeded); a resumed shard does not reproduce the
draws it would have made in one run.

  shard     caption rows index, index+count, ... of both datasets (resumable)
  finalize  merge into hypersim/vkitti_depth_descriptions.json in Iris's format
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.synth_data import vkitti_frames

MODEL = 'OpenGVLab/InternVL3-8B'
PROMPT = ("Describe this image in one sentence, focusing on salient attributes in the given image that are "
          "essential for monocular depth estimation, including but not limited to: camera factors, scene "
          "properties, relative distances, object types, object scales, illuminations, texture, visual features, "
          "occlusions, and boundaries to express how near or far different regions appear from the camera. "
          "Limit your response to one descriptive sentence under 77 tokens.")
GENERATION = dict(max_new_tokens=77, do_sample=True)
SEED = 20261109
KINDS = ('hypersim', 'vkitti')
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def hypersim_images(data_dir):
    """Image paths exactly as get_hypersim_dataset_depth_normal builds them (split 'train')."""
    out = []
    for root, dirs, files in os.walk(os.path.join(data_dir, 'train')):
        for file in files:
            if file.endswith('tonemap.jpg'):
                out.append(os.path.join(root, file))
    return sorted(out)


def all_rows(hypersim_dir, vkitti_dir):
    rows = [('hypersim', p) for p in hypersim_images(hypersim_dir)]
    rows += [('vkitti', p) for p in sorted(r for r, _ in vkitti_frames(vkitti_dir))]
    return rows


# --- copied from Iris's inference_hypersim.py (unchanged behaviour) ---
def build_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    import torch
    from PIL import Image
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(i) for i in images])
# --- end of copied code ---


def shard_path(out, index, count):
    return out/'shards'/f'captions_{index:03d}of{count:03d}.jsonl'


def read_shard(path):
    done = {}
    if path.is_file():
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            done[(r['kind'], r['key'])] = r
    return done


def run_shard(a):
    import torch
    from transformers import AutoModel, AutoTokenizer
    rows = all_rows(a.hypersim, a.vkitti)[a.index::a.count]
    path = shard_path(a.out, a.index, a.count)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {k for k, r in read_shard(path).items() if r['status'] == 'ok'}
    todo = [r for r in rows if r not in done]
    print(f'[caption] shard {a.index}/{a.count}: {len(rows)} images, {len(todo)} to caption', flush=True)
    if not todo:
        return
    torch.manual_seed(SEED + a.index)
    flash = importlib.util.find_spec('flash_attn') is not None
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                      use_flash_attn=flash, trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True, use_fast=False)
    print(f'[caption] model loaded, flash_attn={flash}', flush=True)
    bad = 0
    with open(path, 'a', encoding='utf-8') as f:
        for i, (kind, key) in enumerate(todo, 1):
            try:
                pixel_values = load_image(key, max_num=12).to(torch.bfloat16).cuda()
                text = model.chat(tokenizer, pixel_values, f'<image>\n{PROMPT}', dict(GENERATION)).strip()
                rec = dict(kind=kind, key=key, description=text, status='ok' if text else 'empty')
            except Exception as e:
                rec = dict(kind=kind, key=key, description='', status=f'error: {e}')
            bad += rec['status'] != 'ok'
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            f.flush()
            if i % 200 == 0:
                print(f'[caption] {i}/{len(todo)} (failed {bad})', flush=True)
    if bad > 0.02 * len(todo):
        sys.exit(f'{bad}/{len(todo)} captions failed: environment problem, not individual frames')
    print(f'[caption] shard {a.index} done, failed {bad}', flush=True)


def run_finalize(a):
    rows = all_rows(a.hypersim, a.vkitti)
    merged = {}
    for path in sorted((a.out/'shards').glob(f'captions_*of{a.count:03d}.jsonl')):
        for k, r in read_shard(path).items():
            if r['status'] == 'ok':
                merged[k] = r['description']
    missing = [r for r in rows if not merged.get(r)]
    if missing:
        raise ValueError(f'{len(missing)}/{len(rows)} images have no caption, e.g. {missing[:3]}')
    extra = set(merged) - set(rows)
    if extra:
        raise ValueError(f'{len(extra)} captions for images the loaders will not see, e.g. {sorted(extra)[:3]}')
    for kind in KINDS:
        table = {key: {'description': merged[(k, key)]} for k, key in rows if k == kind}
        target = a.out/f'{kind}_depth_descriptions.json'
        tmp = target.with_suffix('.tmp')
        tmp.write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding='utf-8')
        os.replace(tmp, target)
        print(f'[finalize] {kind}: {len(table)} captions -> {target}', flush=True)
    (a.out/'caption_config.json').write_text(json.dumps(dict(
        model=MODEL, prompt=PROMPT, generation=GENERATION, seed=SEED, shards=a.count,
        hypersim_dir=str(a.hypersim), vkitti_dir=str(a.vkitti),
        note='keys are the loader path strings for exactly these two data roots'), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=('shard', 'finalize'))
    p.add_argument('--hypersim', required=True, help='TRAIN_DATA_DIR_HYPERSIM exactly as training will pass it')
    p.add_argument('--vkitti', required=True, help='TRAIN_DATA_DIR_VKITTI exactly as training will pass it')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--count', type=int, default=1)
    a = p.parse_args()
    if not 0 <= a.index < a.count:
        sys.exit(f'Shard {a.index} of {a.count}')
    run_shard(a) if a.mode == 'shard' else run_finalize(a)
