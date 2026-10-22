"""Validation-only protocol and deterministic, resumable image stream."""
import hashlib
import json
import os
import re
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
from models.anythermal_encoder import AnyThermalEncoder
from ms2_eval.io import load_manifest, load_ms2_gt

TRAIN = set('''2021-08-06-10-59-33 2021-08-06-11-37-46 2021-08-06-17-21-04
2021-08-06-17-44-55 2021-08-13-16-50-57 2021-08-13-17-06-04
2021-08-13-21-36-10 2021-08-13-22-36-41'''.split())
VAL = set('2021-08-13-16-08-46 2021-08-13-16-14-48 2021-08-13-21-58-13'.split())


def checked_manifest(path, root, split, expected=None):
    # Legacy key allowed only with the strict /thr/ and sequence checks below.
    samples, info = load_manifest(path, root, allow_legacy_depth_path=True)
    allowed = TRAIN if split == 'train' else VAL
    ids = set()
    for s in samples:
        if s.sample_id in ids or not re.fullmatch(r'\d{4}(?:-\d{2}){5}_\d+', s.sample_id):
            raise ValueError('duplicate/unsafe sample ID')
        ids.add(s.sample_id)
        seq = s.sample_id.rsplit('_', 1)[0]
        if seq not in allowed or s.sequence not in (seq, 'unknown'):
            raise ValueError(f'Forbidden split/sequence: {s.sample_id}')
        for field in (s.thermal_path, s.thermal_gt_path):
            found = re.findall(r'\d{4}(?:-\d{2}){5}', field)
            if found != [seq] or Path(field).stem != s.sample_id.rsplit('_', 1)[1]:
                raise ValueError(f'ID/path disagreement: {s.sample_id}')
            if not Path(field).is_file():
                raise FileNotFoundError(field)
        if '/thr/' not in s.thermal_gt_path.replace('\\', '/'):
            raise ValueError('GT must be thermal-view /thr/ filtered LiDAR')
        if not s.caption or not s.caption.strip():
            raise ValueError(f'Missing caption: {s.sample_id}')
    if expected is not None and len(samples) != expected:
        raise ValueError(f'{split}: expected {expected} frames, got {len(samples)}')
    return samples, info


class ThermalLidar(Dataset):
    def __init__(self, samples, token_cache=None):
        self.samples, self.tokens = samples, token_cache
        os.environ['IRIS_THERMAL_STRETCH'] = 'percentile'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        with Image.open(s.thermal_path) as im:
            raw = np.asarray(im)
        if raw.ndim != 2 or raw.dtype != np.uint16:
            raise ValueError(f'Expected uint16 thermal: {s.sample_id}')
        gray = AnyThermalEncoder._array_to_uint8(raw)
        if gray.std() < 1e-3:
            raise ValueError(f'Constant converted thermal: {s.sample_id}')
        x = np.repeat(gray[None], 3, axis=0).astype(np.float32) / 255.
        x = (x - np.array([.485, .456, .406], np.float32)[:, None, None]) / np.array([.229, .224, .225], np.float32)[:, None, None]
        _, gt = load_ms2_gt(s.thermal_gt_path, 256.)
        if gt.shape != gray.shape:
            raise ValueError('Native image/GT dimensions differ')
        item = dict(image=torch.from_numpy(x), gt=torch.from_numpy(gt), id=s.sample_id,
                    sequence=s.sample_id.rsplit('_', 1)[0],
                    thermal_stats=torch.tensor([gray.min(), gray.max(), gray.std()], dtype=torch.float32))
        if self.tokens is not None:
            item['token_ids'] = self.tokens['input_ids'][index]
            item['attention_mask'] = self.tokens['attention_mask'][index]
        return item


class ImageStream(Sampler):
    """Cursor counts CONSUMED images, never prefetched images; no epoch tails dropped."""
    def __init__(self, size, seed, start, count):
        self.size, self.seed, self.start, self.count = size, seed, start, count
        if size < 1 or min(start, count) < 0:
            raise ValueError('invalid image stream')

    def __len__(self):
        return self.count

    def __iter__(self):
        cursor, remaining = self.start, self.count
        while remaining:
            epoch, offset = divmod(cursor, self.size)
            order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + epoch)).tolist()
            take = min(remaining, self.size - offset)
            yield from order[offset:offset + take]
            cursor += take
            remaining -= take


def caption_hash(samples):
    return hashlib.sha256(json.dumps([(s.sample_id, s.caption) for s in samples], ensure_ascii=False).encode()).hexdigest()
