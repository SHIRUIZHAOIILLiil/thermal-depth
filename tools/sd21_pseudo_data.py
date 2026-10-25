"""Read-only existing calibrated pseudo GT, exclusively for training frames.

Audit every required file once, freeze SHA256 inventory, verify bytes whenever
training consumes a file. No generation, clipping, interpolation or relabelling.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import numpy as np
import torch
from ms2_eval.io import load_ms2_gt, sha256_file
from tools.da2_caption_data import TRAIN
from tools.train_sd21_tadp_pair import ThermalDirect


def checked_ids(samples):
    ids = [s.sample_id for s in samples]
    if len(ids) != len(set(ids)) or not ids: raise ValueError('Empty/duplicate training IDs')
    if any(s.rsplit('_',1)[0] not in TRAIN or not s.rsplit('_',1)[1].isdigit() for s in ids):
        raise ValueError('Pseudo GT may only be read for official TRAIN IDs')
    return ids


def load_pseudo(path, digest=None, shape=(256,640)):
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if digest is not None and actual != digest: raise ValueError(f'Pseudo GT changed: {path}')
    array = np.load(io.BytesIO(raw), allow_pickle=False)
    if array.shape != shape or array.dtype.kind != 'f':
        raise ValueError(f'Pseudo GT must be floating-point metres of shape {shape}: {path}')
    # Reject corruption/format errors above. Invalid VALUES are legitimately masked.
    return array.astype(np.float32), actual


def prepare_inventory(samples, manifest_sha, pseudo_dir, destination):
    ids = checked_ids(samples)
    root = Path(pseudo_dir).resolve()
    config = root.parent/'build_config.json'
    if not config.is_file():
        raise FileNotFoundError(f'Need existing pseudo-GT provenance (do not regenerate): {config}')
    source = json.loads(config.read_text(encoding='utf-8'))
    if source.get('depth_scale') != 256. or source.get('stored') != 'calibrated depth in metres, no range handling applied':
        raise ValueError('Pseudo source metadata does not certify existing calibrated metric depth')
    identity = dict(schema=1, root=str(root), manifest_sha256=manifest_sha,
                    source_config_sha256=sha256_file(config), samples=len(ids))
    if destination.exists():
        index = json.loads(destination.read_text(encoding='utf-8'))
        if index['identity'] != identity or list(index['files']) != ids:
            raise ValueError('Frozen pseudo inventory differs; never silently rebuild it')
        missing = [name for name in ids if not (root/f'{name}.npy').is_file()]
        if missing: raise FileNotFoundError(f'{len(missing)} pseudo files disappeared; first: {missing[:5]}')
        if index['stats']['usable_missing_pixels'] <= 0: raise ValueError('No usable pseudo supervision')
        return index
    missing = [name for name in ids if not (root/f'{name}.npy').is_file()]
    if missing: raise FileNotFoundError(f'Missing {len(missing)}/{len(ids)} pseudo files; first: {missing[:10]}')
    files = {}
    stats = dict(pixels=0, nonfinite=0, nonpositive_or_too_near=0, at_or_above_80=0,
                 usable_missing_pixels=0, frames_without_usable_pseudo=0)
    for i,s in enumerate(samples):
        pseudo, digest = load_pseudo(root/f'{s.sample_id}.npy')
        _, gt = load_ms2_gt(s.thermal_gt_path, 256.)
        if gt.shape != pseudo.shape: raise ValueError(f'Pseudo/LiDAR shape mismatch: {s.sample_id}')
        valid = np.isfinite(pseudo) & (pseudo>.001) & (pseudo<80)
        usable = valid & (~np.isfinite(gt) | (gt<=.001))
        n = int(usable.sum())
        stats['pixels'] += pseudo.size
        stats['nonfinite'] += int((~np.isfinite(pseudo)).sum())
        stats['nonpositive_or_too_near'] += int((np.isfinite(pseudo) & (pseudo<=.001)).sum())
        stats['at_or_above_80'] += int((np.isfinite(pseudo) & (pseudo>=80)).sum())
        stats['usable_missing_pixels'] += n
        stats['frames_without_usable_pseudo'] += int(n==0)
        files[s.sample_id] = digest
        if (i+1)%2000 == 0 or i+1 == len(ids):
            print(f'[pseudo-audit] {i+1}/{len(ids)} files read, shaped and hashed', flush=True)
    if stats['usable_missing_pixels'] == 0:
        raise ValueError('Complete file set has no usable missing-pixel pseudo supervision')
    index = dict(identity=identity, source_config=source, stats=stats, files=files)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(destination)+'.tmp')
    tmp.write_text(json.dumps(index, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, destination)
    print('[pseudo-audit] PASS '+json.dumps(stats), flush=True)
    return index


class ThermalMixed(ThermalDirect):
    def __init__(self, samples, tokens, inventory):
        ids = checked_ids(samples)
        if list(inventory['files']) != ids: raise ValueError('Inventory/data IDs differ')
        super().__init__(samples, tokens)
        self.inventory = inventory

    def __getitem__(self, index):
        item = super().__getitem__(index)
        path = Path(self.inventory['identity']['root'])/f"{item['id']}.npy"
        pseudo,_ = load_pseudo(path, self.inventory['files'][item['id']])
        item['pseudo'] = torch.from_numpy(pseudo)
        return item
