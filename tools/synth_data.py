"""Fetch the two synthetic training sets Iris trains on, in the layout its loaders read.

Iris (`lotus/train_scripts/train_iris_g_depth.sh`) trains on Hypersim (90% of steps) and
Virtual KITTI 2 (10%). This script reproduces the inputs of `lotus/utils/process_hypersim.py`
and `lotus/utils/vkitti_dataset.py` without downloading the 1.9 TB of Hypersim zips: each scene
zip is read remotely with HTTP range requests and only the three files a frame needs are kept.

  probe             one scene + vKITTI headers: reachability, bytes per frame, archive layout
  hypersim          scenes index, index+count, ... of the public train split (resumable)
  hypersim-finalize every frame present or dropped for NaN depth; write metadata.jsonl
  vkitti-check      the extracted vKITTI tree has what the loader and depth2normal read

process_hypersim.py keeps a frame only if its depth file has no NaN; so does this script.
Unlike it, a depth file that cannot be read stops the job instead of silently dropping the frame.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
import zipfile

CSV_URLS = (
    'https://raw.githubusercontent.com/apple/ml-hypersim/main/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv',
    'https://raw.githubusercontent.com/apple/ml-hypersim/master/evermotion_dataset/analysis/metadata_images_split_scene_v1.csv',
)
SCENE_URL = 'https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/{}.zip'
VKITTI_URL = 'https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_{}'
VKITTI_PARTS = ('rgb.tar', 'depth.tar', 'textgt.tar.gz')
# vkitti_dataset.VKITTIDataset reads these; depth2normal also needs <scene>/<cond>/intrinsic.txt.
VKITTI_SCENES = ('02', '06', '18', '20')
VKITTI_CONDITIONS = ('15-deg-left', '15-deg-right', '30-deg-left', '30-deg-right', 'clone',
                     'fog', 'morning', 'overcast', 'rain', 'sunset')
VKITTI_CAMERAS = ('0', '1')
CSV_NAME = 'metadata_images_split_scene_v1.csv'
SPLIT = 'train'


def http_get(url, start=None, end=None, retries=6):
    """Bytes of url, or of [start, end] inclusive; retries with backoff."""
    headers = {'User-Agent': 'iris-synth-fetch/1'}
    if start is not None:
        headers['Range'] = f'bytes={start}-{end}'
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                data = r.read()
                if start is not None and (r.status != 206 or len(data) != end - start + 1):
                    raise IOError(f'Range {start}-{end} of {url}: status {r.status}, {len(data)} bytes')
                return data
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def http_size(url):
    req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'iris-synth-fetch/1'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers['Content-Length']), r.headers.get('Accept-Ranges', '')


class RangeFile(io.RawIOBase):
    """Seekable read-only file over fetch(start, end_inclusive) -> bytes."""

    def __init__(self, size, fetch):
        self.size, self.fetch, self.pos = size, fetch, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = (offset, self.pos + offset, self.size + offset)[whence]
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = self.fetch(self.pos, self.pos + n - 1)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def remote_zip(url):
    size, ranges = http_size(url)
    if ranges != 'bytes':
        raise IOError(f'{url} does not advertise byte ranges ({ranges!r})')
    raw = RangeFile(size, lambda s, e: http_get(url, s, e))
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=4 << 20))


def frame_files(row):
    """(member tail, path under <out>/train) for the three files process_hypersim copies."""
    fid, cam, scene = int(row['frame_id']), row['camera_name'], row['scene_name']
    tails = (f'images/scene_{cam}_final_preview/frame.{fid:04d}.tonemap.jpg',
             f'images/scene_{cam}_geometry_hdf5/frame.{fid:04d}.depth_meters.hdf5',
             f'images/scene_{cam}_geometry_hdf5/frame.{fid:04d}.normal_cam.hdf5')
    return [(t, f'{scene}/{t}') for t in tails]


def member_index(zf):
    """Member names keyed by the part from 'images/' on (the zip prefixes the scene name)."""
    out = {}
    for name in zf.namelist():
        i = name.find('images/')
        if i >= 0:
            out[name[i:]] = name
    return out


def has_nan(path):
    import h5py
    import numpy as np
    with h5py.File(path, 'r') as f:
        return any(np.isnan(f[k][:]).any() for k in f.keys())


def train_rows(csv_path):
    with open(csv_path, newline='') as f:
        rows = [r for r in csv.DictReader(f)
                if r['included_in_public_release'] == 'True' and r['split_partition_name'] == SPLIT]
    if not rows:
        raise ValueError(f'No public {SPLIT} rows in {csv_path}')
    return rows


def scenes_of(rows):
    seen = []
    for r in rows:
        if r['scene_name'] not in seen:
            seen.append(r['scene_name'])
    return seen


def fetch_csv(out):
    path = out/CSV_NAME
    if path.is_file():
        return path
    for url in CSV_URLS:
        try:
            data = http_get(url)
        except Exception as e:
            print(f'[csv] {url}: {e}', flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp')
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return path
    raise IOError('Hypersim split CSV not reachable')


def write_member(zf, name, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + '.part')
    with zf.open(name) as src, open(tmp, 'wb') as dst:
        while chunk := src.read(4 << 20):
            dst.write(chunk)
    os.replace(tmp, target)


def fetch_scene(scene, rows, root, open_zip=remote_zip):
    """Download a scene's frames into root (= <out>/train). Returns the done record."""
    done = root.parent/'_done'/f'{scene}.json'
    if done.is_file():
        return json.loads(done.read_text())
    zf = open_zip(SCENE_URL.format(scene))
    index = member_index(zf)
    kept, dropped, absent, nbytes = [], [], [], 0
    for r in rows:
        files = frame_files(r)
        if any(t not in index for t, _ in files):
            absent.append(files[0][1])
            continue
        targets = [root/p for _, p in files]
        if not all(t.is_file() for t in targets):
            write_member(zf, index[files[1][0]], targets[1])
            if has_nan(targets[1]):
                targets[1].unlink()
                dropped.append(files[0][1])
                continue
            for (t, _), target in zip(files, targets):
                if not target.is_file():
                    write_member(zf, index[t], target)
        kept.append(files[0][1])
        nbytes += sum(t.stat().st_size for t in targets)
    record = dict(scene=scene, rows=len(rows), kept=kept, dropped_nan=dropped, absent=absent, bytes=nbytes)
    done.parent.mkdir(parents=True, exist_ok=True)
    tmp = done.with_suffix('.tmp')
    tmp.write_text(json.dumps(record))
    os.replace(tmp, done)
    return record


def run_hypersim(a):
    rows = train_rows(fetch_csv(a.out))
    scenes = scenes_of(rows)[a.index::a.count]
    print(f'[hypersim] shard {a.index}/{a.count}: {len(scenes)} scenes', flush=True)
    for i, scene in enumerate(scenes, 1):
        rec = fetch_scene(scene, [r for r in rows if r['scene_name'] == scene], a.out/SPLIT)
        print(f'[hypersim] {i}/{len(scenes)} {scene}: kept {len(rec["kept"])} dropped_nan '
              f'{len(rec["dropped_nan"])} absent {len(rec["absent"])} {rec["bytes"]/2**20:.0f} MiB', flush=True)


def run_finalize(a):
    rows = train_rows(a.out/CSV_NAME)
    records, missing = {}, []
    for scene in scenes_of(rows):
        path = a.out/'_done'/f'{scene}.json'
        if path.is_file():
            records[scene] = json.loads(path.read_text())
        else:
            missing.append(scene)
    if missing:
        raise ValueError(f'{len(missing)} scenes not fetched, e.g. {missing[:5]}')
    kept = {p for r in records.values() for p in r['kept']}
    absent = sum(len(r['absent']) for r in records.values())
    if absent:
        raise ValueError(f'{absent} CSV frames missing from their scene zips')
    meta, bad = [], []
    for r in rows:
        files = frame_files(r)
        if files[0][1] not in kept:
            continue
        if not all((a.out/SPLIT/p).is_file() for _, p in files):
            bad.append(files[0][1])
        meta.append(dict(file_name=files[0][1], depth=files[1][1], normal_cam=files[2][1]))
    if bad:
        raise ValueError(f'{len(bad)} kept frames lack a file, e.g. {bad[:3]}')
    with open(a.out/SPLIT/'metadata.jsonl', 'w') as f:
        for m in meta:
            f.write(json.dumps(m) + '\n')
    dropped = sum(len(r['dropped_nan']) for r in records.values())
    summary = dict(rows=len(rows), kept=len(meta), dropped_nan=dropped, scenes=len(records),
                   gib=sum(r['bytes'] for r in records.values())/2**30)
    (a.out/'fetch_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'[finalize] {summary}', flush=True)


def vkitti_frames(root):
    """(rgb path, depth path) exactly as VKITTIDataset builds them."""
    out = []
    for scene in VKITTI_SCENES:
        for cond in VKITTI_CONDITIONS:
            for cam in VKITTI_CAMERAS:
                d = os.path.join(root, f'Scene{scene}/{cond}/frames/rgb/Camera_{cam}')
                for image in os.listdir(d):
                    p = os.path.join(d, image)
                    out.append((p, p.replace('rgb', 'depth').replace('.jpg', '.png')))
    return out


def run_vkitti_check(a):
    root = str(a.out)
    if 'rgb' in root or 'depth' in root:
        raise ValueError('The loader rewrites every "rgb"/"depth" in the path; pick a root without them')
    if 'normal' in root:
        raise ValueError('depth2normal rewrites "depth" to "normal" in the path; pick a root without them')
    frames = vkitti_frames(root)
    lost = [d for _, d in frames if not os.path.isfile(d)]
    intr = [f'Scene{s}/{c}' for s in VKITTI_SCENES for c in VKITTI_CONDITIONS
            if not os.path.isfile(os.path.join(root, f'Scene{s}/{c}/intrinsic.txt'))]
    # VKITTIDataset opens img_path.replace("rgb","normal").replace(".jpg",".png") for every frame.
    normals = [r for r, _ in frames if a.normals and not os.path.isfile(r.replace('rgb', 'normal').replace('.jpg', '.png'))]
    if not frames or lost or intr or normals:
        raise ValueError(f'{len(frames)} frames, {len(lost)} without depth, {len(intr)} without intrinsic.txt, '
                         f'{len(normals)} without normal')
    print(f'[vkitti] {len(frames)} rgb/depth pairs, intrinsics{" and normals" if a.normals else ""} '
          f'present under {root}', flush=True)


def tar_first_name(url):
    head = http_get(url, 0, 511)
    return head[:100].split(b'\0', 1)[0].decode()


def run_probe(a):
    rows = train_rows(fetch_csv(a.out))
    scenes = scenes_of(rows)
    print(f'[probe] hypersim public train: {len(rows)} frames in {len(scenes)} scenes', flush=True)
    scene = scenes[0]
    t0 = time.time()
    rec = fetch_scene(scene, [r for r in rows if r['scene_name'] == scene], a.out/SPLIT)
    n = len(rec['kept'])
    per = rec['bytes'] / max(n, 1)
    print(f'[probe] {scene}: kept {n}/{rec["rows"]}, dropped_nan {len(rec["dropped_nan"])}, '
          f'absent {len(rec["absent"])}, {time.time()-t0:.0f} s, {per/2**20:.2f} MiB/frame '
          f'-> about {per*len(rows)/2**30:.0f} GiB for the split', flush=True)
    for part in VKITTI_PARTS:
        url = VKITTI_URL.format(part)
        size, ranges = http_size(url)
        first = tar_first_name(url) if part.endswith('.tar') else '(gzip)'
        print(f'[probe] vkitti {part}: {size/2**30:.1f} GiB, ranges={ranges!r}, first entry {first!r}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=('probe', 'hypersim', 'hypersim-finalize', 'vkitti-check'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--count', type=int, default=1)
    p.add_argument('--normals', action='store_true', help='vkitti-check: also require the generated normals')
    a = p.parse_args()
    if not 0 <= a.index < a.count:
        sys.exit(f'Shard {a.index} of {a.count}')
    dict(probe=run_probe, hypersim=run_hypersim, **{'hypersim-finalize': run_finalize,
         'vkitti-check': run_vkitti_check})[a.mode](a)
