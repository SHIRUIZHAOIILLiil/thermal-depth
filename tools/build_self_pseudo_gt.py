"""Second-round targets: tmax1 16k's own prediction fills the frame where LiDAR is absent.

Each training frame's prediction is aligned to that frame's LiDAR with SSI-log, the same
per-frame calibration role the affine fit plays for AnyThermal, and stored in metres where
the dataset reads calibrated pseudo depth. The dataset still writes real LiDAR over it.

Only worth running if the holdout arm (docs/TMAX1_HOLDOUT_ARM_V1.md) returns
`learns_beyond_target`: the model's prediction where it never saw LiDAR must beat the
AnyThermal target there, or this round only teaches the model its own errors.

  shard:    predict rows i, i+n, i+2n, ... (resumable; existing files are skipped)
  finalize: check every training frame has a finite map, then write build_config.json
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ms2_eval.io import sha256_file
from ms2_eval.official_protocol import official_valid_mask
from tools.diagnose_error_sources import aligned_depth

SOURCE_RUN = 'sd2_cap_pct_logtn_img_l0100_tmax1_s43'
STEP = 16000
TRAIN_FRAMES = 75688
TRAIN_MANIFEST = 'ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl'
MIN_FIT_PIXELS = 100


def calibrated(raw, gt):
    """Metres: the prediction under the frame's SSI-log fit on all of its LiDAR."""
    valid = official_valid_mask(gt, .001, 80.)
    if valid.sum() < MIN_FIT_PIXELS:
        raise ValueError(f'Only {int(valid.sum())} LiDAR pixels to fit on')
    depth = aligned_depth(raw, gt, valid, 'ssi_log').astype(np.float32)
    if depth.shape != gt.shape or not np.isfinite(depth).all():
        raise ValueError('Invalid calibrated prediction')
    return depth


def shard(rows, index, count):
    if not 0 <= index < count:
        raise ValueError(f'Shard {index} of {count}')
    return rows[index::count]


def load_rows(a, contract):
    from tools.da2_caption_data import checked_manifest
    if Path(contract['train']['path']).name != TRAIN_MANIFEST:
        raise ValueError('Recipe training manifest differs from the one the arms train on')
    samples, info = checked_manifest(Path(contract['train']['path']), a.data_root, 'train', TRAIN_FRAMES)
    if info != contract['train']:
        raise ValueError('Training manifest differs from the frozen recipe')
    return samples


def weights_path(a):
    return a.runs/'iris_ms2'/SOURCE_RUN/'converted'/f'step{STEP}_weights.pt'


def run_shard(a):
    import torch
    from tools import diagnose_far_gap_curve as curve
    from tools import eval_iris_decoder_only as prior
    from tools.train_route_suite import RouteModel

    if not torch.cuda.is_available():
        raise RuntimeError('Requires CUDA')
    c = prior.parent.parent_inputs(a.runs)['contract']
    rows = shard(load_rows(a, c), a.index, a.count)
    depth_dir = a.out/'calibrated_pseudo_depth'
    depth_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in rows if not (depth_dir/f'{s.sample_id}.npy').is_file()]
    print(f'[shard] {a.index}/{a.count}: {len(rows)} frames, {len(todo)} to predict', flush=True)
    recipe = SimpleNamespace(**c['recipe'])
    model = RouteModel(recipe, torch.device('cuda'), torch.float16)
    path = weights_path(a)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    curve.check_payload(payload, STEP)
    model.load_state_dicts(payload['state_dicts'], str(path))
    model.unet.requires_grad_(False)
    model.set_train(False)
    del payload
    for i, (s, (ident, raw, gt)) in enumerate(zip(todo, prior.pilot.iris.predict_samples(model, todo, 'frame', recipe), strict=True), 1):
        if ident != s.sample_id:
            raise ValueError('Prediction order differs')
        target = depth_dir/f'{ident}.npy'
        partial = target.with_suffix('.tmp.npy')
        np.save(partial, calibrated(raw, gt))
        os.replace(partial, target)
        if i % 1000 == 0:
            print(f'[shard] {a.index}: {i}/{len(todo)}', flush=True)
    print(f'[shard] {a.index} done', flush=True)


def run_finalize(a):
    from tools import eval_iris_decoder_only as prior
    c = prior.parent.parent_inputs(a.runs)['contract']
    rows = load_rows(a, c)
    depth_dir = a.out/'calibrated_pseudo_depth'
    digest, bad = hashlib.sha256(), []
    for s in rows:
        path = depth_dir/f'{s.sample_id}.npy'
        if not path.is_file():
            bad.append(f'missing {s.sample_id}')
            continue
        depth = np.load(path, allow_pickle=False)
        if depth.dtype != np.float32 or depth.ndim != 2 or not np.isfinite(depth).all():
            bad.append(f'invalid {s.sample_id}')
        digest.update(s.sample_id.encode())
    extra = {p.stem for p in depth_dir.glob('*.npy')} - {s.sample_id for s in rows}
    if bad or extra:
        raise ValueError(f'{len(bad)} bad and {len(extra)} unexpected maps, e.g. {(bad + sorted(extra))[:5]}')
    weights = weights_path(a)
    (a.out/'build_config.json').write_text(json.dumps(dict(
        manifest=c['train']['path'], frames=len(rows), ids_sha256=digest.hexdigest(),
        source=str(weights), source_sha256=sha256_file(weights), alignment='ssi_log on all LiDAR of the frame',
        depth_scale=256., gt_min_depth=.001, gt_max_depth=80., min_fit_pixels=MIN_FIT_PIXELS, dtype='float32',
        stored='aligned tmax1 16k prediction in metres; the dataset writes LiDAR over it',
        script_sha256=sha256_file(__file__)), indent=2))
    print(f'[finalize] {len(rows)} maps checked; wrote {a.out/"build_config.json"}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('shard', 'finalize'))
    for name in ('runs', 'out', 'data-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--count', type=int, default=1)
    os.environ.update(IRIS_THERMAL_STRETCH='percentile', IRIS_DECODE_CLAMP='0', IRIS_TEXT_PADDING='legacy')
    a = p.parse_args()
    run_shard(a) if a.mode == 'shard' else run_finalize(a)
