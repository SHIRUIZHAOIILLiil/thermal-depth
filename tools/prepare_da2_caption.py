"""Download ONLY the official metric ViT-B weights and SD2 tokenizer/text encoder.

No training. Creates an immutable asset descriptor with file hashes and upstream commit.
An existing descriptor is verified, not silently updated to newer upstream weights.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ms2_eval.io import sha256_file


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets-dir', type=Path, required=True)
    p.add_argument('--text-repo', default='Manojb/stable-diffusion-2-1-base')
    args = p.parse_args()
    root = args.assets_dir.resolve()
    descriptor = root / 'assets.json'
    if descriptor.exists():
        from tools.train_da2_caption import verify_assets
        verify_assets(descriptor)
        print(f'[verified-existing] {descriptor}')
        return
    root.mkdir(parents=True, exist_ok=True)
    repo = root / 'Depth-Anything-V2'
    if not repo.exists():
        subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/DepthAnything/Depth-Anything-V2.git', str(repo)], check=True)
    remote = subprocess.check_output(['git', '-C', str(repo), 'remote', 'get-url', 'origin'], text=True).strip()
    if remote.rstrip('/') != 'https://github.com/DepthAnything/Depth-Anything-V2.git':
        raise ValueError('Unexpected DA V2 remote; use a dedicated clean assets directory')
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise ValueError('DA V2 asset checkout must be clean')
    from huggingface_hub import hf_hub_download, snapshot_download
    weights = hf_hub_download('depth-anything/Depth-Anything-V2-Metric-VKITTI-Base',
                              'depth_anything_v2_metric_vkitti_vitb.pth')
    text = snapshot_download(args.text_repo, allow_patterns=['tokenizer/*', 'text_encoder/*'])
    paths = [Path(weights), *sorted((Path(text) / 'tokenizer').glob('*')), *sorted((Path(text) / 'text_encoder').glob('*'))]
    hashes = {str(f): sha256_file(f) for f in paths if f.is_file()}
    if len(hashes) < 4:
        raise ValueError('Text assets incomplete')
    fingerprint = hashlib.sha256(json.dumps({k: v for k, v in hashes.items() if k != str(weights)}, sort_keys=True).encode()).hexdigest()
    result = dict(root=str(root), da2_repo=str(repo), weights=str(weights), text_path=str(text),
                  da2_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                  hashes=hashes, text_fingerprint=fingerprint, text_repo=args.text_repo,
                  da2_source='https://github.com/DepthAnything/Depth-Anything-V2',
                  license_note='DA V2 Base checkpoint CC-BY-NC-4.0; SD2 checkpoint license applies separately')
    descriptor.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f'[prepared] {descriptor}; no training submitted')


if __name__ == '__main__':
    main()
