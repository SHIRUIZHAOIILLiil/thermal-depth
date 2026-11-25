"""Login-node preflight (stdlib only): every file a run contract hash-pins must still match.

Decoder jobs refuse to start when a pinned file changed (eval_iris_decoder_only.selection); this
check runs before sbatch so the failure shows up at submission instead of after a day in the queue.
Usage: python3 tools/check_contract_code.py <contract.json> [...]
"""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def changed(contract, root=ROOT):
    code = json.loads(Path(contract).read_text())['code']
    return [name for name, digest in code.items()
            if not (Path(root)/name).is_file() or sha256(Path(root)/name) != digest]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    bad = [(c, n) for c in sys.argv[1:] for n in changed(c)]
    for contract, name in bad:
        print(f'!! {name} differs from the hash pinned by {contract}', file=sys.stderr)
    if bad:
        sys.exit(1)
    print(f'[preflight] pinned code unchanged for {len(sys.argv)-1} contract(s)')
