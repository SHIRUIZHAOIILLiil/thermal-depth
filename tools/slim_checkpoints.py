"""Strip optimizer state from finished training checkpoints.

Every `*_end.pt` in `outputs/lotus_line_v2` is ~9.8 GB, but only ~3.4 GB of that
is the model: the rest is the AdamW moment pair, which is useless once training
has finished (resuming a completed run is not a thing).  Slimming keeps every
archived model fully loadable for inference and evaluation while giving back
roughly two thirds of the space.

Safety: the tool writes `<name>.slim.pt` first, reloads it, checks that every
weight tensor is bit-identical to the original, and only then replaces the
original (and only with `--apply`).  Without `--apply` it just reports.

    python tools/slim_checkpoints.py                       # dry run, shows the plan
    python tools/slim_checkpoints.py --apply               # slim everything it listed
    python tools/slim_checkpoints.py --apply --keep-original   # write .slim.pt, keep both
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

GB = 1024**3
# keys that exist only to resume an interrupted run
DROP_KEYS = ("optimizer_state_dict", "optimizer", "torch_rng_state", "python_rng_state", "permutation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="*",
        type=Path,
        default=[Path("outputs/lotus_line_v2"), Path("outputs/route_suite")],
        help="Directories to scan recursively.",
    )
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=["*_end.pt", "end_weights.pt", "best_weights.pt"],
        help="Which checkpoints count as finished artefacts.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually rewrite the files.")
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="With --apply: leave the fat original in place next to the .slim.pt.",
    )
    parser.add_argument("--min-gb", type=float, default=1.0, help="Skip checkpoints smaller than this.")
    return parser.parse_args()


def find_checkpoints(roots, patterns) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            found.extend(sorted(root.rglob(pattern)))
    # a checkpoint may match several patterns
    return sorted(set(found))


def tensors_equal(left, right) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.shape == right.shape and left.dtype == right.dtype and bool(torch.equal(left, right))
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(tensors_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(tensors_equal(a, b) for a, b in zip(left, right))
    return left == right


def slim_one(path: Path, args: argparse.Namespace) -> dict:
    original_bytes = path.stat().st_size
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        return {"path": str(path), "skipped": "not a dict payload"}

    dropped = [key for key in DROP_KEYS if key in payload]
    if not dropped:
        return {"path": str(path), "skipped": "nothing to drop", "gb": original_bytes / GB}

    slim = {key: value for key, value in payload.items() if key not in DROP_KEYS}
    slim["slimmed_from"] = path.name
    slim["dropped_keys"] = dropped

    target = path.with_suffix(".slim.pt")
    torch.save(slim, target)
    slim_bytes = target.stat().st_size

    # verify: every retained entry must survive the round trip untouched
    reloaded = torch.load(target, map_location="cpu", weights_only=False)
    for key, value in slim.items():
        if key in ("slimmed_from", "dropped_keys"):
            continue
        if not tensors_equal(value, reloaded[key]):
            target.unlink(missing_ok=True)
            raise RuntimeError(f"Verification failed for {path}: key {key!r} changed")
    del payload, slim, reloaded

    result = {
        "path": str(path),
        "dropped": dropped,
        "original_gb": original_bytes / GB,
        "slim_gb": slim_bytes / GB,
        "saved_gb": (original_bytes - slim_bytes) / GB,
    }
    if not args.keep_original:
        path.unlink()
        target.rename(path)
        result["replaced"] = True
    return result


def main() -> None:
    args = parse_args()
    checkpoints = [p for p in find_checkpoints(args.roots, args.patterns) if p.stat().st_size >= args.min_gb * GB]
    if not checkpoints:
        print("No checkpoints matched.")
        return

    total = sum(path.stat().st_size for path in checkpoints)
    print(f"{len(checkpoints)} finished checkpoints, {total / GB:.1f} GB total")
    if not args.apply:
        for path in checkpoints:
            print(f"  {path.stat().st_size / GB:6.2f} GB  {path}")
        print(
            "\nDry run. Model weights are ~1/3 of each file, so expect to recover roughly "
            f"{total * 2 / 3 / GB:.0f} GB. Re-run with --apply to do it."
        )
        return

    saved = 0.0
    for index, path in enumerate(checkpoints, start=1):
        print(f"[{index}/{len(checkpoints)}] {path}", flush=True)
        result = slim_one(path, args)
        if "skipped" in result:
            print(f"    skipped: {result['skipped']}")
            continue
        saved += result["saved_gb"]
        print(
            f"    {result['original_gb']:.2f} -> {result['slim_gb']:.2f} GB "
            f"(dropped {', '.join(result['dropped'])})",
            flush=True,
        )
    print(f"\nRecovered {saved:.1f} GB")


if __name__ == "__main__":
    main()
