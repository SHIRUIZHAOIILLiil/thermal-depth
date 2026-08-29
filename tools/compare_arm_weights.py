"""How far apart did the caption push a pair of arms?

Two arms of one backbone start from the same checkpoint, see the same frames in
the same order under the same seed, and differ in exactly one input: the text
embedding. Whatever separates their weights after training is therefore
attributable to the caption and to nothing else.

Reports the relative L2 distance per pair, plus the layers that moved most, so
"the caption changed this backbone less than that one" becomes a number rather
than an impression. CPU only, no GPU and no data needed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def load_state(path: Path) -> dict:
    """Unwrap whatever convert_iris_ms2_checkpoint.py wrapped the tensors in.

    That converter nests them two deep, as payload["state_dicts"]["unet"], and
    the singular "state_dict" other tools use is not present -- looking only for
    the singular name finds no tensors at all and the comparison reports an empty
    intersection rather than failing.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and isinstance(payload.get("state_dicts"), dict):
        inner = payload["state_dicts"]
        payload = inner.get("unet") or next(
            (v for v in inner.values() if isinstance(v, dict)), inner
        )
    else:
        for key in ("state_dict", "unet", "model"):
            if isinstance(payload, dict) and isinstance(payload.get(key), dict):
                payload = payload[key]
                break
    tensors = {k: v for k, v in payload.items()
               if isinstance(v, torch.Tensor) and v.is_floating_point()}
    if not tensors:
        raise SystemExit(
            f"{path.name}: no float tensors found. Top-level keys: "
            f"{sorted(k for k in payload)[:8]}"
        )
    return tensors


def compare(a: Path, b: Path, top: int) -> None:
    sa, sb = load_state(a), load_state(b)
    shared = sorted(set(sa) & set(sb))
    if not shared:
        raise SystemExit(f"no shared float tensors between {a.name} and {b.name}")
    missing = (set(sa) ^ set(sb))
    if missing:
        print(f"  ! {len(missing)} tensors present in only one of the two")

    num = den = 0.0
    per_layer: list[tuple[float, str]] = []
    for k in shared:
        x, y = sa[k].float(), sb[k].float()
        if x.shape != y.shape:
            print(f"  ! shape differs at {k}: {tuple(x.shape)} vs {tuple(y.shape)}")
            continue
        d2 = torch.sum((x - y) ** 2).item()
        n2 = torch.sum(y ** 2).item()
        num += d2
        den += n2
        if n2 > 0:
            per_layer.append(((d2 / n2) ** 0.5, k))

    print(f"  tensors compared : {len(per_layer)}")
    print(f"  relative L2      : {(num / den) ** 0.5:.6e}   (||A-B|| / ||B||, whole U-Net)")
    per_layer.sort(reverse=True)
    print(f"  top {top} layers by relative change:")
    for r, k in per_layer[:top]:
        print(f"    {r:.4e}  {k}")
    # Cross-attention is where the text enters; report it separately.
    attn2 = [r for r, k in per_layer if "attn2" in k]
    if attn2:
        print(f"  cross-attention (attn2, {len(attn2)} tensors): "
              f"mean {sum(attn2)/len(attn2):.4e}  max {max(attn2):.4e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"), required=True,
                    help="Two checkpoints to compare. Repeat for several pairs.")
    ap.add_argument("--label", action="append", default=None,
                    help="Name for each --pair, in the same order.")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    labels = args.label or [f"pair {i+1}" for i in range(len(args.pair))]
    for label, (a, b) in zip(labels, args.pair):
        print(f"=== {label} ===")
        print(f"  A = {a}")
        print(f"  B = {b}")
        compare(Path(a), Path(b), args.top)
        print()


if __name__ == "__main__":
    main()
