"""Emit an untrained checkpoint so the zero-training baseline is measured our way.

The c-route adapter is zero-initialised and residual, so an untrained adapter is
an exact identity: the route then reduces to feeding the frozen VAE latent
straight into the frozen U-Net, i.e. the zero-training thermal baseline.

That baseline already has a number in the frozen document (0.1291), but it came
from a different runner. It anchors the whole "did the adapter help at all"
claim, so it should be re-measured through the same evaluation path as every
other checkpoint in this suite rather than compared across tools.

    python tools/make_untrained_checkpoint.py --route c1_vae_adapter --output outputs/route_suite/c1_vae_adapter_20ep/epoch00_untrained.pt
    python tools/train_route_suite.py --route c1_vae_adapter --eval-checkpoint outputs/route_suite/c1_vae_adapter_20ep/epoch00_untrained.pt --output-dir outputs/route_suite/c1_vae_adapter_20ep --val-stride 1 --eval-tag full_ep00
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = ROOT / "lotus"
for path in (ROOT, LOTUS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_route_suite import CHECKPOINT_FORMAT, ROUTES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", required=True, choices=sorted(ROUTES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-hidden-channels", type=int, default=256)
    parser.add_argument("--adapter-blocks", type=int, default=6)
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modality, condition, trains_adapter, trains_unet = ROUTES[args.route]
    if trains_unet:
        raise SystemExit(
            f"{args.route} trains the U-Net, whose untrained state is the pretrained Lotus "
            "weights; build that baseline by evaluating the pipeline directly instead."
        )

    state_dicts: dict[str, dict] = {}
    if condition == "vae_adapter":
        from models.thermal_vae_latent_adapter import ThermalVAELatentAdapter

        adapter = ThermalVAELatentAdapter(
            hidden_channels=args.adapter_hidden_channels, blocks=args.adapter_blocks
        )
    elif condition == "anythermal_adapter":
        from models.anythermal_lotus_adapter_v2_3 import AnyThermalLotusAdapterV23

        adapter = AnyThermalLotusAdapterV23()
    else:
        raise SystemExit(f"{args.route} has no adapter to leave untrained")

    state_dicts["adapter"] = {key: value.detach().cpu() for key, value in adapter.state_dict().items()}

    if condition == "vae_adapter":
        probe = torch.randn(1, 4, 32, 40)
        with torch.no_grad():
            out = adapter(probe)
        drift = float((out - probe).abs().max())
        if drift != 0.0:
            raise SystemExit(f"Adapter is not an exact identity at init (max drift {drift:.3e})")
        print("[check] untrained adapter is an exact identity (max drift 0.0)")

    payload = {
        "format": CHECKPOINT_FORMAT,
        "route": args.route,
        "epoch": 0,
        "manifest_sha256": None,
        "caption_mode": "empty",
        "val_metrics": None,
        "untrained": True,
        "state_dicts": state_dicts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"[done] untrained {args.route} checkpoint -> {args.output}")


if __name__ == "__main__":
    main()
