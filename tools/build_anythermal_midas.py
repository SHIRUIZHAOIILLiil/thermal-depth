"""Rebuild AnyThermal's MS2 depth model (calibration experiment A).

The released checkpoint (``pretrained_checkpoints/depth/Midas_anythermal``)
is a MiDaS/DPT depth net whose model class lives in the private castacks
BridgeMultiSpectralDepth fork.  Its state dict fingerprints as:

* backbone  = facebookresearch DINOv2 ViT-B/14 (pos_embed 1370x768, layer
  scale, no register tokens) under ``depth_net.pretrained.model``;
* head      = the public BMSD DPT scaffolding (param-free readout at indices
  0-2, 1x1 conv at .3, resample conv at .4; RefineNet-256 scratch).

So we rebuild it from the public BMSD midas package + torch.hub DINOv2 and
validate with a strict state-dict load.  Run this file directly for the
strict-load self-test:

    python tools/build_anythermal_midas.py --checkpoint <ckpt> [--device cpu]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
import os
from pathlib import Path

import torch

# 集群上这两样都不在本地路径下，用环境变量覆盖；不设时保持原有本地默认值。
BMSD_CANDIDATES = tuple(
    Path(p) for p in (
        os.environ.get("IRIS_BMSD_ROOT"),
        "/mnt/e/project/AnyThermal/baselines/depth/BridgeMultiSpectralDepth",
        "E:/project/AnyThermal/baselines/depth/BridgeMultiSpectralDepth",
    ) if p
)
DEFAULT_CHECKPOINT_CANDIDATES = tuple(
    Path(p) for p in (
        os.environ.get("IRIS_ANYTHERMAL_CKPT"),
        "/mnt/e/project/AnyThermal/_download/pretrained_checkpoints/depth/Midas_anythermal/ckpt_epoch=28_step=145000.ckpt",
        "E:/project/AnyThermal/_download/pretrained_checkpoints/depth/Midas_anythermal/ckpt_epoch=28_step=145000.ckpt",
    ) if p
)


def find_first(paths) -> Path:
    for path in paths:
        if Path(path).exists():
            return Path(path)
    raise FileNotFoundError(f"None of the candidate paths exist: {list(map(str, paths))}")


def install_mmcv_stubs() -> None:
    """The lightning checkpoint pickles mmcv Config objects; stub them out."""
    if "mmcv" in sys.modules:
        return

    class _Stub:
        def __new__(cls, *args, **kwargs):
            return super().__new__(cls)

        def __setstate__(self, state):
            self._state = state

    class _StubDict(dict):
        def __new__(cls, *args, **kwargs):
            return super().__new__(cls)

        def __setstate__(self, state):
            pass

    config = types.ModuleType("mmcv.utils.config")
    config.Config = _Stub
    config.ConfigDict = _StubDict
    utils = types.ModuleType("mmcv.utils")
    utils.config = config
    mmcv = types.ModuleType("mmcv")
    mmcv.utils = utils
    mmcv.Config = _Stub
    mmcv.ConfigDict = _StubDict
    sys.modules["mmcv"] = mmcv
    sys.modules["mmcv.utils"] = utils
    sys.modules["mmcv.utils.config"] = config


def load_bmsd_midas_modules(bmsd_root: Path):
    """Load the public BMSD midas package directly from files (its top-level
    ``models/__init__`` imports mmcv-heavy trainers we do not want)."""
    package_dir = bmsd_root / "models" / "network" / "midas"
    spec_names = ("base_model", "blocks", "vit", "dpt_depth")
    package = types.ModuleType("bmsd_midas")
    package.__path__ = [str(package_dir)]
    sys.modules["bmsd_midas"] = package
    modules = {}
    for name in spec_names:
        spec = importlib.util.spec_from_file_location(f"bmsd_midas.{name}", package_dir / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"bmsd_midas.{name}"] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def build_anythermal_midas(checkpoint_path: Path | None = None, *, readout: str = "ignore",
                           device: str = "cpu", bmsd_root: Path | None = None):
    """Rebuild the DPT depth net and strict-load the released weights.

    Returns ``(model, load_info)`` with the model in eval mode on ``device``.
    """
    checkpoint_path = find_first([checkpoint_path] if checkpoint_path else DEFAULT_CHECKPOINT_CANDIDATES)
    bmsd_root = bmsd_root or find_first(BMSD_CANDIDATES)
    modules = load_bmsd_midas_modules(bmsd_root)
    vit = modules["vit"]
    blocks = modules["blocks"]
    dpt = modules["dpt_depth"]

    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

    # Assemble the DPT scaffolding around the DINOv2 trunk, mirroring
    # _make_pretrained_vitb16_384 but with patch size 14.
    pretrained = vit._make_vit_b16_backbone(
        backbone,
        features=[96, 192, 384, 768],
        hooks=[2, 5, 8, 11],
        use_readout=readout,
    )
    # _make_vit_b16_backbone overwrote dinov2's integer patch_size with [16,16];
    # restore the native value its interpolate_pos_encoding depends on.
    pretrained.model.patch_size = 14

    # BMSD's forward_flex targets timm ViTs (pos_drop, _resize_pos_embed);
    # DINOv2 ships its own arbitrary-size token pipeline -- use it instead.
    # The DPT taps are forward hooks on blocks[2,5,8,11], so they fire the same.
    def dinov2_forward_flex(self, x):
        tokens = self.prepare_tokens_with_masks(x)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)

    pretrained.model.forward_flex = types.MethodType(dinov2_forward_flex, pretrained.model)

    class ResizingFusionBlock(blocks.FeatureFusionBlock):
        """Original FeatureFusionBlock (no out_conv) that tolerates odd grids.

        With patch 14 the MS2 grid is 18x45; the x2 upsample yields 46 wide
        against a 45-wide skip.  Interpolating the coarse path onto the skip
        resolution before adding is the standard MiDaS remedy and is exact on
        even grids.
        """

        def forward(self, *xs):
            import torch.nn.functional as F

            output = xs[0]
            if len(xs) == 2:
                skip = self.resConfUnit1(xs[1])
                if skip.shape[-2:] != output.shape[-2:]:
                    output = F.interpolate(output, size=skip.shape[-2:],
                                           mode="bilinear", align_corners=True)
                output = output + skip
            output = self.resConfUnit2(output)
            return F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)

    scratch = blocks._make_scratch([96, 192, 384, 768], 256, groups=1, expand=False)
    # the fork uses the original FeatureFusionBlock (no out_conv), not _custom
    scratch.refinenet1 = ResizingFusionBlock(256)
    scratch.refinenet2 = ResizingFusionBlock(256)
    scratch.refinenet3 = ResizingFusionBlock(256)
    scratch.refinenet4 = ResizingFusionBlock(256)
    scratch.output_conv = torch.nn.Sequential(
        torch.nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
        modules["blocks"].Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        torch.nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(True),
        torch.nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        torch.nn.ReLU(True),
        torch.nn.Identity(),
    )

    class AnyThermalMidasDPT(torch.nn.Module):
        patch_size = 14

        def __init__(self, pretrained, scratch):
            super().__init__()
            self.pretrained = pretrained
            self.scratch = scratch

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            height, width = x.shape[-2:]
            if height % self.patch_size or width % self.patch_size:
                raise ValueError(
                    f"Input {height}x{width} must be divisible by patch size "
                    f"{self.patch_size}; resize first (e.g. 256x640 -> 252x630). "
                    "dinov2's PatchEmbed asserts the same, so the private fork "
                    "must also have resized to a multiple of 14."
                )
            # inline port of vit.forward_vit with an integer dinov2 patch size
            self.pretrained.model.forward_flex(x)
            grid = (height // self.patch_size, width // self.patch_size)
            taps = []
            for index in (1, 2, 3, 4):
                post = getattr(self.pretrained, f"act_postprocess{index}")
                value = post[0:2](self.pretrained.activations[str(index)])
                if value.ndim == 3:
                    value = value.unflatten(2, grid)
                taps.append(post[3:len(post)](value))
            layer_1, layer_2, layer_3, layer_4 = taps
            layer_1_rn = self.scratch.layer1_rn(layer_1)
            layer_2_rn = self.scratch.layer2_rn(layer_2)
            layer_3_rn = self.scratch.layer3_rn(layer_3)
            layer_4_rn = self.scratch.layer4_rn(layer_4)
            path_4 = self.scratch.refinenet4(layer_4_rn)
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
            return self.scratch.output_conv(path_1).squeeze(1)

    model = AnyThermalMidasDPT(pretrained, scratch)

    install_mmcv_stubs()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {key[len("depth_net."):]: value for key, value in payload["state_dict"].items()
             if key.startswith("depth_net.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    load_info = {
        "checkpoint": str(checkpoint_path),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "tensors": len(state),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    if missing or unexpected:
        raise RuntimeError(
            "Strict reconstruction failed.\n"
            f"missing ({len(missing)}): {missing[:10]}\n"
            f"unexpected ({len(unexpected)}): {unexpected[:10]}"
        )
    model.to(device).eval().requires_grad_(False)
    return model, load_info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--readout", default="ignore", choices=("ignore", "add"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, info = build_anythermal_midas(args.checkpoint, readout=args.readout, device=args.device)
    print(f"strict load OK: {info['tensors']} tensors, epoch={info['epoch']}, step={info['global_step']}")

    with torch.no_grad():
        probe = torch.randn(1, 3, 252, 630, device=args.device)
        output = model(probe)
    print(f"forward OK: input (1,3,252,630) -> output {tuple(output.shape)}, "
          f"range [{float(output.min()):.3f}, {float(output.max()):.3f}], finite={bool(torch.isfinite(output).all())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
