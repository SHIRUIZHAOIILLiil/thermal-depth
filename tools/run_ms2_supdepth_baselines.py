"""Run the published MS2 depth baselines ourselves, on the official frame set.

    python tools/run_ms2_supdepth_baselines.py \
        --bmsd-root  $SCRATCH/baselines/SupDepth4Thermal \
        --ckpt-dir   $SCRATCH/baselines/supdepth4thermal \
        --ms2-root   $IRIS_MS2_ROOT \
        --test-env   test_day \
        --output-dir $IRIS_RUNS/baseline_bench/day

Why this exists
---------------
`docs/PAPER_EXPERIMENTS_V0_2_20260820.md` currently asserts that no method in the
comparison predicts metric depth and that all of them are aligned by the
benchmark's per-image affine fit.  Reading the released code says otherwise:
DORN, BTS, AdaBins and NeWCRF are metric-supervised, and at test time they are
given a per-image **median scale** (`compute_depth_errors`'s `align=True`
default), while the MiDaS family gets a per-image **scale+shift fitted in depth
space**.  Ours gets a per-image scale+shift fitted in *disparity* space.  Three
different exams, none of them alignment-free.

Rather than argue about which published number is comparable to which, this runs
the baselines on our machine so every row can be scored under every alignment on
one shared frame set.

Reproducing the official frame set
----------------------------------
The benchmark's evaluation set is not shipped as a list, but the recipe is fully
determined by released code and contains no randomness:

  * sequence order = the line order of `test_<env>_list.txt`;
  * per sequence, candidates = `sorted(sync_data/<seq>/thr/img_left/*.png)` --
    every frame, taken from the image directory rather than from any manifest;
  * the whole concatenation, across sequences, is then strided:
    `sample_set[0:-1:sample_step]` with `sample_step = 10`
    (`configs/Base/Base_Sup_Mono_Depth.yaml`, `MS2_dataset.crawl_folders_depth`).

⚠️ The stride is applied to the **concatenated** list, not per sequence, so the
phase carries across sequence boundaries.  Striding each sequence separately
gives a different set from the second sequence onward.

⚠️ Build the candidate list from `img_left`, never from one of our manifests.
Ours drop frames (five of `2021-08-06-11-23-45`'s 5,810 have no caption), and a
shorter candidate list shifts every index after the first gap.

`--expect-frames` asserts the result: 2332 / 2292 / 2504 for day / night / rain,
which is what the paper reports as "2.3K, 2.3K, and 2.5K pairs".

Preprocessing is theirs, not ours
---------------------------------
`dataloaders/__init__.py:get_augmentations` builds the eval chain, and the
image-wise percentile clip is in it -- it is not a training-only augmentation:

    RescaleTo([256,640], bilinear) -> /2**14 -> clip to the 1st/99th percentile
    of that image and rescale to [0,1] -> (x - 0.45) / 0.225 -> repeat to 3 channels

Our own `thermal_to_lotus_input` does something else entirely.  Using it here
would understate every baseline, and the failure would read as "their models are
worse than published" rather than as our bug.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ms2_eval.official_protocol import evaluate_sample  # noqa: E402

# What the paper reports, to one significant rounding: "2.3K pairs for daytime,
# 2.3K for nighttime, and 2.5K for rainy conditions". These are a cross-check on
# the reproduced frame set, not the frame set itself -- the exact count comes out
# of the crawl, and 2.5K cannot distinguish 2503 from 2504. Checked against a
# tolerance for that reason: an exact assertion here would be asserting a number
# nobody published.
PAPER_FRAMES = {"test_day": 2300, "test_night": 2300, "test_rain": 2500}
FRAME_TOLERANCE = 0.02
SPLIT_LIST_FILE = {
    "test_day": "test_day_list.txt",
    "test_night": "test_night_list.txt",
    "test_rain": "test_rainy_list.txt",
}
ALIGN_MODES = ("none", "median", "ssi")
MODELS = ("dorn", "bts", "adabins", "newcrf")
CKPT_NAME = {"dorn": "DORN", "bts": "BTS", "adabins": "AdaBins", "newcrf": "NeWCRF"}


# --------------------------------------------------------------------------- #
# their network definitions, in our environment
# --------------------------------------------------------------------------- #


def install_mmcv_shim() -> None:
    """Enough `mmcv` for NeWCRF's head and for unpickling their checkpoints.

    Only `uper_crf_head.py` genuinely needs mmcv, and only for `ConvModule`
    (`swin_transformer.py`'s import is already commented out upstream). DORN, BTS
    and AdaBins import cleanly with no mmcv at all -- verified.

    The submodule names below are mmcv's own (`conv`, the norm under its type's
    short name, `activate`), because the checkpoint's keys were written against
    them. If they are wrong, `strict=True` says so immediately and names the
    keys; that is the intended failure mode, not a silent partial load.
    """
    if "mmcv" in sys.modules:
        return
    import torch.nn as nn

    class ConvModule(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=0, dilation=1, groups=1, bias="auto",
                     conv_cfg=None, norm_cfg=None, act_cfg=None, inplace=True,
                     **kwargs):
            super().__init__()
            if conv_cfg is not None and conv_cfg.get("type", "Conv2d") != "Conv2d":
                raise NotImplementedError(f"conv_cfg {conv_cfg} not supported by the shim")
            use_bias = (norm_cfg is None) if bias == "auto" else bias
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                                  padding=padding, dilation=dilation, groups=groups,
                                  bias=use_bias)
            self.norm_name = None
            if norm_cfg is not None:
                kind = norm_cfg["type"]
                if kind == "BN":
                    norm, self.norm_name = nn.BatchNorm2d(out_channels), "bn"
                elif kind == "SyncBN":
                    norm, self.norm_name = nn.BatchNorm2d(out_channels), "bn"
                elif kind == "GN":
                    norm = nn.GroupNorm(norm_cfg["num_groups"], out_channels)
                    self.norm_name = "gn"
                else:
                    raise NotImplementedError(f"norm_cfg {norm_cfg} not supported by the shim")
                self.add_module(self.norm_name, norm)
            self.activate = None
            if act_cfg is not None:
                kind = act_cfg.get("type", "ReLU")
                if kind != "ReLU":
                    raise NotImplementedError(f"act_cfg {act_cfg} not supported by the shim")
                self.activate = nn.ReLU(inplace=inplace)

        def forward(self, x):
            x = self.conv(x)
            if self.norm_name is not None:
                x = getattr(self, self.norm_name)(x)
            if self.activate is not None:
                x = self.activate(x)
            return x

    class _Stub:
        def __new__(cls, *args, **kwargs):
            return super().__new__(cls)

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.__dict__.update(state)

    class _StubDict(dict):
        def __new__(cls, *args, **kwargs):
            return super().__new__(cls)

        def __setstate__(self, state):
            if isinstance(state, dict):
                self.update(state)

    cnn = types.ModuleType("mmcv.cnn")
    cnn.ConvModule = ConvModule
    config = types.ModuleType("mmcv.utils.config")
    config.Config = _Stub
    config.ConfigDict = _StubDict
    utils = types.ModuleType("mmcv.utils")
    utils.config = config
    mmcv = types.ModuleType("mmcv")
    mmcv.cnn, mmcv.utils = cnn, utils
    mmcv.Config, mmcv.ConfigDict = _Stub, _StubDict
    sys.modules.update({
        "mmcv": mmcv, "mmcv.cnn": cnn,
        "mmcv.utils": utils, "mmcv.utils.config": config,
    })


def load_network_module(bmsd_root: Path, dotted: str):
    """Import one of their network modules without running the package __init__.

    `models/network/__init__.py` imports every architecture, and `models/__init__.py`
    pulls in the pytorch-lightning trainers. Neither is needed to build a network
    and both drag in dependencies this environment does not have.
    """
    if str(bmsd_root) not in sys.path:
        sys.path.insert(0, str(bmsd_root))
    for package in ("models", "models.network"):
        if package not in sys.modules:
            stub = types.ModuleType(package)
            stub.__path__ = [str(bmsd_root / package.replace(".", "/"))]
            sys.modules[package] = stub
    return importlib.import_module(dotted)


def build_network(name: str, bmsd_root: Path):
    """Construct the architecture with the released config's hyper-parameters.

    Values are transcribed from `configs/MonoSupDepth/*.yaml`; the trainer
    constructors they feed are in `models/trainers/mono_depth/*.py`.
    """
    from types import SimpleNamespace

    if name == "dorn":
        module = load_network_module(bmsd_root, "models.network.dorn.dorn")
        return module.DeepOrdinalRegression(
            ord_num=90, beta=80.0, num_layers=101, discretization="SID", num_channel=3
        )
    if name == "bts":
        module = load_network_module(bmsd_root, "models.network.bts.bts")
        return module.BtsModel(
            params=SimpleNamespace(bts_size=512, encoder="resnext101_bts", max_depth=80.0)
        )
    if name == "adabins":
        module = load_network_module(bmsd_root, "models.network.adabin.unet_adaptive_bins")
        return module.UnetAdaptiveBins.build(
            n_bins=256, min_val=1.0e-3, max_val=80.0, norm="linear"
        )
    if name == "newcrf":
        install_mmcv_shim()
        module = load_network_module(bmsd_root, "models.network.newcrf.NewCRFDepth")
        return module.NewCRFDepth(
            version="large07", inv_depth=False, pre_trained=False, ckpt_path=None,
            frozen_stages=-1, min_depth=1.0e-3, max_depth=80.0,
        )
    raise ValueError(f"Unknown model {name!r}")


def forward_depth(name: str, network, image: torch.Tensor) -> torch.Tensor:
    """Mirror each trainer's `inference_depth`, return metric depth (B,H,W).

    The four differ in what they return and it is not guessable: DORN hands back
    a dict, BTS a five-tuple whose last element is the depth, AdaBins a pair
    whose second element is, and NeWCRF the tensor itself.
    """
    if image.shape[1] == 1:                       # BaseModule.forward does this
        image = image.repeat_interleave(3, dim=1)
    if name == "dorn":
        depth = network(image)["target"]
    elif name == "bts":
        depth = network(image)[-1]
    elif name == "adabins":
        depth = network(image)[1]
    else:
        depth = network(image)
    return depth.squeeze(1) if depth.dim() == 4 else depth


def load_weights(network, checkpoint: Path, strict: bool = True) -> dict:
    install_mmcv_shim()                            # their ckpts pickle mmcv Configs
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    prefix = "depth_net."
    stripped = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not stripped:
        raise SystemExit(
            f"{checkpoint.name}: no keys under {prefix!r}; found "
            f"{sorted({k.split('.')[0] for k in state})[:6]}"
        )
    missing, unexpected = network.load_state_dict(stripped, strict=False)
    if strict and (missing or unexpected):
        raise SystemExit(
            f"{checkpoint.name}: state dict does not match the architecture.\n"
            f"  missing {len(missing)}: {list(missing)[:6]}\n"
            f"  unexpected {len(unexpected)}: {list(unexpected)[:6]}"
        )
    return {
        "checkpoint": str(checkpoint),
        "tensors_loaded": len(stripped),
        "epoch": payload.get("epoch"),
        "global_step": payload.get("global_step"),
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


# --------------------------------------------------------------------------- #
# their frame set and their preprocessing
# --------------------------------------------------------------------------- #


def official_frames(ms2_root: Path, test_env: str, sample_step: int) -> list[dict]:
    """Reproduce `crawl_folders_depth`'s sample list for one test environment."""
    list_path = ms2_root / SPLIT_LIST_FILE[test_env]
    if not list_path.is_file():
        raise SystemExit(f"Missing split list {list_path}")
    sequences = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    candidates: list[dict] = []
    for sequence in sequences:                     # list order, not sorted order
        image_dir = ms2_root / "sync_data" / sequence / "thr" / "img_left"
        frames = sorted(image_dir.glob("*.png"))
        if not frames:
            raise SystemExit(f"No thermal frames under {image_dir}")
        for path in frames:                        # every frame is a candidate
            candidates.append({
                "sequence": sequence.lstrip("_"),
                "stem": path.stem,
                "id": f"{sequence.lstrip('_')}_{path.stem}",
                "image_path": path,
                "depth_path": ms2_root / "proj_depth" / sequence / "thr" / "depth_filtered" / f"{path.stem}.png",
            })
        print(f"[frames] {sequence}: {len(frames)} candidates", flush=True)
    # The stride is applied once, to the concatenation, and drops the last element.
    selected = candidates[0:-1:sample_step]
    print(
        f"[frames] {len(candidates)} candidates -> {len(selected)} at stride {sample_step}",
        flush=True,
    )
    return selected


def preprocess_thermal(path: Path, size=(256, 640)) -> torch.Tensor:
    """Their eval transform, in order. See the module docstring for the source."""
    raw = np.asarray(Image.open(path))
    if raw.ndim == 2:
        raw = raw[:, :, None]
    tensor = torch.from_numpy(np.ascontiguousarray(raw.transpose(2, 0, 1))).float()
    if tuple(tensor.shape[-2:]) != tuple(size):
        # RescaleTo uses cv2.INTER_LINEAR; MS2 thermal is already 256x640 so this
        # is a no-op on the real data and exists only so a resized input cannot
        # pass through unnoticed.
        tensor = F.interpolate(tensor[None], size, mode="bilinear", align_corners=False)[0]
    tensor = tensor / (2 ** 14)                    # ArrayToTensor(Itype='thr')
    flat = torch.sort(tensor.reshape(-1)).values   # TensorIWMM: image-wise 1-99% clip
    hi = flat[round(len(flat) * 0.99) - 1]
    lo = flat[round(len(flat) * 0.01)]
    tensor = tensor.clamp(float(lo), float(hi))
    tensor = (tensor - lo) / (hi - lo)
    tensor = (tensor - 0.45) / 0.225               # Normalize()
    return tensor[None]                            # (1, 1, H, W)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bmsd-root", type=Path, required=True,
                        help="Clone of UkcheolShin/SupDepth4Thermal (network definitions).")
    parser.add_argument("--ckpt-dir", type=Path, required=True,
                        help="Directory holding MS2_MD_<model>_THR_ckpt.ckpt.")
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--test-env", choices=sorted(SPLIT_LIST_FILE), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--sample-step", type=int, default=10,
                        help="Official value is 10; 1 evaluates every frame.")
    parser.add_argument("--expect-frames", type=int, default=None,
                        help="Assert the frame count. Defaults to the paper's number "
                             "for this test env when --sample-step is 10.")
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--limit", type=int, default=None, help="Smoke: first N frames only.")
    parser.add_argument(
        "--build-only",
        action="store_true",
        help=(
            "Construct each network, load its checkpoint, and exit without evaluating. "
            "RUN THIS ON A LOGIN NODE FIRST. Three of the four architectures fetch an "
            "ImageNet initialisation on construction -- DORN and BTS from "
            "download.pytorch.org, AdaBins through torch.hub from GitHub -- and a GPU "
            "node with no outbound network would fail there, after queueing. The "
            "weights themselves are then overwritten by the checkpoint; only the cache "
            "matters, and $TORCH_HOME already points at scratch."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-partial-load", action="store_true",
                        help="Report key mismatches instead of aborting. Diagnostic only: "
                             "a partially loaded network still produces numbers.")
    return parser.parse_args()


@torch.no_grad()
def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.build_only:
        for name in [m.strip() for m in args.models.split(",") if m.strip()]:
            checkpoint = args.ckpt_dir / f"MS2_MD_{CKPT_NAME[name]}_THR_ckpt.ckpt"
            network = build_network(name, args.bmsd_root)
            provenance = load_weights(network, checkpoint, strict=not args.allow_partial_load)
            params = sum(p.numel() for p in network.parameters()) / 1e6
            print(f"[build] {name:<8} {params:6.1f}M params, "
                  f"{provenance['tensors_loaded']} tensors loaded strict, "
                  f"epoch {provenance['epoch']} step {provenance['global_step']}", flush=True)
        print("\n[build] all requested models construct and load. Caches are warm.")
        return 0

    frames = official_frames(args.ms2_root, args.test_env, args.sample_step)
    if args.limit is None:
        if args.expect_frames is not None:
            if len(frames) != args.expect_frames:
                raise SystemExit(
                    f"Reproduced {len(frames)} frames for {args.test_env}, "
                    f"--expect-frames said {args.expect_frames}."
                )
            print(f"[frames] matches --expect-frames: {len(frames)}", flush=True)
        elif args.sample_step == 10:
            paper = PAPER_FRAMES[args.test_env]
            drift = abs(len(frames) - paper) / paper
            if drift > FRAME_TOLERANCE:
                raise SystemExit(
                    f"Reproduced {len(frames)} frames for {args.test_env}; the paper "
                    f"reports about {paper}. That is {drift * 100:.1f}% off -- the frame "
                    "set is not the official one. Fix it before reading any metric."
                )
            print(
                f"[frames] {len(frames)} selected; the paper reports about {paper} "
                f"({drift * 100:.1f}% apart, within tolerance)",
                flush=True,
            )
    if args.limit:
        frames = frames[: args.limit]
        print(f"[frames] SMOKE: limited to {len(frames)}", flush=True)

    summary = {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        checkpoint = args.ckpt_dir / f"MS2_MD_{CKPT_NAME[name]}_THR_ckpt.ckpt"
        if not checkpoint.is_file():
            raise SystemExit(f"Missing checkpoint {checkpoint}")
        print(f"\n=== {name} ===", flush=True)
        network = build_network(name, args.bmsd_root)
        provenance = load_weights(network, checkpoint, strict=not args.allow_partial_load)
        print(f"[load] {provenance['tensors_loaded']} tensors, "
              f"epoch {provenance['epoch']}, step {provenance['global_step']}", flush=True)
        network = network.to(device).eval()

        accumulators = {mode: {} for mode in ALIGN_MODES}
        per_sample: list[dict] = []
        started = time.time()
        for index, frame in enumerate(frames):
            image = preprocess_thermal(frame["image_path"]).to(device)
            depth = forward_depth(name, network, image)
            gt = np.asarray(Image.open(frame["depth_path"]), dtype=np.float32) / args.depth_scale
            prediction = depth[None] if depth.dim() == 3 else depth
            if prediction.shape[-2:] != gt.shape:
                prediction = F.interpolate(prediction, gt.shape, mode="bilinear", align_corners=False)
            pred = prediction[0, 0].float().cpu().numpy()
            row = {"id": frame["id"], "sequence": frame["sequence"]}
            for mode in ALIGN_MODES:
                metrics = evaluate_sample(
                    pred, gt, align=mode, min_depth=args.min_depth, max_depth=args.max_depth
                )
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and key != "align_mode":
                        accumulators[mode][key] = accumulators[mode].get(key, 0.0) + float(value)
                        row[f"{mode}_{key}"] = value
            per_sample.append(row)
            if (index + 1) % 250 == 0:
                print(f"[{name}] {index + 1}/{len(frames)}", flush=True)

        count = len(per_sample)
        result = {
            "model": name,
            "test_env": args.test_env,
            "frames": count,
            "sample_step": args.sample_step,
            "frame_set": "official BMSD crawl, concatenated then strided",
            "preprocessing": "SupDepth4Thermal eval chain (percentile clip, /2**14, 0.45/0.225)",
            "provenance": provenance,
            "elapsed_seconds": time.time() - started,
            "by_alignment": {
                mode: {k: v / count for k, v in accumulators[mode].items()} for mode in ALIGN_MODES
            },
        }
        (args.output_dir / f"{name}_{args.test_env}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        with (args.output_dir / f"{name}_{args.test_env}_per_sample.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_sample[0]))
            writer.writeheader()
            writer.writerows(per_sample)
        summary[name] = result["by_alignment"]
        for mode in ALIGN_MODES:
            m = result["by_alignment"][mode]
            print(f"[{name}] {mode:<7} AbsRel {m['abs_rel']:.4f}  SqRel {m['sq_rel']:.4f}  "
                  f"RMSE {m['rmse']:.3f}  RMSElog {m['rmse_log']:.4f}  d1 {m['a1']:.4f}", flush=True)

    print(f"\n=== {args.test_env}, {len(frames)} frames ===")
    header = f"{'model':<10}" + "".join(f"{mode:>26}" for mode in ALIGN_MODES)
    print(header)
    for name, byalign in summary.items():
        cells = "".join(
            f"{byalign[mode]['abs_rel']:>10.4f}{byalign[mode]['rmse']:>8.3f}{byalign[mode]['a1']:>8.4f}"
            for mode in ALIGN_MODES
        )
        print(f"{name:<10}{cells}")
    print("(each cell: AbsRel / RMSE / delta1)")
    print("\nGATE: the 'median' column is the published protocol. It must reproduce")
    print("      MS2 Table IV(c) before any other column here is worth reading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
