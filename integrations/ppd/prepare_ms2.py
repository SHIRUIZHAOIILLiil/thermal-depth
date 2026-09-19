"""Two things Pixel-Perfect Depth needs before it will train on MS2 thermal.

**A split file.** Their datasets read a JSON of parallel path lists keyed
"{split}_rgb_paths" and "{split}_dpt_paths"; ours adds "{split}_lidar_paths",
because the target is pseudo depth with the real returns written over it and
their base class reads one file per frame.

**A checkpoint their trainer will accept.** This is the part that would have
cost a day at runtime. The repository has two loaders and they disagree:

    load_pretrained_model       torch.load(path)["state_dict"]     training
    load_pretrained_model_eval  torch.load(path), dit.* -> pipeline.dit.*   inference

The released ppd.pth is the second form, and training never reaches it: the
trainer calls find_last_ckpt_path, which globs "*.ckpt" only, then indexes
["state_dict"] on whatever it finds. So fine-tuning from the published weights
fails three ways -- wrong extension, missing key, unprefixed names -- none of
which says what is wrong.

So the conversion applies exactly the remap their own eval loader applies, and
wraps it the way their train loader expects. Doing anything cleverer would mean
loading weights differently from the way the authors load them.

    python integrations/ppd/prepare_ms2.py \
        --manifest $SCRATCH/manifests/.../ms2_train_official8_....jsonl \
        --ms2-root $SCRATCH/data/ms2 \
        --pseudo-dir $SCRATCH/runs/pseudo_gt/official_train/calibrated_pseudo_depth \
        --split train --out $SCRATCH/runs/ppd_ms2/splits/train.json \
        --ppd-weights $SCRATCH/models/ppd/ppd.pth \
        --ckpt-dir $SCRATCH/runs/ppd_ms2/v1/checkpoints
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--ms2-root", type=Path)
    parser.add_argument("--pseudo-dir", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", type=Path, help="Where to write the split JSON.")
    parser.add_argument("--ppd-weights", type=Path,
                        help="Their released ppd.pth, to be converted.")
    parser.add_argument("--ckpt-dir", type=Path,
                        help="cfg.callbacks.model_checkpoint.dirpath -- the directory "
                             "find_last_ckpt_path scans. The converted file lands here.")
    return parser.parse_args()


def write_split(args: argparse.Namespace) -> None:
    root = args.ms2_root.resolve()
    rgb, dpt, lidar = [], [], []
    missing = 0
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            pseudo = args.pseudo_dir / f"{row['id']}.npy"
            if not pseudo.is_file():
                missing += 1
                continue
            # Relative to data_root, because that is what their build_metas
            # joins against.
            rgb.append(row["thermal_path"])
            lidar.append(row["thermal_depth_path"])
            dpt.append(str(pseudo.resolve().relative_to(root))
                       if args.pseudo_dir.resolve().is_relative_to(root)
                       else str(pseudo.resolve()))
    if not rgb:
        raise SystemExit("清单里没有一帧配得上伪深度")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        f"{args.split}_rgb_paths": rgb,
        f"{args.split}_dpt_paths": dpt,
        f"{args.split}_lidar_paths": lidar,
    }, indent=1), encoding="utf-8")
    print(f"[split] {len(rgb)} 帧 -> {args.out}")
    if missing:
        print(f"⚠️ {missing} 帧没有伪深度，已跳过")


def convert_weights(args: argparse.Namespace) -> None:
    import torch

    state = torch.load(args.ppd_weights, map_location="cpu")
    if "state_dict" in state:
        raise SystemExit(f"{args.ppd_weights} 已经是训练侧的格式，不用转换")
    # Their load_pretrained_model_eval, verbatim.
    fixed = {(f"pipeline.{k}" if k.startswith("dit.") else k): v
             for k, v in state.items()}
    moved = sum(1 for k in state if k.startswith("dit."))
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    # find_last_ckpt_path sorts the *.ckpt it finds and takes the last, skipping
    # anything named "last", so the name has to sort first and not say "last".
    target = args.ckpt_dir / "e000-s000000.ckpt"
    torch.save({"state_dict": fixed}, target)
    print(f"[ckpt] {len(fixed)} 个张量（{moved} 个加了 pipeline. 前缀）-> {target}")
    print("⚠️ pipeline.sem_encoder.* 本来就不在这份权重里（语义编码器单独加载），"
          "训练日志里它们出现在 missing 是正常的。")


def main() -> None:
    args = parse_args()
    did = False
    if args.manifest is not None:
        for required in ("ms2_root", "pseudo_dir", "out"):
            if getattr(args, required) is None:
                raise SystemExit(f"写 split 需要 --{required.replace('_', '-')}")
        write_split(args)
        did = True
    if args.ppd_weights is not None:
        if args.ckpt_dir is None:
            raise SystemExit("转换权重需要 --ckpt-dir")
        convert_weights(args)
        did = True
    if not did:
        raise SystemExit("没事可做：给 --manifest 或 --ppd-weights")


if __name__ == "__main__":
    main()
