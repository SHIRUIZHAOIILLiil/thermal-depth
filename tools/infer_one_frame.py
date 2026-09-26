"""一帧，多个 prompt，其余全部相同。

问的是「换一句话，这一帧的预测会变成什么样」。为了让看到的差别真的只来自文本，
噪声种子、condition 的取法、checkpoint、分辨率全部锁死，逐个 prompt 复用同一份
bundle —— 只改 bundle.prompts 里的那一条。

⛔ 一帧的数字不是证据。这个项目在找的效应是 0.001–0.002 量级，而帧与帧之间的
   差异比它大两三个数量级；单帧的 AbsRel 变好变坏都在噪声里。这个工具是拿来
   **看图**的，不是拿来选 prompt 的。要选 prompt 得在 val 上跑几百帧，且判据
   事先写下来。

⚠️ 同理，⛔ 不要拿 test 帧反复试 prompt —— 那是拿 test 选方法。

    python tools/infer_one_frame.py \
        --ms2-root E:/dataset/ms2 \
        --manifest manifests/ms2_test_day3_common_thermalcap_20260821.jsonl \
        --frame 2021-08-06-11-23-45_000805 \
        --unet-checkpoint <...>/converted/step16000_weights.pt \
        --prompt "A thermal street scene showing ..." \
        --out-dir runs/one_frame
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "lotus", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_ms2_lotus_thermal_vae_official import (  # noqa: E402
    ThermalVAEPipelineBundle,
    generate_thermal_vae_prediction,
)
from pipeline import LotusGPipeline  # noqa: E402

D_MIN, D_MAX = 1e-3, 80.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--frame", required=True, help="清单里的 id")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--unet-checkpoint", type=Path, default=None)
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument(
        "--align", required=True,
        choices=("ssi_disparity", "ssi", "ssi_log"),
        help="必须和这个 checkpoint 训练时的 norm_type 配套，**没有默认值**："
             "trunc_disparity→ssi_disparity，truncnorm→ssi，log_truncnorm→ssi_log。"
             "三个族互不嵌套，用错了误差差 30 倍，而且不报错，只是数字难看。",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--prompt", action="append", default=[],
        help="要试的句子，可以给多次。空文本与清单里那一句会自动加上作对照。",
    )
    parser.add_argument(
        "--prompt-file", type=Path, default=None,
        help="每行一句，和 --prompt 合并。",
    )
    return parser.parse_args()


def load_row(manifest: Path, frame: str) -> dict:
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if frame in line:
            row = json.loads(line)
            if row.get("id") == frame:
                return row
    raise SystemExit(f"{manifest} 里没有 id == {frame}")


def score(y: np.ndarray, lidar: np.ndarray, align: str) -> tuple[dict, np.ndarray]:
    """逐帧拟合两个参数，在有激光的像素上打分。

    ⚠️ 这是评估协议的一部分（所有已发表的 MS2 数字都做逐帧对齐），不是调参。
    ⚠️ 三个对齐族互不嵌套，必须和训练时的目标空间一致：
         ssi_disparity   D = 1 / (a·y + b)
         ssi             D = a·y + b
         ssi_log         D = exp(a·y + b)
    """
    real = np.isfinite(lidar) & (lidar > D_MIN) & (lidar < D_MAX)
    if real.sum() < 100:
        raise SystemExit(f"这一帧只有 {int(real.sum())} 个有效激光点，不够拟合")
    src = y[real].astype(np.float64)
    gt = lidar[real].astype(np.float64)
    if align == "ssi_log":
        a, b = np.polyfit(src, np.log(gt), 1)
        metres = np.exp(np.clip(a * y + b, -9.0, 9.0))
    elif align == "ssi":
        a, b = np.polyfit(src, gt, 1)
        metres = np.clip(a * y + b, D_MIN, D_MAX)
    else:  # ssi_disparity
        a, b = np.polyfit(src, 1.0 / gt, 1)
        metres = 1.0 / np.clip(a * y + b, 1.0 / D_MAX, None)
    p, g = metres[real], lidar[real]
    ratio = np.maximum(p / g, g / p)
    return (
        {
            "abs_rel": float(np.mean(np.abs(p - g) / g)),
            "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
            "a1": float(np.mean(ratio < 1.25)),
            "a": float(a),
            "b": float(b),
            "valid_px": int(real.sum()),
        },
        metres,
    )


def main() -> None:
    args = parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    row = load_row(args.manifest.resolve(), args.frame)
    image_path = row["thermal_path"]
    lidar = (
        np.asarray(Image.open(args.ms2_root / row["thermal_depth_path"]), dtype=np.float32)
        / 256.0
    )

    prompts: list[tuple[str, str]] = [("empty", ""), ("manifest", row.get("caption", ""))]
    extra = list(args.prompt)
    if args.prompt_file is not None:
        extra += [
            line.strip()
            for line in args.prompt_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    prompts += [(f"new {i + 1}", text) for i, text in enumerate(extra)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("⚠️ 没有 CUDA，CPU 上会很慢")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    lotus = LotusGPipeline.from_pretrained(
        args.lotus_model_path, torch_dtype=dtype, local_files_only=args.local_files_only
    ).to(device)

    checkpoint_note = "零样本（已发布权重，未适配）"
    if args.unet_checkpoint is not None:
        ckpt = torch.load(args.unet_checkpoint.resolve(), map_location="cpu", weights_only=False)
        if "lotus_unet_state_dict" in ckpt:
            state = ckpt["lotus_unet_state_dict"]
        elif isinstance(ckpt.get("state_dicts"), dict) and "unet" in ckpt["state_dicts"]:
            state = ckpt["state_dicts"]["unet"]
        else:
            raise SystemExit(f"认不出的 checkpoint 结构，顶层键：{sorted(ckpt)[:20]}")
        lotus.unet.load_state_dict(state, strict=True)
        checkpoint_note = f"{args.unet_checkpoint.name}  step {ckpt.get('global_step', '?')}"
        del ckpt
    for module in (lotus.vae, lotus.text_encoder, lotus.unet):
        module.requires_grad_(False).eval()

    # 种子按帧固定，不按 prompt 变 —— 三次前向拿到的是同一份噪声，
    # 所以看到的差别只来自文本。condition 也取 mode，不采样。
    bundle = ThermalVAEPipelineBundle(
        lotus=lotus,
        ms2_root=args.ms2_root.resolve(),
        seeds={image_path: args.seed},
        prompts={image_path: ""},
        condition_posterior="mode",
    )

    print(f"\n帧      {args.frame}")
    print(f"权重    {checkpoint_note}")
    print(f"对齐    {args.align}")
    print(f"种子    {args.seed}（所有 prompt 共用，condition 取 mode）\n")

    results = []
    for label, text in prompts:
        bundle.prompts[image_path] = text
        y = generate_thermal_vae_prediction(None, bundle, image_path=image_path)
        stats, metres = score(np.asarray(y, np.float32), lidar, args.align)
        np.save(out / f"pred_{label}.npy", y)
        results.append((label, text, stats, metres))
        print(f"{label:<10s} AbsRel {stats['abs_rel']:.5f}  RMSE {stats['rmse']:.3f}  "
              f"δ1 {stats['a1']:.5f}   ({len(text)} 字符)")

    (out / "result.json").write_text(
        json.dumps(
            {
                "frame": args.frame,
                "checkpoint": checkpoint_note,
                "seed": args.seed,
                "valid_px": results[0][2]["valid_px"],
                "coverage": results[0][2]["valid_px"] / lidar.size,
                "arms": [
                    {"label": lab, "prompt": txt, **st} for lab, txt, st, _ in results
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 对齐族用错不抛异常，只让数字难看。这一帧的同伴臂都在 0.07-0.11，
    # 所以 0.3 以上基本只可能是 --align 选错了族。
    worst = max(st["abs_rel"] for _, _, st, _ in results)
    if worst > 0.3:
        print()
        print(f"⛔ 最差的一臂 AbsRel {worst:.3f} —— 这个量级几乎只会是 --align 选错族；"
              f"现在用的是 {args.align}，去确认这个 checkpoint 训练时的 norm_type。")

    render(out, args, row, lidar, results)
    print(f"\n[完成] {out}")
    print("⛔ 一帧的数字在噪声里，不能据此选 prompt。这里要看的是图。")


def render(out, args, row, lidar, results) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for name in ("Microsoft YaHei", "SimHei", "DengXian"):
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False

    raw = np.asarray(Image.open(args.ms2_root / row["thermal_path"]), dtype=np.float32)
    lo, hi = (float(v) for v in np.percentile(raw, (1.0, 99.0)))
    thermal = np.clip((raw - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(raw)

    # 所有臂共用一条刻度，否则每格各自 min-max，差别会被归一化抹掉。
    stack = np.concatenate([m.ravel() for _, _, _, m in results])
    v_lo, v_hi = (float(v) for v in np.percentile(stack, (1.0, 99.0)))

    n = len(results) + 1
    cols = 2
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(7.6 * cols, 2.0 * rows + 0.9), dpi=150)
    gs = fig.add_gridspec(rows, cols, wspace=0.03, hspace=0.34,
                          left=0.004, right=0.996, top=0.93, bottom=0.01)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(thermal, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("thermal input", fontsize=13, pad=6, loc="left")
    ax.set_xticks([]); ax.set_yticks([])

    for i, (label, text, stats, metres) in enumerate(results, start=1):
        ax = fig.add_subplot(gs[i // cols, i % cols])
        ax.imshow(np.clip((metres - v_lo) / (v_hi - v_lo + 1e-9), 0, 1),
                  cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{label}　AbsRel {stats['abs_rel']:.4f}", fontsize=13, pad=6, loc="left")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"{args.frame}   lidar coverage "
        f"{results[0][2]['valid_px'] / lidar.size:.1%}   same seed, text only"
        f"   ONE FRAME IS NOT EVIDENCE",
        fontsize=13, y=0.985,
    )
    path = out / "one_frame.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print(f"图 -> {path}")


if __name__ == "__main__":
    main()
