"""热像进 VAE 再出来，还剩多少。

b 线的前端就是这只冻结的 VAE 编码器：热像 → encode → condition latent → U-Net。
如果它在编码那一步就把热像的内容丢了，那么后面无论 U-Net 解不解冻、前端换成
什么，丢掉的都回不来。这里不训练任何东西，只让图像走一个来回。

⚠️ 一个孤零零的 PSNR 说明不了任何事。所以每帧都同时跑三条，全在同一只 VAE、
同一个尺寸、同一批帧上：

    thermal-vae   热像过 VAE 来回        ← 要测的
    rgb-vae       同一帧的 RGB 过 VAE    ← 对照：这只 VAE 本来能做多好
    thermal-8x    热像只做 8 倍降采样再升回来，不过 VAE
                                          ← 对照：纯分辨率瓶颈的下界

读法：
  · thermal-vae ≈ rgb-vae    → 编码器没有歧视热像，问题不在这
  · thermal-vae ≫ rgb-vae    → 编码器确实丢热像内容，前端再强也白搭
  · thermal-vae ≈ thermal-8x → 丢的就是 8 倍下采样那点空间分辨率，不是别的

主指标是 **相对误差 RMSE / std(输入)**，不是 PSNR。PSNR 跟对比度走，而热像
经过分位数拉伸之后的对比度和 RGB 不是一回事，直接比 PSNR 会把对比度差异读成
质量差异。

细节保留 = 输出的平均梯度幅值 ÷ 输入的。小于 1 就是被抹平了。

    python tools/probe_vae_thermal_roundtrip.py \
        --manifest <test.jsonl> --ms2-root <root> \
        --output-dir <runs>/analysis/vae_thermal_probe --limit 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "lotus", ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from models.anythermal_lotus_v2 import thermal_to_lotus_input  # noqa: E402

ROWS = ("thermal-vae", "rgb-vae", "thermal-8x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument(
        "--dtype",
        choices=("fp16", "fp32"),
        default="fp32",
        help="默认 fp32：这里量的是编码器丢了多少，不该和半精度的舍入混在一起。",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def uniform(rows: list[dict], limit: int) -> list[dict]:
    if limit <= 0 or limit >= len(rows):
        return rows
    idx = np.linspace(0, len(rows) - 1, limit).round().astype(int)
    return [rows[i] for i in sorted(set(idx.tolist()))]


def rgb_to_tensor(path: Path, size: tuple[int, int]) -> torch.Tensor:
    """RGB 读成和热像同一个尺寸、同一个 [-1,1] 约定。

    尺寸必须对齐，因为要比的是「同一个 8 倍瓶颈下谁掉得多」，两边分辨率不同
    就不是一个对照了。

    ⚠️ 但不能直接 resize：MS2 的 RGB 是 1224x1024、热像是 640x256，长宽比差
    一倍多，硬缩会把 RGB 压扁 —— 那样量到的是形变，不是模态。所以走「等比
    缩放到能盖住 + 中心裁剪」，RGB 的局部结构保持原样。

    代价是 RGB 被降采样了（1224 宽 → 640），原生细节先掉一截，所以 rgb-vae
    这一行是**偏保守**的对照：它只会低估 VAE 对 RGB 的优待。
    """
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    height, width = (int(v) for v in tensor.shape[-2:])
    target_h, target_w = size
    if (height, width) != size:
        ratio = max(target_h / height, target_w / width)
        scaled = (max(target_h, round(height * ratio)), max(target_w, round(width * ratio)))
        tensor = F.interpolate(
            tensor, size=scaled, mode="bilinear", align_corners=False,
            antialias=ratio < 1.0,
        )
        top = (scaled[0] - target_h) // 2
        left = (scaled[1] - target_w) // 2
        tensor = tensor[..., top:top + target_h, left:left + target_w]
    if tuple(tensor.shape[-2:]) != size:
        raise RuntimeError(f"RGB 裁到了 {tuple(tensor.shape[-2:])}，应为 {size}")
    return tensor


def grad_magnitude(x: torch.Tensor) -> torch.Tensor:
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx = x[..., :, 1:] - x[..., :, :-1]
    return torch.cat([dy.abs().flatten(), dx.abs().flatten()])


def compare(before: torch.Tensor, after: torch.Tensor) -> dict:
    a = before.float().mean(dim=1)   # 热像三通道相同；RGB 取通道平均
    b = after.float().mean(dim=1)
    if a.shape != b.shape:
        raise RuntimeError(f"形状对不上：{tuple(a.shape)} vs {tuple(b.shape)}")
    if not (torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise RuntimeError("出现了 NaN/Inf")
    rmse = float(torch.sqrt(torch.mean((a - b) ** 2)))
    std = float(a.std(unbiased=False))
    ga, gb = grad_magnitude(a), grad_magnitude(b)
    ga_mean, gb_mean = float(ga.mean()), float(gb.mean())
    va, vb = ga - ga.mean(), gb - gb.mean()
    denom = float(torch.sqrt((va * va).sum() * (vb * vb).sum()))
    return {
        # 主指标：相对于输入自身起伏的误差，对比度无关
        "rel_rmse": rmse / std if std > 0 else float("nan"),
        "rmse": rmse,
        "input_std": std,
        "psnr_db": float(10.0 * np.log10(4.0 / max(rmse ** 2, 1e-12))),
        "detail_kept": gb_mean / ga_mean if ga_mean > 0 else float("nan"),
        "grad_corr": float((va * vb).sum()) / denom if denom > 0 else float("nan"),
    }


def main() -> None:
    args = parse_args()
    from diffusers import AutoencoderKL

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = (
        AutoencoderKL.from_pretrained(
            args.lotus_model_path, subfolder="vae", torch_dtype=dtype
        )
        .to(device)
        .eval()
        .requires_grad_(False)
    )
    scale = int(2 ** (len(vae.config.block_out_channels) - 1))

    rows = uniform(read_manifest(args.manifest.resolve()), args.limit)
    per_frame: list[dict] = []
    skipped_no_rgb = 0

    for row in rows:
        thermal = thermal_to_lotus_input(args.ms2_root / row["thermal_path"], processing_res=0)
        x = thermal.tensor
        height, width = (int(v) for v in x.shape[-2:])
        if height % scale or width % scale:
            raise RuntimeError(f"{height}x{width} 不是 {scale} 的整数倍")

        rgb_rel = row.get("rgb_path")
        if not rgb_rel or not (args.ms2_root / rgb_rel).is_file():
            skipped_no_rgb += 1
            continue
        rgb = rgb_to_tensor(args.ms2_root / rgb_rel, (height, width))

        entry: dict = {"image_path": row["thermal_path"], "rgb_path": rgb_rel}
        with torch.inference_mode():
            for name, source in (("thermal-vae", x), ("rgb-vae", rgb)):
                # mode()，不是 sample() —— 推理时 condition 走的就是确定性那条。
                latent = vae.encode(source.to(device=device, dtype=dtype)).latent_dist.mode()
                back = vae.decode(latent).sample.float().cpu()
                entry[name] = compare(source, back)
            small = F.interpolate(
                x, scale_factor=1.0 / scale, mode="bilinear",
                align_corners=False, antialias=True,
            )
            entry["thermal-8x"] = compare(
                x,
                F.interpolate(small, size=(height, width), mode="bilinear",
                              align_corners=False),
            )
        per_frame.append(entry)

    if not per_frame:
        raise SystemExit("一帧都没跑成 —— manifest 里没有可用的 rgb_path？")
    # 三行必须同批帧，否则这不是一个对照。上面的循环保证了这点，这里把它钉死。
    assert all(set(ROWS) <= set(entry) for entry in per_frame)

    summary = {
        name: {
            key: float(np.mean([entry[name][key] for entry in per_frame]))
            for key in per_frame[0][name]
        }
        for name in ROWS
    }
    (out / "per_frame.json").write_text(json.dumps(per_frame, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps(
            {
                "frames": len(per_frame),
                "skipped_no_rgb": skipped_no_rgb,
                "dtype": args.dtype,
                "vae_scale": scale,
                "model": args.lotus_model_path,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    note = f"（另有 {skipped_no_rgb} 帧没有可用 RGB，已跳过）" if skipped_no_rgb else ""
    print(f"\n共 {len(per_frame)} 帧，三行同一批{note}")
    print(f"{'':>14s}{'相对误差':>12s}{'细节保留':>12s}{'梯度相关':>12s}{'PSNR(dB)':>11s}")
    label = {
        "thermal-vae": "热像过 VAE",
        "rgb-vae": "RGB 过 VAE",
        "thermal-8x": "热像仅 8 倍降采",
    }
    for name in ROWS:
        s = summary[name]
        print(
            f"{label[name]:>14s}{s['rel_rmse']:>12.4f}{s['detail_kept']:>12.4f}"
            f"{s['grad_corr']:>12.4f}{s['psnr_db']:>11.2f}"
        )

    t, r, d = (summary[name]["rel_rmse"] for name in ROWS)
    print()
    print(f"热像 ÷ RGB   = {t / r:.2f}   （≈1 ＝ 编码器没有歧视热像）")
    print(f"热像 ÷ 8倍降 = {t / d:.2f}   （≈1 ＝ 丢的就是那 8 倍空间分辨率）")
    print("\n⚠️ 这一页量的是**编码器还原输入的能力**，不是深度质量。")
    print("⚠️ RGB 那行先被缩到了热像尺寸，所以它是偏保守的对照。")


if __name__ == "__main__":
    main()
