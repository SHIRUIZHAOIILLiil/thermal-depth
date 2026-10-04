"""Per-frame cost of our depth inference, component by component, batch 1, 256x640.

Mirrors `train_route_suite.RouteModel.predict_disparity` at evaluation, including its
precisions: CLIP text encoder and VAE encoder in fp16 (`--frozen-dtype fp16`, the
default every reported number used), the U-Net in fp32 (it is deep-copied to fp32),
one step at t=999 with the task embedding, VAE decode under fp16 autocast, channel
mean. Timed with CUDA events after warm-up; preprocessing on the CPU is timed apart.

The U-Net weights are the released Lotus-G ones: same architecture and shapes as our
checkpoint (only the values differ), so the arithmetic, and the time, are the same.
That keeps the benchmark runnable on a machine without our 3.5 GB checkpoint.

    HF_HOME=/mnt/e/AI_Cache/huggingface HF_HUB_OFFLINE=1 \
    python tools/bench_depth_inference.py --frames-dir /mnt/e/dataset/ms2/sync_data/_2021-08-06-11-23-45/thr/img_left
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CAPTION = ("The thermal image depicts an urban street scene with two pedestrians walking side by "
           "side in the foreground, a car on the left, and buildings lining the street, with "
           "distant vehicles and structures further away.")


def stretch(path: Path) -> torch.Tensor:
    raw = np.asarray(Image.open(path), np.float32)
    lo, hi = np.percentile(raw, (1.0, 99.0))
    u8 = np.clip((raw - lo) / (hi - lo) * 255.0, 0, 255).round()
    x = torch.from_numpy(u8 / 127.5 - 1.0).float()
    return x[None, None].repeat(1, 3, 1, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--model", default="jingheya/lotus-depth-g-v2-1-disparity")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--unet-dtype", choices=("fp32", "fp16"), default="fp32",
                    help="fp32 is what evaluation ran; fp16 only to show the headroom.")
    args = ap.parse_args()

    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    dev = torch.device("cuda")
    f16 = torch.float16
    u_dt = torch.float32 if args.unet_dtype == "fp32" else torch.float16
    tok = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder", torch_dtype=f16).to(dev).eval()
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=f16).to(dev).eval()
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet").to(dev, u_dt).eval()
    sf = vae.config.scaling_factor
    task = torch.tensor([[1.0, 0.0]], device=dev, dtype=u_dt)
    task = torch.cat([torch.sin(task), torch.cos(task)], dim=-1)
    t999 = torch.tensor(999, dtype=torch.long)

    frames = sorted(args.frames_dir.glob("*.png"))
    step = max(1, len(frames) // (args.warmup + args.frames))
    frames = frames[::step][: args.warmup + args.frames]
    print(f"[gpu] {torch.cuda.get_device_name(0)}  torch {torch.__version__}  unet {args.unet_dtype}")
    print(f"[data] {len(frames)} frames from {args.frames_dir}  (first {args.warmup} are warm-up)")

    parts = ("text", "vae_enc", "unet", "vae_dec")
    times = {k: [] for k in parts}
    cpu_ms = []
    gen = torch.Generator(device=dev)
    for i, path in enumerate(frames):
        t0 = time.perf_counter()
        x = stretch(path).to(dev, f16)
        cpu_ms.append((time.perf_counter() - t0) * 1e3)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(len(parts) + 1)]
        with torch.inference_mode():
            ev[0].record()
            ids = tok([CAPTION], padding="max_length", max_length=tok.model_max_length,
                      truncation=True, return_tensors="pt").input_ids.to(dev)
            prompt = te(ids)[0]
            ev[1].record()
            z = vae.encode(x).latent_dist.mode() * sf
            ev[2].record()
            gen.manual_seed(20260701 + i)
            noise = torch.randn(z.shape, generator=gen, device=dev, dtype=torch.float32)
            x0 = unet(torch.cat([z.float(), noise], 1).to(u_dt), t999,
                      encoder_hidden_states=prompt.to(u_dt), class_labels=task, return_dict=False)[0]
            ev[3].record()
            with torch.autocast("cuda", dtype=f16):
                depth = vae.decode(x0.float() / sf, return_dict=False)[0].mean(1)
            ev[4].record()
        torch.cuda.synchronize()
        if i >= args.warmup:
            for k, name in enumerate(parts):
                times[name].append(ev[k].elapsed_time(ev[k + 1]))
    assert depth.shape[-2:] == (256, 640), depth.shape

    total = [sum(v) for v in zip(*times.values())]
    print(f"\n{'component':10s} {'mean ms':>9s} {'median ms':>10s}")
    for name in parts:
        print(f"{name:10s} {statistics.fmean(times[name]):9.2f} {statistics.median(times[name]):10.2f}")
    print(f"{'GPU total':10s} {statistics.fmean(total):9.2f} {statistics.median(total):10.2f}"
          f"   -> {1000 / statistics.median(total):.1f} frames/s")
    print(f"{'CPU prep':10s} {statistics.fmean(cpu_ms[args.warmup:]):9.2f} "
          f"{statistics.median(cpu_ms[args.warmup:]):10.2f}   (16-bit read + percentile stretch)")
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print("note: the text encoder runs once per caption; with a fixed caption it can be cached.")


if __name__ == "__main__":
    main()
