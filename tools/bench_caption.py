"""Per-frame cost of generating one thermal caption with InternVL3-8B, batch 1.

Uses the captioner class that wrote the 28,302 v3_1 captions
(`E:/project/captioning/scripts/generate_captions.py`, `InternVLCaptioner`) with the
settings the Aire jobs used: bf16 on a bf16-capable GPU (dtype "auto"), greedy
decoding, max_new_tokens 160 (the CLI default), up to 12 tiles of 448 plus a
thumbnail, the thermal frame rendered grayscale through the same 16-bit conversion,
prompt `thermal_depth_v3_1`. Generation time depends on caption length, so the number
of generated tokens is reported with it.

    HF_HOME=/mnt/e/AI_Cache/huggingface HF_HUB_OFFLINE=1 \
    python tools/bench_caption.py --frames-dir /mnt/e/dataset/ms2/sync_data/_2021-08-06-11-23-45/thr/img_left
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--captioning-scripts", type=Path, default=Path("/mnt/e/project/captioning/scripts"))
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--frames", type=int, default=30)
    args = ap.parse_args()

    sys.path.insert(0, str(args.captioning_scripts))
    import torch
    import transformers
    from generate_captions import InternVLCaptioner, load_image, resolve_prompt

    version, prompt = resolve_prompt("thermal_depth_v3_1", "thermal")
    cap = InternVLCaptioner(device="cuda", dtype="auto", max_new_tokens=160)
    t0 = time.perf_counter()
    cap.load()
    print(f"[gpu] {torch.cuda.get_device_name(0)}  torch {torch.__version__}  "
          f"transformers {transformers.__version__}  dtype {cap._model_dtype}")
    print(f"[load] {time.perf_counter() - t0:.1f} s (one-off, not per frame)")

    frames = sorted(args.frames_dir.glob("*.png"))
    step = max(1, len(frames) // (args.warmup + args.frames))
    frames = frames[::step][: args.warmup + args.frames]
    gen_s, prep_s, ntok = [], [], []
    for i, path in enumerate(frames):
        t0 = time.perf_counter()
        image = load_image(path, thermal_render="grayscale")
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        text = cap.generate(image, prompt=prompt, prompt_type=version)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        if i >= args.warmup:
            prep_s.append(t1 - t0)
            gen_s.append(t3 - t2)
            ntok.append(len(cap.tokenizer(text, add_special_tokens=False).input_ids))
        if i == args.warmup:
            print(f"[sample] {path.name}: {text}")

    print(f"\n[data] {len(gen_s)} timed frames (+{args.warmup} warm-up) from {args.frames_dir}")
    print(f"generate  mean {statistics.fmean(gen_s):.2f} s   median {statistics.median(gen_s):.2f} s"
          f"   min {min(gen_s):.2f}  max {max(gen_s):.2f}")
    print(f"tokens    mean {statistics.fmean(ntok):.1f}   median {statistics.median(ntok):.0f}"
          f"   -> {sum(ntok) / sum(gen_s):.1f} tokens/s")
    print(f"image prep (16-bit read + grayscale render) mean {statistics.fmean(prep_s) * 1e3:.1f} ms")
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
