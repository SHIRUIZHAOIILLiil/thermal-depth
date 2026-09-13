"""Measure how much each backbone's cross-attention actually moves when the text changes.

The claim this exists to test: Lotus fine-tuned SD2 for 20k steps with an empty
prompt on every sample, so its cross-attention learned to ignore text -- which
would explain why captions do nothing on a Lotus-G start and something on an SD2
start. Until now that has been an inference from three indirect facts
(docs/HANDOFF_20260910.md section 12.8); this measures it directly.

Text enters a UNet2DConditionModel only through the cross-attention blocks, and
there only through `attn2.to_k` and `attn2.to_v`, which consume
`encoder_hidden_states`. Both backbones carry the same frozen text encoder, so
the embeddings E are identical between them and any difference in response comes
from those two projections alone.

Per layer, against the empty-prompt embedding E0:

  drift      ||W_lotus - W_sd2||_F / ||W_sd2||_F        how far Lotus moved the weights
  response   mean_i ||W E_i - W E0||_F / ||W E0||_F     how far a real caption pushes K/V
                                                        away from where empty text puts it
  spread     mean_i ||W E_i - W Ebar||_F / ||W Ebar||_F  how much K/V varies between captions

A backbone that ignores text has a small `response`: every caption lands close to
where the empty prompt lands. The comparison between backbones is the point; the
absolute values mean little on their own.

CPU only, no GPU, a few minutes. Weights are read straight out of the HF cache as
state dicts -- no UNet is constructed, so this runs in a couple of GB.

  python tools/text_pathway_response.py \
      --manifest $SCRATCH/manifests/sequence_level_internvl3_8b/ms2_train_official8_thermalcap_v3_1_untrimmed_20260821.jsonl \
      --out $SCRATCH/runs/analysis/text_pathway_response.json
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer

SD2 = os.environ.get("SD2_REPO", "Manojb/stable-diffusion-2-1-base")
LOTUS = os.environ.get("LOTUS_REPO", "jingheya/lotus-depth-g-v2-1-disparity")


def unet_state(repo):
    path = hf_hub_download(repo, "unet/diffusion_pytorch_model.safetensors",
                           local_files_only=True)
    return load_file(path)


def captions(manifest, n, seed):
    rows = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = (json.loads(line).get("caption") or "").strip()
            if c:
                rows.append(c)
    random.Random(seed).shuffle(rows)
    return rows[:n]


def embed(texts, repo, batch):
    tok = CLIPTokenizer.from_pretrained(repo, subfolder="tokenizer", local_files_only=True)
    enc = CLIPTextModel.from_pretrained(repo, subfolder="text_encoder",
                                        local_files_only=True).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            ids = tok(texts[i:i + batch], padding="max_length",
                      max_length=tok.model_max_length, truncation=True,
                      return_tensors="pt").input_ids
            out.append(enc(ids, return_dict=False)[0].float())
    return torch.cat(out, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--num-captions", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out")
    args = ap.parse_args()

    texts = captions(args.manifest, args.num_captions, args.seed)
    print(f"captions: {len(texts)}")
    # The empty prompt is the only text Lotus ever saw, so it is the reference
    # point: a backbone that ignores text keeps every caption near it.
    E = embed([""] + texts, SD2, args.batch)
    E0, Ec = E[:1], E[1:]
    Ebar = Ec.mean(0, keepdim=True)

    sd2, lotus = unet_state(SD2), unet_state(LOTUS)
    keys = sorted(k for k in sd2
                  if k.endswith(("attn2.to_k.weight", "attn2.to_v.weight")))
    print(f"cross-attention projections: {len(keys)}\n")

    rows = []
    for k in keys:
        W2, Wl = sd2[k].float(), lotus[k].float()
        rec = {"layer": k,
               "drift": float((Wl - W2).norm() / W2.norm())}
        for name, W in (("sd2", W2), ("lotus", Wl)):
            K0 = E0 @ W.T
            Kc = Ec @ W.T
            Kbar = Ebar @ W.T
            rec[f"response_{name}"] = float(
                (Kc - K0).flatten(1).norm(dim=1).mean() / K0.norm())
            rec[f"spread_{name}"] = float(
                (Kc - Kbar).flatten(1).norm(dim=1).mean() / Kbar.norm())
        rows.append(rec)

    hdr = f"{'layer':<62}{'drift':>8}{'resp sd2':>10}{'resp lotus':>12}{'ratio':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ratio = r["response_lotus"] / r["response_sd2"] if r["response_sd2"] else float("nan")
        print(f"{r['layer'].replace('.weight',''):<62}{r['drift']:>8.3f}"
              f"{r['response_sd2']:>10.4f}{r['response_lotus']:>12.4f}{ratio:>8.2f}")

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    print("\n=== 汇总（所有 cross-attention 投影的均值）===")
    print(f"权重漂移 ||lotus-sd2||/||sd2||      : {mean('drift'):.4f}")
    print(f"响应度 response  sd2 / lotus        : {mean('response_sd2'):.4f} / {mean('response_lotus'):.4f}")
    print(f"离散度 spread    sd2 / lotus        : {mean('spread_sd2'):.4f} / {mean('spread_lotus'):.4f}")
    r = mean("response_lotus") / mean("response_sd2")
    print(f"\n响应度之比 lotus/sd2 = {r:.3f}")
    print("  < 1  Lotus 的 cross-attention 对换文本更迟钝 —— 支持「被训哑」")
    print("  ~ 1  两者响应相当 —— 不支持，移植没有靶子")
    print("  > 1  Lotus 反而更敏感 —— 需要另找解释")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"num_captions": len(texts), "sd2": SD2, "lotus": LOTUS,
                       "layers": rows}, f, indent=2)
        print(f"\n写入 {args.out}")


if __name__ == "__main__":
    main()
