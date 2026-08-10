"""How well the frozen VAE can even represent a dense depth target.

The latent-target objective -- encode the ground truth into a latent, add noise,
train the U-Net to undo it -- needs a dense depth map, which sparse lidar cannot
provide but completed pseudo depth can. Before writing that trainer, this asks
the question that caps it: if the U-Net predicted the target latent *perfectly*,
how good would the decoded depth be?

The answer is the VAE's round trip, because the target is only ever seen through
it. Nothing is trained here; the map is encoded and decoded once, aligned to the
lidar the way every number in this project is aligned, and scored on the pixels
lidar actually covers.

Normalisation follows the checkpoint in use (lotus-depth-g-v2-1-disparity), so
depth is inverted to disparity and min-max mapped to [-1, 1] per image, matching
`HypersimImageDepthNormalTransform` under `norm_type="disparity"`. Getting this
wrong would measure a normalisation mismatch and call it a VAE limit.

    python tools/probe_vae_roundtrip_ceiling.py \
        --manifest <train.jsonl> --ms2-root <root> \
        --pseudo-dir <runs>/pseudo_gt/train_full/calibrated_pseudo_depth \
        --output-dir <runs>/analysis/vae_ceiling --limit 200
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
for path in (ROOT, ROOT / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ms2_eval.official_protocol import fit_scale_shift, official_valid_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="Training split.")
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--pseudo-dir", type=Path, required=True, help="calibrated_pseudo_depth/")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--stride", type=int, default=0, help="0 = spread --limit evenly over the split.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def read_rows(manifest: Path, ms2_root: Path, limit: int, stride: int) -> list[dict]:
    rows = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            relative = row.get("thermal_depth_path") or row.get("depth_path")
            if not relative:
                raise SystemExit(f"Row {row.get('id')} lacks GT depth")
            rows.append({"id": str(row["id"]), "gt_path": ms2_root / relative})
    step = stride if stride > 0 else max(1, len(rows) // max(1, limit))
    rows = rows[::step][:limit] if limit else rows[::step]
    if not rows:
        raise SystemExit("No rows selected")
    return rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from diffusers import AutoencoderKL

    device = torch.device(args.device)
    vae = AutoencoderKL.from_pretrained(args.lotus_model_path, subfolder="vae",
                                        local_files_only=args.local_files_only)
    vae.to(device).eval()
    scaling = float(vae.config.scaling_factor)
    print(f"[vae] {args.lotus_model_path} scaling_factor={scaling}", flush=True)

    rows = read_rows(args.manifest.resolve(), args.ms2_root.resolve(), args.limit, args.stride)
    print(f"[data] {len(rows)} frames", flush=True)

    records = []
    for index, row in enumerate(rows):
        pseudo_path = args.pseudo_dir / f"{row['id']}.npy"
        if not pseudo_path.is_file():
            continue
        gt = np.asarray(Image.open(row["gt_path"]), dtype=np.float64) / args.depth_scale
        valid = official_valid_mask(gt, args.min_depth, args.max_depth)
        if valid.sum() < 100:
            continue
        pseudo = np.load(pseudo_path, allow_pickle=False).astype(np.float64)

        # The target the latent objective would encode: lidar where it spoke,
        # completed pseudo depth everywhere else.
        dense = np.clip(np.where(valid, gt, pseudo), args.min_depth, args.max_depth)
        disparity = 1.0 / dense
        dmin, dmax = float(disparity.min()), float(disparity.max())
        normalised = ((disparity - dmin) / (dmax - dmin + 1e-5) - 0.5) * 2.0

        tensor = torch.from_numpy(normalised).float()[None, None].repeat(1, 3, 1, 1).to(device)
        with torch.no_grad():
            latent = vae.encode(tensor).latent_dist.mode() * scaling
            decoded = vae.decode(latent / scaling).sample
        out = decoded.float().mean(dim=1)[0].cpu().numpy().astype(np.float64)

        # Undo the normalisation, then align exactly as every other number here is
        # aligned -- otherwise this would measure a scale convention, not the VAE.
        round_trip_disparity = (out / 2.0 + 0.5) * (dmax - dmin + 1e-5) + dmin
        gt_disparity = np.zeros_like(gt)
        gt_disparity[valid] = 1.0 / gt[valid]
        scale, shift = fit_scale_shift(round_trip_disparity.astype(np.float32),
                                       gt_disparity.astype(np.float32), valid)
        aligned = np.clip(1.0 / np.clip(round_trip_disparity * scale + shift, 1e-3, None),
                          args.min_depth, args.max_depth)

        abs_rel = float(np.mean(np.abs(aligned[valid] - gt[valid]) / gt[valid]))
        # Fidelity to the map that went in, with no alignment involved at all.
        fidelity = float(np.mean(np.abs(round_trip_disparity - disparity) / np.maximum(disparity, 1e-6)))
        records.append({"id": row["id"], "abs_rel_vs_lidar": abs_rel,
                        "disparity_round_trip_rel": fidelity,
                        "valid_pixels": int(valid.sum())})
        if index % 50 == 0:
            print(f"[{index + 1}/{len(rows)}] {row['id']} abs_rel {abs_rel:.4f} "
                  f"disp_rt {fidelity:.4f}", flush=True)

    if not records:
        raise SystemExit("No frame produced a measurement")
    a = np.asarray([r["abs_rel_vs_lidar"] for r in records])
    f = np.asarray([r["disparity_round_trip_rel"] for r in records])
    summary = {
        "frames": len(records), "lotus_model_path": args.lotus_model_path,
        "normalisation": "disparity, per-image min-max to [-1,1] (norm_type='disparity')",
        "abs_rel_vs_lidar": {"p50": float(np.median(a)), "mean": float(a.mean()),
                             "p90": float(np.percentile(a, 90)), "max": float(a.max())},
        "disparity_round_trip_rel": {"p50": float(np.median(f)), "mean": float(f.mean())},
    }
    (args.output_dir / "vae_ceiling.json").write_text(
        json.dumps({"summary": summary, "per_frame": records}, indent=2), encoding="utf-8")

    print(f"\n[frames] {len(records)}")
    print(f"[ceiling] AbsRel of the perfectly-predicted target vs lidar: "
          f"p50 {np.median(a):.4f}  mean {a.mean():.4f}  p90 {np.percentile(a, 90):.4f}")
    print(f"[fidelity] disparity round-trip relative error: p50 {np.median(f):.4f}")
    print("\nReading it: this is what the latent objective could reach if the U-Net were")
    print("perfect, because the target is only ever seen through this round trip. Compare")
    print("it against 0.0849, which the current pixel-space objective already achieves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
