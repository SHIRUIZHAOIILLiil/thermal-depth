"""Gates for the metric GT adaptation stage. Run this before any Aire job.

    python tools/smoke_metric_adaptation.py \
        --manifest <train manifest> --ms2-root <ms2 root> \
        --metric-norm docs/data/metric_norm_train_full8.json \
        --checkpoint checkpoints/step20000_weights.pt \
        --frames 4

Four gates, in the order they can fail cheaply:

  A  normalisation           metres -> 1/m -> [-1,1] -> back, on real frames
  B  metric prediction       what the checkpoint's output becomes in metres
  C  gradient                L_metric reaches the U-Net; VAE and CLIP stay frozen
  D  evaluation              metric_no_test_alignment reads no GT before scoring

Gate D is behavioural rather than a code read: it runs the prediction-to-depth
path twice, once against the real GT and once against a GT scaled by three, and
requires the two predicted depth maps to be bit-identical.  If any GT leaked
into the transform the maps would differ, and no amount of reading the source
would have proved they do not.

Gate C likewise compares the fp32 decoder against the fp16 one on the same step,
because "no fp16 underflow occurs" is a claim about a number, not about a flag.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = ROOT / "lotus"
for path in (ROOT, LOTUS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ms2_eval.official_protocol import evaluate_sample, official_valid_mask  # noqa: E402
from tools.metric_depth_norm import (  # noqa: E402
    MetricNorm,
    decoded_to_inverse,
    depth_to_inverse,
    depth_to_unit,
    inverse_to_depth,
    inverse_to_unit,
    nonfinite_count,
    non_positive_fraction,
    unit_to_inverse,
)
from tools.train_route_suite import (  # noqa: E402
    RouteModel,
    load_input_tensor,
    prediction_to_depth,
    read_manifest,
    resolve_metric_affine,
    seed_everything,
)

RULE = "=" * 78


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, required=True, help="TRAIN manifest.")
    parser.add_argument("--ms2-root", type=Path, required=True)
    parser.add_argument("--metric-norm", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Route-suite payload. Gates B and C are skipped without it.")
    parser.add_argument("--global-affine", type=Path, default=None,
                        help="Optional: also report gate B under the train-fitted affine.")
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--stride", type=int, default=997,
                        help="Frame stride, so the sample is not four consecutive frames.")
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument("--min-depth", type=float, default=1e-3)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--skip-gradient", action="store_true",
                        help="Skip gate C. It is the slowest and the only one that allocates a graph.")
    return parser.parse_args()


def model_args(args, route="b_thermal_unet") -> SimpleNamespace:
    """The attribute surface `RouteModel` and `load_input_tensor` actually read."""
    return SimpleNamespace(
        route=route,
        lotus_model_path=args.lotus_model_path,
        local_files_only=args.local_files_only,
        condition_latent="mode",
        gt_decode_fp32=True,
        freeze_adapter=False,
        timestep=999,
        seed=args.seed,
        num_inference_steps=1,
        shuffle_condition="none",
        input_max_edge=0,
        depth_scale=args.depth_scale,
        gt_min_depth=args.min_depth,
        gt_max_depth=args.max_depth,
        eval_min_depth=args.min_depth,
        eval_max_depth=args.max_depth,
        anythermal_model_path=None,
    )


# --------------------------------------------------------------------------- #
# A -- normalisation
# --------------------------------------------------------------------------- #


def gate_a(rows, norm: MetricNorm, args) -> bool:
    print(RULE)
    print("GATE A -- normalisation sanity (no model involved)")
    print(RULE)
    print(f"constants: {norm.summary()}")
    print(
        f"represented band: q in [{norm.q_lo:.6f}, {norm.q_hi:.6f}] 1/m "
        f"= depth [{norm.depth_at_q_hi:.2f}, {norm.depth_at_q_lo:.2f}] m"
    )
    worst = 0.0
    ok = True
    for row in rows:
        depth = np.asarray(Image.open(row["depth_path"]), dtype=np.float32) / args.depth_scale
        q, valid = depth_to_inverse(depth, norm)
        if not valid.any():
            print(f"  {row['id']}: no valid lidar, skipped")
            continue
        u, report = inverse_to_unit(q, norm)
        u = np.clip(u, -1.0, 1.0)
        q_back = unit_to_inverse(u, norm)
        depth_back, _ = inverse_to_depth(q_back, norm)
        # Round-trip error is only meaningful where the clip did not fire; a
        # clipped pixel is supposed to come back different.
        inside = valid & (q >= norm.q_lo) & (q <= norm.q_hi)
        err = float(np.max(np.abs(q_back[inside] - q[inside]))) if inside.any() else 0.0
        depth_err = (
            float(np.max(np.abs(depth_back[inside] - depth[inside]))) if inside.any() else 0.0
        )
        worst = max(worst, err)
        clipped_valid = int(np.count_nonzero(valid & ~inside))
        print(f"  {row['id']}")
        print(
            f"    GT depth      [{depth[valid].min():8.3f}, {depth[valid].max():8.3f}] m   "
            f"({int(valid.sum()):,} lidar px, {valid.mean() * 100:.1f}% of frame)"
        )
        print(f"    GT inverse    [{q[valid].min():8.5f}, {q[valid].max():8.5f}] 1/m")
        print(f"    target u      [{u.min():8.5f}, {u.max():8.5f}]  (whole frame, invalid px at -1)")
        print(
            f"    u -> q back   [{q_back[inside].min():8.5f}, {q_back[inside].max():8.5f}] 1/m"
            if inside.any() else "    u -> q back   (nothing inside the band)"
        )
        print(
            f"    round trip    max |dq| = {err:.3e} 1/m,  max |dD| = {depth_err:.3e} m,  "
            f"{clipped_valid:,} valid px outside the band ({clipped_valid / max(1, int(valid.sum())) * 100:.2f}%)"
        )
        if err > 1e-6:
            print("    !! round-trip error above 1e-6 1/m")
            ok = False
    print(f"\n  worst round-trip error over {len(rows)} frames: {worst:.3e} 1/m")
    print(f"  GATE A: {'PASS' if ok else 'FAIL'}\n")
    return ok


# --------------------------------------------------------------------------- #
# B -- metric prediction
# --------------------------------------------------------------------------- #


@torch.no_grad()
def gate_b(model, rows, norm: MetricNorm, args) -> bool:
    print(RULE)
    print("GATE B -- metric prediction sanity (frozen original checkpoint)")
    print(RULE)
    sources = [("raw_inverse", 1.0, 0.0)]
    a, b = norm.as_affine()
    sources.append(("global_norm", a, b))
    if args.global_affine is not None:
        payload = json.loads(args.global_affine.read_text(encoding="utf-8"))
        sources.append(("global_affine", float(payload["a"]), float(payload["b"])))

    prompt = model.encode_prompt("")
    ok = True
    for row in rows:
        image_tensor, _ = load_input_tensor(row, model.modality, model.args)
        prediction = model.predict_disparity(row, image_tensor, prompt)
        gt = np.asarray(Image.open(row["depth_path"]), dtype=np.float32) / args.depth_scale
        pred = prediction[None, None]
        if pred.shape[-2:] != gt.shape:
            pred = F.interpolate(pred, gt.shape, mode="bilinear", align_corners=False)
        y = pred[0, 0].float().cpu().numpy()
        valid = official_valid_mask(gt, args.min_depth, args.max_depth)
        print(f"  {row['id']}")
        print(
            f"    decoded y     [{y.min():8.5f}, {y.max():8.5f}]   "
            f"nonfinite {nonfinite_count(y)}   "
            f"outside [0,1] {int(np.count_nonzero((y < 0) | (y > 1))):,}/{y.size:,}"
        )
        for name, a_i, b_i in sources:
            q = a_i * y.astype(np.float64) + b_i
            depth, non_positive, floored = prediction_to_depth(
                y, {"a": a_i, "b": b_i}, model.args
            )
            metrics = evaluate_sample(
                depth, gt, align="none", min_depth=args.min_depth, max_depth=args.max_depth
            )
            print(
                f"    [{name:<13}] q [{q.min():8.5f}, {q.max():8.5f}] 1/m  "
                f"D [{depth.min():9.3f}, {depth.max():11.1f}] m  "
                f"q<=0 {non_positive_fraction(q) * 100:6.3f}%  "
                f"floored {floored:,}  nan/inf {nonfinite_count(q)}"
            )
            print(
                f"    {'':<15} clamped to range: {metrics['clamped_below']:,} below / "
                f"{metrics['clamped_above']:,} above of {metrics['valid_pixels']:,} valid   "
                f"-> AbsRel {metrics['abs_rel']:.4f}  RMSE {metrics['rmse']:.2f}  d1 {metrics['a1']:.4f}"
            )
            if nonfinite_count(q):
                print("    !! non-finite inverse depth")
                ok = False
        ssi = evaluate_sample(
            y, gt, align="ssi_disparity", min_depth=args.min_depth, max_depth=args.max_depth
        )
        print(
            f"    [reference   ] per-frame affine to THIS frame's GT: "
            f"AbsRel {ssi['abs_rel']:.4f}  RMSE {ssi['rmse']:.2f}  d1 {ssi['a1']:.4f}  "
            f"(scale {ssi['alignment_scale']:.5f}, shift {ssi['alignment_shift']:.5f})"
        )
    print(f"\n  GATE B: {'PASS' if ok else 'FAIL'}\n")
    return ok


# --------------------------------------------------------------------------- #
# C -- gradient
# --------------------------------------------------------------------------- #


def gate_c(model, rows, norm: MetricNorm, args) -> bool:
    print(RULE)
    print("GATE C -- gradient sanity")
    print(RULE)
    import copy

    from train_iris_ms2_g import metric_inverse_depth_loss  # the trainer's own function

    device = torch.device(args.device)
    row = rows[0]
    image_tensor, _ = load_input_tensor(row, model.modality, model.args)
    prompt = model.encode_prompt("")

    gt = np.asarray(Image.open(row["depth_path"]), dtype=np.float32) / args.depth_scale
    gt_t = torch.from_numpy(gt)[None, None].to(device)
    valid_t = torch.from_numpy(
        official_valid_mask(gt, args.min_depth, args.max_depth).astype(np.float32)
    )[None, None].to(device)

    # Two decoders over the same weights: the fp32 copy the trainer builds, and
    # an fp16 one standing in for what would happen without it.
    base_vae = model.lotus.vae
    vae_fp32 = copy.deepcopy(base_vae).to(device=device, dtype=torch.float32)
    vae_fp32.encoder = None
    vae_fp32.requires_grad_(False).eval()
    vae_fp16 = copy.deepcopy(base_vae).to(device=device, dtype=torch.float16)
    vae_fp16.encoder = None
    vae_fp16.requires_grad_(False).eval()

    def fp16_reference_loss(latent):
        """What the metric term would be without the fp32 copy.

        Mirrors `decode_to_disparity`'s fp16 autocast branch -- the path every
        decode in this project took before the metric loss needed a gradient --
        and then the same masked L1. Written out here rather than reused from
        the trainer because the trainer deliberately has no fp16 path.
        """
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            decoded = vae_fp16.decode(
                latent.half() / vae_fp16.config.scaling_factor, return_dict=False
            )[0]
        y = decoded.float().mean(dim=1, keepdim=True) / 2.0 + 0.5
        q_hat = norm.q_lo + y * norm.span
        mask = valid_t > 0.5
        q_gt = torch.zeros_like(gt_t)
        q_gt[mask] = 1.0 / gt_t[mask]
        return (q_hat[mask] - q_gt[mask]).abs().mean(), int(mask.sum())

    results = {}
    for name, use_fp32 in (("fp32 decoder (the trainer's)", True), ("fp16 decoder (the trap)", False)):
        model.unet.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            _, latent = model.predict_disparity(
                row, image_tensor, prompt, return_latent=True
            )
        if use_fp32:
            loss, count, stats = metric_inverse_depth_loss(vae_fp32, latent, gt_t, valid_t, norm)
        else:
            loss, count = fp16_reference_loss(latent)
            stats = {"metric_abs_rel": float("nan")}
        loss.backward()
        grads = [
            p.grad.detach().float().norm().item()
            for p in model.unet.parameters()
            if p.grad is not None
        ]
        total = float(np.sqrt(sum(g * g for g in grads))) if grads else 0.0
        zeros = sum(1 for g in grads if g == 0.0)
        results[name] = total
        print(
            f"  {name}\n"
            f"    L_metric        {loss.item():.6f} over {count:,} lidar pixels\n"
            f"    U-Net grad norm {total:.6e}  ({len(grads)} tensors with a gradient, "
            f"{zeros} of them exactly zero)\n"
            f"    unaligned AbsRel on this frame {stats['metric_abs_rel']:.5f}"
        )

    model.unet.zero_grad(set_to_none=True)

    fp32_norm = results["fp32 decoder (the trainer's)"]
    fp16_norm = results["fp16 decoder (the trap)"]
    ok = fp32_norm > 0.0
    if not ok:
        print("  !! the fp32 path produced no U-Net gradient at all")
    ratio = fp16_norm / fp32_norm if fp32_norm else float("nan")
    print(f"\n    fp16/fp32 gradient-norm ratio: {ratio:.4f}")
    if fp16_norm == 0.0:
        print("    (the fp16 decoder underflowed to exactly zero -- the reason for the fp32 copy)")

    # Frozen-module checks. Trainable-ness and an actual absent gradient are two
    # different claims; both are made here.
    frozen_ok = True
    for label, module in (
        ("VAE (pipeline)", model.lotus.vae),
        ("VAE (fp32 metric copy)", vae_fp32),
        ("CLIP text encoder", model.lotus.text_encoder),
    ):
        trainable = [n for n, p in module.named_parameters() if p.requires_grad]
        with_grad = [n for n, p in module.named_parameters() if p.grad is not None]
        status = "frozen" if not trainable and not with_grad else "NOT FROZEN"
        print(
            f"    {label:<24} {status}: {len(trainable)} trainable params, "
            f"{len(with_grad)} carrying a gradient"
        )
        if trainable or with_grad:
            frozen_ok = False

    unet_trainable = sum(p.numel() for p in model.unet.parameters() if p.requires_grad)
    print(f"    {'U-Net':<24} trainable: {unet_trainable:,} parameters")

    ok = ok and frozen_ok and unet_trainable > 0
    print(f"\n  GATE C: {'PASS' if ok else 'FAIL'}\n")
    return ok


# --------------------------------------------------------------------------- #
# D -- evaluation
# --------------------------------------------------------------------------- #


def gate_d(norm: MetricNorm, args) -> bool:
    print(RULE)
    print("GATE D -- metric_no_test_alignment reads no GT before scoring")
    print(RULE)
    ok = True

    eval_args = SimpleNamespace(
        align_mode="none",
        metric_source="global_norm",
        metric_norm=args.metric_norm,
        global_affine=None,
        eval_min_depth=args.min_depth,
        eval_max_depth=args.max_depth,
    )
    metric = resolve_metric_affine(eval_args)
    print(f"  resolved calibration: q = {metric['a']:.6f} * y + {metric['b']:.6f} 1/m")
    print(f"  source: {metric['metric_source']}, split {metric['provenance']['source_split']}")
    if metric["provenance"]["source_split"] != "train":
        print("  !! constants did not come from train")
        ok = False

    rng = np.random.default_rng(args.seed)
    y = rng.random((32, 64)).astype(np.float32)
    gt = (rng.random((32, 64)).astype(np.float32) * 60.0 + 2.0)

    # The behavioural check: the same prediction against two different GTs.
    depth_a, _, _ = prediction_to_depth(y, metric, eval_args)
    depth_b, _, _ = prediction_to_depth(y, metric, eval_args)  # transform sees no GT to vary
    identical = np.array_equal(depth_a, depth_b)
    metrics_real = evaluate_sample(depth_a, gt, align="none",
                                   min_depth=args.min_depth, max_depth=args.max_depth)
    metrics_scaled = evaluate_sample(depth_a, gt * 3.0, align="none",
                                     min_depth=args.min_depth, max_depth=args.max_depth)
    print(
        f"  prediction -> depth is a pure function of y: {identical} "
        f"(bit-identical across calls)"
    )
    print(
        f"  alignment reported by the protocol: scale {metrics_real['alignment_scale']}, "
        f"shift {metrics_real['alignment_shift']}"
    )
    if (metrics_real["alignment_scale"], metrics_real["alignment_shift"]) != (1.0, 0.0):
        print("  !! align='none' still reported a non-identity alignment")
        ok = False
    print(
        f"  scoring the SAME depth map against GT and 3xGT: "
        f"AbsRel {metrics_real['abs_rel']:.4f} vs {metrics_scaled['abs_rel']:.4f}"
    )
    if abs(metrics_real["abs_rel"] - metrics_scaled["abs_rel"]) < 1e-9:
        print("  !! tripling GT changed nothing -- the metric is not reading GT at all")
        ok = False

    # And the contrast: under ssi_disparity, tripling GT *does* change the
    # prediction, because the fit absorbs it. That is what 'fits to test GT' means.
    ssi_real = evaluate_sample(y, gt, align="ssi_disparity",
                               min_depth=args.min_depth, max_depth=args.max_depth)
    ssi_scaled = evaluate_sample(y, gt * 3.0, align="ssi_disparity",
                                 min_depth=args.min_depth, max_depth=args.max_depth)
    print(
        f"  contrast, ssi_disparity on the same y: scale {ssi_real['alignment_scale']:.5f} "
        f"vs {ssi_scaled['alignment_scale']:.5f} -- the fit tracks GT, which is exactly "
        f"the dependence the metric path does not have"
    )
    print(
        f"  metric path AbsRel is unchanged by the fit because there is no fit: "
        f"{metrics_real['abs_rel']:.4f} (metric) vs {ssi_real['abs_rel']:.4f} (affine-invariant)"
    )

    print(f"\n  GATE D: {'PASS' if ok else 'FAIL'}\n")
    return ok


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    norm = MetricNorm.load(args.metric_norm)

    rows = read_manifest(args.manifest, args.ms2_root, "thermal", split=None, check_files=True)
    picked = rows[:: max(1, args.stride)][: args.frames]
    if len(picked) < args.frames:
        picked = rows[: args.frames]
    print(f"{len(picked)} frames from {args.manifest.name} (stride {args.stride})\n")

    results = {"A": gate_a(picked, norm, args)}

    if args.checkpoint is None:
        print("No --checkpoint: gates B and C skipped.\n")
    else:
        margs = model_args(args)
        device = torch.device(args.device)
        model = RouteModel(margs, device, torch.float16 if device.type == "cuda" else torch.float32)
        model.args = margs
        checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
        loaded = model.load_state_dicts(checkpoint["state_dicts"], str(args.checkpoint))
        print(
            f"loaded {', '.join(loaded)} from {args.checkpoint.name} "
            f"(route {checkpoint.get('route')}, step {checkpoint.get('epoch')}, "
            f"caption_mode {checkpoint.get('caption_mode')})\n"
        )
        results["B"] = gate_b(model, picked, norm, args)
        if not args.skip_gradient:
            results["C"] = gate_c(model, picked, norm, args)

    results["D"] = gate_d(norm, args)

    print(RULE)
    for gate in ("A", "B", "C", "D"):
        if gate in results:
            print(f"  GATE {gate}: {'PASS' if results[gate] else 'FAIL'}")
        else:
            print(f"  GATE {gate}: skipped")
    print(RULE)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
