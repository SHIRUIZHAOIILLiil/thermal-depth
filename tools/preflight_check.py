"""Verify a machine can run the route suite before a long job is queued.

Everything here is something that has silently broken a run before, or that
would only surface hours into a job on a new cluster: a missing weight shard, a
diffusers version that renamed an argument, a manifest whose paths do not resolve
under the new root, an optimizer state that will not round-trip through
torch.save.  Each check is cheap; the whole script is a couple of minutes.

The resume check is the important one. Cluster walltimes are shorter than the
longer routes, so `--resume` is load-bearing, and it had never actually been
exercised.

    python tools/preflight_check.py --route b_thermal_unet \\
        --ms2-root /scratch/$USER/ms2 \\
        --train-manifest /scratch/$USER/manifests/ms2_train_day2seq_20260725.jsonl \\
        --val-manifest   /scratch/$USER/manifests/..._val_..._clip75_....jsonl \\
        --local-files-only

Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import traceback

import torch

ROOT = Path(__file__).resolve().parents[1]
LOTUS_ROOT = ROOT / "lotus"
for path in (ROOT, LOTUS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from train_route_suite import (  # noqa: E402
    DEFAULT_TRAIN_MANIFEST,
    DEFAULT_VAL_MANIFEST,
    ROUTES,
    RouteModel,
    environment_fingerprint,
    load_sample,
    masked_ssi_l1,
    read_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", default="b_thermal_unet", choices=sorted(ROUTES))
    parser.add_argument("--ms2-root", type=Path, default=Path("/mnt/e/dataset/ms2"))
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frozen-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--sample-files", type=int, default=200, help="How many manifest rows to stat.")
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


class Checks:
    def __init__(self) -> None:
        self.results: list[dict] = []

    def run(self, name: str, function):
        print(f"[....] {name}", flush=True)
        try:
            detail = function()
            self.results.append({"check": name, "ok": True, "detail": detail})
            print(f"[ OK ] {name}: {detail}", flush=True)
            return True
        except Exception as error:  # noqa: BLE001 - a preflight reports, it does not raise
            self.results.append({"check": name, "ok": False, "detail": f"{type(error).__name__}: {error}"})
            print(f"[FAIL] {name}: {type(error).__name__}: {error}", flush=True)
            traceback.print_exc(limit=3)
            return False


def main() -> None:
    args = parse_args()
    # fields RouteModel and the loaders expect but preflight does not vary
    args.gt_decode_fp32 = True
    args.input_max_edge = 0
    args.gt_min_depth, args.gt_max_depth, args.depth_scale = 0.1, 80.0, 256.0
    args.seed, args.timestep = 20260703, 999

    device = torch.device(args.device)
    frozen_dtype = torch.float16 if args.frozen_dtype == "fp16" else torch.float32
    modality = ROUTES[args.route][0]
    checks = Checks()
    state: dict = {}

    def check_environment():
        fingerprint = environment_fingerprint(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda but torch.cuda.is_available() is False")
        return json.dumps(fingerprint)

    def check_manifests():
        rows = read_manifest(args.train_manifest, args.ms2_root, modality, split="train", check_files=False)
        val_rows = read_manifest(args.val_manifest, args.ms2_root, modality, split=None, check_files=False)
        state["rows"] = rows
        state["val_rows"] = val_rows
        return f"train {len(rows)} rows, val {len(val_rows)} rows"

    def check_files_resolve():
        rows = state["rows"]
        step = max(1, len(rows) // args.sample_files)
        sampled = rows[::step][: args.sample_files]
        missing = [
            str(path)
            for row in sampled
            for path in (row["image_path"], row["depth_path"])
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"{len(missing)} of {2 * len(sampled)} sampled files absent; first: {missing[:2]}")
        return f"{2 * len(sampled)} sampled paths all resolve under {args.ms2_root}"

    def check_weights():
        model = RouteModel(args, device, frozen_dtype)
        state["model"] = model
        audit = {
            name: int(sum(p.numel() for p in module.parameters()))
            for name, module in model.trainable_modules().items()
        }
        return f"loaded; trainable {audit}"

    def check_forward_backward():
        model = state["model"]
        row = state["rows"][0]
        image_tensor, gt_disparity, valid_mask, _ = load_sample(row, modality, args)
        prompt = model.encode_prompt("")
        prediction = model.predict_disparity(row, image_tensor, prompt)
        loss, abs_rel, _ = masked_ssi_l1(
            prediction[None], gt_disparity.to(device), valid_mask.to(device)
        )
        loss.backward()
        grads = [
            p.grad.norm().item()
            for module in model.trainable_modules().values()
            for p in module.parameters()
            if p.grad is not None
        ]
        if not grads:
            raise RuntimeError("no gradients reached the trainable modules")
        if all(g == 0 for g in grads):
            raise RuntimeError("every gradient is exactly zero (fp16 underflow?)")
        for module in model.trainable_modules().values():
            module.zero_grad(set_to_none=True)
        return f"loss {float(loss):.5f}, abs_rel {float(abs_rel):.4f}, grad tensors {len(grads)}, max {max(grads):.3g}"

    def check_checkpoint_roundtrip():
        model = state["model"]
        modules = model.trainable_modules()
        optimizer = torch.optim.AdamW(
            [{"params": module.parameters(), "lr": 1e-4} for module in modules.values()]
        )
        # one real step so the optimizer carries non-trivial moments
        for module in modules.values():
            for parameter in module.parameters():
                parameter.grad = torch.randn_like(parameter) * 1e-3
        optimizer.step()
        before = {
            name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
            for name, module in modules.items()
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roundtrip.pt"
            torch.save(
                {
                    "format": "preflight",
                    "route": args.route,
                    "state_dicts": before,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
                path,
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
        for name, module in modules.items():
            module.load_state_dict(payload["state_dicts"][name], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        torch.set_rng_state(payload["torch_rng_state"])
        mismatched = [
            f"{name}.{key}"
            for name, module in modules.items()
            for key, value in module.state_dict().items()
            if not torch.equal(value.detach().cpu(), before[name][key])
        ]
        if mismatched:
            raise RuntimeError(f"{len(mismatched)} tensors changed across save/load; first: {mismatched[:2]}")
        moments = sum(1 for group in optimizer.state.values() for key in group if key in ("exp_avg", "exp_avg_sq"))
        return f"weights bit-identical; optimizer restored with {moments} moment tensors"

    order = [
        ("environment", check_environment),
        ("manifests parse", check_manifests),
        ("data paths resolve", check_files_resolve),
        ("model weights load", check_weights),
        ("forward + backward", check_forward_backward),
        ("checkpoint round-trip", check_checkpoint_roundtrip),
    ]
    ok = True
    for name, function in order:
        if not checks.run(name, function):
            ok = False
            break  # later checks depend on earlier ones

    print("\n" + "=" * 60)
    for entry in checks.results:
        print(f"{'PASS' if entry['ok'] else 'FAIL'}  {entry['check']}")
    print("=" * 60)
    print("READY" if ok else "NOT READY -- fix the failure above before queueing a job")

    if args.report:
        args.report.write_text(json.dumps(checks.results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report -> {args.report}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
