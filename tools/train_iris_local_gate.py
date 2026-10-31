"""Matched caption / no-caption training with a bounded, visual-only local gate."""
import argparse
import csv
import hashlib
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import iris_text_correction as old
from tools.da2_caption_data import ImageStream
from tools.train_da2_iris_fusion_v2 import tensor_hash

NAME = "iris_visual_gate_matched_v1"
STEPS = (0, 1000, 2000, 4000, 6000)
ACCUM = 12
W_MAX = .10
W_INIT = .005
LR = 3e-4


class LocalGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(128, 16, 1), nn.GroupNorm(4, 16), nn.GELU(),
                                 nn.Conv2d(16, 1, 3, padding=1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, math.log(W_INIT / (W_MAX-W_INIT)))

    def forward(self, visual, hw):
        h, w = hw
        padded = (h + (-h) % 14, w + (-w) % 14)
        logits = F.interpolate(self.net(visual.float()), size=padded, mode="bilinear", align_corners=False)
        return W_MAX * logits[0, 0, :h, :w].sigmoid()


def shape_delta(raw, base):
    if raw.shape != base.shape or not torch.isfinite(raw).all():
        raise ValueError("Invalid native Iris depth map")
    if not torch.isfinite(base).all() or not (base > 0).all():
        raise ValueError("Invalid native metric base")
    def standardize(x):
        return (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-6)
    return standardize(raw.float()) - standardize(base.float().log())


def estimate(gate, base, visual, delta):
    if gate is None:  # Strict step-zero candidate, independent of the sigmoid initialization.
        return base
    pred = base * (gate(visual, base.shape) * delta).exp()
    if not torch.isfinite(pred).all() or (pred <= 0).any():
        raise ValueError("Invalid corrected depth")
    return pred


def ssi_absrel(pred, gt):
    valid = gt.isfinite() & (gt > .001) & (gt < 80.)
    if not valid.any():
        return None
    p, g = pred[valid].double(), gt[valid].double()
    dp, dg = p - p.mean(), g - g.mean()
    denom = dp.square().sum()
    if denom <= 0:
        raise ValueError("Constant depth on valid pixels")
    aligned = g.mean() + dp * ((dp * dg).sum() / denom)
    return ((aligned.clamp(.001, 80.) - g).abs() / g).mean()


def prompt_for(arm, sample):
    if arm == "empty":
        return ""
    if arm == "frame" and sample.caption:
        return sample.caption
    raise ValueError("Expected frame caption or explicit empty arm")


class FrozenInputs:
    def __init__(self, base, iris, arm):
        self.base, self.iris, self.arm = base, iris, arm
        self.empty = iris.encode_prompt("") if arm == "empty" else None
        # Input of the second spatial fusion, BEFORE the empty-Iris residual.
        self.visual = None
        self.handle = base.fusions[1].register_forward_pre_hook(self.capture)

    def capture(self, module, inputs):
        self.visual = inputs[0].detach()

    @torch.no_grad()
    def get(self, data, i):
        s = data.samples[i]
        batch = default_collate([data[i]])
        self.visual = None
        base = old.forward_batch(self.base, batch)[0].float()
        if self.visual is None or self.visual.shape[1] != 128:
            raise ValueError("Missing frozen DPT visual feature")
        visual = self.visual
        self.visual = None
        text = self.empty if self.arm == "empty" else self.iris.encode_prompt(prompt_for(self.arm, s))
        raw = self.iris.predict_disparity({"manifest_index": s.manifest_line-1}, batch["iris_image"], text)
        return base, visual, shape_delta(raw, base), batch["gt"][0].to(base.device)


def load_inputs(args, contract, sources, snapshot):
    from tools.train_route_suite import RouteModel
    from transformers import CLIPTokenizer
    assets = old.verify_assets(args.assets)
    iris_assets = old.verify_iris_assets(args.base / "iris_assets.json")
    for path, key in ((args.assets, "assets_sha256"), (args.base/"iris_assets.json", "iris_assets_sha256"),
                      (Path(contract["base_path"]), "base_sha256")):
        if old.sha256_file(path) != contract[key]:
            raise ValueError(f"Changed base asset: {path}")
    state = torch.load(contract["base_path"], map_location="cpu", weights_only=False)
    old.validate_state(state, state["contract"], 18000)
    old.check_source(state["contract"])
    core = old.build_model(assets, "no-caption", 43)
    core.load_state_dict(state["model"], strict=True)
    del state
    base = old.FusionDepth(core.core, old.load_iris(iris_assets["path"], sources["empty"]["path"], "empty"))
    if old.restore(args.base/"empty_s43/step006000.pt", base, contract) != 6000:
        raise ValueError("Wrong frozen gate base")
    base.requires_grad_(False).cuda().eval()
    tokenizer = CLIPTokenizer.from_pretrained(iris_assets["path"], subfolder="tokenizer", local_files_only=True)
    recipe = SimpleNamespace(route="b_thermal_unet", backbone="g", lotus_model_path=snapshot, local_files_only=True,
                             gt_decode_fp32=True, condition_latent="mode", timestep=999, num_inference_steps=1,
                             seed=20260703)
    iris = RouteModel(recipe, torch.device("cuda"), torch.float16)
    source = sources[args.arm]["path"]
    state = torch.load(source, map_location="cpu", weights_only=False)
    old.validate_iris_payload(state, args.arm)
    iris.load_state_dicts(state["state_dicts"], source)
    iris.unet.requires_grad_(False).eval()
    return FrozenInputs(base, iris, args.arm), tokenizer


@torch.no_grad()
def evaluate(inputs, data, indices, gates, csv_path):
    totals = {s: {m: np.zeros(len(old.OFFICIAL_METRICS)) for m in ("ssi", "none")} for s in gates}
    tmp = csv_path.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "step", "alignment", *old.OFFICIAL_METRICS])
        for i in indices:
            base, visual, delta, gt = inputs.get(data, i)
            gt = gt.cpu().numpy()
            for s, gate in gates.items():
                pred = estimate(gate, base, visual, delta).cpu().numpy()
                for mode, total in totals[s].items():
                    row = old.evaluate_sample(pred, gt, align=mode, min_depth=.001, max_depth=80.)
                    values = [row[k] for k in old.OFFICIAL_METRICS]
                    if not np.isfinite(values).all():
                        raise ValueError("Nonfinite validation metric")
                    total += values
                    writer.writerow([data.samples[i].sample_id, s, mode, *values])
    os.replace(tmp, csv_path)
    return {s: {m: dict(zip(old.OFFICIAL_METRICS, total/len(indices))) for m, total in modes.items()}
            for s, modes in totals.items()}


def select(curve):
    return min(STEPS, key=lambda s: (curve[s]["ssi"]["abs_rel"], s))


def mark_equal(path, value):
    if path.exists() and old.read(path) != value:
        raise ValueError(f"Configuration changed: {path}")
    old.json_write(path, value)


def run(args):
    from huggingface_hub import snapshot_download
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requires BF16 CUDA")
    os.environ.update(IRIS_THERMAL_STRETCH="percentile", IRIS_DECODE_CLAMP="0", IRIS_TEXT_PADDING="legacy")
    previous = args.base / "empty_s43"
    contract, summary = old.read(previous/"contract.json"), old.read(previous/"summary.json")
    if (contract["experiment"] != old.EXPERIMENT or contract["arm"] != "empty" or summary["contract"] != contract
            or summary["selected_step"] != 6000 or summary["selected_full"]["step"] != 6000
            or summary["selected_full"]["samples"] != 14249):
        raise ValueError("Expected matched empty 6k")
    for filename, digest in contract["code"].items():
        if old.sha256_file(ROOT/filename) != digest:
            raise ValueError(f"Changed fusion source: {filename}")
    sources = old.matched_sources(*(Path(contract["iris_sources"][a]["path"]) for a in ("frame", "empty")))
    if sources != contract["iris_sources"]:
        raise ValueError("Changed source weights")
    train, ti = old.checked_manifest(args.train, args.data_root, "train", 75688)
    val, vi = old.checked_manifest(args.val, args.data_root, "val", 14249)
    if ti != contract["train"] or vi != contract["val"]:
        raise ValueError("Changed manifests")
    snapshot = snapshot_download("jingheya/lotus-depth-g-v2-1-disparity", local_files_only=True)
    inputs, tokenizer = load_inputs(args, contract, sources, snapshot)
    def dataset(samples):
        return old.FusionData(samples, old.make_tokens(samples, tokenizer, None, None, "empty", None, False))
    training, validation = dataset(train), dataset(val)
    torch.manual_seed(43)
    gate = LocalGate().cuda()
    common = dict(experiment=NAME, train=ti, val=vi, base_sha=old.sha256_file(previous/"step006000.pt"),
                  sources=sources, snapshot=snapshot, seed=43, accum=ACCUM, steps=list(STEPS), w_max=W_MAX,
                  w_init=W_INIT, lr=LR, gate_init=tensor_hash(gate), objective="per-frame SSI AbsRel",
                  code={f:old.sha256_file(ROOT/f) for f in ("tools/train_iris_local_gate.py",
                        "tools/iris_text_correction.py", "tools/train_route_suite.py", "tools/train_ms2_joint_gt_v3.py",
                        "lotus/pipeline.py", "models/anythermal_lotus_v2_4.py", "models/anythermal_lotus_v2.py")})
    run_dir = args.out/args.arm
    run_dir.mkdir(parents=True, exist_ok=True)
    mark_equal(run_dir/"contract.json", dict(arm=args.arm, **common))
    if args.mode == "train" and old.read(args.smoke/"passed.json") != common:
        raise ValueError("Matching paired smoke required")
    opt = torch.optim.AdamW(gate.parameters(), lr=LR, weight_decay=.01)
    stop = 2 if args.mode == "smoke" else STEPS[-1]
    start = 0
    last = run_dir/"last.pt"
    if last.exists() and args.mode == "train":
        ck = torch.load(last, map_location="cpu", weights_only=True)
        if ck["arm"] != args.arm or ck["common"] != common:
            raise ValueError("Resume mismatch")
        gate.load_state_dict(ck["gate"])
        opt.load_state_dict(ck["optimizer"])
        start = ck["step"]
    order = iter(ImageStream(len(train), 43, start*ACCUM, (stop-start)*ACCUM))
    trace = hashlib.sha256()
    for step in range(start+1, stop+1):
        opt.zero_grad(set_to_none=True)
        valid_frames, loss_sum = 0, 0.
        for _ in range(ACCUM):
            i = next(order)
            base, visual, delta, gt = inputs.get(training, i)
            if args.mode == "smoke":
                trace.update(train[i].sample_id.encode())
                for x in (base, visual, gt):
                    trace.update(x.float().cpu().contiguous().numpy().tobytes())
            loss = ssi_absrel(estimate(gate, base, visual, delta), gt)
            if loss is not None:
                loss.backward()
                loss_sum += loss.item()
                valid_frames += 1
        if not valid_frames:
            raise ValueError("No training LiDAR in batch")
        for p in gate.parameters():
            p.grad.div_(valid_frames)
        norm = nn.utils.clip_grad_norm_(gate.parameters(), 1., error_if_nonfinite=True)
        if args.mode == "smoke" and norm.item() == 0:
            raise ValueError("Gate has no gradient")
        opt.step()
        if step % 50 == 0 or step == stop:
            print(f"[train] {args.arm} step={step} loss={loss_sum/valid_frames:.7f}", flush=True)
        if args.mode == "train" and step % 250 == 0:
            ck = dict(gate=gate.state_dict(), step=step, arm=args.arm, common=common)
            if step in STEPS:
                torch.save(ck, run_dir/"milestone.tmp")
                os.replace(run_dir/"milestone.tmp", run_dir/f"step{step:06d}.pt")
            torch.save(dict(**ck, optimizer=opt.state_dict()), run_dir/"last.tmp")
            os.replace(run_dir/"last.tmp", last)
    if args.mode == "smoke":
        if tensor_hash(gate) == common["gate_init"]:
            raise ValueError("Gate unchanged")
        smoke_path = run_dir/"smoke.csv"
        evaluate(inputs, validation, [0, len(val)-1], {0:None, stop:gate}, smoke_path)
        with smoke_path.open(newline="") as handle:
            measured = {r["id"]: float(r["abs_rel"]) for r in csv.DictReader(handle)
                        if r["step"] == "0" and r["alignment"] == "ssi"}
        with (previous/"selected_full/val_step006000_frames.csv").open(newline="") as handle:
            expected = {r["id"]: float(r["abs_rel"]) for r in csv.DictReader(handle)
                        if r["alignment"] == "ssi" and r["id"] in measured}
        if len(expected) != 2 or any(not np.isclose(v, expected[s], atol=5e-5, rtol=1e-4)
                                     for s, v in measured.items()):
            raise ValueError("Paired smoke did not reproduce the saved empty 6k validation frames")
        old.json_write(run_dir/"smoke.json", dict(common=common, trace=trace.hexdigest(), arm=args.arm))
        return
    gates = {0: None}
    for s in STEPS[1:]:
        ck = torch.load(run_dir/f"step{s:06d}.pt", map_location="cpu", weights_only=True)
        if (ck["arm"], ck["common"], ck["step"]) != (args.arm, common, s):
            raise ValueError("Curve checkpoint mismatch")
        gates[s] = LocalGate().cuda().eval()
        gates[s].load_state_dict(ck["gate"])
    curve = evaluate(inputs, validation, range(0,len(val),4), gates, run_dir/"curve.csv")
    chosen = select(curve)
    full = evaluate(inputs, validation, range(len(val)), {0:None, chosen:gates[chosen]}, run_dir/"full.csv")
    for mode in ("ssi","none"):
        for k, actual in full[0][mode].items():
            if not np.isclose(actual, summary["selected_full"]["metrics"][mode][k], rtol=1e-4, atol=1e-6):
                raise ValueError(f"Frozen baseline reproduction failed for {mode}/{k}")
    old.json_write(run_dir/"summary.json", dict(arm=args.arm, common=common, curve=curve, selected_step=chosen,
                   selected_full=full[chosen], baseline=full[0], samples=len(val)))
    print(f"[selected] {args.arm} step={chosen}", full[chosen]["ssi"], flush=True)


def pair_smoke(out):
    a, b = [old.read(out/arm/"smoke.json") for arm in ("frame", "empty")]
    if (a["arm"], b["arm"]) != ("frame","empty") or a["common"] != b["common"] or a["trace"] != b["trace"]:
        raise ValueError("Paired smoke mismatch in recipe, initialization, images, baseline features or GT")
    old.json_write(out/"passed.json", a["common"])
    print("[paired-smoke] PASS", flush=True)


def report(out):
    a, b = [old.read(out/arm/"summary.json") for arm in ("frame","empty")]
    if (a["arm"], b["arm"]) != ("frame","empty") or a["common"] != b["common"] or a["samples"] != b["samples"]:
        raise ValueError("Unpaired reports")
    for row in (a, b):
        print(row["arm"], row["selected_step"], row["selected_full"]["ssi"])
    print("zero", a["baseline"]["ssi"])
    return {k:a["selected_full"]["ssi"][k]-b["selected_full"]["ssi"][k] for k in old.OFFICIAL_METRICS}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("smoke","train","pair-smoke","report"):
        p = sub.add_parser(mode)
        p.add_argument("--out", type=Path, required=True)
        if mode in ("train","smoke"):
            p.add_argument("--arm", choices=("frame","empty"), required=True)
            for arg in ("base","assets","train","val","data-root"):
                p.add_argument("--"+arg, type=Path, required=True)
        if mode == "train":
            p.add_argument("--smoke", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "pair-smoke":
        pair_smoke(args.out)
    elif args.mode == "report":
        print("[frame-minus-empty]", report(args.out))
    else:
        run(args)
