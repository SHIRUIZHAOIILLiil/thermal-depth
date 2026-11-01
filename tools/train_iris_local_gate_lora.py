"""Continue matched local gates with bounded, native-Iris K/V LoRA adaptation.

Independent experiment: old trainers/checkpoints are read-only. Pilot official
SSI/none evaluation, not the final-paper unified evaluation protocol.
"""
import argparse
from contextlib import contextmanager
import copy
import csv
import hashlib
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import train_iris_local_gate as g

NAME = "iris_local_gate_lora_kv_r4_v1"
STEPS = (0, 500, 1000, 2000, 4000)
PARENTS = {"frame": 2000, "empty": 6000}
RANK = ALPHA = 4
EPSILON = math.log(1.05)
LORA_LR, GATE_LR = 1e-5, 3e-5
TARGETS = tuple(f"up_blocks.{b}.attentions.{a}.transformer_blocks.0.attn2.to_{kv}"
                for b in (2, 3) for a in range(3) for kv in ("k", "v"))


class LoRALinear(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(RANK, base.in_features, device=base.weight.device,
                                          dtype=base.weight.dtype))
        self.b = nn.Parameter(torch.zeros(base.out_features, RANK, device=base.weight.device,
                                          dtype=base.weight.dtype))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.enabled = True

    def forward(self, x):
        result = self.base(x)
        if self.enabled:
            result = result + (x @ self.a.T @ self.b.T) * (ALPHA / RANK)
        return result


def inject(unet, strict=True):
    unet.requires_grad_(False).eval()
    targets = {}
    for name in TARGETS:
        layer = unet.get_submodule(name)
        expected_out = 640 if name.startswith("up_blocks.2.") else 320
        if not isinstance(layer, nn.Linear) or (strict and
                (layer.in_features != 1024 or layer.out_features != expected_out)):
            raise ValueError(f"Unexpected LoRA target: {name}")
        targets[name] = layer
    for name, layer in targets.items():
        parent, child = name.rsplit(".", 1)
        setattr(unet.get_submodule(parent), child, LoRALinear(layer))
    return {name: unet.get_submodule(name) for name in TARGETS}


@contextmanager
def disabled(layers):
    previous = [m.enabled for m in layers.values()]
    try:
        for m in layers.values():
            m.enabled = False
        yield
    finally:
        for m, value in zip(layers.values(), previous):
            m.enabled = value


def lora_state(layers):
    return {f"{name}.{key}": getattr(layer, key).detach().cpu().clone()
            for name, layer in layers.items() for key in ("a", "b")}


def load_lora(layers, state):
    if set(state) != {f"{n}.{k}" for n in layers for k in ("a", "b")}:
        raise ValueError("Wrong LoRA state keys")
    with torch.no_grad():
        for name, layer in layers.items():
            for key in ("a", "b"):
                value, dest = state[f"{name}.{key}"], getattr(layer, key)
                if value.shape != dest.shape or not torch.isfinite(value).all():
                    raise ValueError("Invalid LoRA tensor")
                dest.copy_(value)


def digest_tensors(tensors):
    digest = hashlib.sha256()
    for name, value in tensors:
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def bounded_prediction(base, r0, candidate):
    residual = r0 + EPSILON * torch.tanh((candidate-r0) / EPSILON)
    pred = base * residual.exp()
    if not torch.isfinite(pred).all() or not (pred > 0).all():
        raise ValueError("Invalid bounded prediction")
    return pred


def lr_factor(step):
    if not 1 <= step <= STEPS[-1]:
        raise ValueError("Invalid optimizer step")
    if step <= 100:
        return step / 100
    phase = (step-100) / (STEPS[-1]-100)
    return .1 + .9 * .5 * (1 + math.cos(math.pi * phase))


def select(curve):
    zero = curve[0]["ssi"]
    eligible = [s for s in STEPS if s == 0 or
                (curve[s]["ssi"]["rmse"] <= zero["rmse"] and
                 curve[s]["ssi"]["a1"] >= zero["a1"])]
    return min(eligible, key=lambda s: (curve[s]["ssi"]["abs_rel"], s))


class Adaptation:
    def __init__(self, inputs, gate, layers):
        self.inputs, self.gate, self.layers = inputs, gate, layers
        self.anchor_gate = copy.deepcopy(gate).requires_grad_(False).eval()

    @torch.no_grad()
    def anchor(self, data, i):
        with disabled(self.layers):
            base, visual, delta, gt = self.inputs.get(data, i)
        r0 = self.anchor_gate(visual, base.shape) * delta
        return base, visual, r0, gt

    def predict(self, data, i, anchor):
        base, visual, r0, _ = anchor
        sample = data.samples[i]
        batch = default_collate([data[i]])
        with torch.no_grad():
            text = (self.inputs.empty if self.inputs.arm == "empty" else
                    self.inputs.iris.encode_prompt(g.prompt_for("frame", sample)))
        # No no_grad/detach here: frozen FP32 VAE decode must carry the gradient.
        raw = self.inputs.iris.predict_disparity(
            {"manifest_index": sample.manifest_line-1}, batch["iris_image"], text)
        delta = g.shape_delta(raw, base)
        return bounded_prediction(base, r0, self.gate(visual, base.shape) * delta)

    def state(self):
        return dict(gate={k:v.detach().cpu().clone() for k,v in self.gate.state_dict().items()},
                    lora=lora_state(self.layers))

    def restore(self, state):
        self.gate.load_state_dict(state["gate"], strict=True)
        load_lora(self.layers, state["lora"])

    def frozen_hash(self):
        trainable = {id(p) for m in self.layers.values() for p in (m.a, m.b)}
        tensors = []
        for prefix, module in (("base", self.inputs.base), ("iris", self.inputs.iris.unet),
                               ("vae", self.inputs.iris.gt_vae),
                               ("condition_vae", self.inputs.iris.lotus.vae),
                               ("clip", self.inputs.iris.lotus.text_encoder),
                               ("anchor_gate", self.anchor_gate)):
            for name, p in module.named_parameters():
                if id(p) not in trainable:
                    if p.requires_grad or p.grad is not None:
                        raise ValueError(f"Frozen parameter acquired gradient: {prefix}.{name}")
                    tensors.append((prefix+"."+name, p))
        return digest_tensors(tensors)


@torch.no_grad()
def evaluate(model, data, indices, states, path):
    totals = {s:{m:np.zeros(len(g.old.OFFICIAL_METRICS)) for m in ("ssi", "none")} for s in states}
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "step", "alignment", *g.old.OFFICIAL_METRICS])
        for count, i in enumerate(indices, 1):
            anchor = model.anchor(data, i)
            gt = anchor[3].cpu().numpy()
            for step, state in states.items():
                if step == 0:
                    pred = anchor[0] * anchor[2].exp()
                else:
                    model.restore(state)
                    pred = model.predict(data, i, anchor)
                for mode in ("ssi", "none"):
                    row = g.old.evaluate_sample(pred.cpu().numpy(), gt, align=mode, min_depth=.001, max_depth=80.)
                    values = [row[k] for k in g.old.OFFICIAL_METRICS]
                    if not np.isfinite(values).all():
                        raise ValueError("Nonfinite evaluation")
                    totals[step][mode] += values
                    writer.writerow([data.samples[i].sample_id, step, mode, *values])
            if count % 500 == 0:
                print(f"[val] {path.name} {count}/{len(indices)}", flush=True)
    os.replace(tmp, path)
    return {s:{m:dict(zip(g.old.OFFICIAL_METRICS, total/len(indices))) for m,total in modes.items()}
            for s,modes in totals.items()}


def parent_recipe(args):
    parents, common = {}, None
    for arm, step in PARENTS.items():
        folder = args.parents / arm
        summary = g.old.read(folder/"summary.json")
        contract = g.old.read(folder/"contract.json")
        ckpath = folder/f"step{step:06d}.pt"
        ck = torch.load(ckpath, map_location="cpu", weights_only=True)
        recipe = summary["common"]
        if (summary["arm"] != arm or summary["selected_step"] != step or summary["samples"] != 14249 or
                recipe["experiment"] != g.NAME or contract != dict(arm=arm, **recipe) or
                ck["arm"] != arm or ck["step"] != step or ck["common"] != recipe):
            raise ValueError(f"Wrong selected parent: {arm}")
        if common is not None and common != recipe:
            raise ValueError("Unmatched parent recipes")
        common = recipe
        parents[arm] = dict(step=step, checkpoint=str(ckpath), sha256=g.old.sha256_file(ckpath),
                            summary_sha256=g.old.sha256_file(folder/"summary.json"))
    for name, digest in common["code"].items():
        if g.old.sha256_file(ROOT/name) != digest:
            raise ValueError(f"Changed parent implementation: {name}")
    return parents, common


def setup(args):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA required")
    os.environ.update(IRIS_THERMAL_STRETCH="percentile", IRIS_DECODE_CLAMP="0", IRIS_TEXT_PADDING="legacy")
    parents, parent_common = parent_recipe(args)
    contract = g.old.read(args.base/"empty_s43/contract.json")
    if (contract["experiment"] != g.old.EXPERIMENT or contract["arm"] != "empty" or
            parent_common["base_sha"] != g.old.sha256_file(args.base/"empty_s43/step006000.pt")):
        raise ValueError("Wrong common empty6k base")
    for name, digest in contract["code"].items():
        if g.old.sha256_file(ROOT/name) != digest:
            raise ValueError(f"Changed base implementation: {name}")
    sources = g.old.matched_sources(*(Path(contract["iris_sources"][a]["path"]) for a in PARENTS))
    if sources != parent_common["sources"] or sources != contract["iris_sources"]:
        raise ValueError("Changed matched Iris sources")
    train, ti = g.old.checked_manifest(args.train, args.data_root, "train", 75688)
    val, vi = g.old.checked_manifest(args.val, args.data_root, "val", 14249)
    if ti != parent_common["train"] or vi != parent_common["val"] or ti != contract["train"] or vi != contract["val"]:
        raise ValueError("Changed data manifests")
    inputs, tokenizer = g.load_inputs(args, contract, sources, parent_common["snapshot"])
    gate = g.LocalGate().cuda()
    ck = torch.load(parents[args.arm]["checkpoint"], map_location="cpu", weights_only=True)
    gate.load_state_dict(ck["gate"], strict=True)
    torch.manual_seed(43)
    layers = inject(inputs.iris.unet)
    model = Adaptation(inputs, gate, layers)
    def dataset(samples):
        return g.old.FusionData(samples, g.old.make_tokens(samples, tokenizer, None, None, "empty", None, False))
    common = dict(experiment=NAME, parents=parents, parent_common=parent_common, seed=43,
                  targets=list(TARGETS), rank=RANK, alpha=ALPHA, dropout=0, epsilon=EPSILON,
                  steps=list(STEPS), accum=g.ACCUM, lora_lr=LORA_LR, gate_lr=GATE_LR,
                  lora_weight_decay=0, gate_weight_decay=.01, warmup=100, final_lr_fraction=.1,
                  clip_grad=1., lora_init=digest_tensors(lora_state(layers).items()),
                  objective="per-frame official SSI AbsRel; real thermal-view LiDAR only",
                  selection="stride4 min AbsRel; RMSE<=own0 and a1>=own0; earlier tie; 0 always eligible",
                  protocol="pilot official_protocol; .001<GT<80; frame mean; ssi/none separate",
                  code={"tools/train_iris_local_gate_lora.py":g.old.sha256_file(Path(__file__))})
    return model, dataset(train), dataset(val), common


def save(path, payload):
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def check_checkpoint(ck, arm, common, step=None):
    if ck["arm"] != arm or ck["common"] != common or (step is not None and ck["step"] != step):
        raise ValueError("Checkpoint/arm/recipe mismatch")


def check_initial(model, data, i):
    anchor = model.anchor(data, i)
    pred = model.predict(data, i, anchor)
    expected = anchor[0] * anchor[2].exp()
    torch.testing.assert_close(pred.detach(), expected, rtol=1e-6, atol=1e-6)
    return anchor, pred


def verify_saved_zero(args, validation, measured):
    ids = {validation.samples[i].sample_id for i in (0, len(validation)-1)}
    with (args.parents/args.arm/"full.csv").open(newline="") as f:
        expected = {r["id"]:float(r["abs_rel"]) for r in csv.DictReader(f)
                    if r["id"] in ids and r["alignment"] == "ssi" and int(r["step"]) == PARENTS[args.arm]}
    if set(expected) != ids or any(not np.isclose(measured[k], expected[k], atol=1e-6, rtol=1e-4) for k in ids):
        raise ValueError("Initial output failed saved parent reproduction")


def run(args):
    started = time.monotonic()
    model, train, val, common = setup(args)
    folder = args.out/args.arm
    folder.mkdir(parents=True, exist_ok=True)
    g.mark_equal(folder/"contract.json", dict(arm=args.arm, **common))
    if args.mode == "train" and g.old.read(args.smoke/"passed.json") != common:
        raise ValueError("Matching paired smoke required")
    lora_params = [p for m in model.layers.values() for p in (m.a, m.b)]
    gate_params = list(model.gate.parameters())
    params = lora_params + gate_params
    opt = torch.optim.AdamW([dict(params=lora_params, lr=LORA_LR, weight_decay=0),
                            dict(params=gate_params, lr=GATE_LR, weight_decay=.01)])
    initial = model.state()
    start = 0
    if args.mode == "train" and (folder/"last.pt").exists():
        ck = torch.load(folder/"last.pt", map_location="cpu", weights_only=True)
        check_checkpoint(ck, args.arm, common)
        if not 0 <= ck["step"] <= STEPS[-1] or ck["step"] % 250:
            raise ValueError("Invalid resume cursor")
        model.restore(ck)
        opt.load_state_dict(ck["optimizer"])
        start = ck["step"]
    smoke = args.mode == "smoke"
    frozen_before = model.frozen_hash() if smoke else None
    if smoke:
        measured = {}
        with torch.no_grad():
            for i in (0, len(val)-1):
                anchor, pred = check_initial(model, val, i)
                measured[val.samples[i].sample_id] = g.old.evaluate_sample(
                    pred.cpu().numpy(), anchor[3].cpu().numpy(), align="ssi")["abs_rel"]
        verify_saved_zero(args, val, measured)
    stop = 2 if smoke else STEPS[-1]
    order = iter(g.ImageStream(len(train), 43, start*g.ACCUM, (stop-start)*g.ACCUM))
    trace = hashlib.sha256()
    evidence = []
    for step in range(start+1, stop+1):
        opt.zero_grad(set_to_none=True)
        for group, peak in zip(opt.param_groups, (LORA_LR, GATE_LR)):
            group["lr"] = peak * lr_factor(step)
        count, total = 0, 0.
        for _ in range(g.ACCUM):
            i = next(order)
            anchor = model.anchor(train, i)
            if smoke:
                trace.update(train.samples[i].sample_id.encode())
                for tensor in (anchor[0], anchor[1], anchor[3]):
                    trace.update(tensor.cpu().contiguous().numpy().tobytes())
                if step == 1:
                    pred = model.predict(train, i, anchor)
                    torch.testing.assert_close(pred.detach(), anchor[0]*anchor[2].exp(), rtol=1e-6, atol=1e-6)
                else:
                    pred = model.predict(train, i, anchor)
            else:
                pred = model.predict(train, i, anchor)
            loss = g.ssi_absrel(pred, anchor[3])
            if loss is not None:
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss")
                loss.backward()
                count += 1
                total += loss.item()
            del pred, loss, anchor
        if not count:
            raise ValueError("No valid LiDAR in update")
        for p in params:
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise ValueError("Missing/nonfinite trainable gradient")
            p.grad.div_(count)
        norms = {kv:sum(float(m.b.grad.double().square().sum()) for n,m in model.layers.items()
                        if n.endswith("to_"+kv))**.5 for kv in ("k", "v")}
        if smoke and (min(norms.values()) <= 0 or sum(float(p.grad.square().sum()) for p in gate_params) <= 0):
            raise ValueError("LoRA K/V or gate has zero gradient")
        norm = nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        opt.step()
        if smoke:
            evidence.append(norms)
        if step % 50 == 0 or step == stop:
            print(f"[train] {args.arm} step={step} loss={total/count:.8f} lr={opt.param_groups[0]['lr']:.3g} "
                  f"grad={float(norm):.5g} lora={norms}", flush=True)
        if not smoke and step % 250 == 0:
            payload = dict(**model.state(), arm=args.arm, common=common, step=step)
            if step in STEPS:
                save(folder/f"step{step:06d}.pt", payload)
            save(folder/"last.pt", dict(**payload, optimizer=opt.state_dict()))
    if smoke:
        if frozen_before != model.frozen_hash():
            raise ValueError("Frozen model parameters changed")
        state = model.state()
        if not any(not torch.equal(v, initial["lora"][k]) for k,v in state["lora"].items() if k.endswith(".b")):
            raise ValueError("LoRA update absent")
        if all(torch.equal(v, initial["gate"][k]) for k,v in state["gate"].items()):
            raise ValueError("Gate update absent")
        with torch.no_grad():
            anchor = model.anchor(val, 0)
            before = model.predict(val, 0, anchor)
            model.restore(initial)
            model.restore(state)
            torch.testing.assert_close(before, model.predict(val, 0, anchor), rtol=0, atol=0)
            ratio = before / (anchor[0]*anchor[2].exp())
            if float(ratio.min()) < 1/1.05-1e-6 or float(ratio.max()) > 1.05+1e-6:
                raise ValueError("Correction exceeded trust bound")
        g.old.json_write(folder/"smoke.json", dict(arm=args.arm, common=common, trace=trace.hexdigest(),
                         initial_reproduction=True, frozen_unchanged=True, gradients=evidence,
                         max_gpu_gb=torch.cuda.max_memory_allocated()/2**30))
        print(f"[smoke] {args.arm} PASS", flush=True)
        return
    states = {0:None}
    for step in STEPS[1:]:
        ck = torch.load(folder/f"step{step:06d}.pt", map_location="cpu", weights_only=True)
        check_checkpoint(ck, args.arm, common, step)
        states[step] = ck
    curve = evaluate(model, val, range(0,len(val),4), states, folder/"curve.csv")
    chosen = select(curve)
    full = evaluate(model, val, range(len(val)), {s:states[s] for s in sorted({0, chosen, 4000})}, folder/"full.csv")
    expected = g.old.read(args.parents/args.arm/"summary.json")["selected_full"]
    for mode in ("ssi", "none"):
        for key, value in full[0][mode].items():
            if not np.isclose(value, expected[mode][key], rtol=1e-4, atol=1e-6):
                raise ValueError(f"Full parent reproduction failed: {mode}/{key}")
    summary = dict(arm=args.arm, common=common, samples=len(val), curve=curve, selected_step=chosen,
                   selected_full=full[chosen], fixed4000=full[4000], baseline=full[0])
    g.old.json_write(folder/"summary.json", summary)
    g.old.json_write(folder/"completed.json", dict(step=4000, seconds=time.monotonic()-started,
                     max_gpu_gb=torch.cuda.max_memory_allocated()/2**30))
    print(f"[selected] {args.arm} new_step={chosen} {full[chosen]['ssi']}", flush=True)


def pair_smoke(out):
    a, b = [g.old.read(out/arm/"smoke.json") for arm in PARENTS]
    if (a["arm"], b["arm"]) != ("frame", "empty") or a["common"] != b["common"] or a["trace"] != b["trace"]:
        raise ValueError("Paired recipe / common inputs mismatch")
    if not all(x["initial_reproduction"] and x["frozen_unchanged"] for x in (a,b)):
        raise ValueError("Incomplete smoke")
    g.old.json_write(out/"passed.json", a["common"])
    print("[paired-smoke] PASS")


def report(out):
    rows = [g.old.read(out/arm/"summary.json") for arm in PARENTS]
    if rows[0]["common"] != rows[1]["common"] or [r["arm"] for r in rows] != list(PARENTS):
        raise ValueError("Unpaired results")
    for key in ("baseline", "fixed4000", "selected_full"):
        print(f"=== {key} ===")
        for r in rows:
            print(r["arm"], "selected", r["selected_step"], r[key]["ssi"])
        print("[frame-minus-empty]", {k:rows[0][key]["ssi"][k]-rows[1][key]["ssi"][k]
                                     for k in rows[0][key]["ssi"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("smoke", "train", "pair-smoke", "report"):
        p = sub.add_parser(mode)
        p.add_argument("--out", type=Path, required=True)
        if mode in ("smoke", "train"):
            p.add_argument("--arm", choices=list(PARENTS), required=True)
            for name in ("base", "parents", "assets", "train", "val", "data-root"):
                p.add_argument("--"+name, type=Path, required=True)
        if mode == "train":
            p.add_argument("--smoke", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "pair-smoke":
        pair_smoke(args.out)
    elif args.mode == "report":
        report(args.out)
    else:
        run(args)
