"""Test validation-selected LoRA arms; step zero reuses the SAME arm's parent."""
import argparse
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import train_iris_local_gate_lora as train
from tools import test_iris_local_gate as shared

read, write, sha256_file = shared.read, shared.write, shared.sha256_file
PROTOCOL, TESTS = shared.PROTOCOL, shared.TESTS


def selected_step(arm, contract, summary):
    common = summary["common"]
    if (contract != dict(arm=arm, **common) or summary["arm"] != arm or
            common["experiment"] != train.NAME or summary["samples"] != 14249 or common["protocol"] != PROTOCOL):
        raise ValueError("Wrong LoRA arm/experiment/protocol")
    curve = {int(k):v for k,v in summary["curve"].items()}
    if set(curve) != set(train.STEPS) or not all(np.isfinite(v["ssi"][k])
            for v in curve.values() for k in ("abs_rel", "rmse", "a1")):
        raise ValueError("Incomplete or nonfinite selection curve")
    step = train.select(curve)
    if step != summary["selected_step"] or (step == 0 and summary["selected_full"] != summary["baseline"]):
        raise ValueError("Selection disagrees with frozen validation rule")
    return step


def verify_sources(args, common):
    parents, parent_common = train.parent_recipe(args)
    if parents != common["parents"] or parent_common != common["parent_common"]:
        raise ValueError("Changed gate parents")
    contract = read(args.base/"empty_s43/contract.json")
    if (contract["experiment"] != train.g.old.EXPERIMENT or contract["arm"] != "empty" or
            sha256_file(args.base/"empty_s43/step006000.pt") != parent_common["base_sha"] or
            contract["iris_sources"] != parent_common["sources"]):
        raise ValueError("Changed shared base")
    for hashes in (contract["code"], common["code"]):
        for name, digest in hashes.items():
            if sha256_file(ROOT/name) != digest:
                raise ValueError(f"Changed implementation: {name}")
    sources = train.g.old.matched_sources(*(Path(parent_common["sources"][a]["path"]) for a in train.PARENTS))
    if sources != parent_common["sources"]:
        raise ValueError("Changed native Iris weights")
    return contract, sources


def parent_test(args, common, manifests, official_ids):
    """LoRA step0 is frame2k or empty6k LOCAL GATE, never bare fusion empty6k."""
    folder = args.parents/"test_selected_v1"/args.arm
    provenance = read(folder/"provenance.json")
    summary = read(folder/"summary.json")
    parent = common["parents"][args.arm]
    expected = dict(arm=args.arm, step=train.PARENTS[args.arm], protocol=PROTOCOL, manifests=manifests,
                    base_sha=common["parent_common"]["base_sha"],
                    checkpoint_sha256=parent["sha256"], selection_sha256=parent["summary_sha256"],
                    iris_source=common["parent_common"]["sources"][args.arm], official_ids=official_ids)
    if any(provenance.get(k) != v for k,v in expected.items()):
        raise ValueError("Parent test provenance mismatch")
    if (summary["arm"] != args.arm or summary["step"] != train.PARENTS[args.arm] or
            summary["manifests"] != manifests or set(summary["results"]) != set(TESTS) or
            summary.get("reused_empty6k") is not False or
            summary["provenance_sha256"] != sha256_file(folder/"provenance.json")):
        raise ValueError("Parent test summary mismatch")
    return folder, summary


def verify_fold(result, cached, step):
    if cached["step"] != step or cached["samples"] != result["samples"]:
        raise ValueError("Parent result step/coverage mismatch")
    for mode, scores in result["metrics"].items():
        for key,value in scores.items():
            if not np.isclose(value, cached["metrics"][mode][key], rtol=1e-10, atol=1e-12):
                raise ValueError("Parent CSV/full summary mismatch")
    for key in ("frames", "official"):
        if result["official_ssi"][key] != cached["official_ssi"][key]:
            raise ValueError("Parent official coverage mismatch")
    for key,value in result["official_ssi"]["metrics"].items():
        if not np.isclose(value, cached["official_ssi"]["metrics"][key], rtol=1e-10, atol=1e-12):
            raise ValueError("Parent official metrics mismatch")


def load_model(args, common, contract, sources):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 GPU required for nonzero selected LoRA")
    inputs, tokenizer = train.g.load_inputs(args, contract, sources, common["parent_common"]["snapshot"])
    parent = common["parents"][args.arm]
    state = torch.load(parent["checkpoint"], map_location="cpu", weights_only=True)
    gate = train.g.LocalGate().cuda().eval()
    gate.load_state_dict(state["gate"], strict=True)
    torch.manual_seed(43)
    layers = train.inject(inputs.iris.unet)
    if train.digest_tensors(train.lora_state(layers).items()) != common["lora_init"]:
        raise ValueError("LoRA layout/initialization mismatch")
    return train.Adaptation(inputs, gate, layers), tokenizer


@torch.no_grad()
def run(args):
    os.environ.update(IRIS_THERMAL_STRETCH="percentile", IRIS_DECODE_CLAMP="0", IRIS_TEXT_PADDING="legacy")
    arm_dir = args.run/args.arm
    summary = read(arm_dir/"summary.json")
    common = summary["common"]
    step = selected_step(args.arm, read(arm_dir/"contract.json"), summary)
    contract, sources = verify_sources(args, common)
    bundles, manifests = shared.load_test_sets(args)
    official_ids = {c:hashlib.sha256("\n".join(sorted(wanted)).encode()).hexdigest()
                    for c,(_,wanted) in bundles.items()}
    out = args.run/"test_selected_v1"/args.arm
    out.mkdir(parents=True, exist_ok=True)
    provenance = dict(arm=args.arm, step=step, protocol=PROTOCOL, manifests=manifests,
                      selection_sha256=sha256_file(arm_dir/"summary.json"),
                      contract_sha256=sha256_file(arm_dir/"contract.json"),
                      parents=common["parents"], base_sha=common["parent_common"]["base_sha"],
                      iris_source=sources[args.arm], official_ids=official_ids,
                      code={name:sha256_file(ROOT/name) for name in
                            ("tools/test_iris_local_gate_lora.py", "tools/test_iris_local_gate.py",
                             "tools/test_da2_fixed.py", "tools/run_ms2_supdepth_baselines.py", "tools/test_sd21_tadp.py")})
    if step == 0:
        src, cached = parent_test(args, common, manifests, official_ids)
        provenance.update(reused_from=str(src), source_summary_sha256=sha256_file(src/"summary.json"),
                          source_provenance_sha256=sha256_file(src/"provenance.json"))
    else:
        checkpoint = arm_dir/f"step{step:06d}.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        train.check_checkpoint(state, args.arm, common, step)
        provenance["checkpoint_sha256"] = sha256_file(checkpoint)
    train.g.mark_equal(out/"provenance.json", provenance)
    # Lazy load: repeated invocations with verified complete CSVs need no GPU model.
    model, tokenizer, results = None, None, {}
    for condition,(samples,wanted) in bundles.items():
        if step == 0:
            csv_path = src/f"{condition}_frames.csv"
            prior = cached["results"][condition]
            if Path(prior["csv"]).resolve() != csv_path.resolve() or prior["csv_sha256"] != sha256_file(csv_path):
                raise ValueError("Parent CSV provenance mismatch")
            row = shared.fold(csv_path, samples, wanted, train.PARENTS[args.arm])
            verify_fold(row, prior, train.PARENTS[args.arm])
        else:
            csv_path = out/f"{condition}_frames.csv"
            if not csv_path.exists():
                if model is None:
                    model, tokenizer = load_model(args, common, contract, sources)
                    model.restore(state)
                tokens = train.g.old.make_tokens(samples, tokenizer, None, None, "empty", None, False)
                data = train.g.old.FusionData(samples, tokens)
                print(f"[test] {args.arm} new_step={step} {condition} frames={len(data)}", flush=True)
                train.evaluate(model, data, range(len(data)), {step:state}, csv_path)
            row = shared.fold(csv_path, samples, wanted, step)
        results[condition] = dict(**row, step=step, csv=str(csv_path), csv_sha256=sha256_file(csv_path))
        for mode,scores in row["metrics"].items():
            print("[test-full]", args.arm, condition, mode, row["samples"],
                  {k:round(scores[k],5) for k in shared.METRICS}, flush=True)
        official = row["official_ssi"]
        print("[test-official]", args.arm, condition,
              f"{official['frames']}/{official['official']}", official["metrics"], flush=True)
    write(out/"summary.json", dict(arm=args.arm, step=step, parent_step=train.PARENTS[args.arm],
           reused_parent=step==0, manifests=manifests, results=results,
           provenance_sha256=sha256_file(out/"provenance.json")))
    print("[completed-test]", out/"summary.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=list(train.PARENTS))
    for name in ("base", "parents", "run", "assets", "manifest-dir", "data-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    run(parser.parse_args())
