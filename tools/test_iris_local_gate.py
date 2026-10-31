"""Evaluate a validation-selected local gate, or verify and reuse the frozen base."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import train_iris_local_gate as train
from tools.da2_caption_data import caption_hash
from tools.run_ms2_supdepth_baselines import official_frames
from tools.test_da2_fixed import TESTS, validate_samples
from tools.test_sd21_tadp import ENVIRONMENTS, METRICS
from ms2_eval.io import load_manifest, sha256_file

read, write = train.old.read, train.old.json_write
PROTOCOL = "pilot official_protocol; .001<GT<80; frame mean; ssi/none separate"
OFFICIAL_COUNTS = {"day": (2331, 2332), "night": (2292, 2292), "rain": (2503, 2503)}


def selected_step(arm, contract, summary):
    common = summary["common"]
    if (contract != dict(arm=arm, **common) or summary["arm"] != arm or common["experiment"] != train.NAME
            or summary["samples"] != 14249):
        raise ValueError("Wrong arm, experiment or validation summary")
    curve = {int(k): v for k, v in summary["curve"].items()}
    if set(curve) != set(train.STEPS) or not all(np.isfinite(v["ssi"]["abs_rel"]) for v in curve.values()):
        raise ValueError("Incomplete checkpoint selection curve")
    step = train.select(curve)
    if step != summary["selected_step"] or (step == 0 and summary["selected_full"] != summary["baseline"]):
        raise ValueError("Selected step disagrees with validation")
    return step


def load_test_sets(args):
    bundles, manifests, seen = {}, {}, set()
    for condition, (suffix, prefix, count, sequences) in TESTS.items():
        path = args.manifest_dir / f"ms2_test_{suffix}_thermalcap_20260821.jsonl"
        samples, info = load_manifest(path, args.data_root, allow_legacy_depth_path=True)
        if len(samples) != count or not info["sha256"].startswith(prefix):
            raise ValueError(f"Wrong test manifest: {condition}")
        ids = validate_samples(samples, sequences)
        if ids & seen:
            raise ValueError("Overlapping test conditions")
        seen |= ids
        wanted = {f["id"] for f in official_frames(args.data_root, ENVIRONMENTS[condition], 10)}
        if (len(ids & wanted), len(wanted)) != OFFICIAL_COUNTS[condition]:
            raise ValueError(f"Unexpected official subset: {condition}")
        bundles[condition] = samples, wanted
        manifests[condition] = dict(**info, caption_sha256=caption_hash(samples))
    return bundles, manifests


def fold(path, samples, wanted, csv_step):
    """Both CSV formats store exactly one SSI and one unaligned row per image."""
    keys = train.old.OFFICIAL_METRICS
    totals = {m: np.zeros(len(keys)) for m in ("ssi", "none")}
    subset, hits = np.zeros(len(keys)), 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for sample in samples:
            for mode, total in totals.items():
                row = next(reader, None)
                if (row is None or (row["id"], row["alignment"]) != (sample.sample_id, mode)
                        or (csv_step is not None and int(row["step"]) != csv_step)):
                    raise ValueError(f"Frame/step mismatch in {path}")
                values = np.array([float(row[k]) for k in keys])
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite metrics in {path}")
                total += values
                if mode == "ssi" and sample.sample_id in wanted:
                    subset += values
                    hits += 1
        if next(reader, None) is not None or not hits:
            raise ValueError(f"Unexpected CSV coverage: {path}")
    return dict(samples=len(samples), metrics={m: dict(zip(keys, t/len(samples))) for m,t in totals.items()},
                official_ssi=dict(frames=hits, official=len(wanted),
                                  metrics={k: subset[keys.index(k)]/hits for k in METRICS}))


def check_baseline_result(condition, result, cached):
    if cached["step"] != 6000 or cached["samples"] != result["samples"]:
        raise ValueError(f"Wrong empty 6k result: {condition}")
    for mode, scores in result["metrics"].items():
        for k,v in scores.items():
            if not np.isclose(v, cached["metrics"][mode][k], rtol=1e-10, atol=1e-12):
                raise ValueError(f"Old full-frame CSV differs from summary: {condition}/{mode}/{k}")
    official = cached["official_ssi"]
    for k in ("frames","official"):
        if official[k] != result["official_ssi"][k]:
            raise ValueError(f"Old official frame count differs: {condition}")
    for k,v in result["official_ssi"]["metrics"].items():
        if not np.isclose(v, official["metrics"][k], rtol=1e-10, atol=1e-12):
            raise ValueError(f"Old official fold differs: {condition}/{k}")


def baseline_source(args, common, manifests):
    path = args.base / "test_selected_20261030/empty"
    previous = args.base / "empty_s43"
    source, summary = read(path/"provenance.json"), read(path/"summary.json")
    checks = dict(arm="empty", step=6000, checkpoint_sha256=common["base_sha"],
                  selection_sha256=sha256_file(previous/"summary.json"),
                  contract_sha256=sha256_file(previous/"contract.json"), iris_source=common["sources"]["empty"],
                  manifests=manifests, protocol=PROTOCOL)
    if any(source.get(k) != v for k,v in checks.items()):
        raise ValueError("Old test provenance does not match the frozen empty 6k")
    if (summary["arm"] != "empty" or summary["step"] != 6000 or summary["manifests"] != manifests
            or set(summary["results"]) != set(TESTS)):
        raise ValueError("Incomplete or mismatched empty 6k test summary")
    return path, summary


def restore_gate(path, arm, common, step):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state["step"], state["arm"], state["common"]) != (step, arm, common):
        raise ValueError("Wrong selected gate checkpoint")
    gate = train.LocalGate().eval()
    gate.load_state_dict(state["gate"], strict=True)
    return gate


@torch.no_grad()
def run(args):
    os.environ.update(IRIS_THERMAL_STRETCH="percentile", IRIS_DECODE_CLAMP="0", IRIS_TEXT_PADDING="legacy")
    arm_dir = args.run / args.arm
    summary = read(arm_dir/"summary.json")
    common, step = summary["common"], selected_step(args.arm, read(arm_dir/"contract.json"), summary)
    previous = args.base/"empty_s43"
    contract = read(previous/"contract.json")
    if (contract["arm"] != "empty" or contract["experiment"] != train.old.EXPERIMENT
            or read(previous/"summary.json")["selected_step"] != 6000
            or contract["iris_sources"] != common["sources"]
            or sha256_file(previous/"step006000.pt") != common["base_sha"]):
        raise ValueError("Changed empty 6k base")
    for hashes in (contract["code"], common["code"]):
        for name, expected in hashes.items():
            if sha256_file(ROOT/name) != expected:
                raise ValueError(f"Frozen source changed: {name}")
    bundles, manifests = load_test_sets(args)
    out = args.run/"test_selected_v1"/args.arm
    out.mkdir(parents=True, exist_ok=True)
    provenance = dict(arm=args.arm, step=step, base_sha=common["base_sha"], protocol=PROTOCOL, manifests=manifests,
                      selection_sha256=sha256_file(arm_dir/"summary.json"),
                      code={n:sha256_file(ROOT/n) for n in ("tools/test_iris_local_gate.py", "tools/test_da2_fixed.py",
                            "tools/run_ms2_supdepth_baselines.py", "tools/test_sd21_tadp.py")},
                      official_ids={c:hashlib.sha256("\n".join(sorted(wanted)).encode()).hexdigest()
                                    for c, (_,wanted) in bundles.items()})
    if step == 0:
        source_dir, cached = baseline_source(args, common, manifests)
        provenance.update(reused_from=str(source_dir), source_summary_sha256=sha256_file(source_dir/"summary.json"),
                          source_provenance_sha256=sha256_file(source_dir/"provenance.json"))
    else:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Selected gate inference needs a BF16 GPU")
        checkpoint = arm_dir/f"step{step:06d}.pt"
        gate = restore_gate(checkpoint, args.arm, common, step).cuda()
        sources = train.old.matched_sources(*(Path(common["sources"][a]["path"]) for a in ("frame","empty")))
        if sources != common["sources"]:
            raise ValueError("Iris weights changed since training")
        inputs, tokenizer = train.load_inputs(args, contract, sources, common["snapshot"])
        provenance.update(checkpoint_sha256=sha256_file(checkpoint), iris_source=sources[args.arm])
    train.mark_equal(out/"provenance.json", provenance)
    results = {}
    for condition, (samples, wanted) in bundles.items():
        if step == 0:
            csv_path = source_dir/condition/"val_step006000_frames.csv"
            row = fold(csv_path, samples, wanted, None)
            check_baseline_result(condition, row, cached["results"][condition])
        else:
            csv_path = out/f"{condition}_frames.csv"
            if not csv_path.exists():
                tokens = train.old.make_tokens(samples, tokenizer, None, None, "empty", None, False)
                data = train.old.FusionData(samples, tokens)  # base always empty; FrozenInputs supplies the arm prompt.
                print(f"[test] {args.arm} step={step} {condition} frames={len(data)}", flush=True)
                train.evaluate(inputs, data, range(len(data)), {step:gate}, csv_path)
            row = fold(csv_path, samples, wanted, step)
        results[condition] = dict(**row, step=step, csv=str(csv_path), csv_sha256=sha256_file(csv_path))
        for mode, scores in row["metrics"].items():
            print("[test-full]", args.arm, condition, mode, row["samples"], {k: round(scores[k],5) for k in METRICS}, flush=True)
        sub = row["official_ssi"]
        print("[test-official]", args.arm, condition, f"{sub['frames']}/{sub['official']}", sub["metrics"], flush=True)
    write(out/"summary.json", dict(arm=args.arm, step=step, reused_empty6k=step==0, manifests=manifests,
                                   results=results, provenance_sha256=sha256_file(out/"provenance.json")))
    print("[completed-test]", out/"summary.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("frame","empty"))
    for name in ("base","run","assets","manifest-dir","data-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    run(parser.parse_args())
