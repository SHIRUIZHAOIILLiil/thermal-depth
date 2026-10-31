"""One train-fitted scalar on decoded Iris caption differences; paired validation."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The legacy route module imports its tools/ siblings as top-level modules.
if str(ROOT / "tools") not in sys.path:
    sys.path.append(str(ROOT / "tools"))
from ms2_eval.io import sha256_file
from ms2_eval.official_protocol import evaluate_sample, official_valid_mask, OfficialProtocolError, OFFICIAL_METRICS
from tools.da2_caption_data import checked_manifest, TRAIN
from tools.da2_continuation import validate_state, check_source
from tools.train_da2_caption import build_model, verify_assets, json_write
from tools.train_da2_iris_fusion_v2 import (
    FusionDepth, FusionData, EXPERIMENT, forward_batch, load_iris, make_tokens,
    matched_sources, read, restore, validate_iris_payload, verify_iris_assets,
)

NAME = "iris_text_correction_empty6k_ssi_v2"
STEP = 6000
CONDITIONS = ("zero", "frame", "permuted")
MODES = ("ssi", "none")
CHUNK = 128


def pick_train(samples, per_sequence=256, nrefs=4):
    rng = np.random.default_rng(43)
    groups = defaultdict(list)
    for i, s in enumerate(samples):
        seq = s.sample_id.rsplit("_", 1)[0]
        if seq not in TRAIN:
            raise ValueError("Reference and fit frames must be training frames")
        groups[seq].append(i)
    if set(groups) != TRAIN:
        raise ValueError("Missing training sequence")
    refs, texts = [], set()
    for seq in rng.permutation(sorted(groups))[:nrefs]:
        for i in rng.permutation(groups[seq]):
            if samples[i].caption not in texts:
                refs.append(int(i))
                texts.add(samples[i].caption)
                break
    if len(refs) != nrefs:
        raise ValueError("Insufficient distinct reference captions")
    fit = []
    for seq in sorted(groups):
        choices = [i for i in groups[seq] if i not in refs]
        fit.extend(rng.choice(choices, per_sequence, replace=False).tolist())
    return refs, sorted(fit)


def unit_map(raw):
    # This checkpoint predicts normalized log-depth: larger means farther.
    # Spatial normalization uses only image predictions, including outside LiDAR.
    raw = np.asarray(raw, np.float32)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError("Invalid decoded Iris map")
    return (raw - raw.mean(dtype=np.float64).item()) / max(raw.std(dtype=np.float64).item(), 1e-6)


def fit_pixels(base, delta, gt):
    valid = official_valid_mask(gt, .001, 80.)
    if base.shape != gt.shape or delta.shape != gt.shape:
        raise ValueError("Map shape mismatch")
    if not valid.any():
        return None
    # SSI fits and scores only on this mask. Keep frame boundaries and the
    # original order, so the official evaluator can run on compact 2D arrays.
    return tuple(array[valid][None, :] for array in (base, delta, gt))


def mean_train_ssi_absrel(frames, beta):
    scores = []
    for base, delta, gt in frames:
        pred = corrected(base, delta, beta)
        scores.append(evaluate_sample(pred, gt, align="ssi", min_depth=.001, max_depth=80.)["abs_rel"])
    return float(np.mean(scores))


def fit_coefficient(frames):
    from scipy.optimize import minimize_scalar
    if not frames:
        raise ValueError("No training LiDAR")
    evaluations = {}

    def objective(beta):
        beta = float(beta)
        if beta not in evaluations:
            try:
                with np.errstate(over="raise", under="ignore", invalid="raise"):
                    score = mean_train_ssi_absrel(frames, beta)
                if not np.isfinite(score):
                    raise FloatingPointError("Nonfinite training SSI AbsRel")
            except (OfficialProtocolError, FloatingPointError):
                if beta == 0.:
                    raise
                score = float("inf")
            evaluations[beta] = score
        return evaluations[beta]

    # The affine fit changes with beta. Bracket sampled local minima, including
    # either boundary, then refine; retain the best actually scored candidate.
    knots = np.linspace(0., 1., 9).tolist()
    scores = [objective(beta) for beta in knots]
    while scores[-1] < min(scores[:-1]):
        knots.append(2. * knots[-1])
        scores.append(objective(knots[-1]))
    if min(scores) != max(scores):
        for i, value in enumerate(scores):
            lo, hi = max(0, i-1), min(len(knots)-1, i+1)
            if np.isfinite(value) and value <= min(scores[lo], scores[hi]):
                minimize_scalar(objective, bounds=(knots[lo], knots[hi]), method="bounded",
                                options={"xatol": 1e-5})
    else:
        # Even equal knot scores can conceal a minimum between knots.
        minimize_scalar(objective, bounds=(0., knots[-1]), method="bounded", options={"xatol": 1e-5})
    beta = min(evaluations, key=lambda b: (evaluations[b], b))
    return dict(beta=beta, train_zero_absrel=evaluations[0.], train_frame_absrel=evaluations[beta],
                evaluated=[dict(beta=b, ssi_absrel=evaluations[b] if np.isfinite(evaluations[b]) else None)
                           for b in sorted(evaluations)])


def corrected(base, delta, beta):
    if beta == 0:
        return base
    result = (base * np.exp(beta * delta.astype(np.float64))).astype(np.float32)
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise FloatingPointError("Invalid corrected depth")
    return result


def donors_for(samples, record, expected):
    pairs = record["pairs"]
    digest = hashlib.sha256(json.dumps(pairs, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    ids = [s.sample_id for s in samples]
    if (digest != expected or [a for a, b in pairs] != ids or
            {b for a, b in pairs} != set(ids) or any(a == b for a, b in pairs)):
        raise ValueError("Frozen validation permutation mismatch")
    by_id = {s.sample_id: s.caption for s in samples}
    return [by_id[b] for a, b in pairs]


def load_models(args, contract, sources, snapshot):
    from transformers import CLIPTokenizer
    from tools.train_route_suite import RouteModel
    assets = verify_assets(args.assets)
    iris_assets = verify_iris_assets(args.base / "iris_assets.json")
    paths = ((args.assets, "assets_sha256"), (args.base / "iris_assets.json", "iris_assets_sha256"),
             (Path(contract["base_path"]), "base_sha256"))
    for path, key in paths:
        if sha256_file(path) != contract[key]:
            raise ValueError(f"Changed asset: {path}")
    state = torch.load(contract["base_path"], map_location="cpu", weights_only=False)
    original = state["contract"]
    validate_state(state, original, 18000)
    check_source(original)
    core = build_model(assets, "no-caption", 43)
    core.load_state_dict(state["model"], strict=True)
    del state
    baseline = FusionDepth(core.core, load_iris(iris_assets["path"], sources["empty"]["path"], "empty"))
    if restore(args.base / "empty_s43/step006000.pt", baseline, contract) != STEP:
        raise ValueError("Expected empty 6k")
    baseline.requires_grad_(False).to("cuda").eval()
    tokenizer = CLIPTokenizer.from_pretrained(iris_assets["path"], subfolder="tokenizer", local_files_only=True)
    # Exactly the legacy showcase evaluation path, including its own cached
    # Lotus VAE/CLIP, dynamic single-prompt encoding, FP32 U-Net and decode.
    recipe = SimpleNamespace(route="b_thermal_unet", backbone="g", lotus_model_path=snapshot,
                             local_files_only=True, gt_decode_fp32=True, condition_latent="mode",
                             timestep=999, num_inference_steps=1, seed=20260703)
    iris = RouteModel(recipe, torch.device("cuda"), torch.float16)
    state = torch.load(sources["frame"]["path"], map_location="cpu", weights_only=False)
    validate_iris_payload(state, "frame")
    iris.load_state_dicts(state["state_dicts"], sources["frame"]["path"])
    iris.unet.requires_grad_(False)
    iris.set_train(False)
    return baseline, tokenizer, iris


class Predictor:
    def __init__(self, baseline, iris, references):
        self.baseline = baseline
        self.iris = iris
        self.refs = [iris.encode_prompt(text) for text in references]

    def maps(self, dataset, index, donor=None):
        sample = dataset.samples[index]
        batch = default_collate([dataset[index]])
        base = forward_batch(self.baseline, batch)[0].float().cpu().numpy()
        if not np.isfinite(base).all() or np.any(base <= 0):
            raise ValueError(f"Invalid metric depth for {sample.sample_id}")
        def decode(prompt):
            row = {"manifest_index": sample.manifest_line - 1}
            pred = self.iris.predict_disparity(row, batch["iris_image"], prompt)
            raw = pred.float().cpu().numpy()
            if raw.shape != base.shape:
                raise ValueError("Iris and DA2 maps must cover the same native pixels")
            return unit_map(raw)
        reference = np.mean([decode(prompt) for prompt in self.refs], axis=0)
        frame = decode(self.iris.encode_prompt(sample.caption)) - reference
        shuffled = decode(self.iris.encode_prompt(donor)) - reference if donor is not None else None
        return base, frame, shuffled, batch["gt"][0].numpy()


def paired_scores(base, frame, shuffled, gt, beta):
    rows = []
    for condition, pred in zip(CONDITIONS, (base, corrected(base, frame, beta), corrected(base, shuffled, beta))):
        for mode in MODES:
            scores = evaluate_sample(pred, gt, align=mode, min_depth=.001, max_depth=80.)
            rows.append(dict(condition=condition, alignment=mode, **{k: float(scores[k]) for k in OFFICIAL_METRICS}))
    return rows


def validate(predictor, data, donors, beta, out, expected):
    totals = defaultdict(list)
    for start in range(0, len(data), CHUNK):
        path = out / f"paired_{start:05d}.csv"
        stop = min(start + CHUNK, len(data))
        if not path.exists():
            tmp = path.with_suffix(".tmp")
            with tmp.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "condition", "alignment", *OFFICIAL_METRICS])
                writer.writeheader()
                for i in range(start, stop):
                    maps = predictor.maps(data, i, donors[i])
                    for row in paired_scores(*maps, beta):
                        writer.writerow(dict(id=data.samples[i].sample_id, **row))
            os.replace(tmp, path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for i in range(start, stop):
                for condition in CONDITIONS:
                    for mode in MODES:
                        row = next(reader)
                        if (row["id"], row["condition"], row["alignment"]) != (data.samples[i].sample_id, condition, mode):
                            raise ValueError(f"Incomplete paired validation: {path}")
                        values = [float(row[k]) for k in OFFICIAL_METRICS]
                        if not np.isfinite(values).all():
                            raise ValueError(f"Nonfinite metrics in {path}")
                        totals[condition, mode].append(values)
            if next(reader, None) is not None:
                raise ValueError(f"Unexpected rows in {path}")
        print(f"[val] {stop}/{len(data)}", flush=True)
    metrics = {c: {m: dict(zip(OFFICIAL_METRICS, np.mean(totals[c, m], axis=0))) for m in MODES} for c in CONDITIONS}
    for mode in MODES:
        for key in OFFICIAL_METRICS:
            if not np.isclose(metrics["zero"][mode][key], expected[mode][key], rtol=1e-4, atol=1e-6):
                raise ValueError(f"Zero correction does not reproduce empty 6k: {mode}/{key}")
    frame = metrics["frame"]["ssi"]
    gates = {}
    for c in ("zero", "permuted"):
        other = metrics[c]["ssi"]
        gates[c] = dict(absrel=frame["abs_rel"] < other["abs_rel"], rmse=frame["rmse"] <= 1.005*other["rmse"],
                        a1=frame["a1"] >= other["a1"]-.001)
    return dict(samples=len(data), beta=beta, metrics=metrics, gates=gates,
                passes=all(bool(v) for g in gates.values() for v in g.values()))


@torch.inference_mode()
def run(args):
    from huggingface_hub import snapshot_download
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA required")
    os.environ.update(IRIS_THERMAL_STRETCH="percentile", IRIS_DECODE_CLAMP="0", IRIS_TEXT_PADDING="legacy")
    run_dir = args.base / "empty_s43"
    contract, selected = read(run_dir / "contract.json"), read(run_dir / "summary.json")
    if (contract["arm"] != "empty" or contract["experiment"] != EXPERIMENT or selected["contract"] != contract
            or selected["selected_step"] != STEP or selected["selected_full"]["step"] != STEP
            or selected["selected_full"]["samples"] != 14249):
        raise ValueError("Requires the validation-selected matched empty 6k model")
    for filename, digest in contract["code"].items():
        if sha256_file(ROOT / filename) != digest:
            raise ValueError(f"Fusion source changed: {filename}")
    sources = matched_sources(*(Path(contract["iris_sources"][arm]["path"]) for arm in ("frame", "empty")))
    if sources != contract["iris_sources"]:
        raise ValueError("Iris checkpoint identity changed")
    train, ti = checked_manifest(args.train, args.data_root, "train", 75688)
    val, vi = checked_manifest(args.val, args.data_root, "val", 14249)
    if ti != contract["train"] or vi != contract["val"]:
        raise ValueError("Manifests differ from the frozen fusion experiment")
    refs, indices = pick_train(train)
    permutation = contract["permutation"]["mapping_sha256"]
    donors = donors_for(val, read(run_dir / "permutation.json"), permutation)
    snapshot = snapshot_download("jingheya/lotus-depth-g-v2-1-disparity", local_files_only=True)
    files = ("tools/iris_text_correction.py", "tools/train_route_suite.py", "tools/train_ms2_joint_gt_v3.py",
             "lotus/pipeline.py", "models/anythermal_lotus_v2_4.py", "models/anythermal_lotus_v2.py")
    recipe = dict(name=NAME, objective="frame_mean_official_ssi_abs_rel", beta_search="nonnegative_adaptive_bracket",
                  fit_ids=[train[i].sample_id for i in indices],
                  references=[dict(id=train[i].sample_id, caption=train[i].caption) for i in refs],
                  permutation_sha256=permutation, train=ti, val=vi, snapshot=snapshot, sources=sources,
                  checkpoint_sha256=sha256_file(run_dir / "step006000.pt"), selection_sha256=sha256_file(run_dir / "summary.json"),
                  code={name: sha256_file(ROOT / name) for name in files})
    out = args.base.parent / NAME
    out.mkdir(exist_ok=True)
    path = out / "recipe.json"
    if path.exists() and read(path) != recipe:
        raise ValueError("Existing output uses a different recipe")
    json_write(path, recipe)
    baseline, tokenizer, iris = load_models(args, contract, sources, snapshot)
    predictor = Predictor(baseline, iris, [train[i].caption for i in refs])
    def dataset(samples):
        return FusionData(samples, make_tokens(samples, tokenizer, None, None, "empty", None, False))

    fit_path = out / "fit.json"
    if not fit_path.exists():
        training = dataset(train)
        frames = []
        for n, i in enumerate(indices, 1):
            base, delta, _, gt = predictor.maps(training, i)
            pixels = fit_pixels(base, delta, gt)
            if pixels is not None:
                frames.append(pixels)
            if n % 128 == 0:
                print(f"[fit] {n}/{len(indices)}", flush=True)
        print("[fit] minimizing training SSI AbsRel", flush=True)
        json_write(fit_path, dict(**fit_coefficient(frames), used=len(frames), total=len(indices),
                                  recipe_sha256=sha256_file(path), objective=recipe["objective"]))
        del frames
    fit = read(fit_path)
    if fit["recipe_sha256"] != sha256_file(path) or not np.isfinite(fit["beta"]) or fit["beta"] < 0:
        raise ValueError("Invalid frozen fit")
    print("[frozen-fit]", {k: v for k, v in fit.items() if k != "evaluated"}, flush=True)
    # A different coefficient can never consume this fit's cached frame rows.
    paired_dir = out / ("val_" + sha256_file(fit_path))
    paired_dir.mkdir(exist_ok=True)
    result = validate(predictor, dataset(val), donors, fit["beta"], paired_dir, selected["selected_full"]["metrics"])
    result["gates"] = {c: {k: bool(v) for k, v in g.items()} for c, g in result["gates"].items()}
    json_write(out / "summary.json", dict(**result, recipe_sha256=sha256_file(path), fit_sha256=sha256_file(fit_path)))
    for c in CONDITIONS:
        print("[val-result]", c, result["metrics"][c]["ssi"], flush=True)
    print("[completed]", out / "summary.json", "passes=", result["passes"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "assets", "train", "val", "data-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    run(parser.parse_args())
