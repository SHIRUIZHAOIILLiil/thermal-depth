"""One trainer for all six routes of the 20-epoch suite.

Why one file instead of six: the task requires "统一数据划分、损失函数和评估方式".
The previous per-route trainers are copy-edited forks of each other, so any
route-to-route difference in the data pipeline, the loss, or the checkpoint rule
is a confound.  Here the only thing that changes between routes is *which
modules exist and which of them get gradients*.

Routes
------
    a_rgb_unet                RGB     -> frozen VAE -> TRAIN U-Net
    b_thermal_unet            Thermal -> frozen VAE -> TRAIN U-Net
    c1_vae_adapter            Thermal -> frozen VAE -> TRAIN Adapter -> frozen U-Net
    c2_vae_adapter_unet       Thermal -> frozen VAE -> TRAIN Adapter -> TRAIN U-Net
    d1_anythermal_adapter     Thermal -> frozen AnyThermal -> TRAIN Adapter -> frozen U-Net
    d2_anythermal_adapter_unet Thermal -> frozen AnyThermal -> TRAIN Adapter -> TRAIN U-Net

Objective (identical for every route, by user instruction 2026-07-25)
---------------------------------------------------------------------
Pure ground-truth supervision: masked scale-shift-invariant L1 against the
LiDAR disparity, on valid pixels only.  **No teacher of any kind** -- no frozen
response-consistency anchor, no condition distillation onto VAE latents, no
dense depth teacher.  Nothing in this file instantiates a second network to
imitate.  Pretrained Lotus/AnyThermal weights are used as *initialisation and
frozen feature extractors*, which is not a teacher-student relationship.

Per-epoch validation runs the official BMSD protocol (ssi_disparity) inline on
a strided subset of the val manifest, so the epoch curve is directly comparable
to every number in the frozen document.

Example (route b, 20 epochs, empty prompt):

    python tools/train_route_suite.py --route b_thermal_unet --epochs 20 --output-dir outputs/route_suite/b_thermal_unet_20ep

Gate discipline: `--smoke-updates 5` first, then `--overfit-steps 300` on a
32-frame subset, only then the full run.
"""

from __future__ import annotations

import argparse
import collections
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

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
from tools.metric_depth_norm import MetricNorm  # noqa: E402

# Written into --fit-global-affine artifacts and checked when one is loaded.
GLOBAL_AFFINE_SCHEMA = "iris.global_affine_calibration.v1"
from models.anythermal_lotus_v2 import thermal_to_lotus_input  # noqa: E402
from models.anythermal_lotus_v2_4 import seeded_noise  # noqa: E402
from models.anythermal_lotus_model import extract_anythermal_feature_pyramid  # noqa: E402
from train_ms2_joint_gt_v3 import (  # noqa: E402
    load_gt_disparity,
    masked_ssi_l1,
    masked_ssi_l1_dense_completion,
    ssi_grad_matching,
    ssi_sky_loss,
    decode_to_disparity,
)

# 集群上路径不同，用环境变量覆盖；不设时保持本地原有默认值不变。
MANIFEST_DIR = Path(
    os.environ.get(
        "IRIS_MANIFEST_DIR",
        "/mnt/e/project/thermal-depth/outputs/manifests/sequence_level_internvl3_8b",
    )
)
DEFAULT_TRAIN_MANIFEST = MANIFEST_DIR / "ms2_train_day2seq_20260725.jsonl"
DEFAULT_VAL_MANIFEST = (
    MANIFEST_DIR
    / "ms2_fixed_sequence_internvl3_8b_filtered_caption_val_rgb_depth_v1_clip75_rerun_20260714.jsonl"
)
CHECKPOINT_FORMAT = "route_suite_multi_epoch_pure_gt"

# route -> (input modality, condition path, trains adapter, trains U-Net)
ROUTES = {
    "a_rgb_unet": ("rgb", "vae", False, True),
    "b_thermal_unet": ("thermal", "vae", False, True),
    "c1_vae_adapter": ("thermal", "vae_adapter", True, False),
    "c2_vae_adapter_unet": ("thermal", "vae_adapter", True, True),
    "d1_anythermal_adapter": ("thermal", "anythermal_adapter", True, False),
    "d2_anythermal_adapter_unet": ("thermal", "anythermal_adapter", True, True),
}


# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", required=True, choices=sorted(ROUTES))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument(
        "--ms2-root",
        type=Path,
        default=Path(os.environ.get("IRIS_MS2_ROOT", "/mnt/e/dataset/ms2")),
    )
    parser.add_argument("--lotus-model-path", default="jingheya/lotus-depth-g-v2-1-disparity")
    # Which Lotus variant --lotus-model-path is. G conditions on
    # [image latent, noisy target latent]; D is the direct variant and takes the
    # image latent alone. Must match what the checkpoint was trained as -- see the
    # same flag in lotus/train_iris_ms2_g.py.
    parser.add_argument("--backbone", choices=["g", "d", "marigold", "e2eft"], default="g")
    parser.add_argument("--anythermal-model-path", default="theairlabcmu/AnyThermal")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--unet-learning-rate", type=float, default=1e-6)
    parser.add_argument("--adapter-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine"),
        default="cosine",
        help="20 epochs at a constant 1-epoch LR is not a fair convergence test; cosine is the default.",
    )
    parser.add_argument("--warmup-updates", type=int, default=200)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)

    parser.add_argument(
        "--gt-loss-weight",
        type=float,
        default=5.0,
        help=(
            "Inherited from the three-term joint objective, where it set the GT "
            "term's share against two distillation regularisers; an overfit-32 sweep "
            "found 0.5/2.0/5.0 endpoint-equivalent and took 5.0 on a hair "
            "(LOTUS_LINE_V2_ROUTE_AND_CAPTION_FREEZE_20260705.md:99). This objective "
            "has one data term, so here it is only a 5x learning-rate scale -- but "
            "--sky-loss-weight, --pseudo-weight and --caption-rank-weight are added "
            "OUTSIDE it, which quietly makes 5.0 the denominator of every one of "
            "them. Changing it would break comparability with every published run."
        ),
    )
    parser.add_argument(
        "--grad-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on MiDaS's multi-scale gradient matching term. 0 (the default) "
            "leaves every existing number untouched. The pointwise data term alone "
            "lets a model hedge boundaries with a smooth ramp, which AbsRel and RMSE "
            "reward because they are pointwise too; this term prices that hedge. "
            "MiDaS pairs alpha 0.5 with a data weight of 1, so 2.5 matches the 5.0 "
            "default here -- but check the two terms' raw magnitudes first."
        ),
    )
    parser.add_argument(
        "--sky-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on the sky term (Jasmine appendix C.3). 0 (the default) leaves "
            "every existing number untouched. Sky returns no lidar, so it gets zero "
            "gradient from the data term and zero weight in the metrics: measured, "
            "the top 32 rows read 41 m frozen and 9.6 m after five epochs, and "
            "AbsRel cannot see the difference. This term is the constraint that "
            "region otherwise lacks. Needs --sky-mask-dir. Calibrate it against the "
            "two terms' GRADIENT magnitudes, not their loss values -- that mistake "
            "is what sank the gradient-matching arm. "
            "NOTE this multiplies the RAW sky term while the data term carries "
            "--gt-loss-weight (5.0), so 0.5 here is a TENTH of the data term, not a "
            "half. Both terms average over their own region too, so a sky pixel at "
            "0.5 pulls 0.56x as hard as a supervised one (7,258 vs 40,363 px); "
            "parity would be about 0.9."
        ),
    )
    parser.add_argument(
        "--sky-loss-mode",
        choices=("l1", "hinge"),
        default="l1",
        help="l1 is Jasmine's two-sided form; hinge only punishes sky judged too near.",
    )
    parser.add_argument(
        "--sky-mask-dir",
        type=Path,
        default=None,
        help=(
            "Directory of <id>.png sky masks (0/255), from tools/build_sky_masks.py. "
            "A missing file means 'no sky in this frame', which is normal (24 of 100 "
            "probe frames: tunnels, canopy, tall buildings) -- not an error."
        ),
    )
    parser.add_argument(
        "--caption-rank-weight",
        type=float,
        default=0.0,
        help=(
            "Lambda on a margin ranking loss between the correct caption and a wrong "
            "one: L = w_gt*L_correct + lambda*relu(margin + L_correct - L_wrong). "
            "0 (the default) skips the second forward entirely, so training is "
            "bit-identical to every published run. Motivation: the cross-attention "
            "probe shows text reaches the features (delta 1.132), that its content is "
            "distinguishable (content fraction 0.279) and that this survives to the "
            "output (0.293) -- but the direction is unrelated to the truth, because "
            "masked_ssi_l1 never prices ignoring the text. This term does. "
            "NOTE lambda multiplies the RAW ranking term while the data term carries "
            "--gt-loss-weight (5.0), so lambda=0.5 gives the ranking term a tenth of "
            "the data term's weight. Requires --caption-mode correct."
        ),
    )
    parser.add_argument(
        "--caption-rank-margin",
        type=float,
        default=0.0,
        help=(
            "m in relu(m + L_correct - L_wrong). 0 means 'the correct caption must "
            "merely not lose', which needs no calibration against the loss scale."
        ),
    )
    parser.add_argument(
        "--caption-rank-refs",
        choices=("wrong", "empty", "both"),
        default="both",
        help=(
            "What a frame's own caption has to beat. 'wrong' is a random other "
            "frame's caption, 'empty' is the null prompt. Both, by default: beating "
            "a wrong caption says the model reads the text at all, beating the empty "
            "prompt says reading it is worth more than ignoring it -- and right now "
            "the empty prompt WINS (0.0869 against 0.0882), so that second one is "
            "the gap that actually has to close."
        ),
    )
    parser.add_argument(
        "--caption-rank-detach-wrong",
        action="store_true",
        help=(
            "Take the wrong-caption branch as a constant reference. Off by default: "
            "a model that genuinely uses the text SHOULD do worse when handed the "
            "wrong one, so that branch moving is the point, not a cheat. The failure "
            "to watch for is L_correct standing still while L_wrong balloons, which "
            "the separate rank c / w log fields make visible."
        ),
    )
    parser.add_argument(
        "--no-caption-rank-detach-empty",
        dest="caption_rank_detach_empty",
        action="store_false",
        default=True,
        help=(
            "Let gradients flow through the empty-prompt reference. Detached by "
            "default, and the asymmetry with the wrong branch is deliberate: the "
            "empty prompt is the configuration holding the best thermal score this "
            "project has (0.0869), so a margin that can be met by degrading it would "
            "trade the one number worth having for a ranking that merely looks "
            "right. Detached, the only way to meet it is a better correct branch."
        ),
    )
    parser.add_argument("--gt-min-depth", type=float, default=0.1)
    parser.add_argument("--gt-max-depth", type=float, default=80.0)
    parser.add_argument("--depth-scale", type=float, default=256.0)
    parser.add_argument(
        "--gt-decode-fp32",
        dest="gt_decode_fp32",
        action="store_true",
        default=True,
        help="Decode through an fp32 VAE copy (fixes the fp16 GT-gradient underflow). On by default.",
    )
    parser.add_argument("--no-gt-decode-fp32", dest="gt_decode_fp32", action="store_false")

    parser.add_argument(
        "--caption-mode",
        choices=("empty", "correct", "permuted"),
        default="empty",
        help=(
            "What text the TRAINING loop sees. 'permuted' hands every frame a "
            "uniformly random other frame's caption, which is the control the "
            "weight effect has never had: the injection effect was shown to be "
            # argparse %-interpolates help text, so a literal percent must be
            # doubled; this single one crashed --help for the whole tool.
            "content-free (shuffled minus correct = -0.00006, 49.5%% win rate), "
            "but nobody has asked whether training's benefit is content-driven "
            "or just the regularisation of a varying text input. Same text "
            "distribution, same token statistics, only the pairing is destroyed."
        ),
    )
    parser.add_argument("--caption-dropout", type=float, default=0.1)
    parser.add_argument(
        "--input-max-edge",
        type=int,
        default=0,
        help="RGB route only. 0 = native 1224x384; positive downscales the longer edge (multiple of 8).",
    )

    parser.add_argument(
        "--eval-min-depth",
        type=float,
        default=1e-3,
        help=(
            "Valid-GT lower bound for the official protocol. The default is MS2's and "
            "every historical number in the frozen document was produced with it -- do "
            "not change it for MS2. RGBDT500 needs 0.1: its millimetre sensor noise "
            "otherwise counts as GT and inflated all four caption cells from 0.36 to 0.82."
        ),
    )
    parser.add_argument(
        "--eval-max-depth",
        type=float,
        default=80.0,
        help="Official upper bound. MS2 = 80 m; RGBDT500's sensor ceiling is 20 m.",
    )
    parser.add_argument("--val-stride", type=int, default=4, help="Val subset stride for the epoch curve.")
    parser.add_argument("--val-every", type=int, default=1, help="Validate every N epochs.")
    parser.add_argument(
        "--val-caption-mode",
        choices=("empty", "correct", "shuffled", "permuted", "fixed"),
        default="empty",
        help=(
            "shuffled: donor caption from half a set away (kilometres along the path). "
            "permuted: donor drawn uniformly at random -- USE THIS. The rotation is a "
            "contaminated control: these drives double back, so a donor kilometres "
            "along the path can be metres away in space, and its caption still predicts "
            "the recipient's median GT depth at R^2 0.32 (MS2) / 0.46 (RGBDT500) versus "
            "~-0.1 for a uniform permutation (probe_caption_scale_information.py). "
            "fixed: the one sentence in --val-caption-text, for every frame."
        ),
    )
    parser.add_argument(
        "--val-caption-text",
        default=None,
        help=(
            "The sentence --val-caption-mode fixed feeds to every frame. Between an "
            "empty prompt and a real caption sit the rungs of Iris Table 5 -- 'An "
            "image', then a generic template -- and they are what separates text "
            "costing us because of what it claims from text costing us because it is "
            "there at all. Only the first reading leaves a caption rewrite anything "
            "to win, so this decides whether regenerating a corpus is worth it."
        ),
    )
    parser.add_argument(
        "--pseudo-gt-dir",
        type=Path,
        default=None,
        help="calibrated_pseudo_depth/ from build_anythermal_pseudo_gt.py: dense depth "
             "in metres, standing in only where lidar returned nothing.",
    )
    parser.add_argument(
        "--pseudo-weight",
        type=float,
        default=0.0,
        help="Weight on the completed region, relative to the real-lidar term inside "
             "the same call. At 0 the run takes the same code path as before this "
             "flag existed. NOTE both terms are then multiplied by --gt-loss-weight "
             "(5.0), and each averages over its own region, so 0.2 here leaves a "
             "pseudo pixel pulling 0.067x as hard as a supervised one (121,650 vs "
             "40,884 px); parity would be about 3.0. Read the share off the gradient, "
             "never off the loss value.",
    )
    parser.add_argument("--pseudo-min-depth", type=float, default=None, help="Defaults to --gt-min-depth.")
    parser.add_argument("--pseudo-max-depth", type=float, default=None, help="Defaults to --gt-max-depth.")
    parser.add_argument(
        "--pseudo-range-mode",
        choices=("drop", "clip"),
        default="drop",
        help="What to do past the range. drop: those pixels carry no target. clip: pin "
             "them to the bound, which asserts everything beyond it sits exactly there "
             "-- a distance prior the sky experiment is meant to test for, not assume. "
             "Measured, 0.12%% of masked sky exceeds 80 m, so drop costs almost nothing.",
    )
    parser.add_argument(
        "--latent-target",
        action="store_true",
        help="Train against the encoded depth latent instead of the decoded disparity, "
             "mirroring lotus/train_iris_g.py: the target map is VAE-encoded, noised at "
             "--timestep, and the U-Net's x0 is scored against it by MSE in latent space "
             "(train_iris_g.py:1038/1073/1137/1151). Sparse lidar cannot supply this -- a "
             "latent cell counts only where all 64 of its pixels are valid (:1054), and "
             "MS2's scattered 26.5%% leaves almost none -- so it needs --pseudo-gt-dir to "
             "fill the gaps, and the whole map, lidar included, becomes one target.",
    )
    parser.add_argument("--skip-val", action="store_true")

    parser.add_argument(
        "--snapshot-epochs",
        default="",
        help="Comma-separated epochs to persist weights for, on top of best/end (e.g. 1,2,5,10).",
    )
    parser.add_argument(
        "--loss-exclude-top-rows",
        type=int,
        default=0,
        help=(
            "Drop the top N image rows from the TRAINING loss mask (validation is "
            "untouched -- it re-reads the GT and scores the official protocol, so "
            "the curve stays comparable). This is the sky experiment: in those rows "
            "97-98%% of pixels have no lidar return at all, and the 2-3%% that do "
            "are near structure (median 8-11 m, >89%% under 15 m), so the only "
            "signal the loss carries up there is 'close'. Removing it separates "
            "two explanations for the sky collapsing from 41 m to 7 m during "
            "fine-tuning: unsupervised drift predicts it still collapses, "
            "near-biased labels predict it stays far. See "
            "docs/SKY_PROBLEM_SUMMARY_20260805.md."
        ),
    )
    parser.add_argument("--timestep", type=int, default=999)
    parser.add_argument(
        "--condition-latent",
        choices=("mode", "sample"),
        default="mode",
        help=(
            "How the input image's latent is taken. mode (default) is what every "
            "published number here used. sample draws from the posterior, which is "
            "what upstream does in training and in LotusGPipeline alike -- pass it "
            "to score a checkpoint trained by lotus/train_iris_ms2_g.py under the "
            "convention it was trained with. Changing this changes the exam, so a "
            "table must not mix the two without saying which each row used."
        ),
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=1,
        help=(
            "Evaluation-only (--eval-checkpoint). 1 (default) is the historical "
            "path: one forward at --timestep, x0 taken as the answer -- bit for "
            "bit what every published number used. >1 unrolls LotusGPipeline's "
            "DDIM loop. THAT IS OUT OF DISTRIBUTION TWICE OVER: this route was "
            "fine-tuned at t=999 only, and so was the Lotus-G checkpoint it "
            "started from (lotus/train_iris_g.py:1073 repeats one constant "
            "timestep instead of sampling). Read a multi-step arm as a "
            "falsification of 'single-step is why text does not help', not as "
            "restoring an ability the model has."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frozen-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--skip-file-check",
        action="store_true",
        help="Trust the manifest instead of stat-ing every referenced file (~50k syscalls).",
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help=(
            "Start from another run's weights (stage 2 of a staged schedule). Unlike "
            "--resume this may cross routes and manifests and carries no optimizer "
            "state: every module the checkpoint and this route share gets loaded, and "
            "training starts at epoch 0 with a fresh optimiser."
        ),
    )
    parser.add_argument(
        "--freeze-adapter",
        action="store_true",
        help=(
            "Keep the adapter fixed and train only the U-Net -- stage 2 of 'train the "
            "adapter first, then the U-Net behind it'. The frozen adapter is still "
            "written into every checkpoint, so evaluation reproduces the same "
            "condition path. Requires --init-from."
        ),
    )
    parser.add_argument(
        "--eval-checkpoint",
        type=Path,
        default=None,
        help="Evaluate this checkpoint instead of training. Use --val-stride 1 for the full val set.",
    )
    parser.add_argument(
        "--shuffle-condition",
        choices=("none", "all", "anythermal"),
        default="none",
        help=(
            "Evaluation-only falsification control. A route can look trained while "
            "its condition branch contributes nothing -- the U-Net alone would carry "
            "it, and no loss curve would show the difference. Feeding a donor frame's "
            "condition breaks the image-condition correspondence: if the metric barely "
            "moves, the branch is decorative. 'all' swaps the whole condition; "
            "'anythermal' swaps only the AnyThermal feature pyramid and keeps the "
            "correct thermal tensor, which isolates that branch from the adapter's "
            "own image input. Donors are drawn uniformly at random."
        ),
    )
    parser.add_argument(
        "--save-raw-pred",
        action="store_true",
        help=(
            "During --eval-checkpoint, also write each prediction to "
            "raw_predictions/<id>.npy at native model resolution, float32 -- the same "
            "convention run_ms2_lotus_*_official.py uses, so analyze_prediction_regions.py "
            "reads either interchangeably. Costs roughly 0.5 MB and one inode per frame."
        ),
    )
    parser.add_argument(
        "--eval-tag",
        default="eval",
        help="Names the output files: eval_<tag>.json and eval_<tag>_per_sample.csv.",
    )

    # ---- metric evaluation (2026-08-26) ----------------------------------- #
    # Everything above this block reads the model affine-invariantly: a scale and
    # a shift are fitted to the test GT of every frame before any metric is
    # computed. That is the right exam for a relative model and the wrong one for
    # a metric claim, so the two live side by side and are never merged.
    parser.add_argument(
        "--align-mode",
        choices=("ssi_disparity", "ssi", "median", "none"),
        default="ssi_disparity",
        help=(
            "How the prediction is brought to metres before the official metrics. "
            "ssi_disparity (default) is the historical path: a per-frame affine fitted "
            "to TEST GT -- every published number in this project used it. 'none' fits "
            "nothing and needs --metric-source to say how the raw output becomes an "
            "inverse depth. 'ssi' and 'median' are the upstream BMSD variants, exposed "
            "for completeness."
        ),
    )
    parser.add_argument(
        "--metric-source",
        choices=("raw_inverse", "global_norm", "global_affine"),
        default=None,
        help=(
            "With --align-mode none, how the decoded [0,1] output y becomes metric "
            "inverse depth q, hence depth 1/q:\n"
            "  raw_inverse    q = y. No constants at all -- a units sanity check on a "
            "checkpoint that was never trained to be metric, not a result.\n"
            "  global_norm    q = q_lo + y*(q_hi - q_lo), from --metric-norm. This is "
            "the metric-adapted model's own convention, and the constants come from the "
            "TRAIN split.\n"
            "  global_affine  q = a*y + b, from --global-affine. One (a, b) fitted once "
            "on TRAIN predictions and frozen -- the calibration baseline.\n"
            "None of the three reads GT at inference. All three are affine in y, so the "
            "evaluator records the (a, b) it actually used and where they came from."
        ),
    )
    parser.add_argument(
        "--metric-norm",
        type=Path,
        default=None,
        help="tools/fit_metric_norm.py artifact. Required by --metric-source global_norm.",
    )
    parser.add_argument(
        "--global-affine",
        type=Path,
        default=None,
        help="tools/train_route_suite.py --fit-global-affine artifact. Required by --metric-source global_affine.",
    )
    parser.add_argument(
        "--fit-global-affine",
        type=Path,
        default=None,
        help=(
            "Fit one dataset-level affine q ~ a*y + b on the manifest given by "
            "--val-manifest, write it here, and exit without evaluating. That manifest "
            "must be the TRAIN split: the whole point of the baseline is that the "
            "calibration never sees val or test. Requires --eval-checkpoint."
        ),
    )
    parser.add_argument(
        "--fit-affine-split-list",
        type=Path,
        default=None,
        help=(
            "train_list.txt. Required by --fit-global-affine and checked row by row. "
            "Not advisory: the manifests in this project carry a `split` field that "
            "says 'train' on test manifests too, so the sequence list is the only "
            "trustworthy statement of which split a file actually holds."
        ),
    )
    parser.add_argument(
        "--fit-affine-stride",
        type=int,
        default=20,
        help=(
            "Frame stride for --fit-global-affine. Two parameters fitted over ~4k "
            "frames x 40k lidar pixels is already 1.5e8 constraints; the default trades "
            "an unmeasurable amount of precision for a job that finishes in an hour."
        ),
    )
    parser.add_argument("--smoke-updates", type=int, default=None)
    parser.add_argument("--overfit-steps", type=int, default=None)
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=32,
        help="With --overfit-steps: how many frames the run is allowed to see.",
    )
    return parser.parse_args()


def validate_metric_args(args: argparse.Namespace) -> None:
    """Guards for the metric evaluation path. Loud, because every failure mode
    here is a number that looks plausible and means something else."""
    if args.align_mode != "ssi_disparity" and args.eval_checkpoint is None and args.fit_global_affine is None:
        raise ValueError(
            "--align-mode is an evaluation knob; training always validates under "
            "ssi_disparity so the epoch curve stays comparable with every run before it."
        )
    if args.align_mode == "none":
        if args.metric_source is None:
            raise ValueError(
                "--align-mode none needs --metric-source: with no fit to GT, something "
                "has to say how the decoded [0,1] output becomes an inverse depth."
            )
    elif args.metric_source is not None:
        raise ValueError(
            f"--metric-source is only read under --align-mode none; got --align-mode "
            f"{args.align_mode}, where the affine is fitted to GT instead."
        )
    if args.metric_source == "global_norm" and args.metric_norm is None:
        raise ValueError("--metric-source global_norm needs --metric-norm")
    if args.metric_source == "global_affine" and args.global_affine is None:
        raise ValueError("--metric-source global_affine needs --global-affine")
    if args.metric_source != "global_norm" and args.metric_norm is not None:
        raise ValueError("--metric-norm is only read by --metric-source global_norm")
    if args.metric_source != "global_affine" and args.global_affine is not None:
        raise ValueError("--global-affine is only read by --metric-source global_affine")
    if args.fit_global_affine is not None:
        if args.eval_checkpoint is None:
            raise ValueError("--fit-global-affine needs the checkpoint to fit, via --eval-checkpoint")
        if args.fit_affine_stride <= 0:
            raise ValueError("--fit-affine-stride must be positive")
        if args.fit_affine_split_list is None:
            raise ValueError("--fit-global-affine needs --fit-affine-split-list (train_list.txt)")
    elif args.fit_affine_split_list is not None:
        raise ValueError("--fit-affine-split-list is only read by --fit-global-affine")


def validate_args(args: argparse.Namespace) -> None:
    validate_metric_args(args)
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.micro_batch_size != 1:
        raise ValueError("This trainer runs micro-batch 1 (variable-length captions, fp32 U-Net).")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive.")
    if args.gt_loss_weight <= 0:
        raise ValueError("--gt-loss-weight must be positive; pure GT is the whole objective.")
    if args.smoke_updates is not None and args.overfit_steps is not None:
        raise ValueError("Choose either --smoke-updates or --overfit-steps, not both.")
    if args.smoke_updates is not None and "smoke" not in args.output_dir.name:
        raise ValueError("--smoke-updates requires an output dir name containing 'smoke'.")
    if args.overfit_steps is not None and "overfit" not in args.output_dir.name:
        raise ValueError("--overfit-steps requires an output dir name containing 'overfit'.")
    if args.input_max_edge and ROUTES[args.route][0] != "rgb":
        raise ValueError("--input-max-edge only applies to the RGB route.")
    if args.freeze_adapter:
        if ROUTES[args.route][1] == "vae":
            raise ValueError(f"Route {args.route} has no adapter to freeze.")
        if not ROUTES[args.route][3]:
            raise ValueError(
                f"Route {args.route} does not train the U-Net, so freezing the adapter "
                "would leave nothing trainable."
            )
        if args.init_from is None:
            raise ValueError(
                "--freeze-adapter without --init-from would freeze a freshly initialised "
                "adapter; pass the stage-1 checkpoint."
            )
    if args.resume is not None and args.init_from is not None:
        raise ValueError("--resume continues a run; --init-from starts a new one. Pick one.")
    if args.caption_rank_weight > 0 and args.caption_mode != "correct":
        raise ValueError(
            "--caption-rank-weight needs --caption-mode correct: the term compares the "
            f"frame's own caption against a wrong one, and mode '{args.caption_mode}' "
            "does not feed the frame its own caption."
        )
    if args.num_inference_steps != 1 and args.eval_checkpoint is None:
        # Every route was fine-tuned at one fixed timestep (--timestep, 999), the
        # same recipe upstream trains Lotus-G with (lotus/train_iris_g.py:1073).
        # A multi-step *training* run would need the timestep sampled, not just
        # the loop unrolled, so this stays an evaluation-only knob.
        raise ValueError(
            "--num-inference-steps > 1 is an evaluation-only knob (--eval-checkpoint). "
            "Training runs one forward at --timestep."
        )
    if args.latent_target:
        if args.pseudo_gt_dir is None:
            raise ValueError(
                "--latent-target needs --pseudo-gt-dir: the target is encoded as a whole "
                "image, and sparse lidar leaves almost every latent cell incomplete"
            )
        if args.pseudo_weight > 0:
            # The pixel-space split has no counterpart here: one latent cell spans 8x8
            # pixels, so lidar and completed depth cannot carry different weights.
            raise ValueError("--latent-target replaces the pixel-space terms; drop --pseudo-weight")
        if args.num_inference_steps != 1:
            raise ValueError("--latent-target trains at a single fixed timestep, as Lotus does")
    if args.pseudo_weight > 0 and args.pseudo_gt_dir is None:
        raise ValueError("--pseudo-weight needs --pseudo-gt-dir")
    if args.pseudo_gt_dir is not None and args.pseudo_weight <= 0 and not args.latent_target:
        # Otherwise the arm reads as completed supervision and trains as the baseline.
        # --latent-target is the one legitimate way to use the directory without a
        # weight: there the completed map is the target itself, not a second term.
        raise ValueError("--pseudo-gt-dir was given but --pseudo-weight is 0")
    if args.pseudo_weight > 0 and args.loss_exclude_top_rows > 0:
        # Excluded rows stop being real GT and would be handed to pseudo depth, so the
        # two switches would silently overwrite each other's region.
        raise ValueError("--pseudo-weight and --loss-exclude-top-rows both claim the top rows")
    if args.val_caption_mode == "fixed" and not (args.val_caption_text or "").strip():
        raise ValueError("--val-caption-mode fixed needs --val-caption-text")
    if args.val_caption_text is not None and args.val_caption_mode != "fixed":
        # Passing the text without the mode would score the real captions under a
        # label that names a sentence never fed to the model.
        raise ValueError("--val-caption-text only applies to --val-caption-mode fixed")
    if args.val_caption_mode in ("shuffled", "permuted") and args.eval_checkpoint is None:
        # The donor reassignment happens in run_evaluation, not in the per-epoch
        # run_validation, so a training run would silently score with each frame's
        # OWN caption and label the curve "shuffled". Fail instead of lying.
        raise ValueError(
            f"--val-caption-mode {args.val_caption_mode} only applies to an evaluation run "
            "(--eval-checkpoint); during training it would silently behave like 'correct'."
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def task_embedding(batch_size, device, dtype):
    task = torch.tensor([[1.0, 0.0]], device=device, dtype=dtype).repeat(batch_size, 1)
    return torch.cat([torch.sin(task), torch.cos(task)], dim=-1)


def environment_fingerprint(device: torch.device) -> dict:
    """Record what the run actually executed on.

    Training is chaotic over ~100k updates, so the same configuration on
    different hardware does not reproduce bit-for-bit. TF32 in particular is on
    by default from Ampere onward and silently drops fp32 matmuls to ~10 mantissa
    bits. Any comparison that crosses machines has to be calibrated first, and
    that is impossible after the fact unless the environment was written down.
    """
    fingerprint = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "platform": sys.platform,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(device)
        fingerprint.update(
            {
                "gpu_name": properties.name,
                "gpu_capability": f"{properties.major}.{properties.minor}",
                "gpu_memory_gb": round(properties.total_memory / 1024**3, 1),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    return fingerprint


def parameter_audit(module) -> dict:
    total = int(sum(p.numel() for p in module.parameters()))
    trainable = int(sum(p.numel() for p in module.parameters() if p.requires_grad))
    return {"parameters": total, "trainable": trainable, "frozen": total - trainable}


def downscaled_size(height: int, width: int, max_edge: int) -> tuple[int, int]:
    if max_edge <= 0 or max(height, width) <= max_edge:
        return height, width
    ratio = max_edge / max(height, width)
    new_h = max(8, int(height * ratio) // 8 * 8)
    new_w = max(8, int(width * ratio) // 8 * 8)
    return new_h, new_w


def learning_rate_factor(update: int, total_updates: int, args: argparse.Namespace) -> float:
    if update < args.warmup_updates:
        return (update + 1) / max(1, args.warmup_updates)
    if args.lr_schedule == "constant":
        return 1.0
    progress = (update - args.warmup_updates) / max(1, total_updates - args.warmup_updates)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine


# --------------------------------------------------------------------------- #
# manifests
# --------------------------------------------------------------------------- #


def read_manifest(
    path: Path, root: Path, modality: str, split: str | None, check_files: bool = True
) -> list[dict]:
    """Rows with the input path and the GT map of the *matching view*.

    Conclusion 15: the RGB route must be scored against RGB-view GT and the
    thermal routes against thermal-view GT. Mixing the two is not a fallback.
    """
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for manifest_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if split is not None and row.get("split") != split:
                raise ValueError(f"Expected split={split!r} but row {row.get('id')} is {row.get('split')!r}")
            if modality == "rgb":
                image_field = row.get("rgb_path")
                depth_field = row.get("rgb_depth_path") or row.get("depth_path")
                view = "rgb"
            else:
                image_field = row.get("thermal_path")
                depth_field = row.get("thermal_depth_path") or row.get("depth_path")
                view = "thr"
            if not image_field or not depth_field:
                raise ValueError(f"Row {row.get('id')} lacks a {modality} input or its {view}-view GT")
            if row.get("rgb_depth_path") and f"/{view}/" not in str(depth_field).replace("\\", "/"):
                raise ValueError(f"Row {row.get('id')}: GT {depth_field} is not the {view} view")
            image_path = root / image_field
            depth_path = root / depth_field
            # ~50k stat calls; cheap locally, slow on a network filesystem. Skip
            # once the payload has been verified on the target machine.
            if check_files:
                if not image_path.is_file():
                    raise FileNotFoundError(f"Missing input image: {image_path}")
                if not depth_path.is_file():
                    raise FileNotFoundError(f"Missing GT depth: {depth_path}")
            rows.append(
                {
                    "id": str(row["id"]),
                    "sequence": str(row.get("sequence", "")),
                    "manifest_index": manifest_index,
                    "image_path": image_path,
                    "depth_path": depth_path,
                    "caption": str(row.get("caption", "")),
                }
            )
    if not rows:
        raise ValueError(f"Manifest {path} produced no rows.")
    return rows


# --------------------------------------------------------------------------- #
# sample loading
# --------------------------------------------------------------------------- #


def load_input_tensor(row: dict, modality: str, args: argparse.Namespace):
    """Return (tensor in [-1,1] as [1,3,H,W], diagnostics)."""
    if modality == "thermal":
        thermal = thermal_to_lotus_input(row["image_path"], processing_res=0)
        if thermal.diagnostics["converted_uint8_std"] <= 0:
            raise RuntimeError(f"Constant thermal conversion: {row['id']}")
        return thermal.tensor, {"thermal": thermal.diagnostics}

    image = np.asarray(Image.open(row["image_path"]).convert("RGB"))
    height, width = image.shape[:2]
    if int(image.min()) == int(image.max()):
        raise RuntimeError(f"Constant RGB image: {row['id']}")
    tensor = torch.from_numpy(np.ascontiguousarray(image.astype(np.float32)))
    tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    target_hw = downscaled_size(height, width, args.input_max_edge)
    if target_hw != (height, width):
        tensor = F.interpolate(tensor, target_hw, mode="bilinear", align_corners=False)
    if target_hw[0] % 8 or target_hw[1] % 8:
        raise RuntimeError(f"RGB resolution {target_hw} not divisible by 8: {row['id']}")
    return tensor, {"native_hw": [height, width], "input_hw": list(target_hw)}


def load_sample(row: dict, modality: str, args: argparse.Namespace):
    tensor, diagnostics = load_input_tensor(row, modality, args)
    gt_disparity, valid_mask = load_gt_disparity(
        row["depth_path"], args.gt_min_depth, args.gt_max_depth, args.depth_scale
    )
    diagnostics.update({"id": row["id"], "gt_valid_pixels": int(valid_mask.sum())})
    sky_mask = None
    if getattr(args, "sky_mask_dir", None) is not None:
        path = args.sky_mask_dir / f"{row['id']}.png"
        if path.exists():
            sky = np.asarray(Image.open(path), dtype=np.uint8) > 127
            sky_mask = torch.from_numpy(sky.astype(np.float32))[None]
            diagnostics["sky_pixels"] = int(sky.sum())
        else:
            # A frame with no sky at all is normal here (24 of 100 in the probe:
            # tunnels, canopy, tall buildings), so a missing file is not an error
            # -- but it must not silently become "sky everywhere" either.
            diagnostics["sky_pixels"] = 0
    pseudo_disparity = pseudo_mask = None
    if getattr(args, "pseudo_gt_dir", None) is not None and args.pseudo_weight > 0:
        path = args.pseudo_gt_dir / f"{row['id']}.npy"
        if not path.is_file():
            # A missing file would quietly turn this arm back into the baseline.
            raise FileNotFoundError(f"No pseudo depth for {row['id']}: {path}")
        depth = np.load(path, allow_pickle=False).astype(np.float32)
        low = args.pseudo_min_depth if args.pseudo_min_depth is not None else args.gt_min_depth
        high = args.pseudo_max_depth if args.pseudo_max_depth is not None else args.gt_max_depth
        usable = np.isfinite(depth) & (depth > low)
        if args.pseudo_range_mode == "clip":
            depth = np.clip(depth, low, high)
        else:
            usable &= depth < high
        # Where lidar spoke, lidar wins: completion only fills what it left empty.
        keep = usable & (valid_mask[0].numpy() < 0.5)
        filled = np.zeros_like(depth)
        filled[keep] = 1.0 / depth[keep]
        pseudo_disparity = torch.from_numpy(filled)[None]
        pseudo_mask = torch.from_numpy(keep.astype(np.float32))[None]
        diagnostics["pseudo_pixels"] = int(keep.sum())
        diagnostics["pseudo_fraction"] = float(keep.mean())
    dense_target = None
    if getattr(args, "latent_target", False):
        path = args.pseudo_gt_dir / f"{row['id']}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"No pseudo depth for {row['id']}: {path}")
        pseudo = np.load(path, allow_pickle=False).astype(np.float32)
        real = valid_mask[0].numpy() > 0.5
        gt_depth = np.zeros_like(pseudo)
        gt_depth[real] = 1.0 / gt_disparity[0].numpy()[real]
        # Encoded as one image, so every pixel must carry a value: out-of-range
        # pseudo depth is clipped rather than dropped -- there is nowhere to drop to.
        dense = np.clip(np.where(real, gt_depth, pseudo), args.gt_min_depth, args.gt_max_depth)
        disparity = 1.0 / dense
        low, high = float(disparity.min()), float(disparity.max())
        # norm_type='disparity' from HypersimImageDepthNormalTransform, the convention
        # this checkpoint was trained under.
        dense_target = torch.from_numpy(((disparity - low) / (high - low + 1e-5) - 0.5) * 2.0)[None]
        diagnostics["latent_target_depth_range_m"] = [float(dense.min()), float(dense.max())]
        diagnostics["latent_target_real_fraction"] = float(real.mean())
    rows_out = int(getattr(args, "loss_exclude_top_rows", 0))
    if rows_out > 0:
        # Only the training loss loses these rows. run_validation reads the GT
        # PNG itself and scores the official protocol untouched, so the curve
        # stays comparable with every other run.
        dropped = int(valid_mask[..., :rows_out, :].sum())
        valid_mask[..., :rows_out, :] = 0.0
        diagnostics["gt_pixels_dropped_top_rows"] = dropped
        diagnostics["gt_valid_pixels_after_exclusion"] = int(valid_mask.sum())
    return (tensor, gt_disparity, valid_mask, sky_mask,
            pseudo_disparity, pseudo_mask, dense_target, diagnostics)


# --------------------------------------------------------------------------- #
# condition construction (the only place routes differ)
# --------------------------------------------------------------------------- #


class RouteModel:
    """Owns the modules a route needs and produces its U-Net condition."""

    def __init__(self, args, device, frozen_dtype):
        self.args = args
        self.device = device
        self.frozen_dtype = frozen_dtype
        modality, condition, train_adapter, train_unet = ROUTES[args.route]
        # Staged schedule: the adapter exists and is used, it just does not get
        # gradients in this stage. Tools that build a RouteModel without this
        # flag (the region analyser) keep the route's own answer.
        if getattr(args, "freeze_adapter", False):
            train_adapter = False
        self.modality = modality
        self.condition_kind = condition
        self.trains_adapter = train_adapter
        self.trains_unet = train_unet

        from pipeline import LotusDPipeline, LotusGPipeline  # noqa: E402

        _repo = str(args.lotus_model_path).lower()
        if args.backbone == "g" and "depth-d" in _repo:
            raise SystemExit(
                f"--backbone g with {args.lotus_model_path}: that is a Lotus-D "
                "checkpoint. Pass --backbone d."
            )
        if args.backbone == "d" and "depth-g" in _repo:
            raise SystemExit(
                f"--backbone d with {args.lotus_model_path}: that is a Lotus-G "
                "checkpoint. Pass --backbone g."
            )
        if (args.backbone == "marigold") != ("marigold" in _repo):
            raise SystemExit(
                f"--backbone {args.backbone} with {args.lotus_model_path}: the backbone "
                "and the repository disagree about whether this is Marigold."
            )
        _pipeline_cls = LotusDPipeline if args.backbone == "d" else LotusGPipeline

        self.lotus = _pipeline_cls.from_pretrained(
            args.lotus_model_path,
            torch_dtype=frozen_dtype,
            local_files_only=args.local_files_only,
        ).to(device)
        for module in (self.lotus.vae, self.lotus.text_encoder, self.lotus.unet):
            module.requires_grad_(False).eval()

        # The U-Net used for the forward pass is always an fp32 copy, trainable
        # or not: on the adapter-only routes the adapter's gradient has to travel
        # back through it, and an fp16 backward is exactly what silently zeroed
        # the GT gradient before (frozen doc 3.11).
        self.unet = copy.deepcopy(self.lotus.unet).to(device=device, dtype=torch.float32)
        if train_unet:
            self.unet.train().requires_grad_(True)
        else:
            self.unet.eval().requires_grad_(False)
        # The pipeline's own fp16 U-Net is never used again (no teacher here), so
        # park it on the CPU instead of paying 1.7 GB of VRAM for a dead copy.
        self.lotus.unet = self.lotus.unet.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # fp32 decoder copy: the fp16 backward through the VAE underflows the GT
        # gradient to exactly zero on most steps.
        self.gt_vae = None
        if args.gt_decode_fp32:
            self.gt_vae = copy.deepcopy(self.lotus.vae).to(device=device, dtype=torch.float32)
            self.gt_vae.requires_grad_(False).eval()
            self.gt_vae.encoder = None

        self.adapter = None
        self.anythermal = None
        if condition == "vae_adapter":
            from models.thermal_vae_latent_adapter import ThermalVAELatentAdapter

            self.adapter = ThermalVAELatentAdapter().to(device=device, dtype=torch.float32)
            self.adapter.train().requires_grad_(train_adapter)
            if not train_adapter:
                self.adapter.eval()
        elif condition == "anythermal_adapter":
            from models.anythermal_encoder import AnyThermalEncoder
            from models.anythermal_lotus_adapter_v2_3 import AnyThermalLotusAdapterV23

            self.anythermal = AnyThermalEncoder(
                model_path=args.anythermal_model_path,
                device=str(device),
                local_files_only=args.local_files_only,
            )
            self.anythermal.model.requires_grad_(False).eval()
            self.adapter = AnyThermalLotusAdapterV23().to(device=device, dtype=torch.float32)
            self.adapter.train().requires_grad_(train_adapter)
            if not train_adapter:
                self.adapter.eval()

    # -- prompts ---------------------------------------------------------- #

    def encode_prompt(self, text: str) -> torch.Tensor:
        encoded, _ = self.lotus.encode_prompt(
            prompt=text,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=None,
        )
        return encoded.detach().float()

    # -- condition -------------------------------------------------------- #

    def vae_latent(self, image_tensor: torch.Tensor) -> torch.Tensor:
        encoder_dtype = next(self.lotus.vae.encoder.parameters()).dtype
        with torch.no_grad():
            posterior = self.lotus.vae.encode(
                image_tensor.to(device=self.device, dtype=encoder_dtype)
            ).latent_dist
            # Upstream draws from the posterior here, in training and in the
            # pipeline alike (`lotus/pipeline.py:910/1121/1297`). Every number
            # this project has published was taken at the mode, so that stays the
            # default; `sample` exists to score a checkpoint trained upstream's
            # way without also changing the exam.
            sampled = getattr(self.args, "condition_latent", "mode") == "sample"
            latent = (posterior.sample() if sampled else posterior.mode())
            latent = latent * self.lotus.vae.config.scaling_factor
        return latent.float()

    def condition(
        self,
        row: dict,
        image_tensor: torch.Tensor,
        donor_row: dict | None = None,
        donor_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the U-Net's condition. A donor, when given, replaces the source
        the shuffle mode names -- see --shuffle-condition."""
        mode = getattr(self.args, "shuffle_condition", "none")
        if donor_row is None:
            mode = "none"

        if self.condition_kind in ("vae", "vae_adapter"):
            if mode == "anythermal":
                raise SystemExit(
                    "--shuffle-condition anythermal needs a route with an AnyThermal "
                    f"branch; {self.args.route} has none. Use 'all'."
                )
            source = donor_tensor if mode == "all" else image_tensor
            latent = self.vae_latent(source)
            if self.condition_kind == "vae_adapter":
                return self.adapter(latent)
            return latent

        # AnyThermal branch: 'all' swaps both the feature source and the adapter's
        # thermal input; 'anythermal' swaps only the features, so a null result
        # cannot be blamed on the adapter losing its image.
        feature_row = donor_row if mode in ("all", "anythermal") else row
        features, _, diagnostics = extract_anythermal_feature_pyramid(
            self.anythermal, feature_row["image_path"], enable_grad=False
        )
        if diagnostics["converted_uint8_std"] <= 0:
            raise RuntimeError(f"Constant AnyThermal conversion: {row['id']}")
        features = [feature.detach().float().to(self.device) for feature in features]
        # target latent grid = whatever the frozen VAE would have produced
        target_size = (image_tensor.shape[-2] // 8, image_tensor.shape[-1] // 8)
        thermal_source = donor_tensor if mode == "all" else image_tensor
        thermal = thermal_source.to(device=self.device, dtype=torch.float32)
        return self.adapter(features, thermal, target_size)

    # -- forward ---------------------------------------------------------- #

    def predict_disparity(
        self,
        row: dict,
        image_tensor: torch.Tensor,
        prompt: torch.Tensor,
        donor_row: dict | None = None,
        donor_tensor: torch.Tensor | None = None,
        target_latent: torch.Tensor | None = None,
        return_latent: bool = False,
    ):
        condition = self.condition(row, image_tensor, donor_row, donor_tensor)
        noise = seeded_noise(
            (1, *condition.shape[1:]),
            seed=self.args.seed + int(row["manifest_index"]),
            device=self.device,
            dtype=torch.float32,
            scale=float(self.lotus.scheduler.init_noise_sigma),
        )
        steps = max(1, int(getattr(self.args, "num_inference_steps", 1)))
        if steps > 1 and getattr(self.args, "backbone", "g") == "d":
            raise SystemExit(
                "--num-inference-steps > 1 with --backbone d: Lotus-D is a direct "
                "single-step predictor, there is no denoising trajectory to unroll."
            )
        if steps == 1:
            timesteps = [torch.full((1,), self.args.timestep, device=self.device, dtype=torch.long)]
        else:
            # Evaluation only -- validate_args refuses this during training.
            self.lotus.scheduler.set_timesteps(steps, device=self.device)
            timesteps = list(self.lotus.scheduler.timesteps)
        unet_dtype = next(self.unet.parameters()).dtype
        if target_latent is not None:
            # train_iris_g.py:1078 -- the input is the target noised at this timestep,
            # not pure noise. At t=999 alpha_bar is about 4.7e-5, so the target
            # contributes under 1% of the input; Lotus trains this way and infers
            # from pure noise, and its released weights work under that mismatch.
            latents = self.lotus.scheduler.add_noise(
                target_latent.to(self.device, torch.float32), noise, timesteps[0]
            )
        else:
            latents = noise
        for timestep in timesteps:
            latent_input = self.lotus.scheduler.scale_model_input(latents, timestep).detach()
            # The one architectural difference, mirroring train_iris_ms2_g.py:
            # G is fed [image latent, noisy target latent], D the image latent alone.
            unet_input = (
                condition if self.args.backbone == "d"
                else torch.cat([condition, latent_input], dim=1)
            )
            # Marigold has no task switcher and its U-Net was built without a
            # class embedding, so it must be called without class_labels.
            x0 = self.unet(
                unet_input.to(unet_dtype),
                timestep,
                encoder_hidden_states=prompt.to(unet_dtype),
                class_labels=(
                    None if self.args.backbone in ("marigold", "e2eft")
                    else task_embedding(1, self.device, unet_dtype)
                ),
                return_dict=False,
            )[0]
            # Same branch as LotusGPipeline.__call__: the checkpoint's DDIM config
            # declares prediction_type='sample', so x0 goes straight into step().
            # The bypass is only valid when the network already emits the clean
            # latent. Lotus declares prediction_type='sample' and does; Marigold
            # and E2E-FT declare 'v_prediction', so what comes out is a velocity
            # and taking it for x0 would be silently wrong -- those go through
            # step(), which undoes the parameterisation, even at one step.
            _needs_step = (
                len(timesteps) > 1
                or self.lotus.scheduler.config.prediction_type != "sample"
            )
            latents = (
                self.lotus.scheduler.step(x0.float(), timestep, latents, return_dict=False)[0]
                if _needs_step
                else x0
            )
        disparity = decode_to_disparity(self.lotus, latents.float(), self.device, gt_vae=self.gt_vae)
        return (disparity, latents.float()) if return_latent else disparity

    def encode_depth_target(self, normalised: torch.Tensor) -> torch.Tensor:
        """VAE-encode a [-1,1] depth map into the latent the loss scores against.

        train_iris_g.py:1038 draws from the posterior rather than taking its mode,
        so a fresh sample per step is part of the recipe; that is also why this runs
        here instead of being precomputed once and frozen.
        """
        image = normalised.to(self.device, torch.float32)[None].repeat(1, 3, 1, 1)
        with torch.no_grad():
            posterior = self.lotus.vae.encode(image.to(self.lotus.vae.dtype)).latent_dist
            return (posterior.sample() * self.lotus.vae.config.scaling_factor).float()

    def trainable_modules(self) -> dict:
        modules = {}
        if self.trains_unet:
            modules["unet"] = self.unet
        if self.trains_adapter:
            modules["adapter"] = self.adapter
        if not modules:
            raise RuntimeError(f"Route {self.args.route} has no trainable module.")
        return modules

    def persisted_modules(self) -> dict:
        """What a checkpoint must carry to reproduce this run's forward path.

        The trainable modules, plus the adapter whenever one exists: a frozen
        adapter in a staged run holds stage 1's weights, and without it the
        checkpoint would silently evaluate against a freshly initialised one.
        For every non-staged route this is exactly `trainable_modules()`.
        """
        modules = dict(self.trainable_modules())
        if self.adapter is not None:
            modules["adapter"] = self.adapter
        return modules

    def load_state_dicts(self, state_dicts: dict, context: str) -> list[str]:
        """Load every module this route shares with `state_dicts`. Returns names."""
        loaded = []
        for name, module in self.persisted_modules().items():
            if name not in state_dicts:
                continue
            module.load_state_dict(state_dicts[name], strict=True)
            loaded.append(name)
        if not loaded:
            raise SystemExit(
                f"{context}: checkpoint carries {sorted(state_dicts)} but this route "
                f"needs {sorted(self.persisted_modules())} -- nothing to load."
            )
        return loaded

    def set_train(self, mode: bool) -> None:
        for module in self.trainable_modules().values():
            module.train(mode)


# --------------------------------------------------------------------------- #
# validation (official BMSD protocol, inline)
# --------------------------------------------------------------------------- #


def rotate_captions(rows: list[dict]) -> dict:
    """Give every frame a distant frame's caption (deterministic half-set rotation).

    A random shuffle is not safe here: MS2 frames sit ~0.5 m apart, so a nearby
    donor would hand back an almost-correct caption and collapse the contrast.
    Rotating by half the set puts the donor kilometres away.
    """
    offset = len(rows) // 2
    captions = [row["caption"] for row in rows]
    missing = sum(1 for caption in captions if not caption.strip())
    if missing:
        raise SystemExit(f"--val-caption-mode shuffled needs captions on every row; {missing} lack one")
    for index, row in enumerate(rows):
        row["donor_id"] = rows[(index + offset) % len(rows)]["id"]
        row["caption"] = captions[(index + offset) % len(rows)]
    return {
        "rotation_offset": offset,
        "frames": len(rows),
        "self_assignments": sum(1 for row in rows if row["donor_id"] == row["id"]),
    }


def permute_captions(rows: list[dict], seed: int) -> dict:
    """Give every frame a UNIFORMLY RANDOM other frame's caption.

    The half-set rotation above turns out to be a contaminated control: measured
    with `probe_caption_scale_information.py`, a rotated caption still predicts
    the recipient frame's median GT depth at R^2 0.32 (MS2) / 0.46 (RGBDT500),
    against 0.55 / 0.69 for the frame's own caption and about -0.1 for a uniform
    permutation. The reason is geometric -- these drives double back, and
    RGBDT500's frames come from only 100 videos -- so a donor "kilometres along
    the path" can be metres away in space and describe the same scene.

    A uniform permutation has no such structure, which makes it the control that
    actually isolates caption content. Self-assignments are swapped away.
    """
    captions = [row["caption"] for row in rows]
    missing = sum(1 for caption in captions if not caption.strip())
    if missing:
        raise SystemExit(f"permuted captions need a caption on every row; {missing} lack one")
    order = np.random.default_rng(seed).permutation(len(rows))
    for index in range(len(rows)):
        if order[index] == index:                       # swap with a neighbour
            partner = (index + 1) % len(rows)
            order[index], order[partner] = order[partner], order[index]
    for index, row in enumerate(rows):
        donor = int(order[index])
        row["donor_id"] = rows[donor]["id"]
        row["caption"] = captions[donor]
    return {
        "permutation_seed": seed,
        "frames": len(rows),
        "self_assignments": sum(1 for row in rows if row["donor_id"] == row["id"]),
        "median_donor_offset_frames": float(
            np.median([abs(int(order[i]) - i) for i in range(len(rows))])
        ),
    }


EVALUATION_LABELS = {
    "ssi_disparity": "affine_invariant",
    "ssi": "affine_invariant_depth_space",
    "median": "metric_median_scaled",
    "none": "metric_no_test_alignment",
}
# Smallest inverse depth let through the reciprocal. Deliberately far below
# 1/max_depth so that a prediction beyond the evaluation range still reaches
# `official_depth_errors`' own clamp and is counted there, rather than being
# silently pinned to 80 m here where nothing would report it.
INVERSE_DEPTH_FLOOR = 1e-6


def resolve_metric_affine(args) -> dict | None:
    """The (a, b) turning a decoded [0, 1] prediction into inverse depth, 1/m.

    All three sources are the same affine `q = a*y + b`; they differ only in
    where the two numbers came from, so the evaluator resolves them once here
    and records the provenance beside the result.  None of them reads GT.
    """
    if args.align_mode != "none":
        return None
    if args.metric_source == "raw_inverse":
        return {
            "metric_source": "raw_inverse",
            "a": 1.0,
            "b": 0.0,
            "provenance": {
                "note": "no calibration: the decoded [0,1] output is read as 1/m directly",
            },
        }
    if args.metric_source == "global_norm":
        norm = MetricNorm.load(args.metric_norm)
        a, b = norm.as_affine()
        return {
            "metric_source": "global_norm",
            "a": a,
            "b": b,
            "provenance": {
                "artifact": str(args.metric_norm),
                "source_split": norm.source_split,
                "source_manifest": norm.source_manifest,
                "q_lo": norm.q_lo,
                "q_hi": norm.q_hi,
                "quantiles": [norm.quantile_lo, norm.quantile_hi],
                "valid_pixels": norm.valid_pixels,
                "frames": norm.frames,
            },
        }
    payload = json.loads(Path(args.global_affine).read_text(encoding="utf-8"))
    if payload.get("schema") != GLOBAL_AFFINE_SCHEMA:
        raise SystemExit(f"{args.global_affine}: schema {payload.get('schema')!r}")
    if payload.get("source_split") != "train":
        raise SystemExit(
            f"{args.global_affine} was fitted on {payload.get('source_split')!r}; the "
            "calibration baseline may only be fitted on train."
        )
    return {
        "metric_source": "global_affine",
        "a": float(payload["a"]),
        "b": float(payload["b"]),
        "provenance": {"artifact": str(args.global_affine), **payload},
    }


def prediction_to_depth(pred_y: np.ndarray, metric: dict, args) -> tuple[np.ndarray, int, int]:
    """Decoded [0, 1] -> metres, with no GT anywhere in the arithmetic.

    Returns `(depth, non_positive_pixels, floored_pixels)`.  The only clamp is
    the positivity floor; the evaluation-range clamp stays where it has always
    been, inside `official_depth_errors`, so its counters keep meaning what they
    have always meant.
    """
    q = metric["a"] * pred_y.astype(np.float64) + metric["b"]
    non_positive = int(np.count_nonzero(q <= 0))
    floored = int(np.count_nonzero(q < INVERSE_DEPTH_FLOOR))
    return 1.0 / np.maximum(q, INVERSE_DEPTH_FLOOR), non_positive, floored


@torch.no_grad()
def run_validation(
    model: RouteModel,
    rows: list[dict],
    prompts: dict,
    args,
    per_sample: list | None = None,
    raw_dir: Path | None = None,
    donors: dict | None = None,
    metric: dict | None = None,
) -> dict:
    model.set_train(False)
    accumulator: dict[str, float] = {}
    count = 0
    for row in rows:
        image_tensor, _ = load_input_tensor(row, model.modality, args)
        if args.val_caption_mode == "empty":
            prompt = prompts["empty"]
        elif args.val_caption_mode == "fixed":
            prompt = prompts["fixed"]
        else:
            # 'shuffled' rows already carry the donor caption (see rotate_captions)
            prompt = model.encode_prompt(row["caption"])
        donor_row = donor_tensor = None
        if donors is not None:
            donor_row = donors[row["id"]]
            donor_tensor, _ = load_input_tensor(donor_row, model.modality, args)
        prediction = model.predict_disparity(row, image_tensor, prompt, donor_row, donor_tensor)
        if raw_dir is not None:
            # Native resolution, before the resize to GT below: that is what
            # run_ms2_lotus_*_official.py --save-raw-pred writes, and
            # analyze_prediction_regions.py resizes to the GT shape itself.
            np.save(raw_dir / f"{row['id']}.npy", prediction.float().cpu().numpy().astype(np.float32))
        gt_metres = np.asarray(Image.open(row["depth_path"]), dtype=np.float32) / args.depth_scale
        pred = prediction[None, None]
        if pred.shape[-2:] != gt_metres.shape:
            pred = F.interpolate(pred, gt_metres.shape, mode="bilinear", align_corners=False)
        pred_y = pred[0, 0].float().cpu().numpy()
        if args.align_mode == "none":
            # The whole of the metric path: two frozen numbers and a reciprocal.
            # `gt_metres` is not read until `evaluate_sample`, and then only to
            # score. Nothing above this line has touched it.
            pred_for_eval, non_positive, floored = prediction_to_depth(pred_y, metric, args)
        else:
            pred_for_eval, non_positive, floored = pred_y, 0, 0
        metrics = evaluate_sample(
            pred_for_eval,
            gt_metres,
            align=args.align_mode,
            min_depth=args.eval_min_depth,
            max_depth=args.eval_max_depth,
        )
        if args.align_mode == "none":
            metrics["non_positive_inverse_depth"] = non_positive
            metrics["inverse_depth_floored"] = floored
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key != "align_mode":
                accumulator[key] = accumulator.get(key, 0.0) + float(value)
        count += 1
        if per_sample is not None:
            per_sample.append({"id": row["id"], "sequence": row["sequence"], **{
                k: v for k, v in metrics.items() if k != "align_mode"
            }})
    model.set_train(True)
    if not count:
        raise RuntimeError("Validation subset is empty.")
    return {key: value / count for key, value in accumulator.items()} | {"val_samples": count}


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #


@torch.no_grad()
def fit_global_affine(model: RouteModel, rows: list[dict], prompts: dict, args, output: Path) -> None:
    """One dataset-level `q_GT ~ a*y + b`, fitted on TRAIN predictions and frozen.

    The baseline this answers is: how much of the metric gap closes with a single
    calibration of the existing network, no retraining at all?  If the adapted
    model cannot beat this, the adaptation bought nothing that a lookup table
    would not have.

    Fitted by accumulating the five sufficient statistics of a least-squares
    line, so every valid pixel of every visited frame contributes exactly once
    and nothing is held in memory.  One fit over the pooled pixels -- not a fit
    per frame averaged afterwards, which is a different and much weaker
    constraint, and not the per-frame fit the affine-invariant path uses.
    """
    model.set_train(False)
    sequences = sorted({row["sequence"] for row in rows})
    allowed = {
        line.strip().lstrip("_")
        for line in args.fit_affine_split_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    stray = [s for s in sequences if s.lstrip("_") not in allowed]
    if stray:
        raise SystemExit(
            f"These sequences are not in {args.fit_affine_split_list.name}: {stray}\n"
            "The calibration baseline may only be fitted on the training split."
        )
    print(
        f"[affine] fitting on {len(rows)} TRAIN frames over {len(sequences)} sequences, "
        f"stride {args.fit_affine_stride}",
        flush=True,
    )
    s_pp = s_p = s_n = s_pq = s_q = 0.0
    frames_used = 0
    for position, row in enumerate(rows):
        image_tensor, _ = load_input_tensor(row, model.modality, args)
        if args.val_caption_mode == "empty":
            prompt = prompts["empty"]
        elif args.val_caption_mode == "fixed":
            prompt = prompts["fixed"]
        else:
            prompt = model.encode_prompt(row["caption"])
        prediction = model.predict_disparity(row, image_tensor, prompt)
        gt_metres = np.asarray(Image.open(row["depth_path"]), dtype=np.float32) / args.depth_scale
        pred = prediction[None, None]
        if pred.shape[-2:] != gt_metres.shape:
            pred = F.interpolate(pred, gt_metres.shape, mode="bilinear", align_corners=False)
        y = pred[0, 0].float().cpu().numpy().astype(np.float64)
        valid = official_valid_mask(gt_metres, args.eval_min_depth, args.eval_max_depth)
        if not valid.any():
            continue
        p = y[valid]
        q = 1.0 / gt_metres[valid].astype(np.float64)
        s_pp += float(np.sum(p * p))
        s_p += float(np.sum(p))
        s_n += float(p.size)
        s_pq += float(np.sum(p * q))
        s_q += float(np.sum(q))
        frames_used += 1
        if (position + 1) % 200 == 0:
            print(f"[affine]   {position + 1}/{len(rows)} frames, {int(s_n):,} pixels", flush=True)

    det = s_pp * s_n - s_p * s_p
    if det <= 0 or not np.isfinite(det):
        raise SystemExit(f"Degenerate global affine fit (det={det})")
    a = (s_n * s_pq - s_p * s_q) / det
    b = (-s_p * s_pq + s_pp * s_q) / det
    if not np.isfinite(a) or not np.isfinite(b):
        raise SystemExit(f"Global affine fit produced non-finite (a, b) = ({a}, {b})")
    payload = {
        "schema": GLOBAL_AFFINE_SCHEMA,
        "a": float(a),
        "b": float(b),
        "form": "q_inverse_depth_per_metre = a * y_decoded + b",
        "source_split": "train",
        "source_manifest": str(args.val_manifest),
        "frame_stride": args.fit_affine_stride,
        "frames_visited": len(rows),
        "frames_used": frames_used,
        "sequences": sequences,
        "valid_pixels": int(s_n),
        "checkpoint": str(args.eval_checkpoint),
        "route": args.route,
        "caption_mode": args.val_caption_mode,
        "condition_latent": args.condition_latent,
        "num_inference_steps": args.num_inference_steps,
        "eval_min_depth": args.eval_min_depth,
        "eval_max_depth": args.eval_max_depth,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": (
            "One affine for the whole dataset, fitted on pooled TRAIN lidar pixels and "
            "frozen. Applied unchanged to val and test. No val or test GT entered this fit."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[affine] a={a:.6f} b={b:.6f} 1/m over {int(s_n):,} pixels from {frames_used} frames\n"
        f"[affine] y=0 -> {1.0 / max(b, INVERSE_DEPTH_FLOOR):.2f} m, "
        f"y=1 -> {1.0 / max(a + b, INVERSE_DEPTH_FLOOR):.2f} m\n"
        f"[affine] -> {output}",
        flush=True,
    )


def run_evaluation(model: RouteModel, val_rows: list[dict], prompts: dict, args, output: Path) -> None:
    """Load a checkpoint and score it with the same forward path training uses."""
    if not val_rows:
        raise SystemExit("--eval-checkpoint needs a validation set; drop --skip-val.")
    checkpoint = torch.load(args.eval_checkpoint.resolve(), map_location="cpu", weights_only=False)
    if checkpoint.get("route") != args.route:
        raise SystemExit(
            f"Checkpoint route {checkpoint.get('route')!r} does not match --route {args.route!r}"
        )
    loaded = model.load_state_dicts(checkpoint["state_dicts"], str(args.eval_checkpoint))
    print(
        f"[eval] loaded {', '.join(loaded)} from checkpoint",
        flush=True,
    )
    print(
        f"[eval] {args.eval_checkpoint} (epoch {checkpoint.get('epoch')}) "
        f"on {len(val_rows)} frames from {args.val_manifest.name}",
        flush=True,
    )

    rotation = None
    if args.val_caption_mode == "fixed":
        # A COMPLETED job has told us nothing about whether a switch took effect,
        # and this one is a string: print what every frame actually received.
        print(f"[eval] fixed caption for all frames: {args.val_caption_text!r}", flush=True)
    if args.val_caption_mode == "shuffled":
        rotation = rotate_captions(val_rows)
        print(
            f"[eval] shuffled captions: rotated by {rotation['rotation_offset']} frames, "
            f"{rotation['self_assignments']} self-assignments",
            flush=True,
        )
    elif args.val_caption_mode == "permuted":
        rotation = permute_captions(val_rows, args.seed)
        print(
            f"[eval] permuted captions: uniform random donors (seed {args.seed}), "
            f"median donor offset {rotation['median_donor_offset_frames']:.0f} frames, "
            f"{rotation['self_assignments']} self-assignments",
            flush=True,
        )

    donors = None
    if args.shuffle_condition != "none":
        # Uniform random donor, self-assignments swapped away -- same construction
        # as permute_captions, so the two controls are read on the same footing.
        order = np.random.default_rng(args.seed).permutation(len(val_rows))
        for index in range(len(val_rows)):
            if order[index] == index:
                partner = (index + 1) % len(val_rows)
                order[index], order[partner] = order[partner], order[index]
        donors = {row["id"]: val_rows[int(order[i])] for i, row in enumerate(val_rows)}
        self_hits = sum(1 for r in val_rows if donors[r["id"]]["id"] == r["id"])
        print(
            f"[eval] shuffle-condition {args.shuffle_condition}: uniform random donors "
            f"(seed {args.seed}), {self_hits} self-assignments",
            flush=True,
        )

    raw_dir = None
    if args.save_raw_pred:
        raw_dir = output / "raw_predictions"
        raw_dir.mkdir(parents=True, exist_ok=True)
        print(f"[eval] saving raw predictions to {raw_dir}", flush=True)

    metric = resolve_metric_affine(args)
    label = EVALUATION_LABELS[args.align_mode]
    if metric is not None:
        print(
            f"[eval] {label} / {metric['metric_source']}: "
            f"q = {metric['a']:.6f} * y + {metric['b']:.6f} 1/m "
            f"(y=0 -> {1.0 / max(metric['b'], INVERSE_DEPTH_FLOOR):.2f} m, "
            f"y=1 -> {1.0 / max(metric['a'] + metric['b'], INVERSE_DEPTH_FLOOR):.2f} m). "
            f"No GT is read before the metrics.",
            flush=True,
        )
    else:
        print(f"[eval] {label}: per-frame affine fitted to the evaluated split's GT", flush=True)

    per_sample: list[dict] = []
    started = time.time()
    metrics = run_validation(
        model, val_rows, prompts, args, per_sample=per_sample, raw_dir=raw_dir,
        donors=donors, metric=metric,
    )
    metrics.update(
        {
            "evaluation_mode": label,
            "align_mode": args.align_mode,
            "metric_calibration": metric,
            "test_gt_used_for_fitting": args.align_mode != "none",
            "route": args.route,
            "checkpoint": str(args.eval_checkpoint),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "raw_predictions_saved": bool(args.save_raw_pred),
            "shuffle_condition": args.shuffle_condition,
            "val_manifest": str(args.val_manifest),
            "val_stride": args.val_stride,
            "val_caption_mode": args.val_caption_mode,
            "val_caption_text": args.val_caption_text,
            "caption_rotation": rotation,
            "num_inference_steps": args.num_inference_steps,
            "timestep": args.timestep if args.num_inference_steps == 1 else None,
            # Part of the exam, not of the model: a result read without it cannot
            # be placed beside the rest of the table.
            "condition_latent": args.condition_latent,
            "environment": environment_fingerprint(model.device),
            "elapsed_seconds": time.time() - started,
            "protocol": (
                f"official BMSD {args.align_mode}, min_depth {args.eval_min_depth}, "
                f"max_depth {args.eval_max_depth}"
            ),
        }
    )
    # An affine-invariant run keeps the historical filenames, so every consumer
    # of eval_<tag>.json -- iris_ms2_pipeline.sbatch's skip logic included --
    # goes on reading what it always read. A metric run writes beside it under
    # its own label: the two exams are not interchangeable and must never land
    # on the same path.
    # The label alone is not enough to name the file. raw_inverse, global_affine
    # and global_norm are all `metric_no_test_alignment`, and on their first
    # outing all three landed on one path and silently overwrote each other --
    # three different exams, one filename, last writer wins. The calibration
    # source is part of what a result is, so it is part of what it is called.
    suffix = "" if args.align_mode == "ssi_disparity" else f"_{label}"
    if metric is not None:
        suffix += f"_{metric['metric_source']}"
    json_path = output / f"eval_{args.eval_tag}{suffix}.json"
    csv_path = output / f"eval_{args.eval_tag}{suffix}_per_sample.csv"
    if json_path.exists():
        old = json.loads(json_path.read_text(encoding="utf-8"))
        previous = old.get("evaluation_mode")
        previous_source = (old.get("metric_calibration") or {}).get("metric_source")
        current_source = metric["metric_source"] if metric else None
        if (previous, previous_source) != (label, current_source) and previous is not None:
            raise SystemExit(
                f"{json_path} already holds a {previous!r}/{previous_source!r} result and "
                f"this run is {label!r}/{current_source!r}. Refusing to overwrite one exam "
                "with the other; change --eval-tag or the output directory."
            )
    json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_sample[0]))
        writer.writeheader()
        writer.writerows(per_sample)
    print(
        f"[eval] {label}: abs_rel {metrics['abs_rel']:.4f}  sq_rel {metrics['sq_rel']:.4f}  "
        f"rmse {metrics['rmse']:.3f}  rmse_log {metrics['rmse_log']:.4f}  "
        f"a1 {metrics['a1']:.4f}  ({metrics['val_samples']} frames, "
        f"{metrics['elapsed_seconds'] / 60:.1f} min) -> {json_path.name}",
        flush=True,
    )


def weights_payload(model: RouteModel, args, epoch: int, manifest_hash: str, metrics: dict | None) -> dict:
    return {
        "format": CHECKPOINT_FORMAT,
        "route": args.route,
        "epoch": epoch,
        "manifest_sha256": manifest_hash,
        "caption_mode": args.caption_mode,
        "val_metrics": metrics,
        "trainable_modules": sorted(model.trainable_modules()),
        "state_dicts": {
            name: {key: value.detach().cpu() for key, value in module.state_dict().items()}
            for name, module in model.persisted_modules().items()
        },
    }


def save_weights(path: Path, model: RouteModel, args, epoch: int, manifest_hash: str, metrics: dict | None) -> None:
    torch.save(weights_payload(model, args, epoch, manifest_hash, metrics), path)


def save_resume_state(path: Path, model, optimizer, args, epoch, update, manifest_hash, best) -> None:
    payload = weights_payload(model, args, epoch, manifest_hash, None)
    payload["optimizer_state_dict"] = optimizer.state_dict()
    payload["update"] = update
    payload["best"] = best
    payload["torch_rng_state"] = torch.get_rng_state()
    payload["python_rng_state"] = random.getstate()
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()
    validate_args(args)
    modality = ROUTES[args.route][0]

    device = torch.device(args.device)
    frozen_dtype = torch.float16 if args.frozen_dtype == "fp16" else torch.float32
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    # Say something before the silent part: hashing the manifest and stat-ing
    # every referenced file is ~50k syscalls, which takes minutes over WSL's
    # /mnt/e, and until it finishes the run looks dead.
    print(f"[init] route {args.route}; hashing + validating {args.train_manifest.name} ...", flush=True)
    manifest_hash = sha256(args.train_manifest)
    train_rows = read_manifest(
        args.train_manifest, args.ms2_root, modality, split="train", check_files=not args.skip_file_check
    )
    if args.overfit_steps is not None:
        train_rows = train_rows[: args.overfit_samples]

    val_rows: list[dict] = []
    if not args.skip_val:
        val_rows = read_manifest(
            args.val_manifest, args.ms2_root, modality, split=None, check_files=not args.skip_file_check
        )
        val_rows = val_rows[:: max(1, args.val_stride)]

    caption_permutation = None
    if args.caption_mode in ("correct", "permuted"):
        missing = [row["id"] for row in train_rows if not row["caption"].strip()]
        if missing:
            raise SystemExit(
                f"--caption-mode {args.caption_mode} but {len(missing)} train rows lack "
                f"captions (first: {missing[:3]})"
            )
    if args.caption_mode == "permuted":
        # Reassign once, before training, so every epoch sees the same wrong
        # pairing -- a fresh permutation each epoch would average the content
        # away by itself and confound the very thing being measured.
        caption_permutation = permute_captions(train_rows, args.seed)
        print(
            f"[data] TRAIN captions permuted: uniform random donors (seed {args.seed}), "
            f"{caption_permutation['self_assignments']} self-assignments, "
            f"median donor offset {caption_permutation['median_donor_offset_frames']:.0f} frames",
            flush=True,
        )

    print(f"[data] train {len(train_rows)} frames from {args.train_manifest.name}", flush=True)
    if val_rows:
        print(f"[data] val   {len(val_rows)} frames (stride {args.val_stride})", flush=True)
    if args.loss_exclude_top_rows > 0:
        # Measure the bite on a sample rather than assert it: a run whose log does
        # not show pixels leaving has not actually run the experiment.
        probe = [load_sample(row, modality, args)[-1] for row in train_rows[:: max(1, len(train_rows) // 20)][:20]]
        kept = sum(d["gt_valid_pixels_after_exclusion"] for d in probe)
        before = sum(d["gt_valid_pixels"] for d in probe)
        print(
            f"[data] loss excludes the top {args.loss_exclude_top_rows} rows: "
            f"{before - kept} of {before} supervised pixels dropped "
            f"({(before - kept) / max(before, 1) * 100:.2f}%, {len(probe)}-frame probe). "
            f"Validation is NOT affected.",
            flush=True,
        )

    model = RouteModel(args, device, frozen_dtype)
    prompts = {"empty": model.encode_prompt("")}
    if args.val_caption_mode == "fixed":
        prompts["fixed"] = model.encode_prompt(args.val_caption_text)
    caption_rng = random.Random(args.seed + 424242)

    if args.fit_global_affine is not None:
        checkpoint = torch.load(args.eval_checkpoint.resolve(), map_location="cpu", weights_only=False)
        if checkpoint.get("route") != args.route:
            raise SystemExit(
                f"Checkpoint route {checkpoint.get('route')!r} does not match --route {args.route!r}"
            )
        model.load_state_dicts(checkpoint["state_dicts"], str(args.eval_checkpoint))
        rows = val_rows[:: args.fit_affine_stride]
        fit_global_affine(model, rows, prompts, args, args.fit_global_affine)
        return

    if args.eval_checkpoint is not None:
        run_evaluation(model, val_rows, prompts, args, output)
        return

    init_from = None
    if args.init_from is not None:
        payload = torch.load(args.init_from.resolve(), map_location="cpu", weights_only=False)
        loaded = model.load_state_dicts(payload["state_dicts"], str(args.init_from))
        init_from = {
            "path": str(args.init_from),
            "sha256": sha256(args.init_from),
            "source_route": payload.get("route"),
            "source_epoch": payload.get("epoch"),
            "source_caption_mode": payload.get("caption_mode"),
            "source_manifest_sha256": payload.get("manifest_sha256"),
            "modules_loaded": loaded,
            "modules_left_at_pretrained_init": sorted(
                set(model.persisted_modules()) - set(loaded)
            ),
        }
        del payload
        print(
            f"[init-from] {args.init_from.name}: loaded {', '.join(loaded)} "
            f"(from route {init_from['source_route']}, epoch {init_from['source_epoch']}); "
            f"fresh optimiser, training restarts at epoch 0",
            flush=True,
        )
        if init_from["modules_left_at_pretrained_init"]:
            print(
                f"[init-from] not in that checkpoint, left at pretrained init: "
                f"{', '.join(init_from['modules_left_at_pretrained_init'])}",
                flush=True,
            )

    parameter_groups = []
    if model.trains_unet:
        parameter_groups.append({"params": model.unet.parameters(), "lr": args.unet_learning_rate})
    if args.latent_target:
        sampled = [load_sample(row, modality, args)[-1]
                   for row in train_rows[:: max(1, len(train_rows) // 6)][:6]]
        ranges = [d["latent_target_depth_range_m"] for d in sampled]
        real = sum(d["latent_target_real_fraction"] for d in sampled) / max(1, len(sampled))
        print(f"[data] latent target on: encoding the completed depth map, dir {args.pseudo_gt_dir}\n"
              f"[data] {len(sampled)}-frame probe: depth range "
              f"{min(r[0] for r in ranges):.1f}-{max(r[1] for r in ranges):.1f} m, "
              f"real lidar covers {real * 100:.1f}% of each map (the rest is completed)",
              flush=True)

    if args.pseudo_weight > 0:
        # A COMPLETED job says nothing about whether a switch took effect, and this one
        # is a directory: read a few frames and report what the loss will actually see.
        sampled = [load_sample(row, modality, args)[-1]
                   for row in train_rows[:: max(1, len(train_rows) // 10)][:10]]
        share = sum(d["pseudo_fraction"] for d in sampled) / max(1, len(sampled))
        real = sum(d["gt_valid_pixels"] for d in sampled) / max(1, len(sampled))
        print(f"[data] dense completion on: weight {args.pseudo_weight}, range mode "
              f"{args.pseudo_range_mode}, dir {args.pseudo_gt_dir}\n"
              f"[data] {len(sampled)}-frame probe: pseudo covers {share * 100:.1f}% of pixels, "
              f"real lidar {real:.0f} px/frame", flush=True)

    if model.trains_adapter:
        parameter_groups.append({"params": model.adapter.parameters(), "lr": args.adapter_learning_rate})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    base_learning_rates = [group["lr"] for group in optimizer.param_groups]

    updates_per_epoch = math.ceil(len(train_rows) / args.gradient_accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    if args.overfit_steps is not None:
        total_updates = args.overfit_steps
    if args.smoke_updates is not None:
        total_updates = args.smoke_updates

    audit = {
        name: parameter_audit(module)
        for name, module in {
            "vae": model.lotus.vae,
            "text_encoder": model.lotus.text_encoder,
            "unet": model.unet,
            **({"adapter": model.adapter} if model.adapter is not None else {}),
            **({"anythermal": model.anythermal.model} if model.anythermal is not None else {}),
        }.items()
    }
    frozen_config = {
        "format": CHECKPOINT_FORMAT,
        "route": args.route,
        "objective": (
            "latent-space MSE against the VAE-encoded completed depth map, mirroring "
            "lotus/train_iris_g.py; lidar and completed depth are one target, since a "
            "latent cell spans 8x8 pixels and cannot separate them"
            if args.latent_target
            else "masked SSI-L1 vs LiDAR disparity, extended to the pixels lidar missed by "
            "offline calibrated pseudo depth under the lidar-fitted alignment"
            if args.pseudo_weight > 0
            else "pure masked SSI-L1 vs LiDAR disparity; no teacher of any kind"
        ),
        "latent_target": args.latent_target,
        "condition_latent": args.condition_latent,
        "pseudo_gt_dir": str(args.pseudo_gt_dir) if args.pseudo_gt_dir else None,
        "pseudo_weight": args.pseudo_weight,
        "pseudo_range_mode": args.pseudo_range_mode,
        "pseudo_min_depth": args.pseudo_min_depth,
        "pseudo_max_depth": args.pseudo_max_depth,
        "loss_exclude_top_rows": args.loss_exclude_top_rows,
        "caption_permutation": caption_permutation,
        "caption_rank_weight": args.caption_rank_weight,
        "caption_rank_margin": args.caption_rank_margin,
        "caption_rank_refs": args.caption_rank_refs,
        "caption_rank_detach_wrong": args.caption_rank_detach_wrong,
        "caption_rank_detach_empty": args.caption_rank_detach_empty,
        "modality": modality,
        "condition": ROUTES[args.route][1],
        "trains_adapter": model.trains_adapter,
        "trains_unet": model.trains_unet,
        "freeze_adapter": args.freeze_adapter,
        "init_from": init_from,
        "train_manifest": str(args.train_manifest),
        "train_manifest_sha256": manifest_hash,
        "train_frames": len(train_rows),
        "val_manifest": str(args.val_manifest),
        "val_frames": len(val_rows),
        "epochs": args.epochs,
        "updates_per_epoch": updates_per_epoch,
        "total_updates": total_updates,
        "effective_batch_size": args.micro_batch_size * args.gradient_accumulation_steps,
        "lr_schedule": args.lr_schedule,
        "warmup_updates": args.warmup_updates,
        "caption_mode": args.caption_mode,
        "seed": args.seed,
        "environment": environment_fingerprint(device),
        "parameter_audit": audit,
        "settings": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
    }
    (output / "frozen_config.json").write_text(
        json.dumps(frozen_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for name, entry in audit.items():
        print(f"[audit] {name:12s} total {entry['parameters']:>12,}  trainable {entry['trainable']:>12,}", flush=True)

    snapshot_epochs = {
        int(token) for token in args.snapshot_epochs.split(",") if token.strip().isdigit()
    }

    start_epoch = 0
    update = 0
    best = {"epoch": None, "abs_rel": float("inf")}
    if args.resume is not None:
        checkpoint = torch.load(args.resume.resolve(), map_location="cpu", weights_only=False)
        if checkpoint.get("format") != CHECKPOINT_FORMAT or checkpoint.get("route") != args.route:
            raise RuntimeError("Resume checkpoint belongs to a different route or format.")
        if checkpoint.get("manifest_sha256") != manifest_hash:
            raise RuntimeError("Resume manifest hash differs from this run.")
        model.load_state_dicts(checkpoint["state_dicts"], str(args.resume))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        update = int(checkpoint["update"])
        best = checkpoint.get("best", best)
        torch.set_rng_state(checkpoint["torch_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        del checkpoint
        print(f"[resume] continuing from epoch {start_epoch}, update {update}", flush=True)

    metrics_path = output / "epoch_metrics.jsonl"
    started = time.time()
    stop = False
    is_gate_run = args.smoke_updates is not None or args.overfit_steps is not None

    for epoch in range(start_epoch, args.epochs):
        rank_rng = np.random.default_rng(args.seed + 20260806 + epoch)
        permutation = list(range(len(train_rows)))
        random.Random(args.seed + epoch).shuffle(permutation)
        epoch_started = time.time()
        running = {"gt_ssi_l1": 0.0, "gt_abs_rel": 0.0, "grad_match": 0.0,
                   "sky": 0.0, "sky_pixels": 0.0, "pseudo": 0.0, "pseudo_pixels": 0.0,
                   "rank": 0.0, "rank_correct": 0.0, "rank_wrong": 0.0,
                   "rank_empty": 0.0}
        # The epoch-cumulative mean stops moving once a few thousand samples are
        # in, which makes a live run look frozen. Keep a short window for the
        # console; the epoch record still uses the full-epoch mean.
        window: collections.deque = collections.deque(maxlen=800)
        seen = 0
        optimizer.zero_grad(set_to_none=True)

        for position, index in enumerate(permutation):
            row = train_rows[index]
            (image_tensor, gt_disparity, valid_mask, sky_mask,
             pseudo_disparity, pseudo_mask, dense_target, _) = load_sample(row, modality, args)
            gt_disparity = gt_disparity.to(device)
            valid_mask = valid_mask.to(device)
            if sky_mask is not None:
                sky_mask = sky_mask.to(device)

            used_own_caption = (
                args.caption_mode in ("correct", "permuted")
                and caption_rng.random() >= args.caption_dropout
            )
            if used_own_caption:
                prompt = model.encode_prompt(row["caption"])
            else:
                prompt = prompts["empty"]

            target_latent = None
            if args.latent_target:
                target_latent = model.encode_depth_target(dense_target)
                prediction, x0_latent = model.predict_disparity(
                    row, image_tensor, prompt, target_latent=target_latent, return_latent=True
                )
            else:
                prediction = model.predict_disparity(row, image_tensor, prompt)
            if prediction.shape != gt_disparity.shape[-2:]:
                prediction = F.interpolate(
                    prediction[None, None], gt_disparity.shape[-2:], mode="bilinear", align_corners=False
                )[0, 0]
            # Dispatched, not degenerated: at pseudo_weight 0 this is the same call the
            # sparse-GT baseline has always made, so that arm needs no rerun to compare.
            pseudo_term, pseudo_px = 0.0, 0
            if args.latent_target:
                # train_iris_g.py:1151 -- MSE against the clean target latent, since
                # this checkpoint's DDIM config predicts the sample, not the noise.
                # Every latent cell counts: completion left no invalid pixel behind.
                gt_loss = F.mse_loss(x0_latent.float(), target_latent.float())
                with torch.no_grad():
                    # Kept on real lidar so the curve stays readable beside every
                    # run that used the pixel-space objective.
                    _, gt_abs_rel, _ = masked_ssi_l1(prediction[None], gt_disparity, valid_mask)
            elif pseudo_mask is None:
                gt_loss, gt_abs_rel, _ = masked_ssi_l1(prediction[None], gt_disparity, valid_mask)
            else:
                gt_loss, _, pseudo_l1, gt_abs_rel, _, pseudo_px = masked_ssi_l1_dense_completion(
                    prediction[None], gt_disparity, valid_mask,
                    pseudo_disparity.to(device), pseudo_mask.to(device), args.pseudo_weight,
                )
                pseudo_term = float(pseudo_l1.detach())
            loss = args.gt_loss_weight * gt_loss
            grad_term = 0.0
            if args.grad_loss_weight > 0:
                grad_loss = ssi_grad_matching(prediction[None], gt_disparity, valid_mask)
                loss = loss + args.grad_loss_weight * grad_loss
                grad_term = float(grad_loss.detach())
            # Margin ranking: the frame's own caption must not lose to a wrong one.
            # Same row, so predict_disparity draws the same seeded noise and the same
            # fixed timestep, and the mask is the same -- only the prompt differs.
            # Skipped on caption-dropout steps, where the "correct" branch saw no
            # caption at all and the comparison would be meaningless.
            rank_term, rank_correct, rank_wrong, rank_empty = 0.0, 0.0, 0.0, 0.0
            if args.caption_rank_weight > 0 and used_own_caption:

                def reference_loss(reference_prompt, detach):
                    """Same frame: same seeded noise, same timestep, same mask."""
                    if detach:
                        with torch.no_grad():
                            other = model.predict_disparity(row, image_tensor, reference_prompt)
                    else:
                        other = model.predict_disparity(row, image_tensor, reference_prompt)
                    if other.shape != gt_disparity.shape[-2:]:
                        other = F.interpolate(
                            other[None, None], gt_disparity.shape[-2:],
                            mode="bilinear", align_corners=False,
                        )[0, 0]
                    value, _, _ = masked_ssi_l1(other[None], gt_disparity, valid_mask)
                    return value.detach() if detach else value

                references = []
                if args.caption_rank_refs in ("wrong", "both"):
                    donor = row
                    while donor["id"] == row["id"] and len(train_rows) > 1:
                        donor = train_rows[int(rank_rng.integers(len(train_rows)))]
                    wrong_loss = reference_loss(
                        model.encode_prompt(donor["caption"]), args.caption_rank_detach_wrong
                    )
                    rank_wrong = float(wrong_loss.detach())
                    references.append(wrong_loss)
                if args.caption_rank_refs in ("empty", "both"):
                    empty_loss = reference_loss(prompts["empty"], args.caption_rank_detach_empty)
                    rank_empty = float(empty_loss.detach())
                    references.append(empty_loss)
                rank_loss = sum(
                    torch.clamp(args.caption_rank_margin + gt_loss - reference, min=0.0)
                    for reference in references
                )
                loss = loss + args.caption_rank_weight * rank_loss
                rank_term = float(rank_loss.detach())
                rank_correct = float(gt_loss.detach())

            sky_term, sky_pixels = 0.0, 0
            if args.sky_loss_weight > 0 and sky_mask is not None:
                sky_loss, sky_pixels = ssi_sky_loss(
                    prediction[None], gt_disparity, valid_mask, sky_mask,
                    args.gt_max_depth, mode=args.sky_loss_mode,
                )
                loss = loss + args.sky_loss_weight * sky_loss
                sky_term = float(sky_loss.detach())
            (loss / args.gradient_accumulation_steps).backward()

            step_loss = float(gt_loss.detach())
            running["gt_ssi_l1"] += step_loss
            running["gt_abs_rel"] += float(gt_abs_rel.detach())
            running["grad_match"] += grad_term
            running["rank"] += rank_term
            running["rank_correct"] += rank_correct
            running["rank_wrong"] += rank_wrong
            running["rank_empty"] += rank_empty
            running["sky"] += sky_term
            running["sky_pixels"] += sky_pixels
            running["pseudo"] += pseudo_term
            running["pseudo_pixels"] += pseudo_px
            window.append(step_loss)
            seen += 1

            is_last = position == len(permutation) - 1
            if (position + 1) % args.gradient_accumulation_steps == 0 or is_last:
                factor = learning_rate_factor(update, total_updates, args)
                for group, base in zip(optimizer.param_groups, base_learning_rates):
                    group["lr"] = base * factor
                trainable = [
                    parameter
                    for module in model.trainable_modules().values()
                    for parameter in module.parameters()
                    if parameter.requires_grad
                ]
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1

                if update % args.log_interval == 0:
                    print(
                        f"[e{epoch + 1}/{args.epochs} u{update}/{total_updates}] "
                        f"gt_ssi_l1 {sum(window) / len(window):.5f} "
                        f"(epoch {running['gt_ssi_l1'] / max(1, seen):.5f}) "
                        f"abs_rel {running['gt_abs_rel'] / max(1, seen):.4f} "
                        + (
                            f"gmatch {running['grad_match'] / max(1, seen):.5f} "
                            if args.grad_loss_weight > 0
                            else ""
                        )
                        + (
                            # The term and how many pixels it saw: a sky loss
                            # averaging over ~0 pixels is off even when weighted on.
                            f"sky {running['sky'] / max(1, seen):.5f} "
                            f"skypx {running['sky_pixels'] / max(1, seen):.0f} "
                            if args.sky_loss_weight > 0
                            else ""
                        )
                        + (
                            # The term and the pixels it covered: completion averaging
                            # over ~0 pixels is off however the weight reads.
                            f"pseudo {running['pseudo'] / max(1, seen):.5f} "
                            f"psdpx {running['pseudo_pixels'] / max(1, seen):.0f} "
                            if args.pseudo_weight > 0
                            else ""
                        )
                        + (
                            # correct and wrong side by side: if the margin is being
                            # met by the wrong branch getting worse rather than the
                            # correct one getting better, it shows here.
                            f"rank {running['rank'] / max(1, seen):.5f} "
                            f"(c {running['rank_correct'] / max(1, seen):.5f} "
                            f"w {running['rank_wrong'] / max(1, seen):.5f} "
                            f"e {running['rank_empty'] / max(1, seen):.5f}) "
                            if args.caption_rank_weight > 0
                            else ""
                        )
                        + f"lr x{factor:.3f} grad {float(grad_norm):.3f}",
                        flush=True,
                    )
                    # Step-level record so a long run is inspectable while it
                    # runs. Key names match tools/watch_training.py.
                    grad_key = "unet_grad_norm" if model.trains_unet else "adapter_grad_norm"
                    with (output / "training_metrics.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "step": update,
                                    "epoch": epoch + 1,
                                    "total": running["gt_ssi_l1"] / max(1, seen),
                                    "window_gt_ssi_l1": sum(window) / len(window),
                                    "gt_abs_rel": running["gt_abs_rel"] / max(1, seen),
                                    "grad_match": running["grad_match"] / max(1, seen),
                                    "sky": running["sky"] / max(1, seen),
                                    "sky_pixels": running["sky_pixels"] / max(1, seen),
                                    grad_key: float(grad_norm),
                                    "lr_factor": factor,
                                    "samples_seen": epoch * len(train_rows) + seen,
                                    "elapsed_seconds": time.time() - started,
                                }
                            )
                            + "\n"
                        )
                if args.smoke_updates is not None and update >= args.smoke_updates:
                    stop = True
                    break
                if args.overfit_steps is not None and update >= args.overfit_steps:
                    stop = True
                    break

        epoch_record = {
            "epoch": epoch + 1,
            "updates": update,
            "train_gt_ssi_l1": running["gt_ssi_l1"] / max(1, seen),
            "train_gt_abs_rel": running["gt_abs_rel"] / max(1, seen),
            # 权重 0 时恒为 0；开启时用来核对两项量级是否失衡
            "train_grad_match": running["grad_match"] / max(1, seen),
            "train_rank": running["rank"] / max(1, seen),
            "train_rank_correct": running["rank_correct"] / max(1, seen),
            "train_rank_wrong": running["rank_wrong"] / max(1, seen),
            "train_rank_empty": running["rank_empty"] / max(1, seen),
            "train_sky": running["sky"] / max(1, seen),
            "train_sky_pixels": running["sky_pixels"] / max(1, seen),
            # A twenty-hour run has to be answerable afterwards from its own record,
            # not from whether a step log happened to land on the right interval.
            "train_pseudo": running["pseudo"] / max(1, seen),
            "train_pseudo_pixels": running["pseudo_pixels"] / max(1, seen),
            "epoch_seconds": time.time() - epoch_started,
        }

        # Also validate when a smoke/overfit run stops early: otherwise the
        # validation path would never execute before the full runs.
        should_validate = bool(val_rows) and (
            stop or (epoch + 1) % args.val_every == 0 or epoch + 1 == args.epochs
        )
        if should_validate:
            validation_started = time.time()
            val_metrics = run_validation(model, val_rows, prompts, args)
            epoch_record["val"] = val_metrics
            epoch_record["val_seconds"] = time.time() - validation_started
            print(
                f"[val e{epoch + 1}] abs_rel {val_metrics['abs_rel']:.4f} "
                f"rmse {val_metrics['rmse']:.3f} a1 {val_metrics['a1']:.4f} "
                f"({val_metrics['val_samples']} frames, {epoch_record['val_seconds'] / 60:.1f} min)",
                flush=True,
            )
            if val_metrics["abs_rel"] < best["abs_rel"]:
                best = {"epoch": epoch + 1, "abs_rel": val_metrics["abs_rel"]}
                if not is_gate_run:
                    save_weights(
                        output / "best_weights.pt", model, args, epoch + 1, manifest_hash, val_metrics
                    )
                    print(f"[val e{epoch + 1}] new best -> best_weights.pt", flush=True)

        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")

        if epoch + 1 in snapshot_epochs:
            save_weights(
                output / f"epoch{epoch + 1:02d}_weights.pt",
                model, args, epoch + 1, manifest_hash, epoch_record.get("val"),
            )

        # Gate runs are throwaway: a resume state carries the optimizer moments
        # and costs ~10 GB for a 5-step smoke. Only real runs get checkpoints.
        if not is_gate_run:
            save_resume_state(
                output / "latest.pt", model, optimizer, args, epoch + 1, update, manifest_hash, best
            )

        if stop:
            break

    if not is_gate_run:
        save_weights(output / "end_weights.pt", model, args, args.epochs, manifest_hash, None)
    summary = dict(frozen_config)
    summary.update(
        {
            "completed_updates": update,
            "elapsed_seconds": time.time() - started,
            "best": best,
            "stopped_early": stop,
            "end_checkpoint_sha256": None if is_gate_run else sha256(output / "end_weights.pt"),
            "gate_run": is_gate_run,
        }
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[done] {args.route}: {update} updates in {(time.time() - started) / 3600:.2f} h; "
        f"best epoch {best['epoch']} abs_rel {best['abs_rel']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
