#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Iris's Lotus-G trainer, run on MS2 thermal against the completed pseudo GT.

A copy of `train_iris_g.py`. Every departure from it is marked `# MS2:` so the
two files diff to a short, readable list. Nothing about the objective, the noise
schedule, the two-task loss, the tokenizer or the optimiser is touched -- the
point of copying rather than reimplementing is that fidelity does not depend on
anyone's checklist being complete.

What changes:

  1. Dataset. Hypersim + VKITTI are replaced by one `MS2ThermalDataset`; the
     mixing logic goes with them. The image is thermal, the depth map is this
     frame's lidar written back over calibrated AnyThermal pseudo depth.
  2. `conv_in` surgery is skipped when the starting weights already have 8 input
     channels. Iris always starts from SD2-base (4 channels) and doubles them;
     starting from a trained Lotus-G checkpoint, doubling again would produce 16
     and load nothing. Both starting points are supported so the two can be
     compared with the starting weights as the only difference.
  3. Validation. `log_validation` renders and scores Iris's own eval datasets,
     which are not on disk here. It is skipped unless `--validation_images` is
     given. Checkpoints are scored out of band by the official MS2 protocol.

The RGB reconstruction branch is *not* changed and *not* renamed: it
reconstructs whatever `pixel_values` holds, which here is the thermal frame. Its
prompt stays empty, as in the original.
"""

import argparse
import copy
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from PIL import Image
from glob import glob
from easydict import EasyDict

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.nn as nn
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo
from packaging import version
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from pipeline import LotusDPipeline, LotusGPipeline
from e2eft_loss import ScaleAndShiftInvariantLoss

# Backbones that train one branch and carry no task switcher. Lotus packs a
# depth branch and a thermal-reconstruction branch into one batch and tells
# them apart with the switcher; neither Marigold nor E2E-FT has either.
SINGLE_BRANCH = ("marigold", "e2eft")
from utils.image_utils import concatenate_images, colorize_depth_map
# MS2: Hypersim and VKITTI are replaced by the MS2 thermal dataset.
from utils.ms2_thermal_dataset import MS2ThermalDataset, MS2ThermalTransform, collate_fn_ms2

from eval import evaluation_depth, evaluation_normal

# MS2 metric adaptation: the one place the inverse-depth convention lives.
import sys as _sys
_IRIS_ROOT = Path(__file__).resolve().parents[1]
if str(_IRIS_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_IRIS_ROOT))
from tools.metric_depth_norm import (  # noqa: E402
    MetricNorm,
    decoded_to_inverse,
    depth_to_inverse,
    inverse_to_depth,
    nonfinite_count,
    range_line,
)

import tensorboard

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

logger = get_logger(__name__, log_level="INFO")
    
TOP5_STEPS_DEPTH = []
TOP5_STEPS_NORMAL = []

def run_example_validation(pipeline, task, args, step, accelerator, generator):
    validation_images = glob(os.path.join(args.validation_images, "*.jpg")) + glob(os.path.join(args.validation_images, "*.png"))
    validation_images = sorted(validation_images)
    print(validation_images)
    
    pred_annos = []
    input_images = []
    
    if task == "depth":
        for i in range(len(validation_images)):
            if torch.backends.mps.is_available():
                autocast_ctx = nullcontext()
            else:
                autocast_ctx = torch.autocast(accelerator.device.type)

            with autocast_ctx:
                # Preprocess validation image
                validation_image = Image.open(validation_images[i]).convert("RGB")
                input_images.append(validation_image)
                validation_image = np.array(validation_image).astype(np.float32)
                validation_image = torch.tensor(validation_image).permute(2,0,1).unsqueeze(0)
                validation_image = validation_image / 127.5 - 1.0 
                validation_image = validation_image.to(accelerator.device)

                task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(accelerator.device)
                task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)

                # Run
                pred_depth = pipeline(
                    rgb_in=validation_image, 
                    task_emb=task_emb,
                    prompt="", 
                    num_inference_steps=1, 
                    timesteps=[args.timestep],
                    generator=generator, 
                    output_type='np',
                    ).images[0]
                
                # Post-process the prediction
                pred_depth = pred_depth.mean(axis=-1)
                is_reverse_color = "disparity" in args.norm_type
                depth_color = colorize_depth_map(pred_depth, reverse_color=is_reverse_color)
                
                pred_annos.append(depth_color)

    elif task == "normal":
        for i in range(len(validation_images)):
            if torch.backends.mps.is_available():
                autocast_ctx = nullcontext()
            else:
                autocast_ctx = torch.autocast(accelerator.device.type)

            with autocast_ctx:
                # Preprocess validation image
                validation_image = Image.open(validation_images[i]).convert("RGB")
                input_images.append(validation_image)
                validation_image = np.array(validation_image).astype(np.float32)
                validation_image = torch.tensor(validation_image).permute(2,0,1).unsqueeze(0)
                validation_image = validation_image / 127.5 - 1.0 
                validation_image = validation_image.to(accelerator.device)

                task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(accelerator.device)
                task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)

                # Run
                pred_normal = pipeline(
                    rgb_in=validation_image, 
                    task_emb=task_emb,
                    prompt="", 
                    num_inference_steps=1, 
                    timesteps=[args.timestep],
                    generator=generator,
                    ).images[0]
                
                pred_annos.append(pred_normal)
                
    else:
        raise ValueError(f"Not Supported Task: {args.task_name}!")

    # Save output
    save_output = concatenate_images(input_images, pred_annos)
    save_dir = os.path.join(args.output_dir,'images')
    os.makedirs(save_dir, exist_ok=True)
    save_output.save(os.path.join(save_dir, f'{step:05d}.jpg'))

def run_evaluation(pipeline, task, args, step, accelerator):
    # Define prediction functions
    # mod
    def gen_depth(rgb_in, pipe, prompt="", num_inference_steps=1, image_path=None, dataset_name=None):
        if torch.backends.mps.is_available():
                autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(pipe.device.type)

        with autocast_ctx:
            rgb_input = rgb_in / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
            task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(pipe.device)
            task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)
            pred_depth = pipe(
                            rgb_in=rgb_input, 
                            task_emb=task_emb,
                            prompt=prompt, 
                            num_inference_steps=num_inference_steps,
                            timesteps=[args.timestep],
                            output_type='np',
                            ).images[0]
            pred_depth = pred_depth.mean(axis=-1) # [0,1]
        return pred_depth
    
    def gen_normal(img, pipe, prompt="", num_inference_steps=1):
        if torch.backends.mps.is_available():
                autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(pipe.device.type)

        with autocast_ctx:
            task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(pipe.device)
            task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)

            pred_normal = pipe(
                            rgb_in=img, # [-1,1] 
                            task_emb=task_emb, 
                            prompt=prompt, 
                            num_inference_steps=num_inference_steps,
                            timesteps=[args.timestep],
                            output_type='pt',
                            ).images[0] # [0,1], (3,h,w)
            pred_normal = (pred_normal*2-1.0).unsqueeze(0) # [-1,1], (1,3,h,w)
        return pred_normal

    if step > 0:
        if task == "depth":
            test_data_dir = os.path.join(args.base_test_data_dir, task)
            test_depth_dataset_configs = {
                "nyuv2": "configs/data_nyu_test.yaml", 
            }
            if args.FULL_EVALUATION:
                print("==> Full Evaluation Mode!")
                test_depth_dataset_configs = {
                "nyuv2": "configs/data_nyu_test.yaml", 
                "kitti": "configs/data_kitti_eigen_test.yaml",
                "scannet": "configs/data_scannet_val.yaml",
                "eth3d": "configs/data_eth3d.yaml",
                "diode": "configs/data_diode_all.yaml",
            }
            LEADER_DATASET = list(test_depth_dataset_configs.keys())[0]
            for dataset_name, config_path in test_depth_dataset_configs.items():
                eval_dir = os.path.join(args.output_dir, f'evaluation-{step:05d}', task, dataset_name)
                test_dataset_config = os.path.join(test_data_dir, config_path)
                alignment_type = "least_square_disparity" if "disparity" in args.norm_type else "least_square"
                metric_tracker = evaluation_depth(eval_dir, test_dataset_config, test_data_dir, eval_mode="generate_prediction",
                            gen_prediction=gen_depth, pipeline=pipeline, save_pred_vis=args.save_pred_vis, alignment=alignment_type)
                print(dataset_name,',', 'abs_relative_difference: ', metric_tracker.result()['abs_relative_difference'], 'delta1_acc: ', metric_tracker.result()['delta1_acc'], 'delta2_acc: ', metric_tracker.result()['delta2_acc'])
                
                if dataset_name == LEADER_DATASET:
                    TOP5_STEPS_DEPTH.append((metric_tracker.result()['abs_relative_difference'], f"step-{step}"))
                    TOP5_STEPS_DEPTH.sort(key=lambda x: x[0])
                    if len(TOP5_STEPS_DEPTH) > 5:
                        TOP5_STEPS_DEPTH.pop()

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        tracker.writer.add_scalar(f"depth_{dataset_name}/rel", metric_tracker.result()['abs_relative_difference'], step)
                        tracker.writer.add_scalar(f"depth_{dataset_name}/delta1", metric_tracker.result()['delta1_acc'], step)
            
            top_five_cycles = [cycle_name for _, cycle_name in TOP5_STEPS_DEPTH]
            print("Top Five:", top_five_cycles)

        elif task == "normal":
            test_data_dir = os.path.join(args.base_test_data_dir, task)
            dataset_split_path = "eval/dataset_normal"
            eval_datasets = [('nyuv2', 'test')]
            if args.FULL_EVALUATION:
                eval_datasets = [('nyuv2', 'test'), ('scannet', 'test'), ('ibims', 'ibims'), ('sintel', 'sintel'), ('oasis','val')]
            eval_dir = os.path.join(args.output_dir, f'evaluation-{step:05d}', task)
            eval_metrics = evaluation_normal(eval_dir, test_data_dir, dataset_split_path, eval_mode="generate_prediction", 
                                                gen_prediction=gen_normal, pipeline=pipeline, eval_datasets=eval_datasets,
                                                save_pred_vis=args.save_pred_vis)
            
            LEADER_DATASET = eval_datasets[0][0]
            mean_value = eval_metrics[LEADER_DATASET]['mean'] if eval_metrics[LEADER_DATASET]['mean'] == eval_metrics[LEADER_DATASET]['mean'] else float('inf')
            TOP5_STEPS_NORMAL.append((mean_value, f"step-{step}"))
            TOP5_STEPS_NORMAL.sort(key=lambda x: x[0])
            if len(TOP5_STEPS_NORMAL) > 5:
                TOP5_STEPS_NORMAL.pop()
            
            top_five_cycles = [cycle_name for _, cycle_name in TOP5_STEPS_NORMAL]
            print("Top Five:", top_five_cycles)

            for dataset_name, metrics in eval_metrics.items():
                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        tracker.writer.add_scalar(f"normal_{dataset_name}/mean", metrics['mean'], step)
                        tracker.writer.add_scalar(f"normal_{dataset_name}/11.25", metrics['a3'], step)

        else:
                raise ValueError(f"Not Supported Task: {task}!")

def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, step):
    logger.info("Running validation for task: %s... " % args.task_name[0])
    task = args.task_name[0]

    # Load pipeline
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    scheduler.register_to_config(prediction_type=args.prediction_type)
    pipeline = (LotusDPipeline if args.backbone == "d" else LotusGPipeline).from_pretrained(
        args.pretrained_model_name_or_path,
        scheduler=scheduler,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    
    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()
  
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    
    
    # Run example-validation
    run_example_validation(pipeline, task, args, step, accelerator, generator)

    # Run evaluation
    run_evaluation(pipeline, task, args, step, accelerator)

    del pipeline
    torch.cuda.empty_cache()

# --------------------------------------------------------------------------- #
# metric GT adaptation (2026-08-26)
# --------------------------------------------------------------------------- #


def decode_metric_inverse_depth(metric_vae, x0_latent, norm):
    """The U-Net's x0 latent -> metric inverse depth, in 1/m.

    Two steps, both taken from code that already exists rather than reinvented:

      1. `decode_to_disparity` (tools/train_ms2_joint_gt_v3.py:420) -- decode,
         channel-mean, `/ 2 + 0.5`.  That is `y`, the decoder's [-1, 1] output
         re-expressed on [0, 1], and it is exactly what every evaluation in this
         project has read off this model.
      2. `decoded_to_inverse` (tools/metric_depth_norm.py) -- `q = q_lo + y*(q_hi - q_lo)`.

    Nothing is clamped here.  The clamp belongs on the reporting path, where it
    is a numerical guard; on the loss path it would zero the gradient of exactly
    the pixels that are most wrong.
    """
    device_type = x0_latent.device.type
    # The surrounding step runs under accelerate's autocast. Decoding inside it
    # would put this back in fp16 and undo the whole point of the fp32 copy.
    with torch.autocast(device_type=device_type, enabled=False):
        decoded = metric_vae.decode(
            x0_latent.float() / metric_vae.config.scaling_factor, return_dict=False
        )[0]
    y = decoded.float().mean(dim=1, keepdim=True) / 2.0 + 0.5   # (B, 1, H, W) in [0, 1]
    return y, decoded_to_inverse(y, norm)


def metric_inverse_depth_loss(metric_vae, x0_latent, gt_depth, gt_valid, norm):
    """Masked L1 between predicted and measured inverse depth, in 1/m.

        L = mean_{i in M} |q_hat_i - q_GT_i|,     M = official valid lidar mask

    `gt_depth` is the *sparse* lidar in metres and `gt_valid` its mask; the
    completed dense map is deliberately not used here.  Completion is a model's
    opinion about where the sensor did not measure, and the term whose job is to
    fix absolute scale should not be anchored to an opinion.

    The sparse map is never VAE-encoded.  Its zeros are holes, not depth, and a
    latent cell pools 8x8 pixels -- encoding it would spread the holes over the
    whole target, which is the confirmed V1 bug recorded in AGENTS.md.
    """
    y, q_hat = decode_metric_inverse_depth(metric_vae, x0_latent, norm)
    if q_hat.shape[-2:] != gt_depth.shape[-2:]:
        raise RuntimeError(
            f"Prediction {tuple(q_hat.shape[-2:])} and lidar {tuple(gt_depth.shape[-2:])} "
            "disagree on resolution"
        )
    mask = gt_valid > 0.5
    count = int(mask.sum())
    stats = {
        "decoded_min": float(y.detach().min()),
        "decoded_max": float(y.detach().max()),
        "q_hat_min": float(q_hat.detach().min()),
        "q_hat_max": float(q_hat.detach().max()),
        "q_hat_nonfinite": nonfinite_count(q_hat.detach()),
        "q_hat_non_positive": int((q_hat.detach() <= 0).sum()),
        "lidar_pixels": count,
    }
    if stats["q_hat_nonfinite"]:
        raise RuntimeError(f"Predicted inverse depth has {stats['q_hat_nonfinite']} NaN/Inf values")
    if not count:
        # A whole batch without a single lidar return is possible in principle
        # and must not become a division by zero. It contributes nothing rather
        # than aborting the run.
        return x0_latent.new_zeros(()), 0, stats
    q_gt, _ = depth_to_inverse(gt_depth, norm, valid=mask)
    loss = (q_hat[mask] - q_gt[mask]).abs().mean()
    with torch.no_grad():
        depth_hat, clip_report = inverse_to_depth(q_hat.detach(), norm)
        stats["depth_hat_min"] = float(depth_hat.min())
        stats["depth_hat_max"] = float(depth_hat.max())
        stats["depth_hat_clip"] = str(clip_report)
        # AbsRel on the measured pixels, with no alignment of any kind. This is
        # the training-time read of the number the metric evaluation reports, so
        # the curve and the final table are talking about the same quantity.
        gt_m = gt_depth[mask]
        pred_m = depth_hat[mask]
        stats["metric_abs_rel"] = float(((pred_m - gt_m).abs() / gt_m).mean())
    return loss, count, stats


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    # MS2: the two source datasets are replaced by one manifest + one pseudo GT
    # directory. The Hypersim/VKITTI flags are gone rather than left dangling,
    # so a stale command line fails loudly instead of training on nothing.
    parser.add_argument(
        "--ms2_manifest",
        type=str,
        required=True,
        help="Training manifest (jsonl): id / thermal_path / thermal_depth_path / caption.",
    )
    parser.add_argument(
        "--ms2_root",
        type=str,
        required=True,
        help="Single root the manifest's relative paths resolve against.",
    )
    parser.add_argument(
        "--pseudo_gt_dir",
        type=str,
        required=True,
        help=(
            "calibrated_pseudo_depth/*.npy from build_anythermal_pseudo_gt.py. The "
            "target is this frame's lidar written back over it; sparse lidar alone "
            "cannot fill an 8x8 latent cell, which is why this is not optional."
        ),
    )
    parser.add_argument(
        "--sky_mask_dir",
        type=str,
        default=None,
        help=(
            "Mask2Former sky masks from tools/build_sky_masks.py. Given one, the "
            "completed target's sky pixels are rewritten to d_max instead of keeping "
            "the pseudo network's guess. Measured, that guess is 26.8 m at the median "
            "and the trained model reproduces it at 24.5 m, so this corrects the "
            "target rather than adding a constraint on top of it. Lidar keeps "
            "precedence over the mask."
        ),
    )
    parser.add_argument(
        "--no_captions",
        action="store_true",
        help=(
            "Feed every frame the empty string. This is the no-text ablation of "
            "the same run, not a fallback for a manifest without captions."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--base_test_data_dir",
        type=str,
        default="datasets/eval/"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=["depth","normal"],
        nargs="+"
    )
    parser.add_argument(
        "--validation_images",
        type=str,
        default=None,
        help=("A set of images evaluated every `--validation_steps` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    # MS2: --resolution_hypersim / --resolution_vkitti / --prob_hypersim /
    # --mix_dataset removed with the datasets they configured. MS2 thermal is
    # 640x256 natively, both edges divisible by 8, so nothing is resized.
    parser.add_argument(
        "--norm_type",
        type=str,
        choices=['instnorm','truncnorm','perscene_norm','disparity','trunc_disparity',
                 'global_metric_disparity'],
        default='trunc_disparity',
        help=(
            'The normalization type for the depth prediction. Every option but the '
            'last normalises each frame by its own statistics and therefore removes '
            'absolute scale; global_metric_disparity uses two constants frozen from '
            'the training split and needs --metric_norm_json.'
        )
    )

    # ---- backbone selection (2026-08-28) ---------------------------------- #
    # Lotus ships two variants. They differ in exactly one thing: what conv_in is
    # fed. G is generative and takes [image latent, noisy target latent] (8 ch);
    # D is the direct variant and takes the image latent alone (4 ch). Both
    # released checkpoints carry the task switcher, so the two-branch loss, the
    # captions, the pseudo-GT and every downstream stage are shared between them.
    #
    # This is a flag rather than a forked trainer on purpose: the claim we are
    # after is "the caption effect reproduces on a second backbone", and that is
    # only true if the two runs differ in the backbone and in nothing else. Two
    # copies of this file would drift silently and void that comparison.
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["g", "d", "marigold", "e2eft"],
        default="g",
        help=(
            "Which Lotus variant the weights are. 'g' (default) reproduces the "
            "existing behaviour line for line. 'd' skips the 4->8 conv_in "
            "expansion and conditions on the image latent alone. 'marigold' is "
            "the multi-step variant: one branch, no task switcher, a random "
            "timestep, and a target chosen by --prediction_type. 'e2eft' keeps "
            "the terminal timestep but scores the decoded image under a "
            "scale-and-shift-invariant L1 instead of the latent."
        ),
    )

    # ---- metric GT adaptation (2026-08-26) -------------------------------- #
    # A second training mode, off unless --metric_adaptation is passed. It changes
    # three things and nothing else: the target's normalisation becomes global,
    # a masked image-space L1 against real lidar is added, and the U-Net starts
    # from a named checkpoint rather than from the base repository.
    parser.add_argument(
        "--metric_adaptation",
        action="store_true",
        help=(
            "Turn on the metric adaptation stage. Implies --norm_type "
            "global_metric_disparity and requires --metric_norm_json."
        ),
    )
    parser.add_argument(
        "--metric_norm_json",
        type=str,
        default=None,
        help=(
            "Artifact from tools/fit_metric_norm.py holding q_lo/q_hi. Estimated on "
            "the training split only; loading one whose source_split is not 'train' "
            "raises."
        ),
    )
    parser.add_argument(
        "--init_unet_from",
        type=str,
        default=None,
        help=(
            "Start the U-Net from this checkpoint instead of --pretrained_model_name_or_path's. "
            "Takes a train_route_suite.py payload (.pt with state_dicts.unet), which is "
            "what tools/convert_iris_ms2_checkpoint.py writes, or a diffusers unet/ "
            "directory. The rest of the pipeline still comes from --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--lambda_metric",
        type=float,
        default=1.0,
        help=(
            "Weight on the masked image-space L1 against real lidar inverse depth. "
            "This is the term that fixes absolute scale; nothing else in the objective "
            "can, because both latent terms are scored against maps the same global "
            "normalisation produced."
        ),
    )
    parser.add_argument(
        "--lambda_dense",
        type=float,
        default=1.0,
        help=(
            "Weight on the existing latent MSE against the completed dense target "
            "(the old `anno_loss`). 1.0 leaves it exactly as the base run had it."
        ),
    )
    parser.add_argument(
        "--lambda_recon",
        type=float,
        default=1.0,
        help=(
            "Weight on the existing thermal reconstruction branch (the old `rgb_loss`). "
            "1.0 leaves it exactly as the base run had it; the objective is not "
            "silently changed by turning metric adaptation on."
        ),
    )
    parser.add_argument(
        "--metric_decode_grad_ckpt",
        type=lambda v: str(v).lower() not in ("0", "false", "no"),
        default=True,
        help=(
            "Gradient-checkpoint the fp32 decoder used by the metric loss. On by "
            "default: without it the decode of a 4x3x256x640 batch in fp32 with "
            "graph retained is the largest allocation in the step."
        ),
    )
    parser.add_argument(
        "--metric_log_every",
        type=int,
        default=200,
        help="Steps between predicted inverse-depth / metric-depth range reports.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--align_cam_normal",
        action="store_true",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--truncnorm_min",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=1
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="sample",
        help="The prediction_type that shall be used for training. ",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument("--FULL_EVALUATION", action="store_true")
    parser.add_argument("--save_pred_vis", action="store_true")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_lotus_g",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    # MS2: the captions travel inside the manifest rather than a side-car JSON
    # keyed by image path, so the two description-path flags are gone with it.

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    # MS2: the manifest, the root and the pseudo GT directory are all required,
    # so argparse has already enforced what this block used to check.

    if args.metric_adaptation:
        if not args.metric_norm_json:
            raise ValueError("--metric_adaptation needs --metric_norm_json")
        if args.norm_type not in ("trunc_disparity", "global_metric_disparity"):
            raise ValueError(
                f"--metric_adaptation cannot run with --norm_type {args.norm_type}: the "
                "target has to be the globally normalised one."
            )
        # Implied rather than required on the command line, so a stale script
        # cannot half-enable the mode.
        args.norm_type = "global_metric_disparity"
        if args.lambda_metric <= 0:
            raise ValueError(
                "--lambda_metric must be positive under --metric_adaptation: it is the "
                "only term that carries measured absolute scale."
            )
    else:
        if args.metric_norm_json:
            raise ValueError("--metric_norm_json only means anything with --metric_adaptation")
        if args.norm_type == "global_metric_disparity":
            raise ValueError("--norm_type global_metric_disparity requires --metric_adaptation")
        if (args.lambda_dense, args.lambda_recon) != (1.0, 1.0):
            # Outside the metric stage this trainer must stay the recipe that
            # produced every published checkpoint, term for term.
            raise ValueError(
                "--lambda_dense / --lambda_recon may only leave 1.0 under "
                "--metric_adaptation; the base recipe's objective is fixed."
            )

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

def main():
    args = parse_args()

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler.register_to_config(prediction_type=args.prediction_type)
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
        )
    
    # Lotus carries a task switcher and its trainer always passes class_labels.
    # Marigold has neither, and a U-Net built with a class embedding refuses a
    # forward that omits class_labels -- so the switcher is added for Lotus only.
    _switcher = {} if args.backbone in SINGLE_BRANCH else dict(
        class_embed_type="projection", projection_class_embeddings_input_dim=4
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision,
        low_cpu_mem_usage=False, device_map=None, **_switcher,
    )
    
    # Replace the first layer to accept 8 in_channels.
    # MS2: only when the starting weights have 4. From SD2-base this runs exactly
    # as Iris wrote it; from a trained Lotus-G checkpoint conv_in is already 8, and
    # doubling it again would build a 16-channel layer that no longer matches the
    # weights it was seeded from. Which of the two we started from is logged, since
    # that is the single difference between the two runs of this line.
    # A Lotus-D checkpoint and SD2-base both arrive with 4 channels, so the shape
    # alone cannot tell them apart. --backbone is what decides, and the name check
    # below is a cheap guard for the one mismatch shapes cannot catch: asking for
    # 'g' while handing over the D checkpoint would expand its conv_in to 8 and
    # train a silently wrong model.
    _repo = str(args.pretrained_model_name_or_path).lower()
    if args.backbone == "g" and "depth-d" in _repo:
        raise ValueError(
            f"--backbone g with {args.pretrained_model_name_or_path}: that looks "
            "like a Lotus-D checkpoint. Pass --backbone d."
        )
    if args.backbone == "d" and "depth-g" in _repo:
        raise ValueError(
            f"--backbone d with {args.pretrained_model_name_or_path}: that looks "
            "like a Lotus-G checkpoint. Pass --backbone g."
        )
    if args.backbone == "marigold" and "lotus" in _repo:
        raise ValueError(
            f"--backbone marigold with {args.pretrained_model_name_or_path}: that "
            "is a Lotus checkpoint. Pass --backbone g or d."
        )
    if args.backbone == "e2eft" and "e2e" not in _repo:
        raise ValueError(
            f"--backbone e2eft with {args.pretrained_model_name_or_path}: that is "
            "not an E2E-FT checkpoint."
        )
    if args.backbone not in ("marigold", "e2eft") and "marigold" in _repo:
        raise ValueError(
            f"--backbone {args.backbone} with {args.pretrained_model_name_or_path}: "
            "that is a Marigold checkpoint. Pass --backbone marigold."
        )

    if args.backbone == "d":
        if unet.conv_in.in_channels != 4:
            raise ValueError(
                f"--backbone d expects conv_in to have 4 input channels, got "
                f"{unet.conv_in.in_channels}."
            )
        logger.info("backbone=d: conv_in stays at 4 input channels, no expansion.")
    elif unet.conv_in.in_channels == 4:
        logger.info("conv_in has 4 input channels: applying Lotus's 4->8 expansion.")
        _weight = unet.conv_in.weight.clone()
        _bias = unet.conv_in.bias.clone()
        _weight = _weight.repeat(1, 2, 1, 1)
        _weight *= 0.5
        # unet.config.in_channels *= 2
        config_dict = EasyDict(unet.config)
        config_dict.in_channels *= 2
        unet._internal_dict = config_dict

        # new conv_in channel
        _n_convin_out_channel = unet.conv_in.out_channels
        _new_conv_in =nn.Conv2d(
            8, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        _new_conv_in.weight = nn.Parameter(_weight)
        _new_conv_in.bias = nn.Parameter(_bias)
        unet.conv_in = _new_conv_in
    elif unet.conv_in.in_channels == 8:
        logger.info(
            "conv_in already has 8 input channels: starting from a trained Lotus-G "
            "checkpoint, expansion skipped."
        )
    else:
        raise ValueError(
            f"conv_in has {unet.conv_in.in_channels} input channels; expected 4 "
            "(SD2-base) or 8 (a trained Lotus-G checkpoint)."
        )

    # MS2 metric adaptation: continue from a named checkpoint rather than from the
    # base repository's weights. Done here, after the conv_in shape is settled and
    # before `accelerator.prepare`, so the loaded tensors are the ones the
    # optimizer is built over.
    if args.init_unet_from:
        init_path = Path(args.init_unet_from)
        if not init_path.exists():
            raise FileNotFoundError(f"--init_unet_from: no such path {init_path}")
        if init_path.is_dir():
            source = UNet2DConditionModel.from_pretrained(
                init_path, subfolder="unet" if (init_path / "unet").is_dir() else None,
                in_channels=8, low_cpu_mem_usage=False, device_map=None,
            )
            init_state = source.state_dict()
            provenance = f"diffusers directory {init_path}"
            del source
        else:
            payload = torch.load(init_path, map_location="cpu", weights_only=False)
            if "state_dicts" not in payload or "unet" not in payload["state_dicts"]:
                raise ValueError(
                    f"{init_path} carries {sorted(payload)}; expected a "
                    "train_route_suite.py payload with state_dicts.unet"
                )
            init_state = payload["state_dicts"]["unet"]
            provenance = (
                f"{init_path} (route {payload.get('route')}, step {payload.get('epoch')}, "
                f"caption_mode {payload.get('caption_mode')})"
            )
        # strict: a silent key mismatch here is a run that trains the wrong
        # tensors and only shows it as a bad metric hours later.
        missing, unexpected = unet.load_state_dict(init_state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"--init_unet_from key mismatch: {len(missing)} missing "
                f"{missing[:5]}, {len(unexpected)} unexpected {unexpected[:5]}"
            )
        logger.info(f"U-Net initialised from {provenance}")

    # Freeze vae and text_encoder and set unet to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            for _ in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(
                    input_dir, subfolder="unet",
                    in_channels=(4 if args.backbone == "d" else 8),
                )
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets and dataloaders.
    # MS2: one dataset in place of the Hypersim/VKITTI pair. `--mix_dataset` and
    # `--prob_hypersim` no longer have anything to mix and are ignored; the loop
    # below reads from this single loader.
    # -------------------- Dataset: MS2 thermal --------------------
    metric_norm = None
    if args.metric_adaptation:
        metric_norm = MetricNorm.load(args.metric_norm_json)
        logger.info(f"Metric adaptation: {metric_norm.summary()}")
        logger.info(f"  constants from {metric_norm.source_manifest}")
    transform_ms2 = MS2ThermalTransform(random_flip=args.random_flip)
    train_dataset_ms2 = MS2ThermalDataset(
        args.ms2_manifest,
        args.ms2_root,
        args.pseudo_gt_dir,
        transform=transform_ms2,
        norm_type=args.norm_type,
        truncnorm_min=args.truncnorm_min,
        use_captions=not args.no_captions,
        metric_norm=metric_norm,
        sky_mask_dir=args.sky_mask_dir,
    )
    train_dataloader_ms2 = torch.utils.data.DataLoader(
        train_dataset_ms2,
        shuffle=True,
        collate_fn=collate_fn_ms2,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True
        )

    # Lr_scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader_ms2) / args.gradient_accumulation_steps)
    assert args.max_train_steps is not None or args.num_train_epochs is not None, "max_train_steps or num_train_epochs should be provided"
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader_ms2, lr_scheduler = accelerator.prepare(  # MS2: one loader
        unet, optimizer, train_dataloader_ms2, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # The metric loss is the first thing in this trainer that needs a gradient to
    # travel back through the VAE decoder, and `vae` is about to become fp16
    # under the shipped recipe. An fp16 backward through this decoder underflows
    # the gradient to exactly zero on most steps -- measured on the route-suite
    # line (tools/train_route_suite.py:899), which is why that file keeps an fp32
    # decoder copy for exactly this purpose. Same fix here: a decode-only fp32
    # copy, still frozen. Taken *before* the cast below, so it holds the weights
    # `from_pretrained` loaded rather than an fp16 round trip of them. The
    # encoder is dropped because the metric term never encodes.
    # E2E-FT scores a decoded image, so its gradient crosses the decoder just as
    # the metric term's does -- and fp16 underflows that backward to exactly zero
    # (see the metric adaptation notes above), so it needs the same fp32 copy.
    metric_vae = None
    if args.metric_adaptation or args.backbone == "e2eft":
        metric_vae = copy.deepcopy(vae).to(device=accelerator.device, dtype=torch.float32)
        metric_vae.encoder = None
        metric_vae.requires_grad_(False)
        if args.metric_decode_grad_ckpt:
            # diffusers only honours checkpointing while the module reports
            # training mode. AutoencoderKL has no dropout and no batch norm, so
            # train() changes nothing about what it computes -- verified by the
            # round-trip check in tools/smoke_metric_adaptation.py, which runs
            # the decode both ways and compares.
            metric_vae.enable_gradient_checkpointing()
            metric_vae.train()
        else:
            metric_vae.eval()
        logger.info(
            f"Metric loss decoder: fp32 copy of the VAE decoder, "
            f"gradient checkpointing {'on' if args.metric_decode_grad_ckpt else 'off'}"
        )

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader_ms2) / args.gradient_accumulation_steps)  # MS2
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # E2E-FT needs the cumulative alphas to undo the v-parameterisation, and the
    # vendored loss object. Built once; both are inert for the other backbones.
    _alpha_prod = noise_scheduler.alphas_cumprod.to(accelerator.device)
    _ssi_loss = ScaleAndShiftInvariantLoss()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("task_name")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    # MS2: a COMPLETED job says nothing about whether the switches took effect, so
    # read a few frames and report what the loss will actually see. If the pseudo
    # GT directory were wrong the depth map would still load and still train -- it
    # would just be supervised by something else.
    _probe = [train_dataset_ms2[i] for i in range(0, len(train_dataset_ms2),
                                                  max(1, len(train_dataset_ms2) // 4))][:4]
    _covered = [float(torch.logical_or(e["valid_mask_values"], e["sky_mask_values"]).float().mean())
                for e in _probe]
    _captioned = sum(1 for e in _probe if e["text_description"].strip())
    logger.info(f"  Num examples MS2 = {len(train_dataset_ms2)}")
    logger.info(f"  Completed depth from = {args.pseudo_gt_dir}")
    logger.info(f"  Normalisation = {args.norm_type} (truncnorm_min {args.truncnorm_min})")
    logger.info(f"  {train_dataset_ms2.sky_summary()}")
    logger.info(f"  {len(_probe)}-frame probe: input {tuple(_probe[0]['pixel_values'].shape)}, "
                f"target covers {min(_covered) * 100:.1f}-{max(_covered) * 100:.1f}% of pixels, "
                f"{_captioned}/{len(_probe)} carry a caption")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Unet timestep = {args.timestep}")
    if args.metric_adaptation:
        logger.info("  --- metric GT adaptation ---")
        logger.info(f"  Init U-Net from = {args.init_unet_from}")
        logger.info(f"  Constants       = {metric_norm.summary()}")
        logger.info(
            f"  Loss            = {args.lambda_metric} * L_metric(sparse lidar, image space) "
            f"+ {args.lambda_dense} * L_dense(completed target, latent) "
            f"+ {args.lambda_recon} * L_I(thermal reconstruction, latent)"
        )
        _probe_lidar = [float(e["gt_valid_values"].mean()) for e in _probe]
        logger.info(
            f"  {len(_probe)}-frame probe: lidar covers "
            f"{min(_probe_lidar) * 100:.1f}-{max(_probe_lidar) * 100:.1f}% of pixels "
            f"(this is what L_metric is averaged over)"
        )
        logger.info(f"  {train_dataset_ms2.clip_summary()}")
    logger.info(f"  Task name: {args.task_name}")
    logger.info(f"  Backbone: lotus-{args.backbone} "
                f"(conv_in {4 if args.backbone == 'd' else 8} channels)")
    logger.info(f"  Is Full Evaluation?: {args.FULL_EVALUATION}")
    logger.info(f"Output Workspace: {args.output_dir}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    
    if accelerator.is_main_process and args.validation_images is not None:
        log_validation(
            vae,
            text_encoder,
            tokenizer,
            unet,
            args,
            accelerator,
            weight_dtype,
            global_step,
        )
            
    for epoch in range(first_epoch, args.num_train_epochs):
        # MS2: one loader, so the alternation between two datasets is gone. Nothing
        # inside `accelerator.accumulate` below changes.
        iter_ms2 = iter(train_dataloader_ms2)

        train_loss = 0.0
        log_ann_loss = 0.0
        log_rgb_loss = 0.0
        log_metric_loss = 0.0

        for _ in range(len(train_dataloader_ms2)):
            batch = next(iter_ms2)

            with accelerator.accumulate(unet):
                # Lotus packs two branches into one batch -- depth prediction and
                # thermal reconstruction -- and tells them apart with the task
                # switcher. Marigold has one branch and no switcher, so nothing
                # here is doubled and the reconstruction term does not exist.
                _two_branch = args.backbone not in SINGLE_BRANCH
                _img = batch["pixel_values"]
                # Convert images to latent space
                rgb_latents = vae.encode(
                    (torch.cat((_img, _img), dim=0) if _two_branch else _img).to(weight_dtype)
                    ).latent_dist.sample()
                rgb_latents = rgb_latents * vae.config.scaling_factor
                # Convert target_annotations to latent space
                assert len(args.task_name) == 1
                if args.task_name[0] == "depth":
                    TAR_ANNO = "depth_values"
                elif args.task_name[0] == "normal":
                    TAR_ANNO = "normal_values"
                else:
                    raise ValueError(f"Do not support {args.task_name[0]} yet. ")
                target_latents = vae.encode(
                    (torch.cat((batch[TAR_ANNO], _img), dim=0) if _two_branch
                     else batch[TAR_ANNO]).to(weight_dtype)
                    ).latent_dist.sample()
                target_latents = target_latents * vae.config.scaling_factor
                
                bsz = target_latents.shape[0]
                bsz_per_task = int(bsz/2) if _two_branch else bsz

                # Get the valid mask for the latent space
                valid_mask_for_latent = batch.get("valid_mask_values", None)
                # E2E-FT scores at image resolution, so it wants the mask before
                # the 8x8 pooling that the latent losses need.
                _e2e_mask = None
                if args.task_name[0] == "depth" and valid_mask_for_latent is not None:
                    sky_mask_for_latent = batch.get("sky_mask_values", None)
                    valid_mask_for_latent = valid_mask_for_latent + sky_mask_for_latent
                if valid_mask_for_latent is not None:
                    valid_mask_for_latent = valid_mask_for_latent.bool()
                    _e2e_mask = valid_mask_for_latent[:bsz_per_task]
                    invalid_mask = ~valid_mask_for_latent
                    valid_mask_down_anno = ~torch.max_pool2d(invalid_mask.float(), 8, 8).bool()
                    valid_mask_down_anno = valid_mask_down_anno.repeat((1, 4, 1, 1))
                else:
                    valid_mask_down_anno = torch.ones_like(target_latents[:bsz_per_task]).to(target_latents.device).bool()
                
                valid_mask_down_rgb = torch.ones_like(target_latents[bsz_per_task:]).to(target_latents.device).bool()

                # Set timestep
                # Hoisted above the noise block so the G path can keep it while the
                # D path skips everything else. It draws no random numbers, so G's
                # random stream is unchanged by the move.
                if args.backbone == "marigold":
                    # marigold_trainer.py.PD_text_prior:259 -- Marigold trains over
                    # the whole schedule, not at one terminal step.
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,),
                        device=target_latents.device,
                    )
                else:
                    timesteps = torch.tensor([args.timestep], device=target_latents.device).repeat(bsz)
                timesteps = timesteps.long()

                # The single architectural difference between the two backbones.
                # G conditions on [image latent, target latent noised at this
                # timestep]; D is the direct variant and sees the image latent
                # alone. train_iris_d.py:1125 computes no noise at all, so the
                # block below is skipped rather than computed and discarded --
                # the D path then draws exactly the random numbers upstream draws,
                # and this branch can be described as following the upstream
                # Lotus-D training path with only the thermal/data changes on top.
                if args.backbone == "d":
                    unet_input = rgb_latents
                else:
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(target_latents)

                    if args.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += args.noise_offset * torch.randn(
                            (target_latents.shape[0], target_latents.shape[1], 1, 1), device=target_latents.device
                        )
                    if args.input_perturbation:
                        new_noise = noise + args.input_perturbation * torch.randn_like(noise)

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    if args.input_perturbation:
                        noisy_latents = noise_scheduler.add_noise(target_latents, new_noise, timesteps)
                    else:
                        noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)

                    # Concatenate rgb and depth
                    unet_input = torch.cat(
                        [rgb_latents, noisy_latents], dim=1
                    )


                # Get text descriptions from batch
                text_descriptions = batch.get("text_descriptions", [""] * bsz)
                
                # Process text descriptions for both annotation and RGB tasks
                all_prompts = []
                for i in range(bsz_per_task):
                    # For annotation task, use the text description
                    if i < len(text_descriptions):
                        all_prompts.append(text_descriptions[i])
                    else:
                        all_prompts.append("")
                
                if _two_branch:
                    for i in range(bsz_per_task):
                        # For RGB reconstruction task, use empty string or same description
                        all_prompts.append("")
                
                # Tokenize all prompts
                text_inputs = tokenizer(
                    all_prompts,
                    padding=True,
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(target_latents.device)
                
                # Debug: Check for truncation
                # if global_step % 100 == 0:  # Print every 100 steps
                #     for i, prompt in enumerate(all_prompts):
                #         if prompt:  # Only check non-empty prompts
                #             # Count non-padding tokens (exclude padding token 0)
                #             input_ids = text_inputs.input_ids[i]
                #             non_padding_tokens = (input_ids != tokenizer.pad_token_id).sum().item()
                #             max_length = tokenizer.model_max_length
                            
                #             if non_padding_tokens >= max_length:
                #                 print(f"Step {global_step}, Prompt {i} TRUNCATED: {prompt[:100]}...")
                #                 print(f"  Original length: {len(prompt)} chars, Token length: {non_padding_tokens}/{max_length}")
                #             else:
                #                 print(f"Step {global_step}, Prompt {i} OK: {prompt[:50]}...")
                #                 print(f"  Token length: {non_padding_tokens}/{max_length}")
                #Debug end
                
                encoder_hidden_states = text_encoder(text_input_ids, return_dict=False)[0]

                #-----------------------mod add text descriptions--------------------------------

                # Get the target for loss
                if args.backbone == "marigold":
                    # marigold_trainer.py.PD_text_prior:366-373, transcribed.
                    _pt = noise_scheduler.config.prediction_type
                    if _pt == "sample":
                        target = target_latents
                    elif _pt == "epsilon":
                        target = noise
                    elif _pt == "v_prediction":
                        target = noise_scheduler.get_velocity(target_latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {_pt}")
                else:
                    target = target_latents

                # Get the task embedding
                task_emb_anno = torch.tensor([1, 0]).float().unsqueeze(0).to(accelerator.device)
                task_emb_anno = torch.cat([torch.sin(task_emb_anno), torch.cos(task_emb_anno)], dim=-1).repeat(bsz_per_task, 1)
                task_emb_rgb = torch.tensor([0, 1]).float().unsqueeze(0).to(accelerator.device)
                task_emb_rgb = torch.cat([torch.sin(task_emb_rgb), torch.cos(task_emb_rgb)], dim=-1).repeat(bsz_per_task, 1)
                task_emb = torch.cat((task_emb_anno, task_emb_rgb), dim=0)

                # Predict
                model_pred = unet(unet_input, timesteps, encoder_hidden_states, return_dict=False,
                                class_labels=task_emb if _two_branch else None)[0]

                # Compute loss
                if args.backbone == "e2eft":
                    # diffusion-e2e-ft/training/train.py:583-625, transcribed.
                    _ap = _alpha_prod[timesteps].view(-1, 1, 1, 1)
                    _bp = (1.0 - _alpha_prod)[timesteps].view(-1, 1, 1, 1)
                    _pt = noise_scheduler.config.prediction_type
                    if _pt == "v_prediction":
                        x0_latent = (_ap ** 0.5) * noisy_latents - (_bp ** 0.5) * model_pred
                    elif _pt == "epsilon":
                        x0_latent = (noisy_latents - (_bp ** 0.5) * model_pred) / (_ap ** 0.5)
                    elif _pt == "sample":
                        x0_latent = model_pred
                    else:
                        raise ValueError(f"Unknown prediction type {_pt}")
                    decoded = metric_vae.decode(
                        x0_latent.float() / metric_vae.config.scaling_factor, return_dict=False
                    )[0]
                    estimate = decoded.mean(dim=1, keepdim=True).clamp(-1.0, 1.0)
                    ground_truth = batch[TAR_ANNO][:, :1].to(estimate.device, torch.float32)
                    if _e2e_mask is None:
                        _e2e_mask = torch.ones_like(estimate, dtype=torch.bool)
                    anno_loss = _ssi_loss(estimate, ground_truth, _e2e_mask)
                else:
                    anno_loss = F.mse_loss(model_pred[:bsz_per_task][valid_mask_down_anno].float(), target[:bsz_per_task][valid_mask_down_anno].float(), reduction="mean")
                # With one branch the second half is empty and mse over it is NaN,
                # so the term is dropped rather than computed.
                rgb_loss = (
                    F.mse_loss(model_pred[bsz_per_task:][valid_mask_down_rgb].float(), target[bsz_per_task:][valid_mask_down_rgb].float(), reduction="mean")
                    if _two_branch else model_pred.new_zeros(())
                )
                # L = lambda_dense * L_dense + lambda_recon * L_I + lambda_metric * L_metric
                # At the default weights of 1 and metric adaptation off this is
                # `anno_loss + rgb_loss`, term for term, as it has always been.
                loss = args.lambda_dense * anno_loss + args.lambda_recon * rgb_loss
                metric_loss = loss.new_zeros(())
                metric_stats = None
                if args.metric_adaptation:
                    metric_loss, metric_pixels, metric_stats = metric_inverse_depth_loss(
                        metric_vae,
                        model_pred[:bsz_per_task],
                        batch["gt_depth_values"].to(accelerator.device),
                        batch["gt_valid_values"].to(accelerator.device),
                        metric_norm,
                    )
                    loss = loss + args.lambda_metric * metric_loss

                # Gather loss
                avg_anno_loss = accelerator.gather(anno_loss.repeat(args.train_batch_size)).mean()
                log_ann_loss += avg_anno_loss.item() / args.gradient_accumulation_steps
                avg_rgb_loss = accelerator.gather(rgb_loss.repeat(args.train_batch_size)).mean()
                log_rgb_loss += avg_rgb_loss.item() / args.gradient_accumulation_steps
                train_loss = log_ann_loss + log_rgb_loss
                if args.metric_adaptation:
                    avg_metric_loss = accelerator.gather(
                        metric_loss.detach().repeat(args.train_batch_size)
                    ).mean()
                    log_metric_loss += avg_metric_loss.item() / args.gradient_accumulation_steps
                    train_loss = train_loss + log_metric_loss

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            logs = {"SL": loss.detach().item(),
                    "SL_A": anno_loss.detach().item(),
                    "SL_R": rgb_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]}
            if args.metric_adaptation:
                # SL_M is the raw term; the run's own gate is that it falls, and
                # that mAbsRel -- unaligned, on measured pixels -- falls with it.
                logs["SL_M"] = metric_loss.detach().item()
                logs["mAbsRel"] = metric_stats.get("metric_abs_rel", float("nan"))
            progress_bar.set_postfix(**logs)
        
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                tracked = {"train_loss": train_loss,
                           "anno_loss": log_ann_loss,
                           "rgb_loss": log_rgb_loss}
                if args.metric_adaptation:
                    tracked["metric_loss"] = log_metric_loss
                    if metric_stats is not None:
                        tracked["metric_abs_rel"] = metric_stats.get("metric_abs_rel", float("nan"))
                accelerator.log(tracked, step=global_step)
                train_loss = 0.0
                log_ann_loss = 0.0
                log_rgb_loss = 0.0
                log_metric_loss = 0.0

                # Step 3's range logging. Cheap, and the only way a scale that has
                # quietly drifted out of the represented band becomes visible
                # before the evaluation says so hours later.
                if (
                    args.metric_adaptation
                    and metric_stats is not None
                    and args.metric_log_every > 0
                    and global_step % args.metric_log_every == 0
                    and accelerator.is_main_process
                ):
                    logger.info(
                        f"[metric step {global_step}] "
                        f"decoded y [{metric_stats['decoded_min']:.4f}, {metric_stats['decoded_max']:.4f}] "
                        f"| q_hat [{metric_stats['q_hat_min']:.5f}, {metric_stats['q_hat_max']:.5f}] 1/m "
                        f"| D_hat [{metric_stats['depth_hat_min']:.2f}, {metric_stats['depth_hat_max']:.2f}] m "
                        f"| nonfinite {metric_stats['q_hat_nonfinite']} "
                        f"non-positive {metric_stats['q_hat_non_positive']} "
                        f"| lidar px {metric_stats['lidar_pixels']:,} "
                        f"| unaligned AbsRel {metric_stats['metric_abs_rel']:.5f}"
                    )
                    logger.info(f"[metric step {global_step}] {metric_stats['depth_hat_clip']}")
                    logger.info(f"[metric step {global_step}] {train_dataset_ms2.clip_summary()}")

                checkpointing_steps = args.checkpointing_steps
                validation_steps = args.validation_steps
                
                if accelerator.is_main_process:
                    if global_step % checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                
                    # MS2: `log_validation` renders Iris's own eval datasets and
                    # scores them with their loaders; none of that is on disk here,
                    # and MS2 checkpoints are scored out of band under the official
                    # BridgeMSD protocol. Skipped unless the images are supplied.
                    if args.validation_images is not None and global_step % validation_steps == 0:
                        log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unwrap_model(unet)

        pipeline = (LotusDPipeline if args.backbone == "d" else LotusGPipeline).from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
