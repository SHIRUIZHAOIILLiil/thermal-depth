import logging
import argparse
import os
import json

from contextlib import nullcontext
import torch
from diffusers.utils import check_min_version
import random
import numpy as np

from pipeline import LotusGPipeline, LotusDPipeline
from utils.seed_all import seed_all
from evaluation.evaluation import evaluation_depth, evaluation_normal

check_min_version('0.28.0.dev0')

def parse_args():
    '''Set the Args'''
    parser = argparse.ArgumentParser(
        description="Run Lotus..."
    )
    # model settings
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="pretrained model path from hugging face or local dir",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="sample",
        help="The used prediction_type. ",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=999,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="regression", # "generation"
        help="Whether to use the generation or regression pipeline."
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="depth", # "normal"
    )
    parser.add_argument(
        "--disparity",
        action="store_true",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    # inference settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--base_test_data_dir",
        type=str,
        default="datasets/eval/"
    )
    parser.add_argument(
        "--eval_config_dir",
        type=str,
        default=None,
        help="Directory containing evaluation YAML configs. Defaults to the depth data directory.",
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )
    parser.add_argument(
        "--processing_res",
        type=int,
        default=None,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 768.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--resample_method",
        choices=["bilinear", "bicubic", "nearest"],
        default="bilinear",
        help="Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`. Default: `bilinear`",
    )
    parser.add_argument(
        "--rng_state_path",
        default=None,
        help="Load the random number generator states from the given path to ensure reproducibility of the results. "
    )
    parser.add_argument(
        "--text_descriptions_path",
        type=str,
        default=None,
        help="Path to JSON file containing text descriptions for the dataset",
    )
    parser.add_argument(
        "--nyuv2_text_descriptions_path",
        type=str,
        default=None,
        help="Path to JSON file containing text descriptions for NYU dataset",
    )
    parser.add_argument(
        "--kitti_text_descriptions_path",
        type=str,
        default=None,
        help="Path to JSON file containing text descriptions for KITTI dataset",
    )
    parser.add_argument(
        "--scannet_text_descriptions_path",
        type=str,
        default=None,
        help="Path to JSON file containing text descriptions for ScanNet dataset",
    )
    parser.add_argument(
        "--eth3d_text_descriptions_path",
        type=str,
        default=None,
        help="Path to JSON file containing text descriptions for ETH3D dataset",
    )
    parser.add_argument(
        "--diode_text_descriptions_path",
        type=str,
        default=None,
        help="Path to JSON file containing text descriptions for DIODE dataset",
    )
    parser.add_argument(
        "--strict_text_mode",
        action="store_true",
        help="Stop evaluation if any image is missing text description",
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        choices=["nyuv2", "kitti", "scannet", "diode", "eth3d"],
        default=["nyuv2", "kitti", "scannet", "diode", "eth3d"],
        help="Depth datasets to evaluate. Defaults to all supported datasets.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Limit each selected depth dataset to its first N samples.",
    )
    
    args = parser.parse_args()

    return args


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Run evaluation...")

    args = parse_args()

    # -------------------- Preparation --------------------
    # Random seed
    if args.seed is not None:
        seed_all(args.seed)

    # Output directories
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output dir = {args.output_dir}")

    # half_precision
    if args.half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32

    # processing_res
    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, due to the padding and pooling properties of conv layers."
        )
    # resample_method = args.resample_method

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"Device = {device}")

    if args.rng_state_path:
        states = torch.load(args.rng_state_path, map_location="cpu")
        
        # 1) Python / NumPy 
        if "random_state" in states:
            random.setstate(states["random_state"])
        if "numpy_random_seed" in states:
            np.random.set_state(states["numpy_random_seed"])
        
        # 2) Torch CPU
        cpu_entry = states.get("torch_manual_seed", None)
        if cpu_entry is not None:
            if isinstance(cpu_entry, torch.Tensor) and cpu_entry.dtype == torch.uint8:
                torch.set_rng_state(cpu_entry)
                logging.info(" Restored Torch CPU RNG STATE (uint8 Tensor).")
            elif isinstance(cpu_entry, (int, np.integer)):
                torch.manual_seed(int(cpu_entry))
                logging.info(f" Restored Torch CPU RNG SEED = {int(cpu_entry)}.")
            else:
                logging.warning(f" Unrecognized type for torch_manual_seed: {type(cpu_entry)}")
        
        # 3) Torch CUDA
        cuda_entry = states.get("torch_cuda_manual_seed", None)
        if torch.cuda.is_available() and cuda_entry is not None:
            if isinstance(cuda_entry, (list, tuple)) and len(cuda_entry) > 0:
                if all(isinstance(x, torch.Tensor) and x.dtype == torch.uint8 for x in cuda_entry):
                    expected_size = torch.cuda.get_rng_state(0).numel()
                    actual_size = cuda_entry[0].numel()
                    
                    if actual_size == expected_size:
                        n = min(len(cuda_entry), torch.cuda.device_count())
                        for i in range(n):
                            torch.cuda.set_rng_state(cuda_entry[i], device=i)
                        logging.info(f"✓ Restored Torch CUDA RNG STATES for {n} device(s).")
                    else:
                        logging.warning(f"⚠ CUDA RNG state size mismatch!")
                        logging.warning(f"  Expected: {expected_size} bytes, Got: {actual_size} bytes")
                        logging.warning(f"  Falling back to seed={args.seed or 42}")
                        torch.cuda.manual_seed_all(args.seed if args.seed else 42)
                        
                elif all(isinstance(x, (int, np.integer)) for x in cuda_entry):
                    n = min(len(cuda_entry), torch.cuda.device_count())
                    for i in range(n):
                        with torch.cuda.device(i):
                            torch.cuda.manual_seed(int(cuda_entry[i]))
                    logging.info(f"✓ Restored Torch CUDA RNG SEEDS for {n} device(s).")
                else:
                    logging.warning(f"⚠ Unrecognized list types in torch_cuda_manual_seed: {[type(x) for x in cuda_entry]}")
                    
            elif isinstance(cuda_entry, torch.Tensor) and cuda_entry.dtype == torch.uint8:
                expected_size = torch.cuda.get_rng_state(0).numel()
                actual_size = cuda_entry.numel()
                
                if actual_size == expected_size:
                    torch.cuda.set_rng_state(cuda_entry, device=0)
                    logging.info("✓ Restored Torch CUDA RNG STATE (device 0).")
                else:
                    logging.warning(f"⚠ CUDA RNG state size mismatch ({actual_size} vs {expected_size})")
                    logging.warning(f"  Falling back to seed={args.seed or 42}")
                    torch.cuda.manual_seed_all(args.seed if args.seed else 42)
                    
            elif isinstance(cuda_entry, (int, np.integer)):
                torch.cuda.manual_seed_all(int(cuda_entry))
                logging.info(f" Restored Torch CUDA RNG SEED (all devices) = {int(cuda_entry)}.")
            else:
                logging.warning(f" Unrecognized type for torch_cuda_manual_seed: {type(cuda_entry)}")
        
        logging.info(f" Loaded RNG snapshot from: {args.rng_state_path}")

    # -------------------- Load Text Descriptions --------------------
    # Map dataset names to their text description file paths
    dataset_text_paths = {
        "nyu_v2": args.nyuv2_text_descriptions_path,  # Note: actual dataset name is nyu_v2
        "kitti": args.kitti_text_descriptions_path,
        "scannet": args.scannet_text_descriptions_path,
        "eth3d": args.eth3d_text_descriptions_path,
        "diode": args.diode_text_descriptions_path,
    }
    
    # Fallback to general text descriptions path if provided
    if args.text_descriptions_path:
        for dataset in dataset_text_paths:
            if dataset_text_paths[dataset] is None:
                dataset_text_paths[dataset] = args.text_descriptions_path
    
    # Load text descriptions for each dataset
    all_text_descriptions = {}
    for dataset_name, text_path in dataset_text_paths.items():
        if text_path and os.path.exists(text_path):
            with open(text_path, 'r') as f:
                all_text_descriptions[dataset_name] = json.load(f)
            logging.info(f"✓ Loaded {len(all_text_descriptions[dataset_name])} text descriptions for {dataset_name} from {text_path}")
        else:
            all_text_descriptions[dataset_name] = {}
            logging.info(f"No text descriptions provided for {dataset_name}, using empty prompts")
    
    # Statistics for missing text descriptions
    missing_text_stats = {}

    # -------------------- Model --------------------
    if args.mode == 'generation':
        pipeline = LotusGPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=dtype,
        )
    elif args.mode == 'regression':
        pipeline = LotusDPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=dtype,
        )
    else:
        raise ValueError(f'Invalid mode: {args.mode}')
    logging.info(f"Successfully loading pipeline from {args.pretrained_model_name_or_path}.")

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    def gen_depth_with_text(rgb_in, pipe, image_path=None, num_inference_steps=1, dataset_name=None):
        """Wrapper function that uses text descriptions"""
        if torch.backends.mps.is_available():
                autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(pipe.device.type)

        with autocast_ctx:
            rgb_input = rgb_in / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]

            # Use text description if available
            prompt = ""
            if image_path and dataset_name and dataset_name in all_text_descriptions:
                text_descriptions = all_text_descriptions[dataset_name]
                if image_path in text_descriptions:
                    text_desc_data = text_descriptions[image_path]
                    if isinstance(text_desc_data, dict) and "description" in text_desc_data:
                        prompt = text_desc_data["description"]
                    elif isinstance(text_desc_data, str):
                        prompt = text_desc_data
                    logging.debug(f"Using text description for {dataset_name}/{image_path}: {prompt[:50]}...")
                else:
                    # Image has no corresponding text description
                    print(f" No text description found for {dataset_name}/{image_path}, using empty prompt")
                    # Track missing text descriptions
                    if dataset_name not in missing_text_stats:
                        missing_text_stats[dataset_name] = 0
                    missing_text_stats[dataset_name] += 1
                    
                    # Stop if strict mode is enabled
                    if args.strict_text_mode:
                        raise ValueError(f"Strict text mode enabled: Missing text description for {dataset_name}/{image_path}")
            elif image_path and dataset_name:
                # Dataset has no text descriptions loaded
                print(f"No text descriptions loaded for dataset {dataset_name}, using empty prompt for {image_path}")
                # Track missing text descriptions
                if dataset_name not in missing_text_stats:
                    missing_text_stats[dataset_name] = 0
                missing_text_stats[dataset_name] += 1
                
                # Stop if strict mode is enabled
                if args.strict_text_mode:
                    raise ValueError(f"Strict text mode enabled: No text descriptions loaded for dataset {dataset_name}")

            task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(pipe.device)
            task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)
            # print(f"Using prompt: {prompt}")
            pred_depth = pipe(
                            rgb_in=rgb_input, 
                            prompt=prompt, 
                            num_inference_steps=num_inference_steps,
                            output_type='np',
                            timesteps=[args.timestep],
                            task_emb=task_emb,
                            processing_res=0, # processing resolution before the pipeline
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
                            prompt=prompt, 
                            num_inference_steps=num_inference_steps,
                            output_type='pt',
                            timesteps=[args.timestep],
                            task_emb=task_emb,
                            processing_res=0, # processing resolution before the pipeline
                            ).images[0] # [0,1], (3,h,w)
            pred_normal = (pred_normal*2-1.0).unsqueeze(0) # [-1,1], (1,3,h,w)
        return pred_normal

    # -------------------- Evaluation --------------------
    with torch.no_grad():
        if args.task_name == 'depth':
            test_data_dir = os.path.join(args.base_test_data_dir, args.task_name)
            eval_config_dir = args.eval_config_dir or test_data_dir
            test_depth_dataset_configs = {
                "nyuv2": "configs/data_nyu_test.yaml", 
                "kitti": "configs/data_kitti_eigen_test.yaml",
                "scannet": "configs/data_scannet_val.yaml",
                "diode": "configs/data_diode_all.yaml",
                "eth3d": "configs/data_eth3d.yaml",
            }
            for dataset_name, config_path in test_depth_dataset_configs.items():
                if dataset_name not in args.eval_datasets:
                    continue
                eval_dir = os.path.join(args.output_dir, args.task_name, dataset_name)
                test_dataset_config = os.path.join(eval_config_dir, config_path)
                alignment_type = "least_square_disparity" if args.disparity else "least_square"
                metric_tracker = evaluation_depth(eval_dir, test_dataset_config, test_data_dir, eval_mode="generate_prediction",
                                                  gen_prediction=gen_depth_with_text, pipeline=pipeline, alignment=alignment_type,
                                                  processing_res=None, max_samples=args.max_eval_samples)
                print(dataset_name,',', 'abs_relative_difference: ', metric_tracker.result()['abs_relative_difference'], 'delta1_acc: ', metric_tracker.result()['delta1_acc'])
        elif args.task_name == 'normal':
            test_data_dir = os.path.join(args.base_test_data_dir, args.task_name)
            dataset_split_path = "evaluation/dataset_normal"
            eval_datasets = [ ('nyuv2', 'test'), ('scannet', 'test'), ('ibims', 'ibims'), ('sintel', 'sintel'),  ('oasis', 'val')]
            eval_dir = os.path.join(args.output_dir, args.task_name)
            evaluation_normal(eval_dir, test_data_dir, dataset_split_path, eval_mode="generate_prediction", 
                              gen_prediction=gen_normal, pipeline=pipeline, eval_datasets=eval_datasets,
                              processing_res=processing_res)
        else:
            raise ValueError(f"Not support predicting {args.task_name} yet. ")
        
        print('==> Evaluation is done. \n==> Results saved to:', args.output_dir)
        
        # Print statistics for missing text descriptions
        if missing_text_stats:
            print(' Text Description Statistics:')
            total_missing = sum(missing_text_stats.values())
            print(f'Total images without text descriptions: {total_missing}')
            for dataset_name, count in missing_text_stats.items():
                print(f'  {dataset_name}: {count} images without text descriptions')
            if total_missing > 0:
                print('  Warning: Some images were processed without text descriptions, which may affect results.')
        else:
            print('\n All images had corresponding text descriptions.')


if __name__ == '__main__':
    main()
