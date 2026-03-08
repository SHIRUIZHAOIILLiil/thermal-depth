import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import os
from pathlib import Path
from tqdm import tqdm
import json
import tarfile
import io

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image_from_tar(tar_obj, image_path, input_size=448, max_num=12):
    """
    Load image from tar file
    """
    # Extract image from tar
    image_file = tar_obj.extractfile("./" + image_path)
    image_data = image_file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def find_all_images(tar_path, filename_list_path):
    """
    Find all RGB images from NYU dataset tar file using the filename list
    """
    image_files = []
    
    # Read the filename list
    with open(filename_list_path, 'r') as f:
        lines = f.readlines()
    
    # Extract RGB image paths from each line
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 1:
            rgb_path = parts[0]  # First part is RGB image path
            image_files.append(rgb_path)
    
    return sorted(image_files)

def main():
    # NYU dataset paths
    tar_path = 'lotus/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar'
    filename_list_path = 'lotus/datasets/eval/depth/data_split/nyu/labeled/filename_list_test.txt'
    output_file = 'outputs/lotus/nyu_depth_descriptions.json'
    
    prompt = "Describe this image in one sentence, focusing on salient attributes in the given image that are essential for monocular depth estimation, including but not limited to: camera factors, scene properties, relative distances, object types, object scales, illuminations, texture, visual features, occlusions, and boundaries to express how near or far different regions appear from the camera. Limit your response to one descriptive sentence under 77 tokens." 
    
    print("Loading model...")
    path = 'OpenGVLab/InternVL3-8B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=77, do_sample=True)
    
    print("Finding all images...")
    image_files = find_all_images(tar_path, filename_list_path)
    print(f"Found {len(image_files)} images")
    
    # Open tar file
    tar_obj = tarfile.open(tar_path)
    
    results = {}
    
    try:
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                pixel_values = load_image_from_tar(tar_obj, img_path, max_num=12).to(torch.bfloat16).cuda()
                
                question = f'<image>\n{prompt}'
                response = model.chat(tokenizer, pixel_values, question, generation_config)
                
                results[img_path] = {
                    'description': response
                }
                
                if len(results) % 100 == 0:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)
                    print(f"\nSaved checkpoint: {len(results)} images processed")
                
            except Exception as e:
                print(f"\nError processing {img_path}: {e}")
                continue
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nDone! Processed {len(results)} images. Results saved to {output_file}")
    
    finally:
        # Close tar file
        tar_obj.close()

if __name__ == '__main__':
    main()