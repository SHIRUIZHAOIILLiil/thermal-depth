import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import json

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

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def find_all_images(base_dir):
    base_path = Path(base_dir)
    image_files = []
    
    for split_dir in ['train', 'test', 'val']:
        split_path = base_path / split_dir
        if not split_path.exists():
            continue
        
        for rgb_file in split_path.rglob('rgb_*.png'):
            image_files.append(rgb_file)
    
    return sorted(image_files)

def write_json_files(results, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    split_results = {'train': {}, 'test': {}, 'val': {}}
    all_results = {}
    
    for img_path_str, data in results.items():
        img_path = Path(img_path_str)
        
        parts = img_path.parts
        split = None
        for s in ['train', 'test', 'val']:
            if s in parts:
                split = s
                break
        
        if split and split in split_results:
            split_results[split][img_path_str] = data
        all_results[img_path_str] = data
    
    for split in ['train', 'test', 'val']:
        json_file = output_path / f'hypersim_rgb_descriptions_{split}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(split_results[split], f, indent=2, ensure_ascii=False)
        print(f"Saved {split} JSON file: {json_file} ({len(split_results[split])} images)")
    
    all_json_file = output_path / 'hypersim_rgb_descriptions_all.json'
    with open(all_json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved all JSON file: {all_json_file} ({len(all_results)} images)")

def main():
    base_dir = 'marigold/data/hypersim'
    output_dir = 'outputs/e2e'
    output_file = Path(output_dir) / 'hypersim_rgb_descriptions_77.json'
    
    prompt = "Describe this image in one sentence, focusing on salient attributes in the given image that are essential for monocular depth estimation, including but not limited to: camera factors, scene properties, relative distances, object types, object scales, illuminations, texture, visual features, occlusions, and boundaries to express how near or far different regions appear from the camera. Limit your response to one descriptive sentence under 77 tokens." 
    
    results = {}
    if output_file.exists():
        print(f"Loading existing results from {output_file}...")
        with open(output_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results")
    else:
        all_json_file = Path(output_dir) / 'hypersim_rgb_descriptions_all.json'
        if all_json_file.exists():
            print(f"Loading existing results from {all_json_file}...")
            with open(all_json_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing results")
    
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
    image_files = find_all_images(base_dir)
    print(f"Found {len(image_files)} images")
    
    processed_count = 0
    remaining_files = []
    for img_path in image_files:
        img_path_str = str(img_path)
        if img_path_str in results:
            processed_count += 1
        else:
            remaining_files.append(img_path)
    
    print(f"Skipping {processed_count} already processed images")
    print(f"Remaining {len(remaining_files)} images to process")
    
    new_results_count = 0
    for img_path in tqdm(remaining_files, desc="Processing images"):
        try:
            pixel_values = load_image(str(img_path), max_num=12).to(torch.bfloat16).cuda()
            
            question = f'<image>\n{prompt}'
            response = model.chat(tokenizer, pixel_values, question, generation_config)
            
            results[str(img_path)] = {
                'description': response
            }
            new_results_count += 1
            
            if new_results_count % 100 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"\nSaved checkpoint: {len(results)} total images ({new_results_count} new)")
            
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue
    
    print("\nGenerating JSON files...")
    write_json_files(results, output_dir)
    
    print(f"\nDone! Total {len(results)} images ({new_results_count} newly processed). Results saved to {output_dir}")

if __name__ == '__main__':
    main()