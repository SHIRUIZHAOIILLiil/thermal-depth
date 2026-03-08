import json
import sys

def convert_vkitti_path(old_path):
    new_path = old_path.replace(
        'path/to/Lotus/data/vkitti/',
        'path/to/Marigold/data/vkitti/vkitti_2.0.3_rgb/'
    )
    return new_path

def convert_json_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    new_data = {}
    for old_path, content in data.items():
        new_path = convert_vkitti_path(old_path)
        new_data[new_path] = content
        print(f"converted: {old_path}\n  ->  {new_path}\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n successfully converted {len(new_data)} paths")
    print(f"results saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "vkitti_output.json"
    else:
        input_file = "internVL/outputs/lotus/vkitti_depth_descriptions.json"
        output_file = "internVL/outputs/marigold/vkitti_depth_descriptions.json"
    
    convert_json_file(input_file, output_file)