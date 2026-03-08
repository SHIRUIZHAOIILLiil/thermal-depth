import json
import re

def convert_path(old_path):
    new_path = old_path.replace(
        'path/to/Lotus/data/',
        'path/to/Marigold/data/'
    )
    
    new_path = re.sub(r'/images/scene_cam_\d+_final_preview/', '/', new_path)
    
    match = re.search(r'frame\.(\d+)\.tonemap\.jpg$', new_path)
    if match:
        frame_num = match.group(1)
        cam_match = re.search(r'scene_cam_(\d+)_final_preview', old_path)
        cam_num = cam_match.group(1) if cam_match else '00'
        
        new_filename = f'rgb_cam_{cam_num}_fr{frame_num}.png'
        new_path = re.sub(r'frame\.\d+\.tonemap\.jpg$', new_filename, new_path)
    
    return new_path

def convert_json_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    new_data = {}
    for old_path, content in data.items():
        new_path = convert_path(old_path)
        new_data[new_path] = content
        print(f"转换: {old_path}\n  ->  {new_path}\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n successfully converted {len(new_data)} paths")
    print(f"results saved to: {output_file}")

if __name__ == "__main__":
    input_file = "internVL/outputs/lotus/hypersim_depth_descriptions.json"
    output_file = "internVL/outputs/marigold/hypersim_depth_descriptions.json"
    
    convert_json_file(input_file, output_file)