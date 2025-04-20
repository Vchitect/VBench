import os
import json
import re
import shutil

json_path = 'prompts/VBench2_full_text_info.json'
source_video_file = '/mnt/petrelfs/zhengdian/zhengdian/VBench2.0/sample_video/vbench2_videos' # CHANGE
target_video_file = '/mnt/petrelfs/zhengdian/zhengdian/VBench2.0/sample_video/Vidu_Q1' # DEFAULT
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for video in os.listdir(source_video_file):
    source_path = os.path.join(source_video_file, video)
    video_suffix = re.split(r'-\d+\.mp4', video)[0]
    info = data[video_suffix[:180]]
    if 'Diversity' in info['dimension'] and len(info['dimension'])==1:
        target_path = os.path.join(target_video_file, 'Diversity', video)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        shutil.copy(source_path, target_path)
    elif 'Diversity' in info['dimension'] and len(info['dimension'])!=1:
        caption = info['caption'][:180]
        for dimension in info['dimension']:
            if video in [f"{caption}-0.mp4", f"{caption}-1.mp4", f"{caption}-2.mp4"]:
                target_path = os.path.join(target_video_file, dimension, video)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy(source_path, target_path)
            else:
                if dimension == 'Diversity':
                    target_path = os.path.join(target_video_file, dimension, video)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    shutil.copy(source_path, target_path)
    else:
        for dimension in info['dimension']:
            target_path = os.path.join(target_video_file, dimension, video)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy(source_path, target_path)