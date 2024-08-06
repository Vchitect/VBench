import os
import json
import logging
import numpy as np
import clip
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from vbench.utils import load_video, load_dimension_info, clip_transform
from tqdm import tqdm

from .distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)


def background_consistency(clip_model, preprocess, video_list, device, read_frame):
    sim = 0.0
    cnt = 0
    video_results = []
    image_transform = clip_transform(224)
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        video_sim = 0.0
        cnt_per_video = 0
        if read_frame:
            video_path = video_path[:-4].replace('videos', 'frames').replace(' ', '_')
            tmp_paths = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path))]
            images = []
            for tmp_path in tmp_paths:
                images.append(preprocess(Image.open(tmp_path)))
            images = torch.stack(images)
        else:
            images = load_video(video_path)
            images = image_transform(images)
        images = images.to(device)
        image_features = clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1, p=2)
        for i in range(len(image_features)):
            image_feature = image_features[i].unsqueeze(0)
            if i == 0:
                first_image_feature = image_feature
            else:
                sim_pre = max(0.0, F.cosine_similarity(former_image_feature, image_feature).item())
                sim_fir = max(0.0, F.cosine_similarity(first_image_feature, image_feature).item())
                cur_sim = (sim_pre + sim_fir) / 2
                video_sim += cur_sim
                cnt += 1
                cnt_per_video += 1
            former_image_feature = image_feature
        sim_per_image = video_sim / (len(image_features) - 1)
        sim += video_sim
        video_results.append({
            'video_path': video_path, 
            'video_results': sim_per_image,
            'video_sim': video_sim,
            'cnt_per_video': cnt_per_video})
    # sim_per_video = sim / (len(video_list) - 1)
    sim_per_frame = sim / cnt
    return sim_per_frame, video_results


def compute_background_consistency(json_dir, device, submodules_list, **kwargs):
    vit_path, read_frame = submodules_list[0], submodules_list[1]
    clip_model, preprocess = clip.load(vit_path, device=device)
    video_list, _ = load_dimension_info(json_dir, dimension='background_consistency', lang='en')
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = background_consistency(clip_model, preprocess, video_list, device, read_frame)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        sim = sum([d['video_sim'] for d in video_results])
        cnt = sum([d['cnt_per_video'] for d in video_results])
        all_results = sim / cnt
    return all_results, video_results

