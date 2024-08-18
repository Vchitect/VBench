import os
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
from urllib.request import urlretrieve
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

batch_size = 32


def get_aesthetic_model(cache_folder):
    """load the aethetic model"""
    path_to_model = cache_folder + "/sa_0_4_vit_l_14_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true"
        )
        # download aesthetic predictor
        if not os.path.isfile(path_to_model):
            try:
                print(f'trying urlretrieve to download {url_model} to {path_to_model}')
                urlretrieve(url_model, path_to_model) # unable to download https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true to pretrained/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth 
            except:
                print(f'unable to download {url_model} to {path_to_model} using urlretrieve, trying wget')
                wget_command = ['wget', url_model, '-P', os.path.dirname(path_to_model)]
                subprocess.run(wget_command)
    m = nn.Linear(768, 1)
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m


def laion_aesthetic(aesthetic_model, clip_model, video_list, device):
    aesthetic_model.eval()
    clip_model.eval()
    aesthetic_avg = 0.0
    num = 0
    video_results = []
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        images = load_video(video_path)
        image_transform = clip_transform(224)

        aesthetic_scores_list = []
        for i in range(0, len(images), batch_size):
            image_batch = images[i:i + batch_size]
            image_batch = image_transform(image_batch)
            image_batch = image_batch.to(device)

            with torch.no_grad():
                image_feats = clip_model.encode_image(image_batch).to(torch.float32)
                image_feats = F.normalize(image_feats, dim=-1, p=2)
                aesthetic_scores = aesthetic_model(image_feats).squeeze()

            aesthetic_scores_list.append(aesthetic_scores)

        aesthetic_scores = torch.cat(aesthetic_scores_list, dim=0)
        normalized_aesthetic_scores = aesthetic_scores / 10
        cur_avg = torch.mean(normalized_aesthetic_scores, dim=0, keepdim=True)
        aesthetic_avg += cur_avg.item()
        num += 1
        video_results.append({'video_path': video_path, 'video_results': cur_avg.item()})

    aesthetic_avg /= num
    return aesthetic_avg, video_results


def compute_aesthetic_quality(json_dir, device, submodules_list, **kwargs):
    vit_path = submodules_list[0]
    aes_path = submodules_list[1]
    if get_rank() == 0:
        aesthetic_model = get_aesthetic_model(aes_path).to(device)
        barrier()
    else:
        barrier()
        aesthetic_model = get_aesthetic_model(aes_path).to(device)
    clip_model, preprocess = clip.load(vit_path, device=device)
    video_list, _ = load_dimension_info(json_dir, dimension='aesthetic_quality', lang='en')
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = laion_aesthetic(aesthetic_model, clip_model, video_list, device)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results
