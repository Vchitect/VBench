import os
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
from urllib.request import urlretrieve
from vbench.utils import load_video, load_dimension_info, clip_transform
from tqdm import tqdm

from abc import ABC, abstractmethod

from vbench.distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)

batch_size = 32

from dataclasses import dataclass

@dataclass
class MemoryEstimate:
    model_size_gb: int = 0
    activation_base_gb: int = 0 # normalize to 512x320x16
    temporal_scaling: float = 1.0
    resolution_scaling: float = 1.0

class DimensionEvaluationBase(ABC):
    def __init__(self, memory_profile: MemoryEstimate):
        self.memory_profile = memory_profile

    @abstractmethod
    def init_model(self, cache_folder):
        pass
    
    @abstractmethod
    def calculate_score(self, json_dir, device, submodules_list, **kwargs):
        pass
    
    def estimate_memory_usage(self, resolution: tuple, timestep: int):

        assert len(resolution) == 2

        activation_scaling = self.memory_profile.temporal_scaling * self.memory_profile.resolution_scaling
        activation_scaling *= (timestep * resolution[0] * resolution[1] / (512 * 320 * 16))
        return activation_scaling * self.memory_profile.activation_base_gb + self.memory_profile.model_size_gb


class AestheticQuality(DimensionEvaluationBase):
    def __init__(self):
        super().__init__(
            memory_profile = MemoryEstimate(
                model_size_gb=0.91,
                activation_base_gb=1.0,
                temporal_scaling=0.0, # scaling logic is incorrect
                resolution_scaling=1.0,
            )
        )
        pass

    def init_model(self, cache_folder):
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

    def _calculate_laion_aesthetic(self, aesthetic_model, clip_model, video_list, device):
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
                    aesthetic_scores = aesthetic_model(image_feats).squeeze(dim=-1)

                aesthetic_scores_list.append(aesthetic_scores)

            aesthetic_scores = torch.cat(aesthetic_scores_list, dim=0)
            normalized_aesthetic_scores = aesthetic_scores / 10
            cur_avg = torch.mean(normalized_aesthetic_scores, dim=0, keepdim=True)
            aesthetic_avg += cur_avg.item()
            num += 1
            video_results.append({'video_path': video_path, 'video_results': cur_avg.item()})

        aesthetic_avg /= num
        return aesthetic_avg, video_results

    def compute_score(self, json_dir, device, submodules_list, **kwargs):
        vit_path = submodules_list[0]
        aes_path = submodules_list[1]
        if get_rank() == 0: # move to callee lvl
            aesthetic_model = self.init_model(aes_path).to(device)
            barrier()
        else:
            barrier()
            aesthetic_model = self.init_model(aes_path).to(device)
        clip_model, preprocess = clip.load(vit_path, device=device)
        video_list, _ = load_dimension_info(json_dir, dimension='aesthetic_quality', lang='en')
        video_list = distribute_list_to_rank(video_list)
        all_results, video_results = self._calculate_laion_aesthetic(aesthetic_model, clip_model, video_list, device)
        if get_world_size() > 1:
            video_results = gather_list_of_dict(video_results)
            all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
        return all_results, video_results
