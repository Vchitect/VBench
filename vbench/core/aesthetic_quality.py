import os
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
from urllib.request import urlretrieve
from vbench.utils import load_video, load_dimension_info, clip_transform, CACHE_DIR, ensure_download
from tqdm import tqdm

from abc import ABC, abstractmethod

from vbench.core import DimensionEvaluationBase, MEMORY_USAGE_PROFILE, EvaluationResult

from vbench.distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)

class AestheticQuality(DimensionEvaluationBase):
    def __init__(self, device="cuda", batch_size=1):
        super().__init__(memory_profile=MEMORY_USAGE_PROFILE["aesthetic_quality"])
        self.batch_size = batch_size
        self.device = device

    def init_model(self, cache_folder=CACHE_DIR):
        path_to_head = os.path.join(cache_folder, "aesthetic_model", "emb_reader", "sa_0_4_vit_l_14_linear.pth")
        path_to_base = os.path.join(cache_folder, "clip_model", "ViT-L-14.pt")

        ensure_download(path_to_head, "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true")
        ensure_download(path_to_base, "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt")
        linear = nn.Linear(768, 1)
        s = torch.load(path_to_head)
        linear.load_state_dict(s)
        lienar.eval()
        clip_base, preprocess = clip.load(path_to_base, device=device)
        self.model = {"aesthetic_model": linear, "clip_vit_L_14": clip_base}

    def _calculate_laion_aesthetic(self, aesthetic_model, clip_model, video_list, batch_size):
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
                image_batch = image_batch.to(self.device)

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

    def compute_score(self, json_dir, submodules_list, **kwargs) -> EvaluationResult:
        clip_model = self.model["clip_vit_L_14"].to(self.device)
        aesthetic_model = self.model["aesthetic_model"].to(self.device)

        assert len(self.model) > 0, "Model not initialized"
        video_list, _ = load_dimension_info(json_dir, dimension='aesthetic_quality', lang='en')
        video_list = distribute_list_to_rank(video_list)
        all_results, video_results = self._calculate_laion_aesthetic(aesthetic_model, clip_model, video_list, self.batch_size)
        if get_world_size() > 1: # move to callee lvl
            video_results = gather_list_of_dict(video_results)
            all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
        return EvaluationResult(
            dimension="aesthetic_quality",
            overall_score=all_results,
            per_video_scores=video_results
        )
