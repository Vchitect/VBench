import os
import json
import logging
import numpy as np
import clip
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from vbench.utils import load_video, load_dimension_info, clip_transform, CACHE_DIR, ensure_download
from tqdm import tqdm

from vbench.distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)

from vbench.core import MEMORY_USAGE_PROFILE, DimensionEvaluationBase

class BackgroundConsistency(DimensionEvaluationBase):
    def __init__(self, memory_profile, device, batch_size):
        super.__init__(memory_profile=MEMORY_USAGE_PROFILE["background_consistency"],
                       device=device,
                       batch_size=batch_size)

    def init_model(self, cache_folder=CACHE_DIR):
        clip_path = os.path.join(cache_folder, "clip_model", "ViT-B-32.pt")
        ensure_download(clip_path, "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt")
        model, preprocess = clip.load(
            device=self.device,
            name=os.path.join(cache_folder, "clip_model/ViT-B-32.pt")
        )
        self.model = { "clip_vit_B_32": model }

    def background_consistency(self, clip_model, preprocess, video_list, read_frame):
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
            images = images.to(self.device)
            image_features = clip_model.encode_image(images) # batch
            image_features = F.normalize(image_features, dim=-1, p=2)

            first_image_feature = image_features[0].unsqueeze(0)
            former_image_feature = image_features[0].unsqueeze(0)

            for i in range(1, len(image_features)):
                image_feature = image_features[i].unsqueeze(0)
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
                'video_results': sim_per_image
            })
        # sim_per_video = sim / (len(video_list) - 1)
        sim_per_frame = sim / cnt
        return sim_per_frame, video_results


    def compute_score(self, json_dir, **kwargs):
        clip_model = self.model["clip_vit_B_32"].to(self.device)
        read_frame = kwargs.get("read_frame", False)

        video_list, _ = load_dimension_info(json_dir, dimension='background_consistency', lang='en')
        video_list = distribute_list_to_rank(video_list)
        all_results, video_results = self.background_consistency(clip_model, preprocess, video_list, read_frame)
        if get_world_size() > 1:
            video_results = gather_list_of_dict(video_results)
            sim = sum([d['video_sim'] for d in video_results])
            cnt = sum([d['cnt_per_video'] for d in video_results])
            all_results = sim / cnt
        return EvaluationResult(
            dimension="background_consistency",
            overall_score=all_results,
            per_video_scores=video_results
        )
