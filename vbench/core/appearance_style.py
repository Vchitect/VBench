import os
import json
import numpy as np
from tqdm import tqdm

import torch
import clip
from PIL import Image
from vbench.utils import load_video, load_dimension_info, clip_transform, read_frames_decord_by_fps, clip_transform, CACHE_DIR, ensure_download

from vbench.distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)

from vbench.core import DimensionEvaluationBase, MemoryEstimate, MEMORY_USAGE_PROFILE, EvaluationResult

class ApperanceStyle(DimensionEvaluationBase):
    def __init__(self, device="cuda", batch_size=1):
        super().__init__(
            memory_profile = MEMORY_USAGE_PROFILE["apperance_style"],
            device=device,
            batch_size=batch_size
        )

    def init_model(self, cache_folder=CACHE_DIR):
        clip_path = os.path.join(cache_folder, "clip_model", "ViT-B-32.pt")
        ensure_download(clip_path, "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt")
        model, preprocess  = clip.load(clip_path, device=self.device)
        self.model["clip_vit_B_32"] = model

    def _appearance_style(self, clip_model, video_dict, sample="rand"):
        sim = 0.0
        cnt = 0
        video_results = []
        image_transform = clip_transform(224)
        for info in tqdm(video_dict, disable=get_rank() > 0):
            if 'auxiliary_info' not in info:
                raise "Auxiliary info is not in json, please check your json."
            query = info['auxiliary_info']['appearance_style']
            text = clip.tokenize([query]).to(self.device)
            video_list = info['video_list']
            for video_path in video_list:
                cur_video = torch.tensor([0])
                with torch.no_grad():
                    video_arrays = load_video(video_path, return_tensor=False)
                    images = [Image.fromarray(i) for i in video_arrays]
                    for batch_idx in range(0, len(images), self.batch_size):
                        images_batch = images_batch[batch_idx * self.batch_size, (batch_idx+1) * self.batch_size]
                        images_batch = images_batch.to(self.device)
                        logits_per_images, logits_per_text = clip_model(images_batch, text)
                        cur_sim = logits_per_text[:, 0].cpu() / 100
                        cur_video = torch.cat((cur_video, cur_sim))
                    cur_video = cur_video[1:]
                    video_sim = torch.mean(cur_video)
                    video_results.append({
                        'video_path': video_path, 
                        'video_results': video_sim, 
                        'frame_results': list(cur_video)
                    })
        sim_per_frame = sim / cnt
        return sim_per_frame, video_results

    def compute_score(self, json_dir, **kwargs) -> Evaluationresult:
        clip_model = self.model["clip_vit_B_32"].to(self.device)

        _, video_dict = load_dimension_info(json_dir, dimension='appearance_style', lang='en')
        video_dict = distribute_list_to_rank(video_dict)
        all_results, video_results = self._appearance_style(clip_model, video_dict)
        if get_world_size() > 1:
            video_results = gather_list_of_dict(video_results)
            all_results = sum([d['cur_sim'] for d in video_results]) / len(video_results)
        return EvaluationResult(
            dimension="apperance_style",
            overall_score=all_results,
            per_video_scores=video_results
        )
        # return all_results, video_results
