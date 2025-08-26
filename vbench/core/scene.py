import os
import json
import torch
import numpy as np
from tqdm import tqdm
from vbench.utils import load_video, load_dimension_info, tag2text_transform, CACHE_DIR, ensure_download
from vbench.third_party.tag2Text.tag2text import tag2text_caption
import logging
from vbench.distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)
from vbench.core import DimensionEvaluationBase, EvaluationResult, MemoryEstimate, MEMORY_USAGE_PROFILE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Scene(DimensionEvaluationBase):
    def __init__(self, device="cuda", batch_size=1):
        super().__init__(
            memory_profile=MEMORY_USAGE_PROFILE["scene"],
            device=device,
            batch_size=batch_size
        )
        
    def init_model(self, cache_folder=CACHE_DIR):
        tag2text_path = os.path.join(cache_folder, "caption_model", "tag2text_swin_14m.pth")
        ensure_download(
            tag2text_path,
            "https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth"
        )
        
        model = tag2text_caption(
            pretrained=tag2text_path,
            image_size=384,
            vit="swin_b"
        )
        model.eval()
        model = model.to(self.device)
        
        self.model["tag2text"] = model
        logger.info("Initialize caption model success")
        
    def _get_caption(self, model, image_arrays):
        caption, tag_predict = model.generate(image_arrays, tag_input=None, return_tag_predict=True)
        return caption
        
    def _check_generate(self, key_info, predictions):
        cur_cnt = 0
        key = key_info['scene']
        for pred in predictions:
            q_flag = [q in pred for q in key.split(' ')]
            if len(q_flag) == sum(q_flag):
                cur_cnt += 1
        return cur_cnt
        
    def _scene_evaluation(self, video_dict):
        success_frame_count, frame_count = 0, 0
        video_results = []
        transform = tag2text_transform(384)
        model = self.model["tag2text"]
        
        for info in tqdm(video_dict, disable=get_rank() > 0):
            if 'auxiliary_info' not in info:
                raise ValueError("Auxiliary info is not in json, please check your json.")
                
            scene_info = info['auxiliary_info']['scene']
            for video_path in info['video_list']:
                video_array = load_video(
                    video_path, 
                    num_frames=16, 
                    return_tensor=False, 
                    width=384, 
                    height=384
                )
                
                video_tensor_list = []
                for i in video_array:
                    video_tensor_list.append(transform(i).to(self.device).unsqueeze(0))
                video_tensor = torch.cat(video_tensor_list)
                
                cur_video_pred = self._get_caption(model, video_tensor)
                cur_success_frame_count = self._check_generate(scene_info, cur_video_pred)
                cur_success_frame_rate = cur_success_frame_count / len(cur_video_pred)
                
                success_frame_count += cur_success_frame_count
                frame_count += len(cur_video_pred)
                
                video_results.append({
                    'video_path': video_path,
                    'video_results': cur_success_frame_rate
                })
                
        success_rate = success_frame_count / frame_count if frame_count > 0 else 0.0
        return success_rate, video_results
        
    def compute_score(self, json_dir, submodules_list, **kwargs) -> EvaluationResult:
        _, prompt_dict_ls = load_dimension_info(json_dir, dimension='scene', lang='en')
        prompt_dict_ls = distribute_list_to_rank(prompt_dict_ls)
        
        all_results, video_results = self._scene_evaluation(prompt_dict_ls)
        
        if get_world_size() > 1:
            video_results = gather_list_of_dict(video_results)
            success_frame_count = sum([d['success_frame_count'] for d in video_results])
            frame_count = sum([d['frame_count'] for d in video_results])
            all_results = success_frame_count / frame_count if frame_count > 0 else 0.0
            
        return EvaluationResult(
            dimension="scene",
            overall_score=all_results,
            per_video_scores=video_results
        )
