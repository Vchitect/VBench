import os
import json
import torch
import numpy as np
from tqdm import tqdm
from vbench.utils import load_video, load_dimension_info, CACHE_DIR, ensure_download
from vbench.third_party.grit_model import DenseCaptioning
from torchvision import transforms
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


class MultipleObjects(DimensionEvaluationBase):
    def __init__(self, device="cuda", batch_size=1):
        super().__init__(
            memory_profile=MEMORY_USAGE_PROFILE["multiple_objects"],
            device=device,
            batch_size=batch_size
        )
        
    def init_model(self, cache_folder=CACHE_DIR):
        grit_path = os.path.join(cache_folder, "grit_model", "grit_b_densecap_objectdet.pth")
        ensure_download(
            grit_path,
            "https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/grit_b_densecap_objectdet.pth"
        )
        
        caption_model = DenseCaptioning(self.device)
        caption_model.initialize_model(grit_path)
        self.model["grit_b"] = caption_model
        logger.info("Initialize detection model success")
        
    def _get_detection_from_grit(self, model, image_arrays):
        pred = []
        pred_batch = []
        if type(image_arrays) is not list:
            image_arrays = image_arrays.numpy()
        with torch.no_grad():

            for i in range(0, len(image_arrays), self.batch_size):
                image_batch = image_arrays[i * self.batch_size : (i+1) * self.batch_size]
                predictions = model.run_caption_tensor_batch(image_batch)

                pred_batch.extend([set(prediction[0][2]) if len(prediction) > 0  else set([])
                                   for prediction in predictions])
                
            for frame in image_arrays:
                ret = model.run_caption_tensor(frame)
                if len(ret) > 0:
                    pred.append(set(ret[0][2]))
                else:
                    pred.append(set([]))

            assert pred == pred_batch, f"should be equal {pred} {pred_batch}"
        return pred
        
    def _check_generate(self, key_info, predictions):
        cur_cnt = 0
        key_a, key_b = key_info.split(' and ')
        key_a = key_a.strip()
        key_b = key_b.strip()
        for pred in predictions:
            if key_a in pred and key_b in pred:
                cur_cnt += 1
        return cur_cnt
        
    def _multiple_objects_evaluation(self, video_dict):
        success_frame_count, frame_count = 0, 0
        video_results = []
        model = self.model["grit_b"]
        
        for info in tqdm(video_dict, disable=get_rank() > 0):
            if 'auxiliary_info' not in info:
                raise ValueError("Auxiliary info is not in json, please check your json.")
                
            object_info = info['auxiliary_info']['object']
            for video_path in info['video_list']:
                video_tensor = load_video(video_path, num_frames=16)
                _, _, h, w = video_tensor.size()
                
                # Resize if video resolution is too high
                if min(h, w) > 768:
                    scale = 720. / min(h, w)
                    output_tensor = transforms.Resize(
                        size=(int(scale * h), int(scale * w))
                    )(video_tensor)
                    video_tensor = output_tensor
                    
                cur_video_pred = self._get_detection_from_grit(
                    model, 
                    video_tensor.permute(0, 2, 3, 1)
                )
                cur_success_frame_count = self._check_generate(object_info, cur_video_pred)
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
        _, prompt_dict_ls = load_dimension_info(json_dir, dimension='multiple_objects', lang='en')
        prompt_dict_ls = distribute_list_to_rank(prompt_dict_ls)
        
        all_results, video_results = self._multiple_objects_evaluation(prompt_dict_ls)

        if get_world_size() > 1:
            video_results = gather_list_of_dict(video_results)
            success_frame_count = sum([x['success_frame_count'] for x in video_results])
            frame_count = sum([x['frame_count'] for x in video_results])
            all_results = success_frame_count / frame_count if frame_count > 0 else 0.0
            
        return EvaluationResult(
            dimension="multiple_objects",
            overall_score=all_results,
            per_video_scores=video_results
        )
