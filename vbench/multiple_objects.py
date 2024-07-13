import os
import json

import torch
import numpy as np
from tqdm import tqdm
from vbench.utils import load_video, load_dimension_info
from vbench.third_party.grit_model import DenseCaptioning

import logging

from .distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)


logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_dect_from_grit(model, image_arrays):
    pred = []
    if type(image_arrays) is not list:
        image_arrays = image_arrays.numpy()
    with torch.no_grad():
        for frame in image_arrays:
            ret = model.run_caption_tensor(frame)
            if len(ret[0])>0:
                pred.append(set(ret[0][0][2]))
            else:
                pred.append(set([]))
    return pred

def check_generate(key_info, predictions):
    cur_cnt = 0
    key_a, key_b = key_info.split(' and ')
    key_a = key_a.strip()
    key_b = key_b.strip()
    for pred in predictions:
        if key_a in pred and key_b in pred:
            cur_cnt+=1
    return cur_cnt

def multiple_objects(model, video_dict, device):
    success_frame_count, frame_count = 0,0
    video_results = []
    for info in tqdm(video_dict, disable=get_rank() > 0):
        if 'auxiliary_info' not in info:
            raise "Auxiliary info is not in json, please check your json."
        object_info = info['auxiliary_info']['object']
        for video_path in info['video_list']:
            video_tensor = load_video(video_path, num_frames=16)
            cur_video_pred = get_dect_from_grit(model, video_tensor.permute(0,2,3,1))
            cur_success_frame_count = check_generate(object_info, cur_video_pred)
            cur_success_frame_rate = cur_success_frame_count/len(cur_video_pred)
            success_frame_count += cur_success_frame_count
            frame_count += len(cur_video_pred)
            video_results.append({
                'video_path': video_path, 
                'video_results': cur_success_frame_rate,
                'success_frame_count': cur_success_frame_count,
                'frame_count': len(cur_video_pred)})
    success_rate = success_frame_count / frame_count
    return success_rate, video_results
        

def compute_multiple_objects(json_dir, device, submodules_dict, **kwargs):
    dense_caption_model = DenseCaptioning(device)
    dense_caption_model.initialize_model_det(**submodules_dict)
    logger.info("Initialize detection model success")
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='multiple_objects', lang='en')
    prompt_dict_ls = distribute_list_to_rank(prompt_dict_ls)
    all_results, video_results = multiple_objects(dense_caption_model, prompt_dict_ls, device)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        success_frame_count = sum([x['success_frame_count'] for x in video_results])
        frame_count = sum([x['frame_count'] for x in video_results])
        all_results = success_frame_count / frame_count
    return all_results, video_results
