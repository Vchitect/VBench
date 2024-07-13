import os
import json

import torch
import numpy as np
from tqdm import tqdm
from vbench.utils import load_video, load_dimension_info, tag2text_transform
from vbench.third_party.tag2Text.tag2text import tag2text_caption

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


def get_caption(model, image_arrays):
    caption, tag_predict = model.generate(image_arrays, tag_input = None, return_tag_predict = True)
    return caption

def check_generate(key_info, predictions):
    cur_cnt = 0
    key = key_info['scene']
    for pred in predictions:
        q_flag = [q in pred for q in key.split(' ')]
        if len(q_flag) == sum(q_flag):
            cur_cnt +=1
    return cur_cnt

def scene(model, video_dict, device):
    success_frame_count, frame_count = 0, 0
    video_results = []
    transform = tag2text_transform(384)
    for info in tqdm(video_dict, disable=get_rank() > 0):
        if 'auxiliary_info' not in info:
            raise "Auxiliary info is not in json, please check your json."
        scene_info = info['auxiliary_info']['scene']
        for video_path in info['video_list']:
            video_array = load_video(video_path, num_frames=16, return_tensor=False, width=384, height=384)
            video_tensor_list = []
            for i in video_array:
                video_tensor_list.append(transform(i).to(device).unsqueeze(0))
            video_tensor = torch.cat(video_tensor_list)
            cur_video_pred = get_caption(model, video_tensor)
            cur_success_frame_count = check_generate(scene_info, cur_video_pred)
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
        

def compute_scene(json_dir, device, submodules_dict, **kwargs):
    model = tag2text_caption(**submodules_dict)
    model.eval()
    model = model.to(device)
    logger.info("Initialize caption model success")
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='scene', lang='en')
    prompt_dict_ls = distribute_list_to_rank(prompt_dict_ls)
    all_results, video_results = scene(model, prompt_dict_ls, device)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        success_frame_count = sum([d['success_frame_count'] for d in video_results])
        frame_count = sum([d['frame_count'] for d in video_results])
        all_results = success_frame_count / frame_count
    return all_results, video_results
