import os
import json

import torch
import numpy as np
from tqdm import tqdm
from vbench.utils import load_video, load_dimension_info, read_frames_decord_by_fps
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
    if type(image_arrays) is not list and type(image_arrays) is not np.ndarray:
        image_arrays = image_arrays.numpy()
    with torch.no_grad():
        for frame in image_arrays:
            ret = model.run_caption_tensor(frame)
            cur_pred = []
            if len(ret[0])<1:
                cur_pred.append(['',''])
            else:
                for idx, cap_det in enumerate(ret[0]):
                    cur_pred.append([cap_det[0], cap_det[2][0]])
            pred.append(cur_pred)
    return pred

def check_generate(color_key, object_key, predictions):
    cur_object_color, cur_object = 0, 0
    for frame_pred in predictions:
        object_flag, color_flag = False, False
        for pred in frame_pred:
            if object_key == pred[1]:
                for color_query in ["white","red","pink","blue","silver","purple","orange","green","gray","yellow","black","grey"]:
                    if color_query in pred[0]:
                        object_flag =True
                if color_key in pred[0]:
                    color_flag = True
        if color_flag:
            cur_object_color+=1
        if object_flag:
            cur_object +=1
    return cur_object, cur_object_color

def color(model, video_dict, device):
    success_frame_count_all, video_count = 0, 0
    video_results = []
    for info in tqdm(video_dict, disable=get_rank() > 0):
        if 'auxiliary_info' not in info:
            raise "Auxiliary info is not in json, please check your json."
        # print(info)
        color_info = info['auxiliary_info']['color']
        object_info = info['prompt']
        object_info = object_info.replace('a ','').replace('an ','').replace(color_info,'').strip()
        for video_path in info['video_list']:
            video_arrays = load_video(video_path, num_frames=16, return_tensor=False)
            cur_video_pred = get_dect_from_grit(model ,video_arrays)
            cur_object, cur_object_color = check_generate(color_info, object_info, cur_video_pred)
            if cur_object>0:
                cur_success_frame_rate = cur_object_color/cur_object
                success_frame_count_all += cur_success_frame_rate
                video_count += 1
                video_results.append({
                    'video_path': video_path, 
                    'video_results': cur_success_frame_rate,
                    'cur_success_frame_rate': cur_success_frame_rate,})
    success_rate = success_frame_count_all / video_count
    return success_rate, video_results
        

def compute_color(json_dir, device, submodules_dict, **kwargs):
    dense_caption_model = DenseCaptioning(device)
    dense_caption_model.initialize_model(**submodules_dict)
    logger.info("Initialize detection model success")
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='color', lang='en')
    prompt_dict_ls = distribute_list_to_rank(prompt_dict_ls)
    all_results, video_results = color(dense_caption_model, prompt_dict_ls, device)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        success_frame_count = sum([x['cur_success_frame_rate'] for x in video_results])
        frame_count = len(video_results)
        all_results = success_frame_count / frame_count
    return all_results, video_results
