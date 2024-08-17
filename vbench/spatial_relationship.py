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


def get_position_score(locality, obj1,obj2, iou_threshold=0.1):
    # input obj1 and obj2 should be [x0,y0,x1,y1]
    # Calculate centers of bounding boxes
    box1 = {
        'x_min': obj1[0],
        'y_min': obj1[1],
        'x_max': obj1[2],
        'y_max': obj1[3],
        'width': obj1[2] - obj1[0],
        'height': obj1[3] - obj1[1]
    }

    box2 = {
        'x_min': obj2[0],
        'y_min': obj2[1],
        'x_max': obj2[2],
        'y_max': obj2[3],
        'width': obj2[2] - obj2[0],
        'height': obj2[3] - obj2[1]
    }
    
    # Get the object center
    box1_center = ((box1['x_min'] + box1['x_max']) / 2, (box1['y_min'] + box1['y_max']) / 2)
    box2_center = ((box2['x_min'] + box2['x_max']) / 2, (box2['y_min'] + box2['y_max']) / 2)

    # Calculate horizontal and vertical distances
    x_distance = box2_center[0] - box1_center[0]
    y_distance = box2_center[1] - box1_center[1]

    # Calculate IoU
    x_overlap = max(0, min(box1['x_max'], box2['x_max']) - max(box1['x_min'], box2['x_min']))
    y_overlap = max(0, min(box1['y_max'], box2['y_max']) - max(box1['y_min'], box2['y_min']))
    intersection = x_overlap * y_overlap
    box1_area = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
    box2_area = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])
    union = box1_area + box2_area - intersection
    iou = intersection / union

    # get max object width and max object height
    max_width = max(box1['width'], box2['width'])
    max_height = max(box1['height'], box2['height'])

    score=0
    if locality in 'on the right of' or locality in 'on the left of':
        if abs(x_distance) > abs(y_distance) and iou < iou_threshold:
            score=1
        elif abs(x_distance) > abs(y_distance) and iou >= iou_threshold:
            score=iou_threshold/iou
        else:
            score=0
    elif locality in 'on the bottom of' or locality in 'on the top of':
        if abs(y_distance) > abs(x_distance) and iou < iou_threshold:
            score=1
        elif abs(y_distance) > abs(x_distance) and iou >= iou_threshold:
            score=iou_threshold/iou
        else:
            score = 0
    return score

def get_dect_from_grit(model, image_arrays):
    pred = []
    if type(image_arrays) is not list:
        image_arrays = image_arrays.numpy()
    with torch.no_grad():
        for frame in image_arrays:
            ret = model.run_caption_tensor(frame)
            pred_cur = []
            if len(ret[0])>0:
                for info in ret[0]:
                    pred_cur.append([info[0],info[1]])
            pred.append(pred_cur)
    return pred

def check_generate(key_info, predictions):
    key_a = key_info['object_a']
    key_b = key_info['object_b']
    relation = key_info['relationship']
    frame_score =[]
    for frame_pred in predictions:
        # filter the target object
        frame_obj_locats = []
        cur_score = [0]
        for item in frame_pred:
            if (key_a == item[0]) or (key_b == item[0]):
                frame_obj_locats.append(item[1])
            for c_obj1 in range(len(frame_obj_locats)-1):
                for c_obj2 in range(c_obj1+1 ,len(frame_obj_locats)):
                    score_obj1_obj2 = get_position_score(relation, frame_obj_locats[c_obj1], frame_obj_locats[c_obj2])
                    cur_score.append(score_obj1_obj2)
        frame_score.append(max(cur_score))
    return frame_score

def spatial_relationship(model, video_dict, device):
    video_results = []
    frame_score_overall = []
    for info in tqdm(video_dict, disable=get_rank() > 0):
        if 'auxiliary_info' not in info:
            raise "Auxiliary info is not in json, please check your json."
        object_info = info['auxiliary_info']['spatial_relationship']
        for video_path in info['video_list']:
            video_tensor = load_video(video_path, num_frames=16)
            cur_video_pred = get_dect_from_grit(model, video_tensor.permute(0,2,3,1))
            cur_video_frame_score = check_generate(object_info, cur_video_pred)
            cur_success_frame_rate = np.mean(cur_video_frame_score)
            frame_score_overall.extend(cur_video_frame_score)
            video_results.append({'video_path': video_path, 'video_results': cur_success_frame_rate, 'frame_results':cur_video_frame_score})
    success_rate = np.mean(frame_score_overall)
    return success_rate, video_results
        

def compute_spatial_relationship(json_dir, device, submodules_dict, **kwargs):
    dense_caption_model = DenseCaptioning(device)
    dense_caption_model.initialize_model_det(**submodules_dict)
    logger.info("Initialize detection model success")
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='spatial_relationship', lang='en')
    prompt_dict_ls = distribute_list_to_rank(prompt_dict_ls)
    all_results, video_results = spatial_relationship(dense_caption_model, prompt_dict_ls, device)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results
