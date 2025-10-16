#############################################################################
# code for paper: Sora Generateds Video with Stunning Geometical Consistency
# arxiv: https://arxiv.org/abs/
# Author: <NAME> xuanyili
# email: xuanyili.edu@gmail.com
# github: https://github.com/meteorshowers/SoraGeoEvaluate
#############################################################################
import os
import cv2
import csv
import numpy as np
from tqdm import tqdm
import torch
from .utils_function import *
from .core_function import *

def get_frames(video_path):
    frame_list = []
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS) # get fps
    while video.isOpened():
        success, frame = video.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to rgb
            frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
            frame = frame[None]
            frame_list.append(frame)
        else:
            break
    video.release()
    assert frame_list != []
    return frame_list, fps

def extract_frame(frame_list, skip_frame, end_frame, match_inter, interval_num):
    extract = []
    for i in range(0, end_frame, match_inter):
        if(i + interval_num > end_frame - 1) or i>skip_frame: 
            break
        extract.append((frame_list[i], frame_list[i+interval_num]))
    return extract
    
def PatchAutoEvaluate(video_path, skip_frame, end_frame, flow, fps):
    interval_nums = 40 
    begin_frame = 0 
    ransac_th = 3
    err_list = []
    if flow!=-1:
        interval_num = int(interval_nums / ((flow/10)* max(1, round(fps / 8))))
    else:
        return -1

    match_inter = int((end_frame - 1 - interval_num) / interval_num)
    if match_inter<1:
        return -1
    
    frame_list, fps = get_frames(video_path)
    if skip_frame==-1:
        frame_list = extract_frame(frame_list, end_frame, end_frame, match_inter, interval_num)
    else:
        frame_list = extract_frame(frame_list, skip_frame, end_frame, match_inter, interval_num)
    if len(frame_list)<=1:
        return -1    
    for idx, (left_img, right_img) in enumerate(frame_list):
        valid_point= EvaluateErrBetweenTwoImage(left_img[0], right_img[0], ransac_th)
        if valid_point is None:
            err_list.append(0)
        else:
            err_list.append(valid_point)
                
    errors_all = np.array(err_list)
    mean_valid_point = np.mean(errors_all)
    max_m=750
    mean_valid_point = np.clip(mean_valid_point, 0, max_m)/max_m
    return mean_valid_point
