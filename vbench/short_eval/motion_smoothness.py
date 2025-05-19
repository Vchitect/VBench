import os
import cv2
import glob
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

from vbench.utils import load_dimension_info

from vbench.third_party.amt.utils.utils import (
    img2tensor, tensor2img,
    check_dim_and_resize
    )
from vbench.third_party.amt.utils.build_utils import build_from_cfg
from vbench.third_party.amt.utils.utils import InputPadder

from vbench.distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)

from vbench.core.motion_smoothness import motion_smoothness

def compute_motion_smoothness(json_dir, device, submodules_list, **kwargs):
    config = submodules_list["config"] # pretrained/amt_model/AMT-S.yaml
    ckpt = submodules_list["ckpt"] # pretrained/amt_model/amt-s.pth
    motion = MotionSmoothness(config, ckpt, device)
    video_list, _ = load_dimension_info(json_dir, dimension='motion_smoothness', lang='en')
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = motion_smoothness(motion, video_list)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results
