import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import cv2
import torch
import vbench2.hack_registry
import numpy as np
from tqdm import tqdm
from vbench2.utils import load_video, load_dimension_info, read_frames_decord_by_fps
from vbench2.third_party.Instance_detector.test import compute_anomaly

import logging

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
            
def compute_instance_preservation(json_dir, device, submodules_dict, **kwargs):
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='instance_preservation', lang='en')
    all_results, video_results = compute_anomaly(prompt_dict_ls, device, submodules_dict)
    return all_results, video_results