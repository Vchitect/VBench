import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import cv2
import torch
import vbench2.hack_registry
import numpy as np
from tqdm import tqdm
from vbench2.utils import load_video, load_dimension_info, read_frames_decord_by_fps
from vbench2.third_party.ViTDetector.detect import compute_abnormality

import logging

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_human_anatomy(json_dir, device, submodules_dict, **kwargs):

    video_list, _ = load_dimension_info(json_dir, dimension='human_anatomy', lang='en')
    all_results, video_results = compute_abnormality(video_list, device, submodules_dict, **kwargs)
    all_results = sum([x['video_results'] for x in video_results]) / len(video_results)
    return all_results, video_results