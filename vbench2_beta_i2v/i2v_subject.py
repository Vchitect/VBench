import io
import os
import cv2
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from vbench2_beta_i2v.utils import load_video, load_i2v_dimension_info, dino_transform_internet, dino_transform_Image_internet
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def i2v_subject(model, video_pair_list, device):
    video_results = []
    sim_list = []

    max_weight = 0.4
    mean_weight = 0.3
    min_weight = 0.3

    image_transform = dino_transform_Image_internet()
    frames_transform = dino_transform_internet()

    for image_path, video_path in tqdm(video_pair_list):
        # input image preprocess & extract feature
        input_image = image_transform(Image.open(image_path))
        input_image = input_image.unsqueeze(0)
        input_image = input_image.to(device)
        input_image_features = model(input_image)
        input_image_features = F.normalize(input_image_features, dim=-1, p=2)

        # get frames from video
        images = load_video(video_path)
        images = frames_transform(images)

        # calculate sim between input image and frames in generated video
        conformity_scores = []
        consec_scores = []
        for i in range(len(images)):
            with torch.no_grad():
                image = images[i].unsqueeze(0)
                image = image.to(device)
                image_features = model(image)
                image_features = F.normalize(image_features, dim=-1, p=2)
                if i != 0:
                    sim_consec = max(0.0, F.cosine_similarity(former_image_features, image_features).item())
                    consec_scores.append(sim_consec)
                sim_to_input = max(0.0, F.cosine_similarity(input_image_features, image_features).item())
                conformity_scores.append(sim_to_input)
                former_image_features = image_features

        video_score = max_weight * np.max(conformity_scores) + \
            mean_weight * np.mean(consec_scores) + \
            min_weight * np.min(consec_scores)

        sim_list.append(video_score)
        video_results.append({'image_path': image_path, 'video_path': video_path, 'video_results': video_score})
    return np.mean(sim_list), video_results


def compute_i2v_subject(json_dir, device, submodules_list, **kwargs):
    dino_model = torch.hub.load(**submodules_list).to(device)
    resolution = submodules_list['resolution']
    logger.info("Initialize DINO success")
    video_pair_list, _ = load_i2v_dimension_info(json_dir, dimension='i2v_subject', lang='en', resolution=resolution)
    all_results, video_results = i2v_subject(dino_model, video_pair_list, device)
    return all_results, video_results
