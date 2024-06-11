import numpy as np
from tqdm import tqdm
import clip

import torch
import torch.nn.functional as F

from vbench2_beta_long.utils import reorganize_clips_results
from vbench.utils import load_dimension_info, clip_transform, read_frames_decord_by_fps
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clip_alignment(clip_model, video_dict, preprocess, device):
    sim = []
    video_results = []
    
    image_transform = clip_transform(224)
    for info in tqdm(video_dict):
        
        query = info["prompt"]
        text = clip.tokenize([query], truncate=True).to(device)
        text_feature = clip_model.encode_text(text)
        text_feature = F.normalize(text_feature, dim=-1)
        
        video_list = info["video_list"]
        for video_path in video_list:
            with torch.no_grad():
                images = read_frames_decord_by_fps(video_path, num_frames=8, sample="middle")
                images = image_transform(images)
                images = images.to(device)
                
                image_features = clip_model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1, p=2)

                video_sim = image_features @ text_feature.T
                video_sim = np.mean(video_sim.cpu().tolist())
                sim.append(video_sim)

                video_results.append({'video_path': video_path, 'video_results': video_sim})
    
    avg_sim = np.mean(sim)
    
    return avg_sim, video_results


def compute_clip_score(json_dir, device, submodules_list, **kwargs):
    
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    logger.info("Initialize CLIP success")
    
    _, video_dict = load_dimension_info(json_dir, dimension='clip_score', lang='en')
    all_results, video_results = clip_alignment(clip_model, video_dict, preprocess, device)
    return all_results, video_results


def compute_long_clip_score(json_dir, device, submodules_list, **kwargs):
    all_results, detailed_results = compute_clip_score(json_dir, device, submodules_list, **kwargs)

    return reorganize_clips_results(detailed_results)
