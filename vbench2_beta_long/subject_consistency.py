import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
from dreamsim import dreamsim
from decord import VideoReader
from torchvision.io import write_video
from vbench.subject_consistency import compute_subject_consistency, subject_consistency
from vbench.utils import load_video, load_dimension_info, dino_transform, dino_transform_Image
from vbench2_beta_long.utils import reorganize_clips_results, create_video_from_first_frames, fuse_inclip_clip2clip
from vbench2_beta_long.utils import dreamsim_transform, dreamsim_transform_Image, dinov2_transform, dinov2_transform_Image
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compute_long_subject_consistency(json_dir, device, submodules_list, **kwargs):
    # compute inclip scores 
    all_results, detailed_results = compute_subject_consistency(json_dir, device, submodules_list)

    inclip_all_results, inclip_detailed_results, inclip_average_scores = reorganize_clips_results(detailed_results)
    inclip_all_results = all_results


    # compute clip2clip scores
    # sample first frames in each clip, and cat them into a new video
    base_path_video = os.path.dirname(list(detailed_results[0].values())[0]).split("split_clip")[0]
    long_video_path = os.path.join(base_path_video, "split_clip")
    new_cat_video_path = os.path.join(base_path_video, 'subject_consistency_cat_firstframes_videos')
    if not os.path.exists(new_cat_video_path):
        os.makedirs(new_cat_video_path, exist_ok=True)
        create_video_from_first_frames(long_video_path, new_cat_video_path, detailed_results)
    else:
        print(f"{new_cat_video_path} has already been created, please check the path")

    # get the new video_list
    video_list = []
    for video_path in os.listdir(new_cat_video_path):
        video_list.append(os.path.join(new_cat_video_path, video_path))
        
    def _compute_subject_consistency(video_list, device, submodules_list, **kwargs):
        if kwargs['sb_clip2clip_feat_extractor'] == 'dino':
            dino_model = torch.hub.load(**submodules_list).to(device)
            read_frame = submodules_list['read_frame']
            logger.info("Initialize DINO success")
            all_results, video_results = subject_consistency(dino_model, video_list, device, read_frame)
        elif kwargs['sb_clip2clip_feat_extractor'] == 'dinov2':
            dinov2_dict = {
                'repo_or_dir': f'facebookresearch/dinov2',
                'model': 'dinov2_vitb14',
                }
            dinov2_model = torch.hub.load(**dinov2_dict).to(device)
            read_frame = submodules_list['read_frame']
            logger.info("Initialize DINOv2 success")
            all_results, video_results = subject_consistency_dinov2(dinov2_model, video_list, device, read_frame)

        elif kwargs['sb_clip2clip_feat_extractor'] == 'dreamsim':
            read_frame = submodules_list['read_frame']
            cache_dir = os.path.expanduser("~/.cache")
            dreamsim_model, _ = dreamsim(pretrained=True, cache_dir=cache_dir)
            all_results, video_results = subject_consistency_dreamsim(dreamsim_model, video_list, device, read_frame)
        return all_results, video_results




    clip2clip_all_results, clip2clip_detailed_results = _compute_subject_consistency(video_list, device, submodules_list, **kwargs)
    dimension = 'subject_consistency'
    fused_all_results, fused_detailed_results = fuse_inclip_clip2clip(inclip_all_results, clip2clip_all_results, inclip_average_scores, clip2clip_detailed_results, dimension, **kwargs)
    # fused_all_results = inclip_all_results * kwargs['w_inclip'] + clip2clip_all_results * kwargs['w_clip2clip']
    return fused_all_results, fused_detailed_results



def subject_consistency_dreamsim(model, video_list, device, read_frame):
    sim = 0.0
    cnt = 0
    video_results = []
    if read_frame:
        image_transform = dreamsim_transform_Image(224)
    else:
        image_transform = dreamsim_transform(224)
    # image_transform = preprocess if read_frame else dino_transform(224)
        
    for video_path in tqdm(video_list):
        video_sim = 0.0
        if read_frame:
            video_path = video_path[:-4].replace('videos', 'frames').replace(' ', '_')
            tmp_paths = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path))]
            images = []
            for tmp_path in tmp_paths:
                images.append(image_transform(Image.open(tmp_path)))
            images = torch.stack(images)
        else:
            images = load_video(video_path)
            images = image_transform(images)
            

        images = images.to(device)

        for i in range(len(images)):
            with torch.no_grad():
                image = images[i].unsqueeze(0)
                image_features = model.embed(image)
                image_features = F.normalize(image_features, dim=-1, p=2)

                if i == 0:
                    first_image_features = image_features
                else:
                    sim_pre = max(0.0, F.cosine_similarity(former_image_features, image_features).item())
                    sim_fir = max(0.0, F.cosine_similarity(first_image_features, image_features).item())
                    cur_sim = (sim_pre + sim_fir) / 2
                    video_sim += cur_sim
                    cnt += 1

            former_image_features = image_features
        sim_per_images = video_sim / (len(images) - 1)
        sim += video_sim
        video_results.append({'video_path': video_path, 'video_results': sim_per_images})
    # sim_per_video = sim / (len(video_list) - 1)
    sim_per_frame = sim / cnt if cnt != 0 else None
    return sim_per_frame, video_results


def subject_consistency_dinov2(model, video_list, device, read_frame):
    sim = 0.0
    cnt = 0
    video_results = []
    if read_frame:
        image_transform = dinov2_transform_Image(224)
    else:
        image_transform = dinov2_transform(224)
    for video_path in tqdm(video_list):
        video_sim = 0.0
        if read_frame:
            video_path = video_path[:-4].replace('videos', 'frames').replace(' ', '_')
            tmp_paths = [os.path.join(video_path, f) for f in sorted(os.listdir(video_path))]
            images = []
            for tmp_path in tmp_paths:
                images.append(image_transform(Image.open(tmp_path)))
        else:
            images = load_video(video_path)
            images = image_transform(images)
        for i in range(len(images)):
            with torch.no_grad():
                image = images[i].unsqueeze(0)
                image = image.to(device)
                image_features = model(image)
                image_features = F.normalize(image_features, dim=-1, p=2)
                if i == 0:
                    first_image_features = image_features
                else:
                    sim_pre = max(0.0, F.cosine_similarity(former_image_features, image_features).item())
                    sim_fir = max(0.0, F.cosine_similarity(first_image_features, image_features).item())
                    cur_sim = (sim_pre + sim_fir) / 2
                    video_sim += cur_sim
                    cnt += 1
            former_image_features = image_features
        sim_per_images = video_sim / (len(images) - 1)
        sim += video_sim
        video_results.append({'video_path': video_path, 'video_results': sim_per_images})
    # sim_per_video = sim / (len(video_list) - 1)
    sim_per_frame = sim / cnt if cnt != 0 else None
    return sim_per_frame, video_results