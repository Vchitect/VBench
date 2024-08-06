import torch
from tqdm import tqdm
from torchvision import transforms
from pyiqa.archs.musiq_arch import MUSIQ
from vbench.utils import load_video, load_dimension_info

from .distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)


def transform(images, preprocess_mode='shorter'):
    if preprocess_mode.startswith('shorter'):
        _, _, h, w = images.size()
        if min(h,w) > 512:
            scale = 512./min(h,w)
            images = transforms.Resize(size=( int(scale * h), int(scale * w) ))(images)
            if preprocess_mode == 'shorter_centercrop':
                images = transforms.CenterCrop(512)(images)

    elif preprocess_mode == 'longer':
        _, _, h, w = images.size()
        if max(h,w) > 512:
            scale = 512./max(h,w)
            images = transforms.Resize(size=( int(scale * h), int(scale * w) ))(images)

    elif preprocess_mode == 'None':
        return images / 255.

    else:
        raise ValueError("Please recheck imaging_quality_mode")
    return images / 255.

def technical_quality(model, video_list, device, **kwargs):
    if 'imaging_quality_preprocessing_mode' not in kwargs:
        preprocess_mode = 'longer'
    else:
        preprocess_mode = kwargs['imaging_quality_preprocessing_mode']
    video_results = []
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        images = load_video(video_path)
        images = transform(images, preprocess_mode)
        acc_score_video = 0.
        for i in range(len(images)):
            frame = images[i].unsqueeze(0).to(device)
            score = model(frame)
            acc_score_video += float(score)
        video_results.append({'video_path': video_path, 'video_results': acc_score_video/len(images)})
    average_score = sum([o['video_results'] for o in video_results]) / len(video_results)
    average_score = average_score / 100.
    return average_score, video_results


def compute_imaging_quality(json_dir, device, submodules_list, **kwargs):
    model_path = submodules_list['model_path']

    model = MUSIQ(pretrained_model_path=model_path)
    model.to(device)
    model.training = False
    
    video_list, _ = load_dimension_info(json_dir, dimension='imaging_quality', lang='en')
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = technical_quality(model, video_list, device, **kwargs)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
        all_results = all_results / 100.
    return all_results, video_results
