import torch
from tqdm import tqdm
from torchvision import transforms
from pyiqa.archs.musiq_arch import MUSIQ
from vbench.utils import load_video, load_dimension_info

def transform(images):
    _, _, h, w = images.size()
    if min(h,w) > 512:
        scale = 512./min(h,w)
        images = transforms.Resize(size=( int(scale * h), int(scale * w) ))(images)
    return images / 255.

def technical_quality(model, video_list, device):
    video_results = []
    for video_path in tqdm(video_list):
        images = load_video(video_path)
        images = transform(images)
        acc_score_video = 0.
        for i in range(len(images)):
            frame = images[i].unsqueeze(0).to(device)
            score = model(frame)
            acc_score_video += float(score)
        video_results.append({'video_path': video_path, 'video_results': acc_score_video/len(images)})
    average_score = sum([o['video_results'] for o in video_results]) / len(video_results)
    average_score = average_score / 100.
    return average_score, video_results


def compute_imaging_quality(json_dir, device, submodules_list):
    model_path = submodules_list['model_path']

    model = MUSIQ(pretrained_model_path=model_path)
    model.to(device)
    model.training = False
    
    video_list, _ = load_dimension_info(json_dir, dimension='imaging_quality', lang='en')
    all_results, video_results = technical_quality(model, video_list, device)
    return all_results, video_results
