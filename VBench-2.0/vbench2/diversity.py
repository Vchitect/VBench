import torch
from torchvision.models import vgg19
from torch import nn
from torchvision import transforms
from collections import OrderedDict
import cv2
from vbench2.utils import get_frames
import numpy as np
import os
import json
from vbench2.utils import load_dimension_info
from tqdm import tqdm


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = vgg19(pretrained=True).features.eval()

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in {0, 5, 10, 19, 28, 30}:
                features.append(x.detach().cpu())
        return features

def gram_matrix(tensor):
    batch_size, channels, height, width = tensor.shape
    features = tensor.view(batch_size, channels, -1)
    gram = torch.bmm(features, features.transpose(1,2))  
    gram = gram / (channels * height * width)
    return gram

def content_loss(content, target_content):
    return torch.mean(torch.abs(content - target_content))

def style_loss(style, target_style):
    gram_style = gram_matrix(style)
    gram_target_style = gram_matrix(target_style)
    return torch.mean(torch.abs(gram_style - gram_target_style))

def evaluate(style_features, content_features):
    content_diversity = 0
    style_diversity = 0
    len_seed = len(content_features)
    for i in range(len_seed):
        for j in range(i+1, len_seed):
            content_diversity += content_loss(content_features[i], content_features[j])
            for k in range(5):
                style_diversity += style_loss(style_features[i][k], style_features[j][k])
    content_diversity/=(0.5*len_seed*(len_seed-1))
    style_diversity/=(2.5*len_seed*(len_seed-1))
    diversity=(content_diversity+1000*style_diversity)/2
    return content_diversity, 1000*style_diversity, diversity / 17.712 # Empirical maximum

def Diversity(prompt_dict_ls, model, device):
    final_score=0
    processed_json=[]
    for prompt_dict in tqdm(prompt_dict_ls):
        video_paths = prompt_dict['video_list']
        style_features=[]
        content_features=[]
        for video_path in video_paths:
            frames=get_frames(video_path)
            frames=torch.cat(frames, dim=0)
            frames=frames.to(device)
            with torch.no_grad():
                features = model(frames)
            style=features[:5] 
            content=features[5]
            style_features.append(style)
            content_features.append(content)
            del style, content, frames
            torch.cuda.empty_cache()

        content_diversity, style_diversity, diversity=evaluate(style_features, content_features)
        diversity = torch.clamp(diversity, min=0, max=1)
        new_item={
                'video_path':video_paths[0],
                'video_results':diversity.tolist()
            }
        processed_json.append(new_item)
        final_score+=diversity
    return final_score/len(prompt_dict_ls), processed_json

def compute_diversity(json_dir, device, submodules_dict, **kwargs):
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='diversity', lang='en')
    model = VGG().to(device)
    
    all_results, video_results = Diversity(prompt_dict_ls, model, device)
    all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results