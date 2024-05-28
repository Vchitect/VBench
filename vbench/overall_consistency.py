import os
import json
import numpy as np

import torch
import clip
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from vbench.utils import load_video, load_dimension_info, clip_transform, read_frames_decord_by_fps, CACHE_DIR, clip_transform_Image
from vbench.third_party.ViCLIP.viclip import ViCLIP
from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer

def get_text_features(model, input_text, tokenizer, text_feature_dict={}):
    if input_text in text_feature_dict:
        return text_feature_dict[input_text]
    text_template= f"{input_text}"
    with torch.no_grad():
        text_features = model.encode_text(text_template).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_feature_dict[input_text] = text_features
    return text_features

def get_vid_features(model, input_frames):
    with torch.no_grad():
        clip_feat = model.encode_vision(input_frames,test=True).float()
        clip_feat /= clip_feat.norm(dim=-1, keepdim=True)
    return clip_feat

def get_predict_label(clip_feature, text_feats_tensor, top=5):
    label_probs = (100.0 * clip_feature @ text_feats_tensor.T).softmax(dim=-1)
    top_probs, top_labels = label_probs.cpu().topk(top, dim=-1)
    return top_probs, top_labels

def overall_consistency(clip_model, video_dict, tokenizer, device, sample="middle"):
    sim = []
    video_results = []
    image_transform = clip_transform(224)
    for info in tqdm(video_dict):
        query = info['prompt']
        text = clip.tokenize([query]).to(device)
        video_list = info['video_list']
        for video_path in video_list:
            cur_video = []
            with torch.no_grad():
                images = read_frames_decord_by_fps(video_path, num_frames=8, sample=sample)
                images = image_transform(images)
                images = images.to(device)
                clip_feat = get_vid_features(clip_model,images.unsqueeze(0))
                text_feat = get_text_features(clip_model, query, tokenizer)
                logit_per_text =  clip_feat @ text_feat.T
                score_per_video =  float(logit_per_text[0][0].cpu())
                sim.append(score_per_video)
                video_results.append({'video_path': video_path, 'video_results': score_per_video})
    avg_score = np.mean(sim)
    return avg_score, video_results


def overall_consistency_t2i(clip_model, image_dict, device):
    sim = []
    video_results = []
    image_transform = clip_transform_Image(224)
    for info in tqdm(image_dict):
        query = info['prompt']
        text = clip.tokenize([query]).to(device)
        video_list = info['video_list']
        for video_path in video_list:
            with torch.no_grad():
                image = load_video(video_path, return_tensor=False)[0]
                image = Image.fromarray(image)
                image = image_transform(image)
                image = image.to(device)
                text = clip.tokenize([query]).to(device)
                _ , logits_per_text = clip_model(image.unsqueeze(0), text)
                score_per_video = float(logits_per_text[0][0].cpu())
                score_per_video = score_per_video / 100
                sim.append(score_per_video)
                video_results.append({'video_path': video_path, 'video_results': score_per_video})

    return score_per_video, video_results

def compute_overall_consistency(json_dir, device, submodules_list, **kwargs):
    _, video_dict = load_dimension_info(json_dir, dimension='overall_consistency', lang='en')

    if Path(video_dict[0]['video_list'][0]).suffix.lower() in ['.png', '.jpg']:
        clip_img, _ = clip.load(device=device, name=submodules_list['name'])
        clip_img = clip_img.to(device)
        return overall_consistency_t2i(clip_img, video_dict, device)
    else:
        tokenizer = SimpleTokenizer(os.path.join(CACHE_DIR, "ViCLIP/bpe_simple_vocab_16e6.txt.gz"))
        viclip = ViCLIP(tokenizer= tokenizer).to(device)
        return overall_consistency(viclip, video_dict, tokenizer, device)
