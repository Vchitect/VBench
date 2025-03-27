# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from vbench2.third_party.LLaVA_NeXT.llava.model.builder import load_pretrained_model
from vbench2.third_party.LLaVA_NeXT.llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from vbench2.third_party.LLaVA_NeXT.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from vbench2.third_party.LLaVA_NeXT.llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
import json
import os
import argparse
from vbench2.utils import load_dimension_info
from tqdm import tqdm

warnings.filterwarnings("ignore")

def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames,frame_time,video_time


def LLaVA_Video(prompt_dict_ls, model, tokenizer, image_processor, device):
    final_score = 0
    valid_num = 0
    processed_json=[]
    base_question=["Is there only one person in the video throughout?", "Is the person in the video the same throughout?", "Does the clothes of the person in the video (color, texture) remain consistent throughout?"]
    for prompt_dict in tqdm(prompt_dict_ls):
        question_num = len(base_question)
        video_paths = prompt_dict['video_list']
        for video_path in video_paths:
        
            max_frames_num = 64
            video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
            conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
            video=[video]
            time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Return yes or no only for the following question."
            score=0
            flag=True
            valid=True
            new_item = {
                "video_path": video_path,
            }
            for i in range(len(base_question)):
                question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n{base_question[i]}"
                conv = copy.deepcopy(conv_templates[conv_template])
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                cont = model.generate(
                        input_ids,
                        images=video,
                        modalities= ["video"],
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=4096,
                    )
            
                text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
                
                if i==0 and "yes" not in text_outputs.lower():
                    valid=False
                    break
                elif i!=0 and "yes" not in text_outputs.lower():
                    flag=False
            if not valid:
                new_item[f"video_results"]=-1
                processed_json.append(new_item)
                continue 
            if flag:
                final_score+=1
                new_item[f"video_results"]=1
            else:
                new_item[f"video_results"]=0
            valid_num+=1

            processed_json.append(new_item)
    return final_score/valid_num, processed_json
        
        
def compute_human_clothes(json_dir, device, submodules_dict, **kwargs):
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='human_clothes', lang='en')
    model_name = "llava_qwen"
    device_map = "auto"
    try:
        pretrained = submodules_dict['llava']
        llava_tokenizer, llava_model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    except:
        pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
        llava_tokenizer, llava_model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    llava_model.eval()
    
    all_results, video_results = LLaVA_Video(prompt_dict_ls, llava_model, llava_tokenizer, image_processor, device)
    score=0
    num=0
    for d in video_results:
        if d['video_results']!=-1:
            num+=1
            score+= d['video_results']
    all_results = score/num
    return all_results, video_results