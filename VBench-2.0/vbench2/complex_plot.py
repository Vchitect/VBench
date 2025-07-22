# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from tqdm import tqdm
import requests
import copy
import torch
import sys
import warnings
import re
from decord import VideoReader, cpu
import numpy as np
import json
import os
import argparse
from vbench2.utils import load_dimension_info
from tqdm import tqdm
warnings.filterwarnings("ignore")

sys_prompt_sum = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant and a brilliant plot summarizer. 
You need to summerize and divide the detailed <video_caption> into several key plots up to the given <template>, each plot should be a complete sentence.
"""

sys_prompt = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant and a brilliant plot consistency judger. 
You need to judge whether prompt2 contains the key elements described in prompt1 , similar semantic should be compromised.
Note that you should not care about the detailed name in prompt when judging the consistency, it is trival.
Simple reasoning is permitted, such as 'A girl is in the forest and a ghost is in the side' is consistent with 'The girl's father was captured by a monster, and the girl ventures into the forest to find the monster and rescue her father.'.
Note that you should only focus on the person or creatures mentioned in prompt1 when assessing the prompt2, unrelated contents should not be judged and considered, such background and the clothes details.
Be sensitive to the consistency, "the runner from one team accelerated in the second-to-last turn, widening the gap and winning the race" is not consistent with "running" only. "holding the swords" only is not consistent with "holding the swords to fight the enermy".
First return yes or no, then giving the reason.
"""

def judge(prompt, sysp, tokenizer, model):
    messages = [
        {
            "role": "system", 
            "content": sysp
        },
        {
            "role": "user", 
            "content": prompt
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        do_sample=False,
        temperature=0,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def split_by_numbered_list(text):
    pattern = r'\d+\.\s*'  # 匹配形如 "1. " 的编号
    parts = re.split(pattern, text)  # 按编号分割
    parts = [part.strip() for part in parts if part.strip()]  # 去除空字符串和多余空白
    return parts

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

def LLaVA_Video(prompt_dict_ls, llava_model, llava_tokenizer, image_processor, qwen_model, qwen_tokenizer, device):
    final_score = 0
    valid_num = 0
    processed_json=[]
    for prompt_dict in tqdm(prompt_dict_ls):
        video_paths = prompt_dict['video_list']
        ground_truth_text = prompt_dict['auxiliary_info']
        length = len(ground_truth_text)
        
        for video_path in video_paths:
            new_item = {
                "video_path": video_path,
            }
            
            max_frames_num = 64
            video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(device).bfloat16()
            video = [video]
            conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
            time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Return the plot in video. Here is the template.."
            if length==5:
                template = "1. ; 2. ; 3. ; 4. ; 5. ."
            elif length==4:
                template = "1. ; 2. ; 3. ; 4. ."
            question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n"+f"{template}"
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt_question, llava_tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            cont = llava_model.generate(
                input_ids,
                images=video,
                modalities= ["video"],
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
            )
            answer_llava = llava_tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
            score=0
            if '1. ' not in answer_llava:
                for q, item in enumerate(ground_truth_text):
                    prompt1 = item.strip()
                    prompt = f"""
                        prompt1: {prompt1}
                        prompt2: {answer_llava}
                        """
                    response = judge(prompt, sys_prompt, qwen_tokenizer, qwen_model)
                    if 'yes' in response.lower():
                        score+=1
                    else:
                        break
            else:
                prompt_list = split_by_numbered_list(answer_llava)
                if len(prompt_list) > length:
                    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Describe the video in detail."
                    question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n"
                    conv = copy.deepcopy(conv_templates[conv_template])
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    input_ids = tokenizer_image_token(prompt_question, llava_tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                    cont = llava_model.generate(
                        input_ids,
                        images=video,
                        modalities= ["video"],
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=4096,
                    )
                    answer_llava = llava_tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
                    prompt = f"""
                        video_caption: {answer_llava}
                        template: {template}
                        """
                    response = judge(prompt, sys_prompt_sum, qwen_tokenizer, qwen_model)
                    prompt_list = split_by_numbered_list(response)
                for q, item in enumerate(prompt_list):
                    prompt1 = ground_truth_text[q]
                    prompt2 = item.strip()
                    prompt = f"""
                        prompt1: {prompt1}
                        prompt2: {prompt2}
                        """
                    response = judge(prompt, sys_prompt, qwen_tokenizer, qwen_model)
                    if 'yes' in response.lower():
                        score+=1
                    else:
                        break
            valid_num+=1
            new_item["video_results"] = score/len(ground_truth_text)
            final_score+=score/len(ground_truth_text)
            processed_json.append(new_item)
        
    return final_score/valid_num, processed_json
        
        
def compute_complex_plot(json_dir, device, submodules_dict, **kwargs):
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='complex_plot', lang='en')
    
    model_name = "llava_qwen"
    device_map = "auto"
    try:
        pretrained = submodules_dict['llava']
        llava_tokenizer, llava_model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    except:
        pretrained = "lmms-lab/LLaVA-Video-7B-Qwen2"
        llava_tokenizer, llava_model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    llava_model.eval()
    
    try:
        qwen_model_name = submodules_dict['qwen']
        qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=submodules_dict['qwen']
        )
        qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, cache_dir=submodules_dict['qwen'])
    except:
        qwen_model_name = 'Qwen/Qwen2.5-7B-Instruct'
        qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=submodules_dict['qwen']
        )
        qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name, cache_dir=submodules_dict['qwen'])
        
    all_results, video_results = LLaVA_Video(prompt_dict_ls, llava_model, llava_tokenizer, image_processor, qwen_model, qwen_tokenizer, device)
    all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results