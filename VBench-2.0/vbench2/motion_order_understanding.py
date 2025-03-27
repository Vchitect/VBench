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

sys_prompt = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant and a brilliant action order judger. 
You need to judge the consistency of the action in two given prompts, you will only focus on the action, similar semantic should be compromised.
Do not make associations, the action should actually have happened, and do not judge the consistency by the subject, for example, "running in the beach" is consistent with "the horse is running in the beach"; "a horse move the head left and then back to right" is consistent with "the horse moves his mane" as the phenomenon is same.
Note that active and passive expressions can be considered consistent. For example, "a horse shake manes" is consistency with "The horse's mane and tail start to move, indicating a gentle breeze or wind." 
Answer yes or no first, then giving the reason.
"""
template_prompt1 = """
prompt1: "sitting"
prompt2: 1. The person is lying on a beige sofa, holding a broom.
"""
template_prompt2 = """
prompt1: playing with a ball
prompt2: 1. The dog is lying on a beige carpet, holding a red ball in its mouth.
"""
template_prompt3 = """
prompt1: organizing the pantry
prompt2: 2. The person moves away from the stove and walks towards the counter.
"""
template_prompt4 = """
prompt1: drinking a glass of water
prompt2: A person is holding a glass of water in their hand
"""

def judge(prompt, model, tokenizer):
    messages = [
        {
            "role": "system", 
            "content": sys_prompt
        },
        {
            "role": "user",
            "content": template_prompt1,
        },
        {
            "role": "assistant",
            "content": 'Yes. "sitting" is similar with lying',
        },
        {
            "role": "user",
            "content": template_prompt2,
        },
        {
            "role": "assistant",
            "content": 'Yes. In prompt1, "holding a ball in its mouth" in prompt2 is "playing with a ball" in prompt for a dog',
        },
        {
            "role": "user",
            "content": template_prompt3,
        },
        {
            "role": "assistant",
            "content": 'No. prompt2 is not consistency with "organizing the pantry" in prompt1.',
        },
        {
            "role": "user",
            "content": template_prompt4,
        },
        {
            "role": "assistant",
            "content": 'No. "holding a glass of water" is not consistency with "drink water" in prompt1.',
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
        
        for video_path in video_paths:
            new_item = {
                "video_path": video_path,
            }
            max_frames_num = 64
            video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
            video = [video]
            conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
            time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Return the action order in video. Here is the template."
            template = "1. ; 2. ."
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
            flag=False
            try:
                prompts=answer_llava.split('2.')
                if len(prompts)<2:
                    flag=True
            except:
                flag=True
            if flag:
                new_item['video_results']=-1
                processed_json.append(new_item)
                continue
            score=0
            for q in range(2):
                prompt1 = ground_truth_text[q]
                prompt2 = prompts[q]
                prompt = f"""
                    prompt1: {prompt1}
                    prompt2: {prompt2}
                """
                response = judge(prompt, qwen_model, qwen_tokenizer)
                if 'yes' in response.lower():
                    score+=1
            if score==2:
                final_score+=1
                new_item['video_results']=1
            else:
                new_item['video_results']=0
            valid_num+=1
            processed_json.append(new_item)
        
    return final_score/valid_num, processed_json
        
        
def compute_motion_order_understanding(json_dir, device, submodules_dict, **kwargs):
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='motion_order_understanding', lang='en')
    
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
    score=0
    num=0
    count=0
    for d in video_results:
        if d['video_results']!=-1:
            num+=1
            score+= d['video_results']
        else:
            count+=1
    if count>0.9*len(video_results):
        return 0, video_results 
    all_results = score/num
    return all_results, video_results