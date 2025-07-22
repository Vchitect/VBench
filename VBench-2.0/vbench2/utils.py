import io
import os
import re
import yaml
import cv2
import json
import random
import numpy as np
import subprocess
import logging

from PIL import Image, ImageSequence
from tqdm import tqdm
from pathlib import Path
from bisect import bisect_left
import gdown

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.io import write_video
from decord import VideoReader, cpu
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from collections import defaultdict
from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR

CACHE_DIR = os.environ.get('VBENCH2_CACHE_DIR')
if CACHE_DIR is None:
    CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'vbench2')
from .distributed import (
    get_rank,
    barrier,
)

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def split_video_into_scenes(video_path, threshold=27.0):
    # Open our video, create a scene manager, and add a detector.
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()
    return scene_list


def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices

def load_video(video_path, data_transform=None, num_frames=None, return_tensor=True, width=None, height=None):
    if video_path.endswith('.gif'):
        frame_ls = []
        img = Image.open(video_path)
        for frame in ImageSequence.Iterator(img):
            frame = frame.convert('RGB')
            frame = np.array(frame).astype(np.uint8)
            frame_ls.append(frame)
        buffer = np.array(frame_ls).astype(np.uint8)
    elif video_path.endswith('.png'):
        frame = Image.open(video_path)
        frame = frame.convert('RGB')
        frame = np.array(frame).astype(np.uint8)
        frame_ls = [frame]
        buffer = np.array(frame_ls)
    elif video_path.endswith('.mp4'):
        import decord
        decord.bridge.set_bridge('native')
        if width:
            video_reader = VideoReader(video_path, width=width, height=height, num_threads=1)
        else:
            video_reader = VideoReader(video_path, num_threads=1)
        frame_indices = range(len(video_reader))
        if num_frames:
            frame_indices = get_frame_indices(
            num_frames, len(video_reader), sample="middle"
            )
        frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
        buffer = frames.asnumpy().astype(np.uint8)
    else:
        raise NotImplementedError
    
    frames = buffer
    if num_frames and not video_path.endswith('.mp4'):
        frame_indices = get_frame_indices(
        num_frames, len(frames), sample="middle"
        )
        frames = frames[frame_indices]
    
    if data_transform:
        frames = data_transform(frames)
    elif return_tensor:
        frames = torch.Tensor(frames)
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8

    return frames

def read_frames_decord_by_fps(
        video_path, sample_fps=2, sample='rand', fix_start=None, 
        max_num_frames=-1,  trimmed30=False, num_frames=8
    ):
    video_reader = VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    if trimmed30 and duration > 30:
        duration = 30
        vlen = int(30 * float(fps))
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )
    frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames

def clip_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def get_frames(video_path):
    frame_list = []
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS) # get fps
    interval = 1
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    while video.isOpened():
        success, frame = video.read()
        if success:
            image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)  # convert to rgb
            image_pil = Image.fromarray(image_np)
            frame = preprocess(image_pil)
            frame = frame[None]
            frame_list.append(frame)
        else:
            break
    video.release()
    assert frame_list != []
    frame_list = extract_frame(frame_list, interval)
    return frame_list 
    
    
def extract_frame(frame_list, interval=1):
    extract = []
    for i in range(0, len(frame_list), interval):
        extract.append(frame_list[i])
    return extract

    
def load_dimension_info(json_dir, dimension, lang):
    """
    Load video list and prompt information based on a specified dimension and language from a JSON file.
    
    Parameters:
    - json_dir (str): The directory path where the JSON file is located.
    - dimension (str): The dimension for evaluation to filter the video prompts.
    - lang (str): The language key used to retrieve the appropriate prompt text.
    
    Returns:
    - video_list (list): A list of video file paths that match the specified dimension.
    - prompt_dict_ls (list): A list of dictionaries, each containing a prompt and its corresponding video list.
    
    The function reads the JSON file to extract video information. It filters the prompts based on the specified
    dimension and compiles a list of video paths and associated prompts in the specified language.
    
    Notes:
    - The JSON file is expected to contain a list of dictionaries with keys 'dimension', 'video_list', and language-based prompts.
    - The function assumes that the 'video_list' key in the JSON can either be a list or a single string value.
    """
    video_list = []
    prompt_dict_ls = []
    full_prompt_list = load_json(json_dir)
    for prompt_dict in full_prompt_list:
        if dimension in prompt_dict['dimension'][0].lower() and 'video_list' in prompt_dict:
            prompt = prompt_dict[f'prompt_{lang}']
            cur_video_list = prompt_dict['video_list'] if isinstance(prompt_dict['video_list'], list) else [prompt_dict['video_list']]
            video_list += cur_video_list
            if 'auxiliary_info' in prompt_dict:
                prompt_dict_ls += [{'prompt': prompt, 'video_list': cur_video_list, 'auxiliary_info': prompt_dict['auxiliary_info']}]
            else:
                prompt_dict_ls += [{'prompt': prompt, 'video_list': cur_video_list}]
    return video_list, prompt_dict_ls

def clone_model(repo_url, target_dir):
    git_clone_command = [
        'git', 'clone', '--recursive',
        repo_url,
        target_dir
    ]
    git_lfs_pull_command = ['git', '-C', target_dir, 'lfs', 'pull']
    try:
        print("Cloning repository...")
        subprocess.run(git_clone_command, check=True)
        print("Pulling LFS files...")
        original_dir = os.getcwd() 
        os.chdir(target_dir)         
        subprocess.run(['git', 'lfs', 'pull'], check=True)
        os.chdir(original_dir)     
        print("Model downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")


def hug_model(model_name, local_dir):
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"File exists: {local_dir}")
        return local_dir
    print(f"File {local_dir} does not exist. Downloading...")
    os.makedirs(local_dir, exist_ok=True)
    download_command = [
        "huggingface-cli", "download",
        model_name,
        "--repo-type", "model",
        "--local-dir", local_dir
    ]
    result = subprocess.run(download_command, capture_output=True, text=True)
    if result.returncode == 0:
        print("Model downloaded successfully!")
    else:
        print("Error occur:", result.stderr)

def google_drive(model, file_id, output_path):
    file = f"{CACHE_DIR}/{model}"
    url = f"https://drive.google.com/uc?id={file_id}"
    os.makedirs(file, exist_ok=True)
    try:
        gdown.download(url, output_path, quiet=False)
        print(f"Model downloaded successfully to: {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
def init_submodules(dimension_list, local=False, read_frame=False):
    submodules_dict = {}
    if local:
        logger.info("\x1b[32m[Local Mode]\x1b[0m Working in local mode, please make sure that the pre-trained model has been fully downloaded.")
    for dimension in dimension_list:
        os.makedirs(CACHE_DIR, exist_ok=True)

        if dimension == 'Multi-View_Consistency':
            submodules_dict[dimension] = {
                'raft': f'{CACHE_DIR}/raft_model/models/raft-things.pth',
                "repo":"facebookresearch/co-tracker",
                "model":"cotracker2"
            }
            details = submodules_dict[dimension]
            if not os.path.isfile(details['raft']):
                print(f"File {details['raft']} does not exist. Downloading...")
                wget_command = ['wget', '-P', f'{CACHE_DIR}/raft_model/', 'https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip']
                unzip_command = ['unzip', '-d', f'{CACHE_DIR}/raft_model/', f'{CACHE_DIR}/raft_model/models.zip']
                remove_command = ['rm', '-r', f'{CACHE_DIR}/raft_model/models.zip']
                try:
                    subprocess.run(wget_command, check=True)
                    subprocess.run(unzip_command, check=True)
                    subprocess.run(remove_command, check=True)
                except subprocess.CalledProcessError as err:
                    print(f"Error during downloading RAFT model: {err}")
                
        elif dimension == 'Camera_Motion':
            submodules_dict[dimension] = {
                "repo":"facebookresearch/co-tracker",
                "model":"cotracker2"
            }
            
        elif dimension == 'Human_Identity':
            submodules_dict[dimension] = {
                "model":f'{CACHE_DIR}/arcface/resnet18_110.pth'
            }
            details = submodules_dict[dimension]
            if not os.path.isfile(details['model']):
                print(f"File {details['model']} does not exist. Downloading...")
                file_id = "1m387vGTQ4GW4I4PQfBsCC-Y9aEp6zjvy"
                url = f"https://drive.google.com/uc?id={file_id}&export=download"
                os.makedirs(f'{CACHE_DIR}/arcface', exist_ok=True)
                wget_command = ['wget', '-O', details['model'], url]
                try:
                    subprocess.run(wget_command, check=True)
                    print("Model downloaded successfully!")
                except subprocess.CalledProcessError as e:
                    print(f"An error occurred: {e}")
        
        elif dimension == 'Instance_Preservation':
            submodules_dict[dimension] = {
                "model":f'{CACHE_DIR}/instance_anomaly_detector/model'
            }
            details = submodules_dict[dimension]
            if not os.path.isdir(details['model']):
                print(f"File {details['model']} does not exist. Downloading...")
                
                file = f"{CACHE_DIR}/instance_anomaly_detector"
                url = 'https://drive.google.com/drive/folders/106rnzZvH-VUKkz8dMPFflD6tqtbSgXwh?usp=sharing'
                os.makedirs(file, exist_ok=True)
                output_path = details['model']
                try:
                    gdown.download_folder(url=url, output=output_path, quiet=False, use_cookies=False)
                    print(f"Model downloaded successfully to: {output_path}")
                except Exception as e:
                    print(f"An error occurred: {e}")
        
        elif dimension == 'Human_Anatomy':
            default_config = 'vbench2/third_party/ViTDetector/simmim_finetune__vit_base__img224__800ep.yaml'
            submodules_dict[dimension] = {
                "detector_config": "vbench2/third_party/YOLO-World/yolo_world_v2_xl_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py",
                "detector_weights":f'{CACHE_DIR}/YOLO-World/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth',
                "analyzer_configs": {
                    "human": {"cfg_path": default_config, "weight_path": f"{CACHE_DIR}/anomaly_detector/human.pth", "threshold": 0.4545454545454546},
                    "face": {"cfg_path": default_config, "weight_path": f"{CACHE_DIR}/anomaly_detector/face.pth", "threshold": 0.30303030303030304},
                    "hand": {"cfg_path": default_config, "weight_path": f"{CACHE_DIR}/anomaly_detector/hand.pth", "threshold": 0.3232}
                },
                "batch_size" : 128
            }
            details = submodules_dict[dimension]
            if not os.path.isfile(details['detector_weights']):
                print(f"File {details['detector_weights']} does not exist. Downloading...")
                google_drive(model="YOLO-World", file_id="1qo-K1kum7yiEwIlN1TWDvABXX6qriUen", output_path=details['detector_weights'])
                
            if not os.path.isfile(details['analyzer_configs']['human']['weight_path']):
                print(f"File {details['analyzer_configs']['human']['weight_path']} does not exist. Downloading...")
                google_drive(model="anomaly_detector", file_id="1mlSURYOi_vN9ST1wzEhJaVO6-ZoES8UT", output_path=details['analyzer_configs']['human']['weight_path'])
                
            if not os.path.isfile(details['analyzer_configs']['face']['weight_path']):
                print(f"File {details['analyzer_configs']['face']['weight_path']} does not exist. Downloading...")
                google_drive(model="anomaly_detector", file_id="1e2qTjrtsYlkWLql0qj8DqaNZ09KuV8Qo", output_path=details['analyzer_configs']['face']['weight_path'])
                
            if not os.path.isfile(details['analyzer_configs']['hand']['weight_path']):
                print(f"File {details['analyzer_configs']['hand']['weight_path']} does not exist. Downloading...")
                google_drive(model="anomaly_detector", file_id="1j3QeAcAtdLe5BFgK-c33UaHV6-iUto0i", output_path=details['analyzer_configs']['hand']['weight_path'])
                    
        elif dimension in ["Human_Clothes", "Composition", "Dynamic_Spatial_Relationship", "Dynamic_Attribute", "Motion_Rationality", "Mechanics", "Thermotics", "Material"]:
            submodules_dict[dimension] = {"llava": f'{CACHE_DIR}/lmms-lab/LLaVA-Video-7B-Qwen2'}
            hug_model(
                model_name='lmms-lab/LLaVA-Video-7B-Qwen2',
                local_dir=submodules_dict[dimension]['llava']
            )
                
        elif dimension in ["Complex_Landscape", "Complex_Plot", "Human_Interaction", "Motion_Order_Understanding"]:
            submodules_dict[dimension] = {
                "llava": f'{CACHE_DIR}/lmms-lab/LLaVA-Video-7B-Qwen2',
                "qwen": f'{CACHE_DIR}/Qwen/Qwen2.5-7B-Instruct'
            }
            hug_model(
                model_name='lmms-lab/LLaVA-Video-7B-Qwen2',
                local_dir=submodules_dict[dimension]['llava']
            )
            hug_model(
                model_name='Qwen/Qwen2.5-7B-Instruct',
                local_dir=submodules_dict[dimension]['qwen']
            )
        else:
            submodules_dict[dimension]={}

    return submodules_dict

def save_json(data, path, indent=4):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)

def load_json(path):
    """
    Load a JSON file from the given file path.
    
    Parameters:
    - file_path (str): The path to the JSON file.
    
    Returns:
    - data (dict or list): The data loaded from the JSON file, which could be a dictionary or a list.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_prompt_from_filename(path: str):
    """
    1. prompt-0.suffix -> prompt
    2. prompt.suffix -> prompt
    """
    prompt = Path(path).stem
    number_ending = r'-\d+$' # checks ending with -<number>
    if re.search(number_ending, prompt):
        return re.sub(number_ending, '', prompt)
    return prompt