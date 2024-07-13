import os
import json
import numpy as np
import logging
import subprocess
import torch
import re
from pathlib import Path
from PIL import Image, ImageSequence
from decord import VideoReader, cpu
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR

CACHE_DIR = os.environ.get('VBENCH_CACHE_DIR')
if CACHE_DIR is None:
    CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'vbench')

from .distributed import (
    get_rank,
    barrier,
)

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clip_transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def clip_transform_Image(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def dino_transform(n_px):
    return Compose([
        Resize(size=n_px),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def dino_transform_Image(n_px):
    return Compose([
        Resize(size=n_px),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

def tag2text_transform(n_px):
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    return Compose([ToPILImage(),Resize((n_px, n_px)),ToTensor(),normalize])

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
    """
    Load a video from a given path and apply optional data transformations.

    The function supports loading video in GIF (.gif), PNG (.png), and MP4 (.mp4) formats.
    Depending on the format, it processes and extracts frames accordingly.
    
    Parameters:
    - video_path (str): The file path to the video or image to be loaded.
    - data_transform (callable, optional): A function that applies transformations to the video data.
    
    Returns:
    - frames (torch.Tensor): A tensor containing the video frames with shape (T, C, H, W),
      where T is the number of frames, C is the number of channels, H is the height, and W is the width.
    
    Raises:
    - NotImplementedError: If the video format is not supported.
    
    The function first determines the format of the video file by its extension.
    For GIFs, it iterates over each frame and converts them to RGB.
    For PNGs, it reads the single frame, converts it to RGB.
    For MP4s, it reads the frames using the VideoReader class and converts them to NumPy arrays.
    If a data_transform is provided, it is applied to the buffer before converting it to a tensor.
    Finally, the tensor is permuted to match the expected (T, C, H, W) format.
    """
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
    import decord
    decord.bridge.set_bridge("torch")
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
        if dimension in prompt_dict['dimension'] and 'video_list' in prompt_dict:
            prompt = prompt_dict[f'prompt_{lang}']
            cur_video_list = prompt_dict['video_list'] if isinstance(prompt_dict['video_list'], list) else [prompt_dict['video_list']]
            video_list += cur_video_list
            if 'auxiliary_info' in prompt_dict and dimension in prompt_dict['auxiliary_info']:
                prompt_dict_ls += [{'prompt': prompt, 'video_list': cur_video_list, 'auxiliary_info': prompt_dict['auxiliary_info'][dimension]}]
            else:
                prompt_dict_ls += [{'prompt': prompt, 'video_list': cur_video_list}]
    return video_list, prompt_dict_ls

def init_submodules(dimension_list, local=False, read_frame=False):
    submodules_dict = {}
    if local:
        logger.info("\x1b[32m[Local Mode]\x1b[0m Working in local mode, please make sure that the pre-trained model has been fully downloaded.")
    for dimension in dimension_list:
        os.makedirs(CACHE_DIR, exist_ok=True)
        if get_rank() > 0:
            barrier()
        if dimension == 'background_consistency':
            # read_frame = False
            if local:
                vit_b_path = f'{CACHE_DIR}/clip_model/ViT-B-32.pt'
                if not os.path.isfile(vit_b_path):
                    wget_command = ['wget', 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt', '-P', os.path.dirname(vit_b_path)]
                    subprocess.run(wget_command, check=True)
            else:
                vit_b_path = 'ViT-B/32'

            submodules_dict[dimension] = [vit_b_path, read_frame]
        elif dimension == 'human_action':
            umt_path = f'{CACHE_DIR}/umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth'
            if not os.path.isfile(umt_path):
                wget_command = ['wget', 'https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/l16_ptk710_ftk710_ftk400_f16_res224.pth', '-P', os.path.dirname(umt_path)]
                subprocess.run(wget_command, check=True)
            submodules_dict[dimension] = [umt_path,]
        elif dimension == 'temporal_flickering':
            submodules_dict[dimension] = []
        elif dimension == 'motion_smoothness':
            CUR_DIR = os.path.dirname(os.path.abspath(__file__))
            submodules_dict[dimension] = {
                    'config': f'{CUR_DIR}/third_party/amt/cfgs/AMT-S.yaml',
                    'ckpt': f'{CACHE_DIR}/amt_model/amt-s.pth'
                }
            details = submodules_dict[dimension]
            # Check if the file exists, if not, download it with wget
            if not os.path.isfile(details['ckpt']):
                print(f"File {details['ckpt']} does not exist. Downloading...")
                wget_command = ['wget', '-P', os.path.dirname(details['ckpt']),
                                'https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth']
                subprocess.run(wget_command, check=True)

        elif dimension == 'dynamic_degree':
            submodules_dict[dimension] = {
                'model': f'{CACHE_DIR}/raft_model/models/raft-things.pth'
            }
            details = submodules_dict[dimension]
            if not os.path.isfile(details['model']):
                # raise NotImplementedError
                print(f"File {details['model']} does not exist. Downloading...")
                wget_command = ['wget', '-P', f'{CACHE_DIR}/raft_model/', 'https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip']
                unzip_command = ['unzip', '-d', f'{CACHE_DIR}/raft_model/', f'{CACHE_DIR}/raft_model/models.zip']
                remove_command = ['rm', '-r', f'{CACHE_DIR}/raft_model/models.zip']
                try:
                    subprocess.run(wget_command, check=True)
                    subprocess.run(unzip_command, check=True)
                    subprocess.run(remove_command, check=True)
                except subprocess.CalledProcessError as err:
                    print(f"Error during downloading RAFT model: {err}")
        # Assign the DINO model path for subject consistency dimension
        elif dimension == 'subject_consistency':
            if local:
                submodules_dict[dimension] = {
                    'repo_or_dir': f'{CACHE_DIR}/dino_model/facebookresearch_dino_main/',
                    'path': f'{CACHE_DIR}/dino_model/dino_vitbase16_pretrain.pth', 
                    'model': 'dino_vitb16',
                    'source': 'local',
                    'read_frame': read_frame
                    }
                details = submodules_dict[dimension]
                # Check if the file exists, if not, download it with wget
                if not os.path.isdir(details['repo_or_dir']):
                    print(f"Directory {details['repo_or_dir']} does not exist. Cloning repository...")
                    subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/dino', details['repo_or_dir']], check=True)

                if not os.path.isfile(details['path']):
                    print(f"File {details['path']} does not exist. Downloading...")
                    wget_command = ['wget', '-P', os.path.dirname(details['path']),
                                    'https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth']
                    subprocess.run(wget_command, check=True)
            else:
                submodules_dict[dimension] = {
                    'repo_or_dir':'facebookresearch/dino:main',
                    'source':'github',
                    'model': 'dino_vitb16',
                    'read_frame': read_frame
                    }
        elif dimension == 'aesthetic_quality':
            aes_path = f'{CACHE_DIR}/aesthetic_model/emb_reader'
            if local:
                vit_l_path = f'{CACHE_DIR}/clip_model/ViT-L-14.pt'
                if not os.path.isfile(vit_l_path):
                    wget_command = ['wget' ,'https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt', '-P', os.path.dirname(vit_l_path)]
                    subprocess.run(wget_command, check=True)
            else:
                vit_l_path = 'ViT-L/14'
            submodules_dict[dimension] = [vit_l_path, aes_path]
        elif dimension == 'imaging_quality':
            musiq_spaq_path = f'{CACHE_DIR}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth'
            if not os.path.isfile(musiq_spaq_path):
                wget_command = ['wget', 'https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth', '-P', os.path.dirname(musiq_spaq_path)]
                subprocess.run(wget_command, check=True)
            submodules_dict[dimension] = {'model_path': musiq_spaq_path}
        elif dimension in ["object_class", "multiple_objects", "color", "spatial_relationship" ]:
            submodules_dict[dimension] = {
                "model_weight": f'{CACHE_DIR}/grit_model/grit_b_densecap_objectdet.pth'
            }
            if not os.path.exists(submodules_dict[dimension]['model_weight']):
                wget_command = ['wget', 'https://datarelease.blob.core.windows.net/grit/models/grit_b_densecap_objectdet.pth', '-P', os.path.dirname(submodules_dict[dimension]["model_weight"])]
                subprocess.run(wget_command, check=True)
        elif dimension == 'scene':
            submodules_dict[dimension] = {
                "pretrained": f'{CACHE_DIR}/caption_model/tag2text_swin_14m.pth',
                "image_size":384, 
                "vit":"swin_b"
            }
            if not os.path.exists(submodules_dict[dimension]['pretrained']):
                wget_command = ['wget', 'https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/tag2text_swin_14m.pth', '-P', os.path.dirname(submodules_dict[dimension]["pretrained"])]
                subprocess.run(wget_command, check=True)
        elif dimension == 'appearance_style':
            if local:
                submodules_dict[dimension] = {"name": f'{CACHE_DIR}/clip_model/ViT-B-32.pt'}
                if not os.path.isfile(submodules_dict[dimension]["name"]):
                    wget_command = ['wget', 'https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt', '-P', os.path.dirname(submodules_dict[dimension]["name"])]
                    subprocess.run(wget_command, check=True)
            else:
                submodules_dict[dimension] = {"name": 'ViT-B/32'}
        elif dimension in ["temporal_style", "overall_consistency"]:
            submodules_dict[dimension] = {
                "pretrain": f'{CACHE_DIR}/ViCLIP/ViClip-InternVid-10M-FLT.pth',
            }
            if not os.path.exists(submodules_dict[dimension]['pretrain']):
                wget_command = ['wget', 'https://huggingface.co/OpenGVLab/VBench_Used_Models/resolve/main/ViClip-InternVid-10M-FLT.pth', '-P', os.path.dirname(submodules_dict[dimension]["pretrain"])]
                subprocess.run(wget_command, check=True)

        if get_rank() == 0:
            barrier()
    return submodules_dict


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
