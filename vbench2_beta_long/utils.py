import io
import os
import re
import yaml
import cv2
import json
import random
import numpy as np

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from bisect import bisect_left

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.io import write_video
from decord import VideoReader

from collections import defaultdict
from vbench.utils import CACHE_DIR, load_video, save_json, load_dimension_info, dino_transform, dino_transform_Image
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from scenedetect import open_video, SceneManager, split_video_ffmpeg
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
from moviepy.editor import VideoFileClip
from scipy.stats import rankdata

###################################################################################################
# Consistency Dimensions' Score Distribution Transformation

def quantile_map(inclip_scores, clip2clip_scores, step=0.01):
    """
    Perform quantile mapping from clip2clip_scores to inclip_scores.

    Parameters:
    inclip_scores (array-like): Array of Inclip scores.
    clip2clip_scores (array-like): Array of Clip2Clip scores.
    step (float): Step size for generating the mapping table. Default is 0.01.

    Returns:
    tuple: Mapped Clip2Clip scores, Mapping table between original Clip2Clip scores and mapped scores.
    """
    # Convert clip2clip_scores to quantiles
    ranks = rankdata(clip2clip_scores, method='ordinal')
    clip2clip_quantiles = ranks / (len(clip2clip_scores) + 1)

    # Use the inverse CDF of inclip_scores to map quantiles to actual values
    inclip_sorted = np.sort(inclip_scores)
    inclip_quantiles = np.linspace(0, 1, len(inclip_scores), endpoint=False)

    # Interpolate to find corresponding inclip values for clip2clip quantiles
    clip2clip_scores_mapped = np.interp(clip2clip_quantiles, inclip_quantiles, inclip_sorted)

    # Generate the mapping table
    mapping_range = np.arange(0, 1, step)
    mapping_table = {}
    
    for score in mapping_range:
        # Find the index of the closest quantile to the current score
        closest_idx = (np.abs(clip2clip_quantiles - score)).argmin()
        # Map the score to the corresponding mapped value
        mapping_table[round(float(score), 2)] = round(float(clip2clip_scores_mapped[closest_idx]), 15)
    
    return clip2clip_scores_mapped, mapping_table



###################################################################################################
# Scene Transition Detection

def split_video_into_scenes(video_path, output_dir, threshold=27.0):
    # Open our video, create a scene manager, and add a detector.
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    if scene_list:
        save_video_by_scene_list(video_path, video_name, scene_list, output_dir=output_dir)
    return True if scene_list else False


def save_video_by_scene_list(video_path, video_name, scene_list, output_dir=None):

    first_video_properties = get_video_properties(video_path)
    if not first_video_properties:
        print("Failed to read the first video.")
        return

    fps = first_video_properties['fps']

    frames = load_video(video_path, return_tensor=True)
    
    for i, (start, end) in enumerate(scene_list):
        # get start & end time of each scene
        start_frame = int(start.get_frames())
        end_frame = int(end.get_frames())

        current_scene_frames = frames[start_frame:end_frame]
        current_scene_frames = current_scene_frames.permute(0, 2, 3, 1)



        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(video_path), "split_scene")
            output_filename = os.path.join(output_dir, f"{video_name}-Scene-{i}.mp4")
        else:
            output_filename = os.path.join(output_dir, f"{video_name}-Scene-{i}.mp4")

        write_video(output_filename, current_scene_frames, fps=fps)



def save_segment(frames, fps, save_path):
    if not save_path.endswith('.mp4'):
        save_path += '.mp4'
    

    if frames.dim() == 4 and frames.shape[1] in [1, 3, 4]:  # (N, C, H, W)
        frames = frames.permute(0, 2, 3, 1) # (N, H, W, C)

    write_video(save_path, frames, fps=fps)
    print(f"Video saved to {save_path}")

def split_video_into_clips(video_path, output_path, duration=2, fps=8):

    first_video_properties = get_video_properties(video_path)
    if not first_video_properties:
        print("Failed to read the video.")
        return

    fps = first_video_properties['fps']

    # Load video frames
    frames = load_video(video_path, return_tensor=True)
    segment_frame_count = fps * duration  # Calculate the number of frames per segment

    
    video_name = os.path.basename(video_path).split('.mp4')[0]
    output_dir = os.path.join(output_path, video_name)
    os.makedirs(output_dir, exist_ok=True)

    if len(frames) < segment_frame_count:
        print("Video is too short to be split. Saving the full video instead.")
        frames = frames.permute(0, 2, 3, 1)
        save_path = os.path.join(output_dir, f"{video_name}_full.mp4")
        write_video(save_path, frames, fps=fps)
        print(f"Saved the full video: {save_path}")
        return output_dir

    # Start splitting
    segment_count = 0
    total_segments = len(frames) // segment_frame_count
    remaining_frames = len(frames) % segment_frame_count
    for i in range(total_segments):
        start_frame = i * segment_frame_count
        end_frame = start_frame + segment_frame_count
        segment_frames = frames[start_frame:end_frame]
        segment_frames = segment_frames.permute(0, 2, 3, 1)

        save_path = os.path.join(output_dir, f"{video_name}_{segment_count:03d}.mp4")

        write_video(save_path, segment_frames, fps=fps)
        print(f"Saved {save_path}")
        segment_count += 1

    # Handle the last segment if it's shorter than the expected duration
    if remaining_frames > 0:
        # If the last segment is shorter, extend it by borrowing frames from the previous segments
        additional_frames_needed = segment_frame_count - remaining_frames
        extended_start_frame = max(0, (total_segments * segment_frame_count) - additional_frames_needed)
        
        extended_segment_frames = frames[extended_start_frame:, :, :, :]
        extended_segment_frames = extended_segment_frames.permute(0, 2, 3, 1)


        save_path = os.path.join(output_dir, f"{video_name}_{segment_count:03d}.mp4")
        write_video(save_path, extended_segment_frames, fps=fps)
        print(f"Extended and saved the last segment: {save_path}")

    return output_dir


######################################################################################################
# reorganize codes.
def reorganize_clips_results(detailed_results, dimension=None):

    prompt_scores = defaultdict(list)
    for video_result in detailed_results:
        # Extracting the prompt name (long video name) from the path
        prompt_name = os.path.basename((video_result['video_path'])).split('_')[0]

        long_video_path = video_result['video_path'].split("filtered_clips")[0]
        prompt_name = os.path.join(long_video_path, prompt_name) + ".mp4"
        prompt_scores[prompt_name].append(video_result['video_results'])

    average_scores_list = []
    for prompt, scores in prompt_scores.items():
        average_score = sum(scores) / len(scores) if scores else 0
        average_scores_list.append({
            'video_path': prompt,
            'video_results': average_score
        })

    # Calculate the overall average of all scores
    # all_scores_flat = [average_score for average_score in prompt_scores.values() for score in scores]
    # all_results = sum(all_scores_flat) / len(all_scores_flat) if all_scores_flat else 0
    all_results = sum([item['video_results'] for item in average_scores_list]) / len(average_scores_list) if average_scores_list else 0
    video_cnt=len([item['video_results'] for item in average_scores_list])
    if dimension == 'temporal_flickering':
        average_scores_list.append({
                'long_video_cnt': video_cnt
            })
    if dimension == 'imaging_quality':
        all_results = all_results / 100

    return all_results, detailed_results, average_scores_list


# clip-clip similarity calculation
# Compute similarity across frames randomly sampled from each clip
def create_video_from_first_frames(video_paths, new_cat_video_path, detailed_results):
    if not video_paths:
        print("No video paths provided.")
        return
    
    dimension_video_list = []
    # get the dimension's video list
    def get_long_video_name(video_info_list):
        descriptions = []
        for video_info in video_info_list:
            video_path = video_info['video_path']
            description = os.path.basename(os.path.dirname(video_path))
            descriptions.append(description)
        return descriptions
    dimension_video_list = get_long_video_name(detailed_results)



    # Initialize variables to store the first video's properties
    first_video_properties = get_video_properties(os.path.join(video_paths, os.listdir(video_paths)[0]))
    if not first_video_properties:
        print("Failed to read the first video.")
        return

    fps = first_video_properties['fps']


    # Iterate through each video path and write the first frame to the output video
    for long_video_dir in sorted(os.listdir(video_paths)):
        if long_video_dir not in dimension_video_list:
            continue
        output_dir = os.path.join(new_cat_video_path, long_video_dir) + ".mp4"
        frames = []
        for video_path in sorted(os.listdir(os.path.join(video_paths, long_video_dir))):
            video_full_path = os.path.join(video_paths, long_video_dir, video_path)
            video_frames = load_video(video_full_path, return_tensor=True)

            first_frame = video_frames[0]
            frames.append(first_frame)

        if len(frames) == 1:
            print(f"{long_video_dir} has only one splitted clip, skipping this video")
            continue
        if len(frames) > 0:
            frames = torch.stack(frames)  # Stack frames along a new dimension
            save_segment(frames, fps, output_dir)
            print(f"Created new video from first frames: {output_dir}")
    return 




# for subject/background consistency
def get_video_properties(video_path):
    """Retrieve fps and frame size from the video."""
    if os.path.isdir(video_path):
        video_file = os.path.join(video_path, os.listdir(video_path)[0])
    elif video_path.endswith(('.mp4', '.avi', '.mov')):
        video_file = video_path
    else:
        raise Exception(f"{video_path} should be a path that contains video clips or a path of a video file!")

    try:
        vr = VideoReader(video_file, num_threads=1)
    except Exception as e:
        print(f"Failed to open video file {video_file}: {e}")
        return None

    fps = vr.get_avg_fps()

    return {'fps': int(fps)}


####################################################################################################
# for temporal flickering
def build_filtered_info_json(videos_path, output_path, name):
    cur_full_info_dict = {} # to save the prompt and video path info for the current dimensions

    # get splitted video paths
    # filtered_clips_path = os.path.join(videos_path, 'split_clip')
    filtered_clips_path = os.path.join(videos_path, 'filtered_videos','filtered_clips')
    for filtered_video_name in os.listdir(filtered_clips_path):
        filtered_video_path = os.path.join(filtered_clips_path, filtered_video_name)
        base_prompt = get_prompt_from_filename(filtered_video_name)

        if base_prompt not in cur_full_info_dict:
            cur_full_info_dict[base_prompt] = {
                "prompt_en": base_prompt, 
                "dimension": 'temporal_flickering',
                "video_list": []
            }
        if filtered_video_path.endswith(('.mp4', '.avi', '.mov')):
            cur_full_info_dict[base_prompt]["video_list"].append(filtered_video_path)
        # if os.path.isdir(filtered_video_path):
        #     for split_clip_name in os.listdir(filtered_video_path):
        #         if split_clip_name.endswith(('.mp4', '.avi', '.mov')):
        #             cur_full_info_dict[base_prompt]["video_list"].append(os.path.join(filtered_video_path, split_clip_name))

    cur_full_info_list = list(cur_full_info_dict.values())


    cur_full_info_path = os.path.join(output_path, name+'_info.json')
    save_json(cur_full_info_list, cur_full_info_path)
    print(f'Evaluation meta data saved to {cur_full_info_path}')
    return cur_full_info_path

def linear_interpolate(x, x0, x1, y0, y1):
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

def fuse_inclip_clip2clip(inclip_avg_results, clip2clip_avg_results, inclip_dict, clip2clip_dict, dimension, **kwargs):
    if clip2clip_avg_results is None:
        return inclip_avg_results, inclip_dict

    fused_detailed_results = [] # to record detailed clip2clip & inclip
    fused_all_results_sum = 0 # to record sum of results for each video
    fused_all_results_count = 0 # to record nummber of results in each detailed dict

    if dimension == 'subject_consistency':
        postfix = 'sb'
    elif dimension == 'background_consistency':
        postfix = 'bg'

    with open(kwargs['slow_fast_eval_config'] , 'r') as f:
        params = yaml.safe_load(f)

    kwargs['inclip_mean'] = params.get(f'inclip_mean_{postfix}')
    kwargs['inclip_std'] = params.get(f'inclip_std_{postfix}')
    kwargs['clip2clip_mean'] = params.get(f'clip2clip_mean_{postfix}')
    kwargs['clip2clip_std'] = params.get(f'clip2clip_std_{postfix}')
    if kwargs['dev_flag']:
        kwargs['w_inclip'] = params.get(f'w_inclip_{postfix}')
        kwargs['w_clip2clip'] = params.get(f'w_clip2clip_{postfix}')


    w_inclip = kwargs['w_inclip']
    w_clip2clip = kwargs['w_clip2clip']
    inclip_mean = kwargs['inclip_mean']
    inclip_std = kwargs['inclip_std']
    clip2clip_mean = kwargs['clip2clip_mean']
    clip2clip_std = kwargs['clip2clip_std']

    # Load the mapping table from the YAML file
    with open(kwargs[f'{postfix}_mapping_file_path'], 'r') as f:
        mapping_table = yaml.safe_load(f)

    # Find the interval in the mapping table for clip2clip_score
    keys = sorted(mapping_table.keys())

    clip2clip_dict = {os.path.basename(item['video_path']): item['video_results'] for item in clip2clip_dict}

    for inclip_item in inclip_dict:
        video_path = inclip_item['video_path']
        inclip_score = inclip_item['video_results']

        clip2clip_score = clip2clip_dict.get(os.path.basename(video_path), 0)


        # Find the interval in the mapping table for clip2clip_score using bisect
        idx = bisect_left(keys, clip2clip_score)
        if idx == 0:
            mapped_clip2clip_score = mapping_table[keys[0]]
        elif idx == len(keys):
            mapped_clip2clip_score = mapping_table[keys[-1]]
        else:
            k0, k1 = keys[idx - 1], keys[idx]
            mapped_clip2clip_score = linear_interpolate(
                clip2clip_score, k0, k1,
                mapping_table[k0], mapping_table[k1]
            )

        # Map clip2clip_score to the scale of inclip_score
        # mapped_clip2clip_score = (clip2clip_score - clip2clip_mean) / clip2clip_std * inclip_std + inclip_mean

        fused_score = inclip_score * w_inclip + mapped_clip2clip_score * w_clip2clip if mapped_clip2clip_score != 0.0 else inclip_score
        # fused_detailed_results[video_path] = fused_score
        fused_detailed_results.append({
            "video_path": video_path,
            'inclip_score': inclip_score,
            'clip2clip_score': clip2clip_score,
            'mapped_clip2clip_score': mapped_clip2clip_score,
            "video_results": fused_score
        })
        fused_all_results_sum += fused_score
        fused_all_results_count += 1
    fused_all_results = fused_all_results_sum / fused_all_results_count

    return fused_all_results, fused_detailed_results


def get_duration_from_json(video_path, full_info_list, clip_lengths):
    
    video_name = os.path.basename(video_path)

    pattern1 = re.compile(r"^(.*?)-\d+\.mp4$")

    pattern2 = re.compile(r"^(.*?)-Scene-\d+\.mp4$")

    match = pattern1.match(video_name) or pattern2.match(video_name)
    if match:
        video_description = match.group(1)
        dimensions = [prompt['dimension'] for prompt in full_info_list if prompt['prompt_en'] == video_description]
        if dimensions:
            # Flatten the list of dimensions and remove duplicates
            unique_dimensions = set(dim for sublist in dimensions for dim in sublist)
            # Retrieve the clip lengths for each dimension and find the maximum length
            length_values = [clip_lengths[dim] for dim in unique_dimensions if dim in clip_lengths]
            max_length = max(length_values) if length_values else None
            assert max_length is not None, f"clip duration get a wrong value, check your video path and prompt info"

            return max_length
        
    
def load_clip_lengths(yaml_file):
    with open(yaml_file, 'r') as file:
        clip_lengths = yaml.safe_load(file)
    return clip_lengths

def get_prompt_from_filename(path: str):
    """
    1. prompt-0.suffix -> prompt
    2. prompt.suffix -> prompt
    3. prompt-0_000.suffix -> prompt
    4. prompt-Scene-0_000.suffix -> prompt
    """
    prompt = Path(path).stem

    # Regular expression to remove trailing scene and numeric patterns
    pattern = re.compile(r'(-Scene-\d+|-\d+)_\d+$')
    prompt = re.sub(pattern, '', prompt)

    number_ending = r'-\d+$' # checks ending with -<number>
    if re.search(number_ending, prompt):
        return re.sub(number_ending, '', prompt)
    return prompt


def dreamsim_transform(n_px):
    t = transforms.Compose([
        transforms.Resize((n_px, n_px),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Lambda(lambda x: x.float().div(255.0)),
    ])

    return t

def dreamsim_transform_Image(n_px):
    t = transforms.Compose([
        transforms.Resize((n_px, n_px),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    return t

def dinov2_transform(n_px):
    t = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    return t

def dinov2_transform_Image(n_px):
    t = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    return t
