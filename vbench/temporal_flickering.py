import numpy as np
from tqdm import tqdm
import cv2
from vbench.utils import load_dimension_info

from .distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)


def get_frames(video_path):
        frames = []
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            success, frame = video.read()
            if success:
                frames.append(frame)
            else:
                break
        video.release()
        assert frames != []
        return frames


def mae_seq(frames):
    ssds = []
    for i in range(len(frames)-1):
        ssds.append(calculate_mae(frames[i], frames[i+1]))
    return np.array(ssds)


def calculate_mae(img1, img2):
    """Computing the mean absolute error (MAE) between two images."""
    if img1.shape != img2.shape:
        print("Images don't have the same shape.")
        return
    return np.mean(cv2.absdiff(np.array(img1, dtype=np.float32), np.array(img2, dtype=np.float32)))


def cal_score(video_path):
    """please ensure the video is static"""
    frames = get_frames(video_path)
    score_seq = mae_seq(frames)
    return (255.0 - np.mean(score_seq).item())/255.0


def temporal_flickering(video_list):
    sim = []
    video_results = []
    for video_path in tqdm(video_list, disable=get_rank() > 0):
        try:
            score_per_video = cal_score(video_path)
        except AssertionError:
            continue
        video_results.append({'video_path': video_path, 'video_results': score_per_video})
        sim.append(score_per_video)
    avg_score = np.mean(sim)
    return avg_score, video_results


def compute_temporal_flickering(json_dir, device, submodules_list, **kwargs):
    video_list, _ = load_dimension_info(json_dir, dimension='temporal_flickering', lang='en')
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = temporal_flickering(video_list)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results









