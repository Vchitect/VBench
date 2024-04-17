import os
import cv2
import glob
import numpy as np
import torch
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import shutil

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from vbench.utils import CACHE_DIR, get_prompt_from_filename, load_json
from vbench.third_party.RAFT.core.raft import RAFT
from vbench.third_party.RAFT.core.utils_core.utils import InputPadder


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = 'cuda'


class StaticFilter:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.load_model()


    def load_model(self):
        self.model = torch.nn.DataParallel(RAFT(self.args))
        self.model.load_state_dict(torch.load(self.args.model))

        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()


    def get_score(self, img, flo):
        img = img[0].permute(1,2,0).cpu().numpy()
        flo = flo[0].permute(1,2,0).cpu().numpy()

        u = flo[:,:,0]
        v = flo[:,:,1]
        rad = np.sqrt(np.square(u) + np.square(v))
        
        h, w = rad.shape
        rad_flat = rad.flatten()
        cut_index = int(h*w*0.02)

        max_rad = np.mean(abs(np.sort(-rad_flat))[:cut_index])

        return max_rad


    def check_static(self, score_list):
        thres = self.params["thres"]
        count_num = self.params["count_num"]
        count = 0
        for score in score_list[:-2]:
            if score > thres:
                count += 1
            if count > count_num:
                return False
        for score in score_list[-2:]:
            if score > thres*count_num*2:
                return False
        return True
    

    def set_params(self, frame, count):
        scale = min(list(frame.shape)[-2:])
        self.params = {"thres":3.0*(scale/256.0), "count_num":round(2*(count/16.0))}


    def infer(self, path):
        with torch.no_grad():
            frames = self.get_frames(path)
            self.set_params(frame=frames[0], count=len(frames))
            static_score = []
            for image1, image2 in zip(frames[:-1]+[frames[0],frames[-1]], frames[1:]+[frames[-1],frames[0]]):
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                _, flow_up = self.model(image1, image2, iters=20, test_mode=True)
                max_rad = self.get_score(image1, flow_up)
                static_score.append(max_rad)
            whether_static = self.check_static(static_score)
            return whether_static


    def get_frames(self, video_path):
        frame_list = []
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            success, frame = video.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to rgb
                frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
                frame = frame[None].to(DEVICE)
                frame_list.append(frame)
            else:
                break
        video.release()
        assert frame_list != []
        return frame_list

def check_and_move(args, filter_results, target_path=None):
    if target_path is None:
         target_path = os.path.join(args.result_path, "filtered_videos")
    os.makedirs(target_path, exist_ok=True)
    for prompt, v in filter_results.items():
        if v["static_count"] < 5 and args.filter_scope=='temporal_flickering':
            logger.warning(f"Prompt: '{prompt}' has fewer than 5 filter results.")
        for i, video_path in enumerate(v["static_path"]):
            target_name = os.path.join(target_path, f"{prompt}-{i}.mp4")
            shutil.copy(video_path, target_name)
    logger.info(f"All filtered videos are saved in the '{target_path}' path")

def static_filter(args):
    static_filter = StaticFilter(args, device=DEVICE)
    prompt_dict = {}
    prompt_list = []
    paths = sorted(glob.glob(os.path.join(args.videos_path, "*.mp4")))
    
    if args.filter_scope=='temporal_flickering':
        full_prompt_list = load_json(f"{CUR_DIR}/vbench/VBench_full_info.json")
        for prompt in full_prompt_list:
            if 'temporal_flickering' in prompt['dimension']:
                prompt_dict[prompt['prompt_en']] = {"static_count":0, "static_path":[]}
                prompt_list.append(prompt['prompt_en'])

    elif args.filter_scope=='all':
        for prompt in paths:
            prompt = get_prompt_from_filename(prompt)
            prompt_dict[prompt] = {"static_count":0, "static_path":[]}
            prompt_list.append(prompt)

    else:
        assert os.path.isfile(args.filter_scope) and Path(args.filter_scope).suffix.lower() == '.json', f"""
        --filter_scope flag is not correctly set, set to 'all' to filter all videos in the --videos_path directory, 
        or provide the correct path to the JSON file
        """
        full_prompt_list = load_json(args.filter_scope)
        for prompt in full_prompt_list:
            prompt = get_prompt_from_filename(prompt)
            prompt_dict[prompt] = {"static_count":0, "static_path":[]}
            prompt_list.append(prompt)
    
    for path in tqdm(paths):
        name = get_prompt_from_filename(path)
        if name in prompt_list:
            if prompt_dict[name]["static_count"] < 5 or args.filter_scope != 'temporal_flickering':
                if static_filter.infer(path):
                    prompt_dict[name]["static_count"] += 1
                    prompt_dict[name]["static_path"].append(path)

    os.makedirs(args.result_path, exist_ok=True)
    info_file = os.path.join(args.result_path, args.store_name)
    json.dump(prompt_dict, open(info_file, "w"))
    logger.info(f"Filtered results info is saved in the '{info_file}' file")
    check_and_move(args, prompt_dict)

def parse_args():
    parser = argparse.ArgumentParser(description='static_filter', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--model', type=str, default=f"{CACHE_DIR}/raft_model/models/raft-things.pth", help="restore checkpoint")
    parser.add_argument('--videos_path', default="", required=True, help="video path for filtering")
    parser.add_argument('--result_path', type=str, default="./filter_results", help='result save path')
    parser.add_argument('--store_name', type=str, default="filtered_static_video.json", help='result file name')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument('--filter_scope', default='temporal_flickering', help=f'''For specifying the scope for filtering videos
        1. 'temporal_flickering' (default): filter videos based on matches with temporal_flickering dimension of VBench.
        2. 'all': filter all video in the current directory.
        3. '$filename': if a filepath to a JSON file is provided, only the filename exists in JSON file will be filtered.
                >       usage: --filter_scope example.json
    ''')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    static_filter(args)
