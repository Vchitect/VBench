import argparse
import os
import cv2
import glob
import numpy as np
import torch
from tqdm import tqdm
import json

from vbench.third_party.RAFT.core.raft import RAFT
from vbench.third_party.RAFT.core.utils_core.utils import InputPadder


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


def filter_static(args):
    static_filter = StaticFilter(args, device=DEVICE)
    prompt_dict = {}
    with open(args.prompt_file, "r") as f:
        lines = [line.strip() for line in f.readlines()]
        for line in lines:
            prompt_dict[line] = {"static_count":0, "static_path":[]}
    
    paths = sorted(glob.glob(os.path.join(args.videos_path, "*.mp4")))
    for path in tqdm(paths):
        name = '-'.join(path.split('/')[-1].split('-')[:-1]) 
        if name in lines:
            if prompt_dict[name]["static_count"] < 5:
                if static_filter.infer(path):
                    prompt_dict[name]["static_count"] += 1
                    prompt_dict[name]["static_path"].append(path)
    os.makedirs(args.result_path, exist_ok=True)
    json.dump(prompt_dict, open(os.path.join(args.result_path, args.store_name), "w"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="./pretrained/raft_model/models/raft-things.pth", help="restore checkpoint")
    parser.add_argument('--videos_path', default="", required=True, help="video path for filtering")
    parser.add_argument('--result_path', type=str, default="./filter_results", help='result save path')
    parser.add_argument('--store_name', type=str, default="filtered_static_video.json", help='result file name')
    parser.add_argument('--prompt_file', type=str, default="./prompts/prompts_per_dimension/temporal_flickering.txt", help='static_prompt')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    filter_static(args)
