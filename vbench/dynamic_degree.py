import argparse
import os
import cv2
import glob
import numpy as np
import torch
from tqdm import tqdm
from easydict import EasyDict as edict
from decord import VideoReader

from vbench.utils import load_dimension_info, read_frames_decord_by_fps

from vbench.third_party.RAFT.core.raft import RAFT
from vbench.third_party.RAFT.core.utils_core.utils import InputPadder

class DynamicDegree:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.load_model()
    

    def load_model(self):
        self.model = torch.nn.DataParallel(RAFT(self.args))
        self.model.load_state_dict(torch.load(self.args.model, map_location=self.device))

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
        cut_index = int(h*w*0.05)

        max_rad = np.mean(abs(np.sort(-rad_flat))[:cut_index])

        return max_rad.item()


    def set_params(self, frame, count):
        scale = min(list(frame.shape)[-2:])
        self.params = {"thres":6.0*(scale/256.0), "count_num":round(4*(count/16.0))}

    def infer(self, video_path):
        with torch.no_grad():
            frames = read_frames_decord_by_fps(video_path, sample=f'fps8.0')
            frames = frames[:len(frames)//8 * 8] # round(fps/8)
            frames = frames.unsqueeze(1)
            self.set_params(frame=frames[0], count=len(frames))
            static_score = []
            for image1, image2 in zip(frames[:-1], frames[1:]):
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                _, flow_up = self.model(image1, image2, iters=20, test_mode=True)
                max_rad = self.get_score(image1, flow_up)
                static_score.append(max_rad)
            whether_move = self.check_move(static_score)
            return whether_move


    def check_move(self, score_list):
        thres = self.params["thres"]
        count_num = self.params["count_num"]
        count = 0
        for score in score_list:
            if score > thres:
                count += 1
            if count >= count_num:
                return True
        return False


def dynamic_degree(dynamic, video_list):
    sim = []
    video_results = []
    for video_path in tqdm(video_list):
        score_per_video = dynamic.infer(video_path)
        video_results.append({'video_path': video_path, 'video_results': score_per_video})
        sim.append(score_per_video)
    avg_score = np.mean(sim)
    return avg_score, video_results



def compute_dynamic_degree(json_dir, device, submodules_list, **kwargs):
    model_path = submodules_list["model"] 
    # set_args
    args_new = edict({"model":model_path, "small":False, "mixed_precision":False, "alternate_corr":False})
    dynamic = DynamicDegree(args_new, device)
    video_list, _ = load_dimension_info(json_dir, dimension='dynamic_degree', lang='en')
    all_results, video_results = dynamic_degree(dynamic, video_list)
    return all_results, video_results
