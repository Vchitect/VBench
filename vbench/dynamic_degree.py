import argparse
import os
import cv2
import glob
import numpy as np
import torch
from tqdm import tqdm
from easydict import EasyDict as edict

from vbench.utils import load_dimension_info

from vbench.third_party.RAFT.core.raft import RAFT
from vbench.third_party.RAFT.core.utils_core.utils import InputPadder


from .distributed import (
    get_world_size,
    get_rank,
    all_gather,
    barrier,
    distribute_list_to_rank,
    gather_list_of_dict,
)


class DynamicDegree:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.load_model()
    

    def load_model(self):
        self.model = RAFT(self.args)
        ckpt = torch.load(self.args.model, map_location="cpu")
        new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        self.model.load_state_dict(new_ckpt)
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
            if video_path.endswith('.mp4'):
                frames = self.get_frames(video_path)
            elif os.path.isdir(video_path):
                frames = self.get_frames_from_img_folder(video_path)
            else:
                raise NotImplementedError
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


    def get_frames(self, video_path):
        frame_list = []
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS) # get fps
        interval = round(fps/8)
        while video.isOpened():
            success, frame = video.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to rgb
                frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
                frame = frame[None].to(self.device)
                frame_list.append(frame)
            else:
                break
        video.release()
        assert frame_list != []
        frame_list = self.extract_frame(frame_list, interval)
        return frame_list 
    
    
    def extract_frame(self, frame_list, interval=1):
        extract = []
        for i in range(0, len(frame_list), interval):
            extract.append(frame_list[i])
        return extract


    def get_frames_from_img_folder(self, img_folder):
        exts = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 
        'tiff', 'JPG', 'PNG', 'JPEG', 'BMP', 
        'TIF', 'TIFF']
        frame_list = []
        imgs = sorted([p for p in glob.glob(os.path.join(img_folder, "*")) if os.path.splitext(p)[1][1:] in exts])
        # imgs = sorted(glob.glob(os.path.join(img_folder, "*.png")))
        for img in imgs:
            frame = cv2.imread(img, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
            frame = frame[None].to(self.device)
            frame_list.append(frame)
        assert frame_list != []
        return frame_list



def dynamic_degree(dynamic, video_list):
    sim = []
    video_results = []
    for video_path in tqdm(video_list, disable=get_rank() > 0):
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
    video_list = distribute_list_to_rank(video_list)
    all_results, video_results = dynamic_degree(dynamic, video_list)
    if get_world_size() > 1:
        video_results = gather_list_of_dict(video_results)
        all_results = sum([d['video_results'] for d in video_results]) / len(video_results)
    return all_results, video_results
