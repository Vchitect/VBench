import argparse
import os
import cv2
import glob
import numpy as np
import torch
from tqdm import tqdm
from easydict import EasyDict as edict
import csv
import decord
decord.bridge.set_bridge('torch')
from .third_party.cotracker.utils.visualizer import Visualizer
from .third_party.Dense_match.PatchAutoEvaluate import PatchAutoEvaluate
from .third_party.Dense_match.utils_function import *
from .third_party.RAFT.core.raft import RAFT
from .third_party.RAFT.core.utils.utils import InputPadder
import json
from vbench2.utils import load_dimension_info, split_video_into_scenes
from tqdm import tqdm

def transform_class360(vector, min_reso, factor=0.008): # 768*0.05
    scale = min_reso * factor
    up, down, y = vector
    if abs(y)<scale:
        if up * down<0 and up>scale:
            return "orbits"  #orbits_counterclockwise
        elif up*down<0 and up<-scale:
            return "orbits"   #orbits_clockwise
        else:
            return None

class CameraPredict:
    def __init__(self, device, submodules_list):
        self.device = device
        self.grid_size = 10
        self.number_points = 1
        try:
            self.model = torch.hub.load(submodules_list["repo"], submodules_list["model"]).to(self.device)
        except:
            # workaround for CERTIFICATE_VERIFY_FAILED (see: https://github.com/pytorch/pytorch/issues/33288#issuecomment-954160699)
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            self.model = torch.hub.load(submodules_list["repo"], submodules_list["model"]).to(self.device)

    def transform360(self, vector):
        up=[]
        down=[]
        for item in vector:
            if item[2]>self.scale/2:
                down.append(item[0])
            else:
                up.append(item[0])
        y = np.mean([item[1] for item in vector])
        if len(up)>0:
            mean_up=sum(up)/len(up)
        else:
            mean_up=0
        if len(down)>0:
            mean_down=sum(down)/len(down)
        else:
           mean_down=0
        return [mean_up, mean_down, y]

    def infer(self, video, fps=16, end_frame=-1, save_video=False, save_dir="./saved_videos"):
        b,_,_,h,w=video.shape
        self.scale=min(h,w)
        self.height=h
        self.width=w
        pred_tracks, pred_visibility = self.model(video, grid_size=self.grid_size) # B T N 2,  B T N 1
        if save_video:
            vis = Visualizer(save_dir=save_dir, pad_value=120, fps=fps, linewidth=3)
            vis.visualize(video, pred_tracks, pred_visibility, filename="temp1")
            raise
        if end_frame!=-1:
            pred_tracks = pred_tracks[:,:end_frame]
            pred_visibility = pred_visibility[:,:end_frame]
        return pred_tracks[0].long().detach().cpu().numpy()
    
    def get_edge_point_360(self, track):
        middle = self.grid_size // 2
        number = 2
        lists=[0,1,self.grid_size-2,self.grid_size-1]
        idx=2
        res=[]
        for i in lists:
            if track[i, idx, 0]<0 or track[i, idx, 1]<0:
                res.append(None)
            else:
                res.append(list(track[i, idx, :]))
        return res
    
    def get_edge_direction_360(self, tracks):
        alls=[]
        for track1, track2 in zip(tracks[:-1], tracks[1:]):
            edge_points1 = self.get_edge_point_360(track1)
            edge_points2 = self.get_edge_point_360(track2)
            vector_results = []
            for points1, points2 in zip(edge_points1, edge_points2):
                if self.check_valid(points1) and self.check_valid(points2):
                    vector_results.append([points2[0]-points1[0], points2[1]-points1[1], points1[1]])
            if len(vector_results)==0:
                continue
            vector_results_360 = self.transform360(vector_results)
            class_results360 = transform_class360(vector_results_360, min_reso=self.scale)
            alls.append(class_results360)
        return alls
    
    def check_valid(self, point):
        if point is not None:
            if point[0]>0 and point[0]<self.width and point[1]>0 and point[1]<self.height:
                return True
            else:
                return False
        else:
            return False

    def camera_classify(self, track1, track2, tracks):
        r360_results = self.get_edge_direction_360(tracks)
        return r360_results
    
    def predict(self, video, fps, end_frame):
        pred_track = self.infer(video, fps, end_frame)
        track1 = pred_track[0].reshape((self.grid_size, self.grid_size, 2))
        track2 = pred_track[-1].reshape((self.grid_size, self.grid_size, 2))
        tracks=[pred_track[i].reshape(self.grid_size, self.grid_size, 2) for i in range(0, len(pred_track), 20)]
        results = self.camera_classify(track1, track2, tracks)
        return results
    
def whether_orbit(video_path, camera):
    label='orbits'
    end_frame=-1
    scene_list = split_video_into_scenes(video_path, 10.0)
    if len(scene_list)!=0:
        end_frame = int(scene_list[0][1].get_frames())
    video_reader = decord.VideoReader(video_path)
    video = video_reader.get_batch(range(len(video_reader))) 
    frame_count, height, width = video.shape[0], video.shape[1], video.shape[2]
    video = video.permute(0, 3, 1, 2)[None].float().cuda() # B T C H W
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    predict_results = camera.predict(video, fps, end_frame)
    flag = label in predict_results
    return flag, end_frame, frame_count
    
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

    def infer(self, video_path, skip_frame):
        with torch.no_grad():
            frames, fps = self.get_frames(video_path, skip_frame)
            self.set_params(frame=frames[0], count=len(frames))
            static_score = []
            for image1, image2 in zip(frames[:-1], frames[1:]):
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                _, flow_up = self.model(image1, image2, iters=20, test_mode=True)
                max_rad = self.get_score(image1, flow_up)
                static_score.append(max_rad)
            mean_score = sum(static_score) / len(static_score)
            if mean_score<5:
                return -1, fps
            else:
                return np.clip(mean_score, 0, 30), fps

    def get_frames(self, video_path, skip_frame):
        frame_list = []
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS) # get fps
        interval = max(1, round(fps / 8))
        while video.isOpened():
            success, frame = video.read()
            if success:
                frame = cv2.resize(frame, (854, 480))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to rgb
                frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
                frame = frame[None].to(self.device)
                frame_list.append(frame)
            else:
                break
        video.release()
        assert frame_list != []
        if skip_frame==-1:
            frame_list = self.extract_frame(frame_list, len(frame_list), interval)
        else:
            frame_list = self.extract_frame(frame_list, skip_frame, interval)
        return frame_list, fps
    
    def extract_frame(self, frame_list, skip_frame, interval=1):
        extract = []
        for i in range(0, len(frame_list), interval):
            if i>skip_frame:
                break
            extract.append(frame_list[i])
        return extract

def dynamic_degree(dynamic, video_path, skip_frame, end_frame, judge):
    score_per_video=-1
    fps=-1
    if judge:
        score_per_video, fps = dynamic.infer(video_path, skip_frame)
    match_score = PatchAutoEvaluate(video_path, skip_frame, end_frame, score_per_video, fps)
    if score_per_video != -1 and match_score!=-1:
        final = np.clip(score_per_video, 0, 10)*match_score/10
    else:
        final = -1
    return final

def multi_view_consistency(prompt_dict_ls, camera, dynamic):
    final_score=0
    valid_num=0
    processed_json=[]
    for prompt_dict in tqdm(prompt_dict_ls):
        video_paths = prompt_dict['video_list']
        for video_path in video_paths:
            judge, skip_frame, end_frame = whether_orbit(video_path, camera)
            score = dynamic_degree(dynamic, video_path, skip_frame, end_frame, judge)
            processed_json.append({'video_path': video_path, 'video_results': score})
            if score!=-1:
                final_score+=score
                valid_num+=1
    return final_score/valid_num, processed_json
            
def compute_multi_view_consistency(json_dir, device, submodules_dict, **kwargs):
    camera = CameraPredict(device, submodules_dict)
    model_path = submodules_dict['raft']
    args_new = edict({"model":model_path, "small":False, "mixed_precision":False, "alternate_corr":False})
    dynamic = DynamicDegree(args_new, device)
    
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='multi-view_consistency', lang='en')
    all_results, video_results = multi_view_consistency(prompt_dict_ls, camera, dynamic)
    score=0
    num=0
    for d in video_results:
        if d['video_results']!=-1:
            num+=1
            score+= d['video_results']
    all_results = score/num
    return all_results, video_results