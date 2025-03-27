import cv2
import numpy as np
import torch
import decord
decord.bridge.set_bridge('torch')
import torchvision
import imageio
from einops import rearrange
from scipy.spatial.distance import cosine
from .third_party.cotracker.utils.visualizer import Visualizer
from .third_party.arcface.models import *
from retinaface.predict_single import Model
from PIL import Image
from torch.utils import model_zoo
import os
import json
from collections import Counter
from vbench2.utils import load_dimension_info
from tqdm import tqdm

def most_frequent_number(numbers):
    count = Counter(numbers)
    most_common_num, count = count.most_common(1)[0]
    return most_common_num

def extract_face_features(face_image, model):
    face_image = cv2.resize(face_image, (128, 128))  
    image_gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)  # 转换为灰度图，shape: (100, 100)
    image = np.dstack((image_gray, np.fliplr(image_gray)))  # shape: (100, 100, 2)
    image = image.transpose((2, 0, 1))  # shape: (2, 100, 100)
    image = image[:, np.newaxis, :, :]  # shape: (2, 1, 100, 100)
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    data = torch.from_numpy(image).cuda()
    with torch.no_grad():
        output = model(data)
        output = output.data.cpu().numpy()
        fe_1 = output[::2]
        fe_2 = output[1::2]
        feature = np.hstack((fe_1, fe_2))
    return feature.flatten() # 返回人脸特征向量

def calculate_similarity(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    
class IDTracker:
    def __init__(self, similarity_threshold=0.4, grid_size=20):
        self.trackers = {}  # 存储每个ID的人脸特征
        self.similarity_threshold = similarity_threshold  # 相似度阈值
        self.track_human=[]

    def update(self, frame, frame_count, retina_model, model):
        frame=frame.astype(np.uint8)
        faces = retina_model.predict_jsons(frame)
        flag=True
        if faces is not None:
            if len(faces)==1:
                box=faces[0]['bbox']
                if len(box)==4:
                    x1, y1, x2, y2 = box
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(frame.shape[1], int(x2))
                    y2 = min(frame.shape[0], int(y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
                    face_image = frame[y1:y2, x1:x2]
                    face_features = extract_face_features(face_image, model)
                    if frame_count<1:
                        self.trackers=face_features
                    else:
                        similarity = calculate_similarity(face_features, self.trackers)
                        if similarity<self.similarity_threshold:
                            flag=False
                elif frame_count==0 and len(box)!=4:
                    return True, False
            else:
                return True, False
        else:
            return True, False
            
        return flag, True

    def check_point(self, point, box):
        return point[1]>=box[0] and point[1]<=box[1] and point[0]>=box[2] and point[0]<=box[3]

def evaluate_id_consistency(prompt_dict_ls, retina_model, model):
    score=0
    num=0
    mini_frame=20
    similarity_threshold=0.4
    processed_json=[]
    for prompt_dict in tqdm(prompt_dict_ls):
        video_paths = prompt_dict['video_list']
        for video_path in video_paths:
        
            video_reader = decord.VideoReader(video_path)
            video = video_reader.get_batch(range(len(video_reader))) 
            frame_count, height, width = video.shape[0], video.shape[1], video.shape[2]
            scale = min(height, width)
            video = video.permute(0, 3, 1, 2)[None].float().cuda() # B T C H W
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            tracker = IDTracker(similarity_threshold=similarity_threshold)
            consistent_frame_count = 0
            frame_num = 0
            video_np = video[0].permute(0,2,3,1).detach().cpu().numpy()
            for coun, frame in enumerate(video_np):
                if coun<1:
                    flag, valid = tracker.update(frame, coun, retina_model, model)
                    if not flag or not valid:
                        consistent_frame_count = 0
                        frame_num = 0
                        break
                else:
                    flag, valid = tracker.update(frame, coun, retina_model, model)
                    if not valid:
                        continue
                    else:
                        if flag:
                            consistent_frame_count += 1
                        frame_num+=1

            new_item={
                'video_path':video_path,
            }
            if frame_num<mini_frame:
                new_item['video_results']=-1
                processed_json.append(new_item)
                continue
            else:
                new_item['video_results']=consistent_frame_count/frame_num
            score+=consistent_frame_count
            num+=frame_num
            processed_json.append(new_item)
            
    return score/num, processed_json

    
    
def compute_human_identity(json_dir, device, submodules_dict, **kwargs):
    _, prompt_dict_ls = load_dimension_info(json_dir, dimension='human_identity', lang='en')
    
    url="https://github.com/ternaus/retinaface/releases/download/0.01/retinaface_resnet50_2020-07-20-f168fae3c.zip"
    retina_state_dict = model_zoo.load_url(url, progress=True, map_location="cpu")
    retina_model = Model(max_size=2048, device=device)
    retina_model.load_state_dict(retina_state_dict)
    model = resnet_face18(use_se=False)
    state_dict=torch.load(submodules_dict['model'])
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    model.load_state_dict(new_state_dict)
    model.to(device).eval()
    
    all_results, video_results = evaluate_id_consistency(prompt_dict_ls, retina_model, model)
    score=0
    num=0
    for d in video_results:
        if d['video_results']!=-1:
            num+=1
            score+= d['video_results']
    all_results = score/num
    return all_results, video_results