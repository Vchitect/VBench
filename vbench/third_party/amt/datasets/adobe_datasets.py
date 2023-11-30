'''
    This code is partially borrowed from IFRNet (https://github.com/ltkong218/IFRNet). 
'''
import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
sys.path.append('.')
from utils.utils import read, img2tensor
from datasets.gopro_datasets import (
    random_resize_woflow, random_crop_woflow, center_crop_woflow,
    random_reverse_channel_woflow, random_vertical_flip_woflow,
    random_horizontal_flip_woflow, random_rotate_woflow, 
    random_reverse_time_woflow
)


class Adobe240_Dataset(Dataset):
    def __init__(self, dataset_dir='data/adobe240/test_frames', interFrames=7, augment=True):
        super().__init__()
        self.augment = augment
        self.interFrames = interFrames
        self.setLength = interFrames + 2
        self.dataset_dir = os.path.join(dataset_dir)
        video_list = os.listdir(self.dataset_dir)[9::10]
        self.frames_list = []
        self.file_list = []
        for video in video_list:
            frames = sorted(os.listdir(os.path.join(self.dataset_dir, video)))
            n_sets = (len(frames) - self.setLength) // (interFrames + 1)  + 1
            videoInputs = [frames[(interFrames + 1) * i: (interFrames + 1) * i + self.setLength] for i in range(n_sets)]
            videoInputs = [[os.path.join(video, f) for f in group] for group in videoInputs]
            self.file_list.extend(videoInputs)

    def __getitem__(self, idx):
        clip_idx = idx // self.interFrames
        embt_idx = idx % self.interFrames
        imgpaths = [os.path.join(self.dataset_dir, fp) for fp in self.file_list[clip_idx]]
        pick_idxs = list(range(0, self.setLength, self.interFrames + 1))
        imgt_beg = self.setLength // 2 - self.interFrames // 2
        imgt_end = self.setLength // 2 + self.interFrames // 2 + self.interFrames % 2
        imgt_idx = list(range(imgt_beg, imgt_end)) 
        input_paths = [imgpaths[idx] for idx in pick_idxs]
        imgt_paths = [imgpaths[idx] for idx in imgt_idx]
        
        img0 = np.array(read(input_paths[0]))
        imgt = np.array(read(imgt_paths[embt_idx]))
        img1 = np.array(read(input_paths[1]))
        embt = torch.from_numpy(np.array((embt_idx  + 1) / (self.interFrames + 1)
                                         ).reshape(1, 1, 1).astype(np.float32))

        if self.augment == True:
            img0, imgt, img1 = random_resize_woflow(img0, imgt, img1, p=0.1)
            img0, imgt, img1 = random_crop_woflow(img0, imgt, img1, crop_size=(224, 224))
            img0, imgt, img1 = random_reverse_channel_woflow(img0, imgt, img1, p=0.5)
            img0, imgt, img1 = random_vertical_flip_woflow(img0, imgt, img1, p=0.3)
            img0, imgt, img1 = random_horizontal_flip_woflow(img0, imgt, img1, p=0.5)
            img0, imgt, img1 = random_rotate_woflow(img0, imgt, img1, p=0.05)
            img0, imgt, img1, embt = random_reverse_time_woflow(img0, imgt, img1, 
                                                                embt=embt, p=0.5)
        else:
            img0, imgt, img1 = center_crop_woflow(img0, imgt, img1, crop_size=(512, 512))
            
        img0 = img2tensor(img0).squeeze(0)
        imgt = img2tensor(imgt).squeeze(0)
        img1 = img2tensor(img1).squeeze(0)
        
        return {'img0': img0.float(), 
                'imgt': imgt.float(), 
                'img1': img1.float(),  
                'embt': embt}

    def __len__(self):
        return len(self.file_list) * self.interFrames
