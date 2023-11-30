'''
    This code is partially borrowed from IFRNet (https://github.com/ltkong218/IFRNet). 
    In the consideration of the difficulty in flow supervision generation, we abort 
    flow loss in the 8x case.
'''
import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from utils.utils import read, img2tensor

def random_resize_woflow(img0, imgt, img1, p=0.1):
    if random.uniform(0, 1) < p:
        img0 = cv2.resize(img0, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        imgt = cv2.resize(imgt, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
    return img0, imgt, img1

def random_crop_woflow(img0, imgt, img1, crop_size=(224, 224)):
    h, w = crop_size[0], crop_size[1]
    ih, iw, _ = img0.shape
    x = np.random.randint(0, ih-h+1)
    y = np.random.randint(0, iw-w+1)
    img0 = img0[x: x + h, y : y + w, :]
    imgt = imgt[x: x + h, y : y + w, :]
    img1 = img1[x: x + h, y : y + w, :]
    return img0, imgt, img1

def center_crop_woflow(img0, imgt, img1, crop_size=(512, 512)):
    h, w = crop_size[0], crop_size[1]
    ih, iw, _ = img0.shape
    img0 = img0[ih // 2 - h // 2: ih // 2 + h // 2, iw // 2 - w // 2: iw // 2 +  w // 2, :]
    imgt = imgt[ih // 2 - h // 2: ih // 2 + h // 2, iw // 2 - w // 2: iw // 2 +  w // 2, :]
    img1 = img1[ih // 2 - h // 2: ih // 2 + h // 2, iw // 2 - w // 2: iw // 2 +  w // 2, :]
    return img0, imgt, img1

def random_reverse_channel_woflow(img0, imgt, img1, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, :, ::-1]
        imgt = imgt[:, :, ::-1]
        img1 = img1[:, :, ::-1]
    return img0, imgt, img1

def random_vertical_flip_woflow(img0, imgt, img1, p=0.3):
    if random.uniform(0, 1) < p:
        img0 = img0[::-1]
        imgt = imgt[::-1]
        img1 = img1[::-1]
    return img0, imgt, img1

def random_horizontal_flip_woflow(img0, imgt, img1, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, ::-1]
        imgt = imgt[:, ::-1]
        img1 = img1[:, ::-1]
    return img0, imgt, img1

def random_rotate_woflow(img0, imgt, img1, p=0.05):
    if random.uniform(0, 1) < p:
        img0 = img0.transpose((1, 0, 2))
        imgt = imgt.transpose((1, 0, 2))
        img1 = img1.transpose((1, 0, 2))
    return img0, imgt, img1

def random_reverse_time_woflow(img0, imgt, img1, embt, p=0.5):
    if random.uniform(0, 1) < p:
        tmp = img1
        img1 = img0
        img0 = tmp
    embt = 1 - embt
    return img0, imgt, img1, embt

class GoPro_Train_Dataset(Dataset):
    def __init__(self, dataset_dir='data/GOPRO', interFrames=7, augment=True):
        self.dataset_dir = dataset_dir + '/train'
        self.interFrames = interFrames
        self.augment = augment
        self.setLength = interFrames + 2
        video_list = [
            'GOPR0372_07_00', 'GOPR0374_11_01', 'GOPR0378_13_00', 'GOPR0384_11_01', 
            'GOPR0384_11_04', 'GOPR0477_11_00', 'GOPR0868_11_02', 'GOPR0884_11_00', 
            'GOPR0372_07_01', 'GOPR0374_11_02', 'GOPR0379_11_00', 'GOPR0384_11_02', 
            'GOPR0385_11_00', 'GOPR0857_11_00', 'GOPR0871_11_01', 'GOPR0374_11_00', 
            'GOPR0374_11_03', 'GOPR0380_11_00', 'GOPR0384_11_03', 'GOPR0386_11_00', 
            'GOPR0868_11_01', 'GOPR0881_11_00']
        self.frames_list = []
        self.file_list = []
        for video in video_list:
            frames = sorted(os.listdir(os.path.join(self.dataset_dir, video)))
            n_sets = (len(frames) - self.setLength) // (interFrames+1)  + 1
            videoInputs = [frames[(interFrames + 1) * i: (interFrames + 1) * i + self.setLength
                                                        ] for i in range(n_sets)]
            videoInputs = [[os.path.join(video, f) for f in group] for group in videoInputs]
            self.file_list.extend(videoInputs)

    def __len__(self):
        return len(self.file_list) * self.interFrames

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
        
        embt = torch.from_numpy(np.array((embt_idx  + 1) / (self.interFrames+1)
                                         ).reshape(1, 1, 1).astype(np.float32))
        img0 = np.array(read(input_paths[0]))
        imgt = np.array(read(imgt_paths[embt_idx]))
        img1 = np.array(read(input_paths[1]))

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
            
        img0 = img2tensor(img0.copy()).squeeze(0)
        imgt = img2tensor(imgt.copy()).squeeze(0)
        img1 = img2tensor(img1.copy()).squeeze(0)
        
        return {'img0': img0.float(), 
                'imgt': imgt.float(), 
                'img1': img1.float(),  
                'embt': embt}

class GoPro_Test_Dataset(Dataset):
    def __init__(self, dataset_dir='data/GOPRO', interFrames=7):
        self.dataset_dir = dataset_dir + '/test'
        self.interFrames = interFrames
        self.setLength = interFrames + 2
        video_list = [
            'GOPR0384_11_00', 'GOPR0385_11_01', 'GOPR0410_11_00', 
            'GOPR0862_11_00', 'GOPR0869_11_00', 'GOPR0881_11_01', 
            'GOPR0384_11_05', 'GOPR0396_11_00', 'GOPR0854_11_00', 
            'GOPR0868_11_00', 'GOPR0871_11_00']
        self.frames_list = []
        self.file_list = []
        for video in video_list:
            frames = sorted(os.listdir(os.path.join(self.dataset_dir, video)))
            n_sets = (len(frames) - self.setLength)//(interFrames+1)  + 1
            videoInputs = [frames[(interFrames + 1) * i:(interFrames + 1) * i + self.setLength
                                                        ] for i in range(n_sets)]
            videoInputs = [[os.path.join(video, f) for f in group] for group in videoInputs]
            self.file_list.extend(videoInputs)

    def __len__(self):
        return len(self.file_list) * self.interFrames

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

        img0, imgt, img1 = center_crop_woflow(img0, imgt, img1, crop_size=(512, 512))

        img0 = img2tensor(img0).squeeze(0)
        imgt = img2tensor(imgt).squeeze(0)
        img1 = img2tensor(img1).squeeze(0)
        
        embt = torch.from_numpy(np.array((embt_idx + 1) / (self.interFrames + 1)
                                         ).reshape(1, 1, 1).astype(np.float32))
        return {'img0': img0.float(), 
                'imgt': imgt.float(), 
                'img1': img1.float(),  
                'embt': embt}