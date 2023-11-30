'''
    This code is partially borrowed from IFRNet (https://github.com/ltkong218/IFRNet). 
'''
import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from utils.utils import read


def random_resize(img0, imgt, img1, flow, p=0.1):
    if random.uniform(0, 1) < p:
        img0 = cv2.resize(img0, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        imgt = cv2.resize(imgt, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        flow = cv2.resize(flow, dsize=None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR) * 2.0
    return img0, imgt, img1, flow

def random_crop(img0, imgt, img1, flow, crop_size=(224, 224)):
    h, w = crop_size[0], crop_size[1]
    ih, iw, _ = img0.shape
    x = np.random.randint(0, ih-h+1)
    y = np.random.randint(0, iw-w+1)
    img0 = img0[x:x+h, y:y+w, :]
    imgt = imgt[x:x+h, y:y+w, :]
    img1 = img1[x:x+h, y:y+w, :]
    flow = flow[x:x+h, y:y+w, :]
    return img0, imgt, img1, flow

def random_reverse_channel(img0, imgt, img1, flow, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, :, ::-1]
        imgt = imgt[:, :, ::-1]
        img1 = img1[:, :, ::-1]
    return img0, imgt, img1, flow

def random_vertical_flip(img0, imgt, img1, flow, p=0.3):
    if random.uniform(0, 1) < p:
        img0 = img0[::-1]
        imgt = imgt[::-1]
        img1 = img1[::-1]
        flow = flow[::-1]
        flow = np.concatenate((flow[:, :, 0:1], -flow[:, :, 1:2], flow[:, :, 2:3], -flow[:, :, 3:4]), 2)
    return img0, imgt, img1, flow

def random_horizontal_flip(img0, imgt, img1, flow, p=0.5):
    if random.uniform(0, 1) < p:
        img0 = img0[:, ::-1]
        imgt = imgt[:, ::-1]
        img1 = img1[:, ::-1]
        flow = flow[:, ::-1]
        flow = np.concatenate((-flow[:, :, 0:1], flow[:, :, 1:2], -flow[:, :, 2:3], flow[:, :, 3:4]), 2)
    return img0, imgt, img1, flow

def random_rotate(img0, imgt, img1, flow, p=0.05):
    if random.uniform(0, 1) < p:
        img0 = img0.transpose((1, 0, 2))
        imgt = imgt.transpose((1, 0, 2))
        img1 = img1.transpose((1, 0, 2))
        flow = flow.transpose((1, 0, 2))
        flow = np.concatenate((flow[:, :, 1:2], flow[:, :, 0:1], flow[:, :, 3:4], flow[:, :, 2:3]), 2)
    return img0, imgt, img1, flow

def random_reverse_time(img0, imgt, img1, flow, p=0.5):
    if random.uniform(0, 1) < p:
        tmp = img1
        img1 = img0
        img0 = tmp
        flow = np.concatenate((flow[:, :, 2:4], flow[:, :, 0:2]), 2)
    return img0, imgt, img1, flow


class Vimeo90K_Train_Dataset(Dataset):
    def __init__(self, 
                 dataset_dir='data/vimeo_triplet', 
                 flow_dir=None, 
                 augment=True, 
                 crop_size=(224, 224)):
        self.dataset_dir = dataset_dir
        self.augment = augment
        self.crop_size = crop_size
        self.img0_list = []
        self.imgt_list = []
        self.img1_list = []
        self.flow_t0_list = []
        self.flow_t1_list = []
        if flow_dir is None:
            flow_dir = 'flow'
        with open(os.path.join(dataset_dir, 'tri_trainlist.txt'), 'r') as f:
            for i in f:
                name = str(i).strip()
                if(len(name) <= 1):
                    continue
                self.img0_list.append(os.path.join(dataset_dir, 'sequences', name, 'im1.png'))
                self.imgt_list.append(os.path.join(dataset_dir, 'sequences', name, 'im2.png'))
                self.img1_list.append(os.path.join(dataset_dir, 'sequences', name, 'im3.png'))
                self.flow_t0_list.append(os.path.join(dataset_dir, flow_dir, name, 'flow_t0.flo'))
                self.flow_t1_list.append(os.path.join(dataset_dir, flow_dir, name, 'flow_t1.flo'))

    def __len__(self):
        return len(self.imgt_list)

    def __getitem__(self, idx):
        img0 = read(self.img0_list[idx])
        imgt = read(self.imgt_list[idx])
        img1 = read(self.img1_list[idx])
        flow_t0 = read(self.flow_t0_list[idx])
        flow_t1 = read(self.flow_t1_list[idx])
        flow = np.concatenate((flow_t0, flow_t1), 2).astype(np.float64)

        if self.augment == True:
            img0, imgt, img1, flow = random_resize(img0, imgt, img1, flow, p=0.1)
            img0, imgt, img1, flow = random_crop(img0, imgt, img1, flow, crop_size=self.crop_size)
            img0, imgt, img1, flow = random_reverse_channel(img0, imgt, img1, flow, p=0.5)
            img0, imgt, img1, flow = random_vertical_flip(img0, imgt, img1, flow, p=0.3)
            img0, imgt, img1, flow = random_horizontal_flip(img0, imgt, img1, flow, p=0.5)
            img0, imgt, img1, flow = random_rotate(img0, imgt, img1, flow, p=0.05)
            img0, imgt, img1, flow = random_reverse_time(img0, imgt, img1, flow, p=0.5)
                
        
        img0 = torch.from_numpy(img0.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        imgt = torch.from_numpy(imgt.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        flow = torch.from_numpy(flow.transpose((2, 0, 1)).astype(np.float32))
        embt = torch.from_numpy(np.array(1/2).reshape(1, 1, 1).astype(np.float32))

        return {'img0': img0.float(), 'imgt': imgt.float(), 'img1': img1.float(), 'flow': flow.float(), 'embt': embt}


class Vimeo90K_Test_Dataset(Dataset):
    def __init__(self, dataset_dir='data/vimeo_triplet'):
        self.dataset_dir = dataset_dir
        self.img0_list = []
        self.imgt_list = []
        self.img1_list = []
        self.flow_t0_list = []
        self.flow_t1_list = []
        with open(os.path.join(dataset_dir, 'tri_testlist.txt'), 'r') as f:
            for i in f:
                name = str(i).strip()
                if(len(name) <= 1):
                    continue
                self.img0_list.append(os.path.join(dataset_dir, 'sequences', name, 'im1.png'))
                self.imgt_list.append(os.path.join(dataset_dir, 'sequences', name, 'im2.png'))
                self.img1_list.append(os.path.join(dataset_dir, 'sequences', name, 'im3.png'))
                self.flow_t0_list.append(os.path.join(dataset_dir, 'flow', name, 'flow_t0.flo'))
                self.flow_t1_list.append(os.path.join(dataset_dir, 'flow', name, 'flow_t1.flo'))

    def __len__(self):
        return len(self.imgt_list)

    def __getitem__(self, idx):
        img0 = read(self.img0_list[idx])
        imgt = read(self.imgt_list[idx])
        img1 = read(self.img1_list[idx])
        flow_t0 = read(self.flow_t0_list[idx])
        flow_t1 = read(self.flow_t1_list[idx])
        flow = np.concatenate((flow_t0, flow_t1), 2)

        img0 = torch.from_numpy(img0.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        imgt = torch.from_numpy(imgt.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        img1 = torch.from_numpy(img1.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        flow = torch.from_numpy(flow.transpose((2, 0, 1)).astype(np.float32))
        embt = torch.from_numpy(np.array(1/2).reshape(1, 1, 1).astype(np.float32))
        
        return {'img0': img0.float(), 
                'imgt': imgt.float(), 
                'img1': img1.float(), 
                'flow': flow.float(), 
                'embt': embt}




