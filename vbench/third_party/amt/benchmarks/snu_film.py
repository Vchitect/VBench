import os
import sys
import tqdm
import torch
import argparse
import numpy as np
import os.path as osp
from omegaconf import OmegaConf

sys.path.append('.')
from utils.build_utils import build_from_cfg
from metrics.psnr_ssim import calculate_psnr, calculate_ssim
from utils.utils import InputPadder, read, img2tensor


def parse_path(path):
    path_list = path.split('/')
    new_path = osp.join(*path_list[-3:])
    return new_path

parser = argparse.ArgumentParser(
                prog = 'AMT',
                description = 'SNU-FILM evaluation',
                )
parser.add_argument('-c', '--config', default='cfgs/AMT-S.yaml') 
parser.add_argument('-p', '--ckpt', default='pretrained/amt-s.pth')
parser.add_argument('-r', '--root', default='data/SNU_FILM') 
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg_path = args.config
ckpt_path = args.ckpt
root = args.root

network_cfg = OmegaConf.load(cfg_path).network
network_name = network_cfg.name
model = build_from_cfg(network_cfg)
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt['state_dict'])
model = model.to(device)
model.eval()

divisor = 20; scale_factor = 0.8
splits = ['easy', 'medium', 'hard', 'extreme']
for split in splits:
    with open(os.path.join(root, f'test-{split}.txt'), "r") as fr:
        file_list = [l.strip().split(' ') for l in fr.readlines()]
    pbar = tqdm.tqdm(file_list, total=len(file_list))
    
    psnr_list = []; ssim_list = []
    for name in pbar:
        img0 = img2tensor(read(osp.join(root, parse_path(name[0])))).to(device)
        imgt = img2tensor(read(osp.join(root, parse_path(name[1])))).to(device)
        img1 = img2tensor(read(osp.join(root, parse_path(name[2])))).to(device)
        padder = InputPadder(img0.shape, divisor)
        img0, img1 = padder.pad(img0, img1)
            
        embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)
        imgt_pred = model(img0, img1, embt, scale_factor=scale_factor, eval=True)['imgt_pred']
        imgt_pred = padder.unpad(imgt_pred)

        psnr = calculate_psnr(imgt_pred, imgt).detach().cpu().numpy()
        ssim = calculate_ssim(imgt_pred, imgt).detach().cpu().numpy()

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        desc_str = f'[{network_name}/SNU-FILM] [{split}] psnr: {avg_psnr:.02f}, ssim: {avg_ssim:.04f}'
        pbar.set_description_str(desc_str)
