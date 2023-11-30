import os
import sys
import cv2
import tqdm
import glob
import torch
import argparse
import numpy as np
import os.path as osp
from omegaconf import OmegaConf

sys.path.append('.')
from utils.utils import InputPadder, read, img2tensor
from utils.build_utils import build_from_cfg
from metrics.psnr_ssim import calculate_psnr, calculate_ssim

parser = argparse.ArgumentParser(
                prog = 'AMT',
                description = 'Xiph evaluation',
                )
parser.add_argument('-c', '--config', default='cfgs/AMT-S.yaml') 
parser.add_argument('-p', '--ckpt', default='pretrained/amt-s.pth') 
parser.add_argument('-r', '--root', default='data/xiph') 
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg_path = args.config
ckpt_path = args.ckpt
root = args.root

network_cfg = OmegaConf.load(cfg_path).network
network_name = network_cfg.name
model = build_from_cfg(network_cfg)
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt['state_dict'], False)
model = model.to(device)
model.eval()

############################################# Prepare Dataset #############################################
download_links = [
    'https://media.xiph.org/video/derf/ElFuente/Netflix_BoxingPractice_4096x2160_60fps_10bit_420.y4m',
    'https://media.xiph.org/video/derf/ElFuente/Netflix_Crosswalk_4096x2160_60fps_10bit_420.y4m',
    'https://media.xiph.org/video/derf/Chimera/Netflix_DrivingPOV_4096x2160_60fps_10bit_420.y4m',
    'https://media.xiph.org/video/derf/ElFuente/Netflix_FoodMarket_4096x2160_60fps_10bit_420.y4m',
    'https://media.xiph.org/video/derf/ElFuente/Netflix_FoodMarket2_4096x2160_60fps_10bit_420.y4m',
    'https://media.xiph.org/video/derf/ElFuente/Netflix_RitualDance_4096x2160_60fps_10bit_420.y4m',
    'https://media.xiph.org/video/derf/ElFuente/Netflix_SquareAndTimelapse_4096x2160_60fps_10bit_420.y4m',
    'https://media.xiph.org/video/derf/ElFuente/Netflix_Tango_4096x2160_60fps_10bit_420.y4m',
]
file_list = ['BoxingPractice', 'Crosswalk', 'DrivingPOV', 'FoodMarket', 'FoodMarket2', 'RitualDance', 
             'SquareAndTimelapse', 'Tango']

for file_name, link in zip(file_list, download_links):
    data_dir = osp.join(root, file_name)
    if osp.exists(data_dir) is False:
        os.makedirs(data_dir)
    if len(glob.glob(f'{data_dir}/*.png')) < 100:
        os.system(f'ffmpeg -i {link} -pix_fmt rgb24 -vframes 100 {data_dir}/%03d.png')
############################################### Prepare End ###############################################


divisor = 32; scale_factor = 0.5
for category in ['resized-2k', 'cropped-4k']:
    psnr_list = []
    ssim_list = []
    pbar = tqdm.tqdm(file_list, total=len(file_list))
    for flie_name in pbar:
        dir_name = osp.join(root, flie_name)
        for intFrame in range(2, 99, 2):
            img0 = read(f'{dir_name}/{intFrame - 1:03d}.png')
            img1 = read(f'{dir_name}/{intFrame + 1:03d}.png')
            imgt = read(f'{dir_name}/{intFrame:03d}.png')

            if category == 'resized-2k':
                img0 = cv2.resize(src=img0, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                img1 = cv2.resize(src=img1, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                imgt = cv2.resize(src=imgt, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

            elif category == 'cropped-4k':
                img0 = img0[540:-540, 1024:-1024, :]
                img1 = img1[540:-540, 1024:-1024, :]
                imgt = imgt[540:-540, 1024:-1024, :]
            img0 = img2tensor(img0).to(device)
            imgt = img2tensor(imgt).to(device)
            img1 = img2tensor(img1).to(device)
            embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(device)
            
            padder = InputPadder(img0.shape, divisor)
            img0, img1 = padder.pad(img0, img1)

            with torch.no_grad():
                imgt_pred = model(img0, img1, embt, scale_factor=scale_factor, eval=True)['imgt_pred']
                imgt_pred = padder.unpad(imgt_pred)

            psnr = calculate_psnr(imgt_pred, imgt)
            ssim = calculate_ssim(imgt_pred, imgt)

            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            desc_str = f'[{network_name}/Xiph] [{category}/{flie_name}] psnr: {avg_psnr:.02f}, ssim: {avg_ssim:.04f}'

            pbar.set_description_str(desc_str)