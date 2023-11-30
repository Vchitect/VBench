import sys
import tqdm
import torch
import argparse
import numpy as np
from omegaconf import OmegaConf

sys.path.append('.')
from utils.build_utils import build_from_cfg
from datasets.gopro_datasets import GoPro_Test_Dataset
from metrics.psnr_ssim import calculate_psnr, calculate_ssim

parser = argparse.ArgumentParser(
                prog = 'AMT',
                description = 'GOPRO evaluation',
                )
parser.add_argument('-c', '--config', default='cfgs/AMT-S_gopro.yaml') 
parser.add_argument('-p', '--ckpt', default='pretrained/gopro_amt-s.pth',) 
parser.add_argument('-r', '--root', default='data/GOPRO',) 
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

dataset = GoPro_Test_Dataset(dataset_dir=root)

psnr_list = []
ssim_list = []
pbar = tqdm.tqdm(dataset, total=len(dataset))
for data in pbar:
    input_dict = {}
    for k, v in data.items():
        input_dict[k] = v.to(device).unsqueeze(0)
    with torch.no_grad():
        imgt_pred = model(**input_dict)['imgt_pred']
        psnr = calculate_psnr(imgt_pred, input_dict['imgt'])
        ssim = calculate_ssim(imgt_pred, input_dict['imgt'])
    psnr_list.append(psnr)
    ssim_list.append(ssim)
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    desc_str = f'[{network_name}/GOPRO] psnr: {avg_psnr:.02f}, ssim: {avg_ssim:.04f}'
    pbar.set_description_str(desc_str)


