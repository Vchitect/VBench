import sys
import time
import torch
import argparse
from omegaconf import OmegaConf

sys.path.append('.')
from utils.build_utils import build_from_cfg

parser = argparse.ArgumentParser(
                prog = 'AMT',
                description = 'Speed&parameter benchmark',
                )
parser.add_argument('-c', '--config', default='cfgs/AMT-S.yaml') 
args = parser.parse_args()

cfg_path = args.config
network_cfg = OmegaConf.load(cfg_path).network
model = build_from_cfg(network_cfg)
model = model.cuda()
model.eval()

img0 = torch.randn(1, 3, 256, 448).cuda()
img1 = torch.randn(1, 3, 256, 448).cuda()
embt = torch.tensor(1/2).float().view(1, 1, 1, 1).cuda()

with torch.no_grad():
    for i in range(100):
        out = model(img0, img1, embt, eval=True)
    torch.cuda.synchronize()
    time_stamp = time.time()
    for i in range(1000):
        out = model(img0, img1, embt, eval=True)
    torch.cuda.synchronize()
    print('Time: {:.5f}s'.format((time.time() - time_stamp) / 1))

total = sum([param.nelement() for param in model.parameters()])
print('Parameters: {:.2f}M'.format(total / 1e6))
