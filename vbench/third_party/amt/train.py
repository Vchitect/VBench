import os
import argparse
from shutil import copyfile
import torch.distributed as dist
import torch
import importlib
import datetime
from utils.dist_utils import (
    get_world_size,
)
from omegaconf import OmegaConf
from utils.utils import seed_all
parser = argparse.ArgumentParser(description='VFI')
parser.add_argument('-c', '--config', type=str)
parser.add_argument('-p', '--port', default='23455', type=str)
parser.add_argument('--local_rank', default='0')

args = parser.parse_args()


def main_worker(rank, config):
    if 'local_rank' not in config:
        config['local_rank'] = config['global_rank'] = rank
    if torch.cuda.is_available():
        print(f'Rank {rank} is available')
        config['device'] = f"cuda:{rank}"
        if config['distributed']:
            dist.init_process_group(backend='nccl', 
                                    timeout=datetime.timedelta(seconds=5400))
    else:
        config['device'] = 'cpu'

    cfg_name = os.path.basename(args.config).split('.')[0]
    config['exp_name'] = cfg_name + '_' + config['exp_name']
    config['save_dir'] = os.path.join(config['save_dir'], config['exp_name'])

    if (not config['distributed']) or rank == 0:
        os.makedirs(config['save_dir'], exist_ok=True)
        os.makedirs(f'{config["save_dir"]}/ckpts', exist_ok=True)
        config_path = os.path.join(config['save_dir'],
                                   args.config.split('/')[-1])
        if not os.path.isfile(config_path):
            copyfile(args.config, config_path)
        print('[**] create folder {}'.format(config['save_dir']))

    trainer_name = config.get('trainer_type', 'base_trainer')
    print(f'using GPU {rank} for training')
    if rank == 0:
        print(trainer_name)
    trainer_pack = importlib.import_module('trainers.' + trainer_name)
    trainer = trainer_pack.Trainer(config)

    trainer.train()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    cfg = OmegaConf.load(args.config)
    seed_all(cfg.seed)
    rank = int(args.local_rank)
    torch.cuda.set_device(torch.device(f'cuda:{rank}'))
    # setting distributed cfgurations
    cfg['world_size'] = get_world_size()
    cfg['local_rank'] = rank
    if rank == 0:
       print('world_size: ', cfg['world_size'])
    main_worker(rank, cfg)
        
