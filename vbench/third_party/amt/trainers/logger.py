import time
import wandb
import shutil
import logging
import os.path as osp
from torch.utils.tensorboard import SummaryWriter


def mv_archived_logger(name):
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S_", time.localtime())
    basename = 'archived_' + timestamp + osp.basename(name)
    archived_name = osp.join(osp.dirname(name), basename)
    shutil.move(name, archived_name) 


class CustomLogger:
    def __init__(self, common_cfg, tb_cfg=None, wandb_cfg=None, rank=0):
        global global_logger
        self.rank = rank

        if self.rank == 0:
            self.logger = logging.getLogger('VFI')
            self.logger.setLevel(logging.INFO)
            format_str = logging.Formatter(common_cfg['format'])

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(format_str)

            if osp.exists(common_cfg['filename']):
                mv_archived_logger(common_cfg['filename'])

            file_handler = logging.FileHandler(common_cfg['filename'],
                                               common_cfg['filemode'])
            file_handler.setFormatter(format_str)

            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
            self.tb_logger = None

            self.enable_wandb = False

            if wandb_cfg is not None:
                self.enable_wandb = True
                wandb.init(**wandb_cfg)

            if tb_cfg is not None:
                self.tb_logger = SummaryWriter(**tb_cfg)

        global_logger = self

    def __call__(self, msg=None, level=logging.INFO, tb_msg=None):
        if self.rank != 0:
            return
        if msg is not None:
            self.logger.log(level, msg)

        if self.tb_logger is not None and tb_msg is not None:
            self.tb_logger.add_scalar(*tb_msg)

    def close(self):
        if self.rank == 0 and self.enable_wandb:
            wandb.finish()
