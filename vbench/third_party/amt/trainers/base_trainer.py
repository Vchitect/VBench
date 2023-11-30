import time
import wandb
import logging
import numpy as np
import os.path as osp
from collections import OrderedDict

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from .logger import CustomLogger
from utils.utils import AverageMeterGroups
from metrics.psnr_ssim import calculate_psnr
from utils.build_utils import build_from_cfg


class Trainer:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rank = self.config['local_rank']
        init_log = self._init_logger()
        self._init_dataset()
        self._init_loss()
        self.model_name = config['exp_name']
        self.model = build_from_cfg(config.network).to(self.config.device)
        
        if config['distributed']:
            self.model = DDP(self.model,
                             device_ids=[self.rank],
                             output_device=self.rank,
                             broadcast_buffers=True,
                             find_unused_parameters=False)

        init_log += str(self.model)
        self.optimizer = AdamW(self.model.parameters(),
                               lr=config.lr, weight_decay=config.weight_decay)
        if self.rank == 0: 
            print(init_log) 
        self.logger(init_log)
        self.resume_training()
    
    def resume_training(self):
        ckpt_path = self.config.get('resume_state')
        if ckpt_path is not None:
            ckpt = torch.load(self.config['resume_state'])
            if self.config['distributed']:
                self.model.module.load_state_dict(ckpt['state_dict'])
            else:
                self.model.load_state_dict(ckpt['state_dict'])
            self.optimizer.load_state_dict(ckpt['optim'])
            self.resume_epoch = ckpt.get('epoch')
            self.logger(
                f'load model from {ckpt_path} and training resumes from epoch {self.resume_epoch}')
        else:
            self.resume_epoch = 0

    def _init_logger(self):
        init_log = ''
        console_cfg = dict(
            level=logging.INFO,
            format="%(asctime)s %(filename)s[line:%(lineno)d]"
            "%(levelname)s %(message)s",
            datefmt="%a, %d %b %Y %H:%M:%S",
            filename=f"{self.config['save_dir']}/log",
            filemode='w')
        tb_cfg = dict(log_dir=osp.join(self.config['save_dir'], 'tb_logger'))
        wandb_cfg = None
        use_wandb = self.config['logger'].get('use_wandb', False)
        if use_wandb:
            resume_id = self.config['logger'].get('resume_id', None)
            if resume_id:
                wandb_id = resume_id
                resume = 'allow'
                init_log += f'Resume wandb logger with id={wandb_id}.'
            else:
                wandb_id = wandb.util.generate_id()
                resume = 'never'

            wandb_cfg = dict(id=wandb_id,
                             resume=resume,
                             name=osp.basename(self.config['save_dir']),
                             config=self.config,
                             project="YOUR PROJECT",
                             entity="YOUR ENTITY",
                             sync_tensorboard=True)
            init_log += f'Use wandb logger with id={wandb_id}; project=[YOUR PROJECT].'
        self.logger = CustomLogger(console_cfg, tb_cfg, wandb_cfg, self.rank)
        return init_log

    def _init_dataset(self):
        dataset_train = build_from_cfg(self.config.data.train)
        dataset_val = build_from_cfg(self.config.data.val)
        
        self.sampler = DistributedSampler(
            dataset_train, num_replicas=self.config['world_size'], rank=self.config['local_rank'])
        self.config.data.train_loader.batch_size //= self.config['world_size']
        self.loader_train = DataLoader(dataset_train,
                                       **self.config.data.train_loader,
                                       pin_memory=True, drop_last=True, sampler=self.sampler)

        self.loader_val = DataLoader(dataset_val, **self.config.data.val_loader,
                                     pin_memory=True, shuffle=False, drop_last=False)

    def _init_loss(self):
        self.loss_dict = dict()
        for loss_cfg in self.config.losses:
            loss = build_from_cfg(loss_cfg)
            self.loss_dict[loss_cfg['nickname']] = loss

    def set_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self, iters):
        ratio = 0.5 * (1.0 + np.cos(iters /
                                    (self.config['epochs'] * self.loader_train.__len__()) * np.pi))
        lr = (self.config['lr'] - self.config['lr_min']
              ) * ratio + self.config['lr_min']
        return lr

    def train(self):
        local_rank = self.config['local_rank']
        best_psnr = 0.0
        loss_group = AverageMeterGroups()
        time_group = AverageMeterGroups()
        iters_per_epoch = self.loader_train.__len__()
        iters = self.resume_epoch * iters_per_epoch
        total_iters = self.config['epochs'] * iters_per_epoch

        start_t = time.time()
        total_t = 0
        for epoch in range(self.resume_epoch, self.config['epochs']):
            self.sampler.set_epoch(epoch)
            for data in self.loader_train:
                for k, v in data.items():
                    data[k] = v.to(self.config['device'])
                data_t = time.time() - start_t

                lr = self.get_lr(iters)
                self.set_lr(self.optimizer, lr)

                self.optimizer.zero_grad()
                results = self.model(**data)
                total_loss = torch.tensor(0., device=self.config['device'])
                for name, loss in self.loss_dict.items():
                    l = loss(**results, **data)
                    loss_group.update({name: l.cpu().data})
                    total_loss += l
                total_loss.backward()
                self.optimizer.step()

                iters += 1

                iter_t = time.time() - start_t
                total_t += iter_t
                time_group.update({'data_t': data_t, 'iter_t': iter_t})

                if (iters+1) % 100 == 0 and local_rank == 0:
                    tpi = total_t / (iters - self.resume_epoch * iters_per_epoch)
                    eta = total_iters * tpi
                    remainder = (total_iters - iters) * tpi
                    eta = self.eta_format(eta)

                    remainder = self.eta_format(remainder)
                    log_str  = f"[{self.model_name}]epoch:{epoch +1}/{self.config['epochs']} "
                    log_str += f"iter:{iters + 1}/{self.config['epochs'] * iters_per_epoch} "
                    log_str += f"time:{time_group.avg('iter_t'):.3f}({time_group.avg('data_t'):.3f}) "
                    log_str += f"lr:{lr:.3e} eta:{remainder}({eta})\n"
                    for name in self.loss_dict.keys():
                        avg_l = loss_group.avg(name)
                        log_str += f"{name}:{avg_l:.3e} "
                        self.logger(tb_msg=[f'loss/{name}', avg_l, iters])
                    log_str += f'best:{best_psnr:.2f}dB\n\n' 
                    self.logger(log_str)
                    loss_group.reset()
                    time_group.reset()
                start_t = time.time()

            if (epoch+1) % self.config['eval_interval'] == 0 and local_rank == 0:
                psnr, eval_t = self.evaluate(epoch)
                total_t += eval_t
                self.logger(tb_msg=['eval/psnr', psnr, epoch])
                if psnr > best_psnr:
                    best_psnr = psnr
                    self.save('psnr_best.pth', epoch)
                    if self.logger.enable_wandb:
                        wandb.run.summary["best_psnr"] = best_psnr
                if (epoch+1) % 50 == 0:
                    self.save(f'epoch_{epoch+1}.pth', epoch)
                self.save('latest.pth', epoch)

        self.logger.close()

    def evaluate(self, epoch):
        psnr_list = []
        time_stamp = time.time()
        for i, data in enumerate(self.loader_val):
            for k, v in data.items():
                data[k] = v.to(self.config['device'])

            with torch.no_grad():
                results = self.model(**data, eval=True)
                imgt_pred = results['imgt_pred']
                for j in range(data['img0'].shape[0]):
                    psnr = calculate_psnr(imgt_pred[j].detach().unsqueeze(
                        0), data['imgt'][j].unsqueeze(0)).cpu().data
                    psnr_list.append(psnr)

        eval_time = time.time() - time_stamp

        self.logger('eval epoch:{}/{} time:{:.2f} psnr:{:.3f}'.format(
            epoch+1, self.config["epochs"], eval_time, np.array(psnr_list).mean()))
        return np.array(psnr_list).mean(), eval_time

    def save(self, name, epoch):
        save_path = '{}/{}/{}'.format(self.config['save_dir'], 'ckpts', name)
        ckpt = OrderedDict(epoch=epoch)
        if self.config['distributed']:
            ckpt['state_dict'] = self.model.module.state_dict()
        else:
            ckpt['state_dict'] = self.model.state_dict()
        ckpt['optim'] = self.optimizer.state_dict()
        torch.save(ckpt, save_path)

    def eta_format(self, eta):
        time_str = ''
        if eta >= 3600:
            hours = int(eta // 3600)
            eta -= hours * 3600
            time_str = f'{hours}'

        if eta >= 60:
            mins = int(eta // 60)
            eta -= mins * 60
            time_str = f'{time_str}:{mins:02}'

        eta = int(eta)
        time_str = f'{time_str}:{eta:02}'
        return time_str
