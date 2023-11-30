import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Loss(nn.Module):
    def __init__(self, loss_weight, keys, mapping=None) -> None:
        '''
            mapping: map the kwargs keys into desired ones.
        '''
        super().__init__()
        self.loss_weight = loss_weight
        self.keys = keys
        self.mapping = mapping
        if isinstance(mapping, dict):
            self.mapping = {k: v for k, v in mapping if v in keys}

    
    def forward(self, **kwargs):
        params = {k: v for k, v in kwargs.items() if k in self.keys}
        if self.mapping is not None:
            for k, v in kwargs.items(): 
                if self.mapping.get(k) is not None: 
                    params[self.mapping[k]] = v 
        
        return self._forward(**params) * self.loss_weight

    def _forward(self, **kwargs):
        pass


class CharbonnierLoss(Loss):
    def __init__(self, loss_weight, keys) -> None:
        super().__init__(loss_weight, keys)
        
    def _forward(self, imgt_pred, imgt):    
        diff = imgt_pred - imgt
        loss = ((diff ** 2 + 1e-6) ** 0.5).mean()
        return loss


class AdaCharbonnierLoss(Loss):
    def __init__(self, loss_weight, keys) -> None:
        super().__init__(loss_weight, keys)
        
    def _forward(self, imgt_pred, imgt, weight):   
        alpha = weight / 2
        epsilon = 10 ** (-(10 * weight - 1) / 3)

        diff = imgt_pred - imgt
        loss = ((diff ** 2 + epsilon ** 2) ** alpha).mean()
        return loss
  
  
class TernaryLoss(Loss):
    def __init__(self, loss_weight, keys, patch_size=7):
        super().__init__(loss_weight, keys)
        self.patch_size = patch_size
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w, dtype=torch.float32)

    def transform(self, tensor):
        self.w = self.w.to(tensor.device)
        tensor_ = tensor.mean(dim=1, keepdim=True)
        patches = F.conv2d(tensor_, self.w, padding=self.patch_size//2, bias=None)
        loc_diff = patches - tensor_
        loc_diff_norm = loc_diff / torch.sqrt(0.81 + loc_diff ** 2)
        return loc_diff_norm

    def valid_mask(self, tensor):
        padding = self.patch_size//2
        b, c, h, w = tensor.size()
        inner = torch.ones(b, 1, h - 2 * padding, w - 2 * padding).type_as(tensor)
        mask = F.pad(inner, [padding] * 4)
        return mask
  
    def _forward(self, imgt_pred, imgt):
        loc_diff_x = self.transform(imgt_pred)
        loc_diff_y = self.transform(imgt)
        diff = loc_diff_x - loc_diff_y.detach()
        dist = (diff ** 2 / (0.1 + diff ** 2)).mean(dim=1, keepdim=True)
        mask = self.valid_mask(imgt_pred)
        loss = (dist * mask).mean()
        return loss
 

class GeometryLoss(Loss):
    def __init__(self, loss_weight, keys, patch_size=3):
        super().__init__(loss_weight, keys)
        self.patch_size = patch_size
        out_channels = patch_size * patch_size
        self.w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
        self.w = np.transpose(self.w, (3, 2, 0, 1))
        self.w = torch.tensor(self.w).float()

    def transform(self, tensor):
        b, c, h, w = tensor.size()
        self.w = self.w.to(tensor.device)
        tensor_ = tensor.reshape(b*c, 1, h, w)
        patches = F.conv2d(tensor_, self.w, padding=self.patch_size // 2, bias=None)
        loc_diff = patches - tensor_
        loc_diff_ = loc_diff.reshape(b, c*(self.patch_size ** 2), h, w)
        loc_diff_norm = loc_diff_ / torch.sqrt(0.81 + loc_diff_ ** 2)
        return loc_diff_norm

    def valid_mask(self, tensor):
        padding = self.patch_size // 2
        b, c, h, w = tensor.size()
        inner = torch.ones(b, 1, h - 2 * padding, w - 2 * padding).type_as(tensor)
        mask = F.pad(inner, [padding] * 4)
        return mask

    def _forward(self, ft_pred, ft_gt):
        loss = 0.
        for pred, gt in zip(ft_pred, ft_gt):
            loc_diff_x = self.transform(pred)
            loc_diff_y = self.transform(gt)
            diff = loc_diff_x - loc_diff_y
            dist = (diff ** 2 / (0.1 + diff ** 2)).mean(dim=1, keepdim=True)
            mask = self.valid_mask(pred)
            loss = loss + (dist * mask).mean()
        return loss
    

class IFRFlowLoss(Loss):
    def __init__(self, loss_weight, keys, beta=0.3) -> None:
        super().__init__(loss_weight, keys)
        self.beta = beta
        self.ada_cb_loss = AdaCharbonnierLoss(1.0, ['imgt_pred', 'imgt', 'weight'])
    
    def _forward(self, flow0_pred, flow1_pred, flow):
        
        robust_weight0 = self.get_robust_weight(flow0_pred[0], flow[:, 0:2])
        robust_weight1 = self.get_robust_weight(flow1_pred[0], flow[:, 2:4])
        loss = 0
        for lvl in range(1, len(flow0_pred)):
            scale_factor = 2**lvl
            loss = loss + self.ada_cb_loss(**{
                'imgt_pred': self.resize(flow0_pred[lvl], scale_factor),
                'imgt': flow[:, 0:2],
                'weight': robust_weight0
            })
            loss = loss + self.ada_cb_loss(**{
                'imgt_pred': self.resize(flow1_pred[lvl], scale_factor),
                'imgt': flow[:, 2:4],
                'weight': robust_weight1
            })
        return loss
    
    def resize(self, x, scale_factor):
        return scale_factor * F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)
    
    def get_robust_weight(self, flow_pred, flow_gt):
        epe = ((flow_pred.detach() - flow_gt) ** 2).sum(dim=1, keepdim=True) ** 0.5
        robust_weight = torch.exp(-self.beta * epe)
        return robust_weight


class MultipleFlowLoss(Loss):
    def __init__(self, loss_weight, keys, beta=0.3) -> None:
        super().__init__(loss_weight, keys)
        self.beta = beta
        self.ada_cb_loss = AdaCharbonnierLoss(1.0, ['imgt_pred', 'imgt', 'weight'])
    
    def _forward(self, flow0_pred, flow1_pred, flow):
        
        robust_weight0 = self.get_mutli_flow_robust_weight(flow0_pred[0], flow[:, 0:2])
        robust_weight1 = self.get_mutli_flow_robust_weight(flow1_pred[0], flow[:, 2:4])
        loss = 0
        for lvl in range(1, len(flow0_pred)):
            scale_factor = 2**lvl
            loss = loss + self.ada_cb_loss(**{
                'imgt_pred': self.resize(flow0_pred[lvl], scale_factor),
                'imgt': flow[:, 0:2],
                'weight': robust_weight0
            })
            loss = loss + self.ada_cb_loss(**{
                'imgt_pred': self.resize(flow1_pred[lvl], scale_factor),
                'imgt': flow[:, 2:4],
                'weight': robust_weight1
            })
        return loss
    
    def resize(self, x, scale_factor):
        return scale_factor * F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)

    def get_mutli_flow_robust_weight(self, flow_pred, flow_gt):
        b, num_flows, c, h, w = flow_pred.shape
        flow_pred = flow_pred.view(b, num_flows, c, h, w)
        flow_gt = flow_gt.repeat(1, num_flows, 1, 1).view(b, num_flows, c, h, w)
        epe = ((flow_pred.detach() - flow_gt) ** 2).sum(dim=2, keepdim=True).max(1)[0] ** 0.5
        robust_weight = torch.exp(-self.beta * epe)
        return robust_weight