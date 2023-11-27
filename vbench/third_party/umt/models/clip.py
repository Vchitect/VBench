#!/usr/bin/env python
import os
from collections import OrderedDict

import torch
from torch import nn


MODEL_PATH = 'your_model_path/clip_visual_encoder'
_MODELS = {
    # extracted from OpenAI, see extract_clip
    "ViT-B/16": os.path.join(MODEL_PATH, "vit_b16.pth"),
    "ViT-L/14": os.path.join(MODEL_PATH, "vit_l14.pth"),
    "ViT-L/14_336": os.path.join(MODEL_PATH, "vit_l14_336.pth"),
}


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x, return_attn=False):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if return_attn:
            return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x, return_attn=False):
        if return_attn:
            x_, attn = self.attention(self.ln_1(x), return_attn=True)
            x = x + x_
            x = x + self.mlp(self.ln_2(x))
            return x, attn
        else:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x


class Transformer(nn.Module):
    def __init__(
            self, width, layers, heads, return_attn=False, 
            clip_return_layer=1, clip_return_interval=1,
        ):
        super().__init__()
        self.layers = layers
        self.return_attn = return_attn
        self.resblocks = nn.ModuleList()
        for _ in range(layers):
            self.resblocks.append(
                ResidualAttentionBlock(
                    width, heads,
                )
            )
        self.return_index = []
        for i in range(clip_return_layer):
            self.return_index.append(layers - int(i * clip_return_interval) - 1)
        print(f'Teacher return index: {self.return_index}')

    def forward(self, x):
        attn = None
        z = []
        for idx, blk in enumerate(self.resblocks):
            if idx == self.layers - 1 and self.return_attn:
                x, attn = blk(x, return_attn=True)
            else:
                x = blk(x)
            if idx in self.return_index:
                z.append(x)
        x = torch.stack(z)
        return x, attn


class VisionTransformer(nn.Module):
    def __init__(
        self, input_resolution, patch_size, width, layers, heads, output_dim, 
        clip_norm_type='l2', kernel_size=1,
        return_attn=False, clip_return_layer=1, clip_return_interval=1,
    ):
        super().__init__()
        self.clip_norm_type = clip_norm_type
        self.return_attn = return_attn
        print(f'Normalization Type: {clip_norm_type}')
        print(f'Return Attention: {return_attn}')
        print(f'Return Layer: {clip_return_layer}')
        print(f'Return Interval: {clip_return_interval}')

        self.output_dim = output_dim
        self.conv1 = nn.Conv3d(
            3, width, 
            (kernel_size, patch_size, patch_size), 
            (kernel_size, patch_size, patch_size), 
            (0, 0, 0), bias=False
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        
        self.transformer = Transformer(
            width, layers, heads, return_attn=return_attn, 
            clip_return_layer=clip_return_layer,
            clip_return_interval=clip_return_interval,
        )

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x, mask=None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        N, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(N * T, H * W, C)

        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        if mask is not None:
            cls_tokens = x[:, :1, :]
            x = x[:, 1:]
            x = x.reshape(N, T * H * W, C)
            x = x[~mask].view(N * T, -1, C)
            HW = x.shape[1]
            x = torch.cat([cls_tokens, x], dim=1)
        else:
            HW = H * W

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, attn = self.transformer(x)

        K = x.shape[0]
        x = self.ln_post(x[:, 1:, :, :])  # [HW, NT, C]
        x = x.view(K, HW, N, T, C).permute(0, 2, 3, 1, 4).reshape(K, N, T * HW, C)  # [K, N, THW, C]
        x = x @ self.proj
        
        if self.clip_norm_type == 'l2':
            x = x / x.norm(dim=-1, keepdim=True)
        elif self.clip_norm_type == 'none':
            pass
        else:
            raise NotImplementedError

        if self.return_attn:
            return x, attn[:, 0, 1:]
        else:
            return x


def inflate_weight(weight_2d, time_dim, center=True):
    print(f'Init center: {center}')
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d


def load_state_dict(model, state_dict, input_resolution=224, patch_size=16, center=True):
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 2:
                print(f'Ignore: {k}')
                continue
            print(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)

    pos_embed_checkpoint = state_dict['positional_embedding']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = (input_resolution // patch_size) ** 2
    orig_size = int((pos_embed_checkpoint.shape[-2] - 1) ** 0.5)
    new_size = int(num_patches ** 0.5)
    if orig_size != new_size:
        print(f'Pos_emb from {orig_size} to {new_size}')
        extra_tokens = pos_embed_checkpoint[:1]
        pos_tokens = pos_embed_checkpoint[1:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(0, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)
        state_dict['positional_embedding'] = new_pos_embed
    
    model.load_state_dict(state_dict, strict=True)


def clip_b16(
    pretrained=True, 
    clip_norm_type='l2', input_resolution=224, kernel_size=1,
    return_attn=False, center=True, clip_return_layer=1,
    clip_return_interval=1
):
    model = VisionTransformer(
        input_resolution=input_resolution, patch_size=16, 
        width=768, layers=12, heads=12, output_dim=512,
        clip_norm_type=clip_norm_type,
        kernel_size=kernel_size, return_attn=return_attn,
        clip_return_layer=clip_return_layer, 
        clip_return_interval=clip_return_interval
    )
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["ViT-B/16"], map_location='cpu')
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=16, center=center)
    return model.eval()


def clip_l14(
    pretrained=True, 
    clip_norm_type='l2', input_resolution=224, kernel_size=1,
    return_attn=False, center=True, clip_return_layer=1,
    clip_return_interval=1
):
    model = VisionTransformer(
        input_resolution=input_resolution, patch_size=14,
        width=1024, layers=24, heads=16, output_dim=768,
        clip_norm_type=clip_norm_type,
        kernel_size=kernel_size, return_attn=return_attn,
        clip_return_layer=clip_return_layer,
        clip_return_interval=clip_return_interval
    )
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["ViT-L/14"], map_location='cpu')
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=14, center=center)
    return model.eval()


def clip_l14_336(
    pretrained=True, 
    clip_norm_type='l2', input_resolution=336, kernel_size=1,
    return_attn=False, center=True, clip_return_layer=1,
    clip_return_interval=1
):
    model = VisionTransformer(
        input_resolution=input_resolution, patch_size=14, 
        width=1024, layers=24, heads=16, output_dim=768,
        clip_norm_type=clip_norm_type,
        kernel_size=kernel_size, return_attn=return_attn,
        clip_return_layer=clip_return_layer,
        clip_return_interval=clip_return_interval,
    )
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["ViT-L/14_336"], map_location='cpu')
        load_state_dict(model, state_dict, input_resolution=input_resolution, patch_size=14, center=center)
    return model.eval()


if __name__ == '__main__':
    import time
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    import numpy as np

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 8

    model = clip_ml_b16(pretrained=True, kernel_size=1, return_attn=False, clip_return_layer=1)
    # print(model)

    # flops = FlopCountAnalysis(model, torch.rand(1, 3, num_frames, 224, 224))
    # s = time.time()
    # print(flop_count_table(flops, max_depth=1))
    # print(time.time()-s)
    print(model(torch.rand(1, 3, num_frames, 224, 224)).shape)