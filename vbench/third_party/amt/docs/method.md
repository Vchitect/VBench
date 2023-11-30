# Illustration of AMT

<p align="center">
<img src="https://user-images.githubusercontent.com/21050959/229420451-65951bd0-732c-4f09-9121-f291a3862d6e.png" width="1200">
</p>

### :rocket: Highlights:

+ [**Good tradeoff**](#good-tradeoff) between performance and efficiency.

+ [**All-pairs correlation**](#all-pairs-correlation) for modeling large motions during interpolation.

+ A [**plug-and-play operator**](#multi-field-refinement) to improve the diversity of predicted task-oriented flows, further **boosting the interpolation performance**.


## Good Tradeoff

<p align="left">
<img src="https://user-images.githubusercontent.com/21050959/229470703-2f386d62-d26c-46a3-af97-ddfc4270678a.png" width="500">
</p>

We examine the proposed AMT on several public benchmarks with different model scales, showing strong performance and high efficiency in contrast to the SOTA methods (see Figure). Our small model outperforms [IFRNet-B](https://arxiv.org/abs/2205.14620), a SOTA lightweight model, by **\+0.17dB PSNR** on Vimeo90K with **only 60% of its FLOPs and parameters**. For large-scale setting, our AMT exceeds the previous SOTA (i.e., [IFRNet-L](https://arxiv.org/abs/2205.14620)) by **+0.15 dB PSNR** on Vimeo90K with **75% of its FLOPs and 65% of its parameters**. Besides, we provide a huge model for comparison
with the SOTA transformer-based method [VFIFormer](https://arxiv.org/abs/2205.07230). Our convolution-based AMT shows a **comparable performance** but only needs **nearly 23× less computational cost** compared to VFIFormer. 

Considering its effectiveness, we hope our AMT could bring a new perspective for the architecture design in efficient frame interpolation.

## All-pairs correlation

We build all-pairs correlation to effectively model large motions during interpolation.

Here is an example about the update operation at a single scale in AMT:

```python
  # Construct bidirectional correlation volumes
  fmap0, fmap1 = self.feat_encoder([img0_, img1_]) # [B, C, H//8, W//8]
  corr_fn = BidirCorrBlock(fmap0, fmap1, radius=self.radius, num_levels=self.corr_levels)
  
  # Correlation scaled lookup (bilateral -> bidirectional)
  t1_scale = 1. / embt
  t0_scale = 1. / (1. - embt)
  coord = coords_grid(b, h // 8, w // 8, img0.device)
  corr0, corr1 = corr_fn(coord + flow1 * t1_scale, coord + flow0 * t0_scale) 
  corr = torch.cat([corr0, corr1], dim=1)
  flow = torch.cat([flow0, flow1], dim=1)
  
  # Update both intermediate feature and bilateral flows
  delta_feat, delta_flow = self.update(feat, flow, corr)
  delta_flow0, delta_flow1 = torch.chunk(delta_flow, 2, 1)
  flow0 = flow0 + delta_flow0
  flow1= flow1 + delta_flow1
  feat = feat + delta_feat

```

Note: we extend above operations to each pyramid scale (except for the last one), which guarantees the consistency of flows on the coarse scale.

### ⏫ performance gain
|                         | Vimeo 90k | Hard  | Extreme |
|-------------------------|-----------|-------|---------|
| Baseline                | 35.60     | 30.39 | 25.06   |
| + All-pairs correlation | 35.97 (**+0.37**)  | 30.60 (**+0.21**) | 25.30 (**+0.24**)  |

More ablations can be found in the [paper](https://arxiv.org/abs/2304.09790).

## Multi-field Refinement

For most frame interpolation methods which are based on backward warping, the common formulation for
interpolating the final intermediate frame $I_{t}$ is:

$I_{t} = M \odot \mathcal{W}(I_{0}, F_{t\rightarrow 0}) + (1 - M) \odot \mathcal{W}(I_{1}, F_{t\rightarrow 1}) + R$

Above formualtion only utilizes **one set of** bilateral optical flows $F_{t\rightarrow 0}$ and $F_{t\rightarrow 1}$, occulusion masks $M$, and residuals $R$.

Multi-field refinement aims to improve the common formulation of backward warping.
Specifically, we first predict **multiple** bilateral optical flows (accompanied by the corresponding masks and residuals) through simply enlarging the output channels of the last decoder. 
Then, we use aforementioned equation to genearate each interpolated candidate frame. Finally, we obtain the final interpolated frame through combining candidate frames using stacked convolutional layers.

Please refer to [this code snippet](../networks/blocks/multi_flow.py#L46) for the details of the first step.
Please refer to [this code snippet](../networks/blocks/multi_flow.py#L10) for the details of the last two steps.

### 🌟 easy to use
The proposed multi-field refinement can be **easily migrated to any frame interpolation model** to improve the performance.

Code examples are shown below:

```python

# (At the __init__ stage) Initialize a decoder that predicts multiple flow fields (accompanied by the corresponding masks and residuals) 
self.decoder1 = MultiFlowDecoder(channels[0], skip_channels, num_flows)
...

# (At the forward stage) Predict multiple flow fields (accompanied by the corresponding masks and residuals) 
up_flow0_1, up_flow1_1, mask, img_res = self.decoder1(ft_1_, f0_1, f1_1, up_flow0_2, up_flow1_2)
# Merge multiple predictions 
imgt_pred = multi_flow_combine(self.comb_block, img0, img1, up_flow0_1, up_flow1_1,  # self.comb_block stacks two convolutional layers
                                                            mask, img_res, mean_)

```

### ⏫ performance gain

| # Number of flow pairs | Vimeo 90k     | Hard          | Extreme       |
|------------------------|---------------|---------------|---------------|
| Baseline (1 pair)      | 35.84         | 30.52         | 25.25         |
| 3 pairs                | 35.97 (**+0.13**) | 30.60 (**+0.08**) | 25.30 (**+0.05**) |
| 5 pairs                | 36.00 (**+0.16**) | 30.63 (**+0.11**) | 25.33 (**+0.08**) |

## Comparison with SOTA methods
<p align="left">
<img src="https://user-images.githubusercontent.com/21050959/230716340-dea52895-1713-4857-97e5-48cdff9c478f.png" width="1200">
</p>


## Discussions 

We encountered the challenges about the novelty issue during the rebuttal process.

We are ready to clarify again here:

1. We consider the estimation of task-oriented flows from **the perspective of architecture formulation rather than loss function designs** in previous works. The detailed analysis can be found in Sec. 1 of the main paper. We introduce all-pairs correlation to strengthen the ability
in motion modeling, which guarantees **the consistency of flows on the coarse scale**. We employ multi-field refinement to **ensure diversity for the flow regions that need to be task-specific at the finest scale**. The two designs also enable our AMT to capture large motions and successfully handle occlusion regions with high efficiency. As a consequence, they both bring noticeable performance improvements, as shown in the ablations. 
2. The frame interpolation task is closely related to the **motion modeling**. We strongly believe that a [RAFT-style](https://arxiv.org/abs/2003.12039) approach to motion modeling would be beneficial for the frame interpolation task. However, such style **has not been well studied** in the recent frame interpolation literature. Experimental results show that **all-pairs correlation is very important for the performance gain**. We also involve many novel and task-specific designs
beyond the original RAFT. For other task-related design choices, our volume design, scaled lookup strategy, content update, and cross-scale update way have good performance gains on challenging cases (i.e., Hard and Extreme). Besides, if we discard all design choices (but remaining multi-field refinement) and follow the original RAFT to retrain a new model, **the PSNR values will dramatically decrease** (-0.20dB on Vimeo, -0.33dB on Hard, and -0.39dB on Extreme). 
3.  [M2M-VFI](https://arxiv.org/abs/2204.03513) is the most relevant to our multi-field refinement. It also generates multiple flows through the decoder and prepares warped candidates in the image domain. However, there are **five key differences** between our multi-field refinement and M2M-VFI. **First**, our method generates the candidate frames by backward warping rather than forward warping in M2M-VFI. The proposed multi-field refinement aims to improve the common formulation of backward warping (see Eqn.~(4) in the main paper). **Second**, while M2M-VFI predicts multiple flows to overcome the hole issue and artifacts in overlapped regions caused by forward warping, we aim to alleviate the ambiguity issue in the occluded areas and motion boundaries by enhancing the diversity of flows. **Third**, M2M-VFI needs to estimate bidirectional flows first through an off-the-shelf optical flow estimator and then predict multiple bilateral flows through a motion refinement network. On the contrary, we directly estimate multiple bilateral flows in a one-stage network. In this network, we first estimate one pair of bilateral flows at the coarse scale and then derive multiple groups of fine-grained bilateral flows from the coarse flow pairs. **Fourth**, M2M-VFI jointly estimates two reliability maps together with all pairs of bilateral flows, which can be further used to fuse the overlapping pixels caused by forward warping. As shown in Eqn. (5) of the main paper, we estimate not only an occlusion mask but a residual content for cooperating with each pair of bilateral flows. The residual content is used to compensate for the unreliable details after warping. This design has been investigated in Tab. 2e of the main paper. **Fifth**, we stack two convolutional layers to adaptively merge candidate frames, while M2M-VFI normalizes the sum of all candidate frames through a pre-computed weighting map 

More discussions and details can be found in the [appendix](https://arxiv.org/abs/2304.09790) of our paper.
