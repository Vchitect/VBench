# AMT: All-Pairs Multi-Field Transforms for Efficient Frame Interpolation


This repository contains the official implementation of the following paper:
> **AMT: All-Pairs Multi-Field Transforms for Efficient Frame Interpolation**<br>
> [Zhen Li](https://paper99.github.io/)<sup>\*</sup>, [Zuo-Liang Zhu](https://nk-cs-zzl.github.io/)<sup>\*</sup>, [Ling-Hao Han](https://scholar.google.com/citations?user=0ooNdgUAAAAJ&hl=en), [Qibin Hou](https://scholar.google.com/citations?hl=en&user=fF8OFV8AAAAJ&view_op=list_works), [Chun-Le Guo](https://scholar.google.com/citations?hl=en&user=RZLYwR0AAAAJ),  [Ming-Ming Cheng](https://mmcheng.net/cmm)<br>
> (\* denotes equal contribution) <br>
> Nankai University <br>
> In CVPR 2023<br>

[[Paper](https://arxiv.org/abs/2304.09790)]
[[Project Page](https://nk-cs-zzl.github.io/projects/amt/index.html)]
[[Web demos](#web-demos)]
[Video]

AMT is a **lightweight, fast, and accurate** algorithm for Frame Interpolation. 
It aims to provide practical solutions for **video generation** from **a few given frames (at least two frames)**.

![Demo gif](assets/amt_demo.gif)
* More examples can be found in our [project page](https://nk-cs-zzl.github.io/projects/amt/index.html).

## Web demos
Integrated into [Hugging Face Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo: [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/NKU-AMT/AMT)

Try AMT to interpolate between two or more images at [![PyTTI-Tools:FILM](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IeVO5BmLouhRh6fL2z_y18kgubotoaBq?usp=sharing)


## Change Log
- **Apr 20, 2023**: Our code is publicly available.


## Method Overview
![pipeline](https://user-images.githubusercontent.com/21050959/229420451-65951bd0-732c-4f09-9121-f291a3862d6e.png)

For technical details, please refer to the [method.md](docs/method.md) file, or read the full report on [arXiv](https://arxiv.org/abs/2304.09790).

## Dependencies and Installation
1. Clone Repo

   ```bash
   git clone https://github.com/MCG-NKU/AMT.git
   ```

2. Create Conda Environment and Install Dependencies

   ```bash
   conda env create -f environment.yaml
   conda activate amt
   ```
3. Download pretrained models for demos from [Pretrained Models](#pretrained-models) and place them to the `pretrained` folder

## Quick Demo

**Note that the selected pretrained model (`[CKPT_PATH]`) needs to match the config file (`[CFG]`).**

 > Creating a video demo, increasing $n$ will slow down the motion in the video. (With $m$ input frames, `[N_ITER]` $=n$ corresponds to $2^n\times (m-1)+1$ output frames.)


 ```bash
 python demos/demo_2x.py -c [CFG] -p [CKPT] -n [N_ITER] -i [INPUT] -o [OUT_PATH] -r [FRAME_RATE]
 # e.g. [INPUT]
 # -i could be a video / a regular expression / a folder contains multiple images
 # -i demo.mp4 (video)/img_*.png (regular expression)/img0.png img1.png (images)/demo_input (folder)

 # e.g. a simple usage
 python demos/demo_2x.py -c cfgs/AMT-S.yaml -p pretrained/amt-s.pth -n 6 -i assets/quick_demo/img0.png assets/quick_demo/img1.png

 ```

 + Note: Please enable `--save_images` for saving the output images (Save speed will be slowed down if there are too many output images)
 + Input type supported: `a video` / `a regular expression` / `multiple images` / `a folder containing input frames`.
 + Results are in the `[OUT_PATH]` (default is `results/2x`) folder.

## Pretrained Models

<p id="Pretrained"></p>

<table>
<thead>
  <tr>
    <th> Dataset </th>
    <th> :link: Download Links </th>
    <th> Config file </th>
    <th> Trained on </th>
    <th> Arbitrary/Fixed </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>AMT-S</td>
    <th> [<a href="https://drive.google.com/file/d/1WmOKmQmd6pnLpID8EpUe-TddFpJuavrL/view?usp=share_link">Google Driver</a>][<a href="https://pan.baidu.com/s/1yGaNLeb9TG5-81t0skrOUA?pwd=f66n">Baidu Cloud</a>][<a href="https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth">Hugging Face</a>] </th>
    <th> [<a href="cfgs/AMT-S.yaml">cfgs/AMT-S</a>] </th>
    <th>Vimeo90k</th>
    <th>Fixed</th>
  </tr>
  <tr>
    <td>AMT-L</td>
    <th>[<a href="https://drive.google.com/file/d/1UyhYpAQLXMjFA55rlFZ0kdiSVTL7oU-z/view?usp=share_link">Google Driver</a>][<a href="https://pan.baidu.com/s/1qI4fBgS405Bd4Wn1R3Gbeg?pwd=nbne">Baidu Cloud</a>][<a href="https://huggingface.co/lalala125/AMT/resolve/main/amt-l.pth">Hugging Face</a>]</th>
    <th> [<a href="cfgs/AMT-L.yaml">cfgs/AMT-L</a>] </th>
    <th>Vimeo90k</th>
    <th>Fixed</th>
  </tr>
  <tr>
    <td>AMT-G</td>
    <th>[<a href="https://drive.google.com/file/d/1yieLtKh4ei3gOrLN1LhKSP_9157Q-mtP/view?usp=share_link">Google Driver</a>][<a href="https://pan.baidu.com/s/1AjmQVziQut1bXgQnDcDKvA?pwd=caf6">Baidu Cloud</a>][<a href="https://huggingface.co/lalala125/AMT/resolve/main/amt-g.pth">Hugging Face</a>] </th>
    <th> [<a href="cfgs/AMT-G.yaml">cfgs/AMT-G</a>] </th>
    <th>Vimeo90k</th>
    <th>Fixed</th>
  </tr>
  <tr>
    <td>AMT-S</td>
    <th>[<a href="https://drive.google.com/file/d/1f1xAF0EDm-rjDdny8_aLyeedfM0QL4-C/view?usp=share_link">Google Driver</a>][<a href="https://pan.baidu.com/s/1eZtoULyduQM8AkXeYEBOEw?pwd=8hy3">Baidu Cloud</a>][<a href="https://huggingface.co/lalala125/AMT/resolve/main/gopro_amt-s.pth">Hugging Face</a>] </th>
    <th> [<a href="cfgs/AMT-S_gopro.yaml">cfgs/AMT-S_gopro</a>] </th>
    <th>GoPro</th>
    <th>Arbitrary</th>
  </tr>
</tbody>
</table>

## Training and Evaluation

Please refer to [develop.md](docs/develop.md) to learn how to benchmark the AMT and how to train a new AMT model from scratch.


## Citation
   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
   @inproceedings{licvpr23amt,
      title={AMT: All-Pairs Multi-Field Transforms for Efficient Frame Interpolation},
      author={Li, Zhen and Zhu, Zuo-Liang and Han, Ling-Hao and Hou, Qibin and Guo, Chun-Le and Cheng, Ming-Ming},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2023}
   }
   ```


## License
This code is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for non-commercial use only.
Please note that any commercial use of this code requires formal permission prior to use.

## Contact

For technical questions, please contact `zhenli1031[AT]gmail.com` and `nkuzhuzl[AT]gmail.com`.

For commercial licensing, please contact `cmm[AT]nankai.edu.cn`

## Acknowledgement

We thank Jia-Wen Xiao, Zheng-Peng Duan, Rui-Qi Wu, and Xin Jin for proof reading.
We thank [Zhewei Huang](https://github.com/hzwer) for his suggestions.

Here are some great resources we benefit from:

- [IFRNet](https://github.com/ltkong218/IFRNet) and [RIFE](https://github.com/megvii-research/ECCV2022-RIFE) for data processing, benchmarking, and loss designs.
- [RAFT](https://github.com/princeton-vl/RAFT), [M2M-VFI](https://github.com/feinanshan/M2M_VFI), and [GMFlow](https://github.com/haofeixu/gmflow) for inspirations.
- [FILM](https://github.com/google-research/frame-interpolation) for Web demo reference.


**If you develop/use AMT in your projects, welcome to let us know. We will list your projects in this repository.**

We also thank all of our contributors.

<a href="https://github.com/MCG-NKU/AMT/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=MCG-NKU/AMT" />
</a>

