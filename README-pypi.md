![vbench_logo](https://raw.githubusercontent.com/Vchitect/VBench/master/asset/vbench_logo_short.jpg)

**VBench** is a comprehensive benchmark suite for video generative models. You can use **VBench** to evaluate video generation models from 16 different ability aspects.

This project is the PyPI implementation of the following research:
> **VBench: Comprehensive Benchmark Suite for Video Generative Models**<br>
> [Ziqi Huang](https://ziqihuangg.github.io/)<sup>∗</sup>, [Yinan He](https://github.com/yinanhe)<sup>∗</sup>, [Jiashuo Yu](https://scholar.google.com/citations?user=iH0Aq0YAAAAJ&hl=zh-CN)<sup>∗</sup>, [Fan Zhang](https://github.com/zhangfan-p)<sup>∗</sup>, [Chenyang Si](https://chenyangsi.top/), [Yuming Jiang](https://yumingj.github.io/), [Yuanhan Zhang](https://zhangyuanhan-ai.github.io/),  [Tianxing Wu](https://tianxingwu.github.io/), [Qingyang Jin](https://github.com/Vchitect/VBench), [Nattapol Chanpaisit](https://nattapolchan.github.io/me), [Yaohui Wang](https://wyhsirius.github.io/), [Xinyuan Chen](https://scholar.google.com/citations?user=3fWSC8YAAAAJ), [Limin Wang](https://wanglimin.github.io), [Dahua Lin](http://dahua.site/)<sup>+</sup>, [Yu Qiao](http://mmlab.siat.ac.cn/yuqiao/index.html)<sup>+</sup>, [Ziwei Liu](https://liuziwei7.github.io/)<sup>+</sup><br>

[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2311.17982)
[![Project Page](https://img.shields.io/badge/VBench-Website-green?logo=googlechrome&logoColor=green)](https://vchitect.github.io/VBench-project/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Leaderboard-blue)](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)
[![Video](https://img.shields.io/badge/YouTube-Video-c4302b?logo=youtube&logoColor=red)](https://www.youtube.com/watch?v=7IhCC8Qqn8Y)
[![Visitor](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVchitect%2FVBench&count_bg=%23FFA500&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)

## Installation
```
pip install vbench
```

To evaluate some video generation ability aspects, you need to install [detectron2](https://github.com/facebookresearch/detectron2) via:
   ```
   pip install detectron2@git+https://github.com/facebookresearch/detectron2.git
   ```
    
If there is an error during [detectron2](https://github.com/facebookresearch/detectron2) installation, see [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

## Usage

### Evaluate Your Own Videos
We support evaluating any video. Simply provide the path to the video file, or the path to the folder that contains your videos. There is no requirement on the videos' names.
- Note: We support customized videos / prompts for the following dimensions: `'subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality'`


To evaluate videos with customed input prompt, run our script with `--mode=custom_input`:
```
python evaluate.py \
    --dimension $DIMENSION \
    --videos_path /path/to/folder_or_video/ \
    --mode=custom_input
```
alternatively you can use our command:
```
vbench evaluate \
    --dimension $DIMENSION \
    --videos_path /path/to/folder_or_video/ \
    --mode=custom_input
```

### Evaluation on the Standard Prompt Suite of VBench

##### command line 
```bash
    vbench evaluate --videos_path $VIDEO_PATH --dimension $DIMENSION
```
For example:
```bash
    vbench evaluate --videos_path "sampled_videos/lavie/human_action" --dimension "human_action"
```
##### python
```python
    from vbench import VBench
    my_VBench = VBench(device, <path/to/VBench_full_info.json>, <path/to/save/dir>)
    my_VBench.evaluate(
        videos_path = <video_path>,
        name = <name>,
        dimension_list = [<dimension>, <dimension>, ...],
    )
```
For example: 
```python
    from vbench import VBench
    my_VBench = VBench(device, "vbench/VBench_full_info.json", "evaluation_results")
    my_VBench.evaluate(
        videos_path = "sampled_videos/lavie/human_action",
        name = "lavie_human_action",
        dimension_list = ["human_action"],
    )
```

### Evaluation on a specific category from VBench

##### command line 
```bash
vbench evaluate \
    --videos_path $VIDEO_PATH \
    --dimension $DIMENSION \
    --mode=vbench_category \
    --category=$CATEGORY
```
or 
```
python evaluate.py \
    --dimension $DIMENSION \
    --videos_path /path/to/folder_or_video/ \
    --mode=vbench_category
```

## Prompt Suite

We provide prompt lists are at `prompts/`. 

Check out [details of prompt suites](https://github.com/Vchitect/VBench/tree/master/prompts), and instructions for [**how to sample videos for evaluation**](https://github.com/Vchitect/VBench/tree/master/prompts).

## Citation

   If you find this package useful for your reports or publications, please consider citing the VBench paper:

   ```bibtex
    @article{huang2023vbench,
        title={{VBench}: Comprehensive Benchmark Suite for Video Generative Models},
        author={Huang, Ziqi and He, Yinan and Yu, Jiashuo and Zhang, Fan and Si, Chenyang and Jiang, Yuming and Zhang, Yuanhan and Wu, Tianxing and Jin, Qingyang and Chanpaisit, Nattapol and Wang, Yaohui and Chen, Xinyuan and Wang, Limin and Lin, Dahua and Qiao, Yu and Liu, Ziwei},
        journal={arXiv preprint arXiv:2311.17982},
        year={2023}
    }
   ```
