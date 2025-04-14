![vbench_logo](https://raw.githubusercontent.com/Vchitect/VBench/master/asset/vbench_logo_short.jpg)

**VBench-2.0** is a comprehensive benchmark suite for video generative models. You can use **VBench-2.0** to evaluate video generation models from 18 different ability aspects.

This project is the PyPI implementation of the following research:
> **VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness**<br>
> [Dian Zheng](https://zhengdian1.github.io/)<sup>∗</sup>, [Ziqi Huang](https://ziqihuangg.github.io/)<sup>∗</sup>, [Hongbo Liu](https://github.com/Alexios-hub), [Kai Zou](https://github.com/Jacky-hate), [Yinan He](https://github.com/yinanhe), [Fan Zhang](https://github.com/zhangfan-p), [Yuanhan Zhang](https://zhangyuanhan-ai.github.io/),  [Jingwen He](https://scholar.google.com/citations?user=GUxrycUAAAAJ&hl=zh-CN), [Wei-Shi Zheng](https://www.isee-ai.cn/~zhwshi/)<sup>+</sup>, [Yu Qiao](http://mmlab.siat.ac.cn/yuqiao/index.html)<sup>+</sup>, [Ziwei Liu](https://liuziwei7.github.io/)<sup>+</sup><br>



[![Paper](https://img.shields.io/badge/VBench-2.0%20Report-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2503.21755)
[![Project Page](https://img.shields.io/badge/Project-Page-green?logo=googlechrome&logoColor=green)](https://vchitect.github.io/VBench-2.0-project/)
[![Video](https://img.shields.io/badge/YouTube-Video-c4302b?logo=youtube&logoColor=red)](https://www.youtube.com/watch?v=kJrzKy9tgAc)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Leaderboard-blue)](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)
[![Visitor](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVchitect%2FVBench&count_bg=%23FFA500&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)

## Installation
```
conda install psutil
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install ninja
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.2.post1
pip install vbench2
pip install -e git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1#egg=clip
pip install -e git+https://github.com/LLaVA-VL/LLaVA-NeXT@79ef45a6d8b89b92d7a8525f077c3a3a9894a87d#egg=llava
pip install -e git+https://github.com/AILab-CVC/YOLO-World.git@4826695a84ce370dad7fe4cb555a6c7162f3751d#egg=yolo_world
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html --no-cache-dir
pip install retinaface_pytorch==0.0.8 --no-deps
```

## Usage

<!-- ### Evaluate Your Own Videos
We support evaluating any video. Simply provide the path to the video file, or the path to the folder that contains your videos. There is no requirement on the videos' names. -->
<!-- - Note: We support customized videos / prompts for the following dimensions: `'subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality'` -->


<!-- To evaluate videos with customed input prompt, run our script with `--mode=custom_input`:
```bash
python evaluate.py \
    --dimension $DIMENSION \
    --videos_path /path/to/folder_or_video/ \
    --mode=custom_input
```
alternatively you can use our command:
```bash
vbench evaluate \
    --dimension $DIMENSION \
    --videos_path /path/to/folder_or_video/ \
    --mode=custom_input
``` -->

### Evaluation on the Standard Prompt Suite of VBench-2.0

##### command line 
```bash
    vbench2 evaluate --videos_path $VIDEO_PATH --dimension $DIMENSION
```
For example:
```bash
    vbench2 evaluate --videos_path "sampled_videos/HunyuanVideo/Human_Interaction" --dimension "Human_Interaction"
```
##### python
```python
    from vbench2 import VBench2
    my_VBench = VBench2(device, <path/to/VBench_full_info.json>, <path/to/save/dir>)
    my_VBench.evaluate(
        videos_path = <video_path>,
        name = <name>,
        dimension_list = [<dimension>, <dimension>, ...],
    )
```
For example: 
```python
    from vbench2 import VBench2
    my_VBench = VBench2(device, "vbench2/VBench2_full_info.json", "evaluation_results")
    my_VBench.evaluate(
        videos_path = "sampled_videos/HunyuanVideo/Human_Interaction",
        name = "HunyuanVideo_Human_Interaction",
        dimension_list = ["Human_Interaction"],
    )
```

## Prompt Suite

We provide prompt lists are at `prompts/`. 

Check out [details of prompt suites](https://github.com/Vchitect/VBench/tree/master/VBench-2.0/prompts), and instructions for [**how to sample videos for evaluation**](https://github.com/Vchitect/VBench/tree/master/VBench-2.0/prompts).

## Citation

   If you find this package useful for your reports or publications, please consider citing the VBench-2.0 paper:

   ```bibtex
    @article{zheng2025vbench2,
        title={{VBench-2.0}: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness},
        author={Zheng, Dian and Huang, Ziqi and Liu, Hongbo and Zou, Kai and He, Yinan and Zhang, Fan and Zhang, Yuanhan and He, Jingwen and Zheng, Wei-Shi and Qiao, Yu and Liu, Ziwei},
        journal={arXiv preprint arXiv:2503.21755},
        year={2025}
    }
   ```
