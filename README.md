![vbench_logo](https://raw.githubusercontent.com/Vchitect/VBench/master/asset/vbench_logo_github_20240605.jpg)

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2311.99999-b31b1b.svg)](https://arxiv.org/abs/2311.99999) -->
[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2311.17982)
[![Project Page](https://img.shields.io/badge/VBench-Website-green?logo=googlechrome&logoColor=green)](https://vchitect.github.io/VBench-project/)
[![Dataset Download](https://img.shields.io/badge/Dataset-Download-red?logo=googlechrome&logoColor=red)](https://drive.google.com/drive/folders/1on66fnZ8atRoLDimcAXMxSwRxqN8_0yS?usp=sharing)
[![PyPI](https://img.shields.io/pypi/v/vbench)](https://pypi.org/project/vbench/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Leaderboard-blue)](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)
[![Video](https://img.shields.io/badge/YouTube-Video-c4302b?logo=youtube&logoColor=red)](https://www.youtube.com/watch?v=7IhCC8Qqn8Y)
[![Visitor](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVchitect%2FVBench&count_bg=%23FFA500&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)


This repository contains the implementation of the following paper and its related serial works in progress. We evaluate video generative models!
> **VBench: Comprehensive Benchmark Suite for Video Generative Models**<br>
> [Ziqi Huang](https://ziqihuangg.github.io/)<sup>∗</sup>, [Yinan He](https://github.com/yinanhe)<sup>∗</sup>, [Jiashuo Yu](https://scholar.google.com/citations?user=iH0Aq0YAAAAJ&hl=zh-CN)<sup>∗</sup>, [Fan Zhang](https://github.com/zhangfan-p)<sup>∗</sup>, [Chenyang Si](https://chenyangsi.top/), [Yuming Jiang](https://yumingj.github.io/), [Yuanhan Zhang](https://zhangyuanhan-ai.github.io/),  [Tianxing Wu](https://tianxingwu.github.io/), [Qingyang Jin](https://github.com/Vchitect/VBench), [Nattapol Chanpaisit](https://nattapolchan.github.io/me), [Yaohui Wang](https://wyhsirius.github.io/), [Xinyuan Chen](https://scholar.google.com/citations?user=3fWSC8YAAAAJ), [Limin Wang](https://wanglimin.github.io), [Dahua Lin](http://dahua.site/)<sup>+</sup>, [Yu Qiao](http://mmlab.siat.ac.cn/yuqiao/index.html)<sup>+</sup>, [Ziwei Liu](https://liuziwei7.github.io/)<sup>+</sup><br>
> IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR**), 2024



### Table of Contents
- [Updates](#updates)
- [Overview](#overview)
- [Evaluation Results](#evaluation_results)
- [Video Generation Models Info](https://github.com/Vchitect/VBench/tree/master/sampled_videos#what-are-the-details-of-the-video-generation-models)
- [Installation](#installation)
- [Usage](#usage)
- [Prompt Suite](#prompt_suite)
- [Sampled Videos](#sampled_videos)
- [Evaluation Method Suite](#evaluation_method_suite)
- [Citation and Acknowledgement](#citation_and_acknowledgement)

<a name="updates"></a>
## :fire: Updates
- [09/2024] **PyPI Updates: PyPI package is updated to version [0.1.4](https://github.com/Vchitect/VBench/releases/tag/v0.1.4):** bug fixes and multi-gpu inference.
- [08/2024] **Longer and More Descriptive Prompts**: [Available Here](https://github.com/Vchitect/VBench/tree/master/prompts/gpt_enhanced_prompts)! We follow [CogVideoX](https://github.com/THUDM/CogVideo?tab=readme-ov-file#prompt-optimization)'s prompt optimization technique to enhance VBench prompts using GPT-4o, making them longer and more descriptive without altering their original meaning.
- [08/2024] **VBench Leaderboard** update: Our leaderboard has 28 *T2V models*, 12 *I2V models* so far. All video generative models are encouraged to participate! [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Leaderboard-blue)](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)
- [06/2024] :fire: **[VBench-Long](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_long)** :fire: is ready to use for evaluating longer Sora-like videos!
- [06/2024] **Model Info Documentation**: Information on video generative models in our [VBench Leaderboard](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard) 
 is documented [HERE](https://github.com/Vchitect/VBench/tree/master/sampled_videos#what-are-the-details-of-the-video-generation-models).
- [05/2024] **PyPI Update**: PyPI package `vbench` is updated to version 0.1.2. This includes changes in the preprocessing for high-resolution images/videos for `imaging_quality`, support for evaluating customized videos, and minor bug fixes.
- [04/2024] We release all the videos we sampled and used for VBench evaluation. [![Dataset Download](https://img.shields.io/badge/Dataset-Download-red?logo=googlechrome&logoColor=red)](https://drive.google.com/drive/folders/13pH95aUN-hVgybUZJBx1e_08R6xhZs5X) See details [here](https://github.com/Vchitect/VBench/tree/master/sampled_videos).
- [03/2024] :fire: **[VBench-Trustworthiness](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_trustworthiness)** :fire: We now support evaluating the **trustworthiness** (*e.g.*, culture, fairness, bias, safety) of video generative models.
- [03/2024] :fire: **[VBench-I2V](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v)** :fire: We now support evaluating **Image-to-Video (I2V)** models. We also provide [Image Suite](https://drive.google.com/drive/folders/1fdOZKQ7HWZtgutCKKA7CMzOhMFUGv4Zx?usp=sharing).
- [03/2024] We support **evaluating customized videos**! See [here](https://github.com/Vchitect/VBench/?tab=readme-ov-file#new-evaluate-your-own-videos) for instructions.
- [01/2024] PyPI package is released! [![PyPI](https://img.shields.io/pypi/v/vbench)](https://pypi.org/project/vbench/). Simply `pip install vbench`.
- [12/2023] :fire: **[VBench](https://github.com/Vchitect/VBench?tab=readme-ov-file#usage)** :fire: Evaluation code released for 16 **Text-to-Video (T2V) evaluation** dimensions. 
    - `['subject_consistency', 'background_consistency', 'temporal_flickering', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality', 'object_class', 'multiple_objects', 'human_action', 'color', 'spatial_relationship', 'scene', 'temporal_style', 'appearance_style', 'overall_consistency']`
- [11/2023] Prompt Suites released. (See prompt lists [here](https://github.com/Vchitect/VBench/tree/master/prompts))
  
<a name="overview"></a>
## :mega: Overview
![overall_structure](./asset/fig_teaser_new.jpg)
We propose **VBench**, a comprehensive benchmark suite for video generative models. We design a comprehensive and hierarchical <b>Evaluation Dimension Suite</b> to decompose "video generation quality" into multiple well-defined dimensions to facilitate fine-grained and objective evaluation. For each dimension and each content category, we carefully design a <b>Prompt Suite</b> as test cases, and sample <b>Generated Videos</b> from a set of video generation models. For each evaluation dimension, we specifically design an <b>Evaluation Method Suite</b>, which uses carefully crafted method or designated pipeline for automatic objective evaluation. We also conduct <b>Human Preference Annotation</b> for the generated videos for each dimension, and show that VBench evaluation results are <b>well aligned with human perceptions</b>. VBench can provide valuable insights from multiple perspectives.

<a name="evaluation_results"></a>
## :mortar_board: Evaluation Results

***See our leaderboard for the most updated ranking and numerical results (with models like Gen-3, Kling, Pika)***. [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Leaderboard-blue)](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)

<p align="center">
  <img src="./asset/radar-open.jpg" width="48%" style="margin-right: 4%;" />
  <img src="./asset/radar-close.jpg" width="48%" />
</p>

We visualize VBench evaluation results of various publicly available video generation models, as well as Gen-2 and Pika, across 16 VBench dimensions. We normalize the results per dimension for clearer comparisons. 

#### :trophy: Leaderboard

See numeric values at our [Leaderboard](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard) :1st_place_medal::2nd_place_medal::3rd_place_medal:

**How to join VBench Leaderboard?**
See the 3 options below:
| Sampling Party | Evaluation Party |              Comments                         |
| :---: |  :---: |        :--------------    | 
| VBench Team | VBench Team | We periodically allocate resources to sample newly released models and perform evaluations. You can request us to perform sampling and evaluation, but the progress depends on our available resources. |
| Your Team | VBench Team | For non-open-source models interested in joining our leaderboard, submit your video samples to us for evaluation. If you prefer to provide the evaluation results directly, see the row below. |
| Your Team | Your Team | If you have already used VBench for full evaluation in your report/paper, submit your `eval_results.zip` files to the [VBench Leaderboard](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard) using the `Submit here!` form. The evaluation results will be automatically updated to the leaderboard. Also, share your model information for our records for any columns [here](https://github.com/Vchitect/VBench/tree/master/sampled_videos#what-are-the-details-of-the-video-generation-models).  |


#### :film_projector: Model Info
See [model info](https://github.com/Vchitect/VBench/tree/master/sampled_videos#what-are-the-details-of-the-video-generation-models) for video generation models we used for evaluation.

<!-- The values have been normalized for better readability of the chart. The normalization process involves scaling each set of performance values to a common scale between 0.3 and 0.8. The formula used for normalization is: (value - min value) / (max value - min value). -->

<a name="installation"></a>
## :hammer: Installation
### Install with pip
```
pip install vbench
```

To evaluate some video generation ability aspects, you need to install [detectron2](https://github.com/facebookresearch/detectron2) via:
   ```
   pip install detectron2@git+https://github.com/facebookresearch/detectron2.git
   ```
    
If there is an error during [detectron2](https://github.com/facebookresearch/detectron2) installation, see [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

Download [VBench_full_info.json](https://github.com/Vchitect/VBench/blob/master/vbench/VBench_full_info.json) to your running directory to read the benchmark prompt suites.

### Install with git clone
    git clone https://github.com/Vchitect/VBench.git
    pip install -r VBench/requirements.txt
    pip install VBench
    
If there is an error during [detectron2](https://github.com/facebookresearch/detectron2) installation, see [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

<a name="usage"></a>
## Usage
Use VBench to evaluate videos, and video generative models.
- A Side Note: VBench is designed for evaluating different models on a standard benchmark. Therefore, by default, we enforce evaluation on the **standard VBench prompt lists** to ensure **fair comparisons** among different video generation models. That's also why we give warnings when a required video is not found. This is done via defining the set of prompts in [VBench_full_info.json](https://github.com/Vchitect/VBench/blob/master/vbench/VBench_full_info.json). However, we understand that many users would like to use VBench to evaluate their own videos, or videos generated from prompts that does not belong to the VBench Prompt Suite, so we also added the function of **Evaluating Your Own Videos**. Simply set `mode=custom_input`, and you can evaluate your own videos.


### **[New]** Evaluate Your Own Videos
We support evaluating any video. Simply provide the path to the video file, or the path to the folder that contains your videos. There is no requirement on the videos' names.
- Note: We support customized videos / prompts for the following dimensions: `'subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality'`


To evaluate videos with customized input prompt, run our script with `--mode=custom_input`:
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

To evaluate using multiple gpus, we can use the following commands:
```
torchrun --nproc_per_node=${GPUS} --standalone evaluate.py ...args...
```
or 
```
vbench evaluate --ngpus=${GPUS} ...args...
```

### Evaluation on the Standard Prompt Suite of VBench

##### Command Line 
```bash
vbench evaluate --videos_path $VIDEO_PATH --dimension $DIMENSION
```
For example:
```bash
vbench evaluate --videos_path "sampled_videos/lavie/human_action" --dimension "human_action"
```
##### Python
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

### Evaluation of Different Content Categories

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

### Example of Evaluating VideoCrafter-1.0
We have provided scripts to download VideoCrafter-1.0 samples, and the corresponding evaluation scripts.
```
# download sampled videos
sh scripts/download_videocrafter1.sh

# evaluate VideoCrafter-1.0
sh scripts/evaluate_videocrafter1.sh
```
#### Get Final Score and Submit to Leaderboard
We have provided scripts for calculating the `Final Score`, `Quality Score`, and `Semantic Score` in the Leaderboard. You can run them locally to obtain the final scores or as a final check before submitting to the Leaderboard.
##### command line 
```bash
# Pack the evaluation results into a zip file.
cd evaluation_results
zip -r ../evaluation_results.zip .

# [Optional] get the final score of your submission file.
python scripts/cal_final_score.py --zip_file {path_to_evaluation_results.zip} --model_name {your_model_name}
```
You can submit the json file to [HuggingFace](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)


<a name="pretrained_models"></a>
## :gem: Pre-Trained Models
[Optional] Please download the pre-trained weights according to the guidance in the `model_path.txt` file for each model in the `pretrained` folder to `~/.cache/vbench`.

<a name="prompt_suite"></a>
## :bookmark_tabs: Prompt Suite

We provide prompt lists are at `prompts/`. 

Check out [details of prompt suites](https://github.com/Vchitect/VBench/tree/master/prompts), and instructions for [**how to sample videos for evaluation**](https://github.com/Vchitect/VBench/tree/master/prompts).

<a name="sampled_videos"></a>
## :bookmark_tabs: Sampled Videos

[![Dataset Download](https://img.shields.io/badge/Dataset-Download-red?logo=googlechrome&logoColor=red)](https://drive.google.com/drive/folders/13pH95aUN-hVgybUZJBx1e_08R6xhZs5X)

To facilitate future research and to ensure full transparency, we release all the videos we sampled and used for VBench evaluation. You can download them on [Google Drive](https://drive.google.com/drive/folders/13pH95aUN-hVgybUZJBx1e_08R6xhZs5X).

See detailed explanations of the sampled videos [here](https://github.com/Vchitect/VBench/tree/master/sampled_videos).

We also provide detailed setting for the models under evaluation [here](https://github.com/Vchitect/VBench/tree/master/sampled_videos#what-are-the-details-of-the-video-generation-models).

<a name="evaluation_method_suite"></a>
## :surfer: Evaluation Method Suite

To perform evaluation on one dimension, run this:
```
python evaluate.py --videos_path $VIDEOS_PATH --dimension $DIMENSION
```
- The complete list of dimensions:
    ```
    ['subject_consistency', 'background_consistency', 'temporal_flickering', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality', 'object_class', 'multiple_objects', 'human_action', 'color', 'spatial_relationship', 'scene', 'temporal_style', 'appearance_style', 'overall_consistency']
    ```

Alternatively, you can evaluate multiple models and multiple dimensions using this script:
```
bash evaluate.sh
```
- The default sampled video paths:
    ```
    vbench_videos/{model}/{dimension}/{prompt}-{index}.mp4/gif
    ```



#### Before evaluating the temporal flickering dimension, it is necessary to filter out the static videos first.
To filter static videos in the temporal flickering dimension, run this:
```
# This only filter out static videos whose prompt matches the prompt in the temporal_flickering.
python static_filter.py --videos_path $VIDEOS_PATH
```
You can adjust the filtering scope by:
```
# 1. Change the filtering scope to consider all files inside videos_path for filtering.
python static_filter.py --videos_path $VIDEOS_PATH --filter_scope all

# 2. Specify the path to a JSON file ($filename) to consider only videos whose prompts match those listed in $filename.
python static_filter.py --videos_path $VIDEOS_PATH --filter_scope $filename
```

<a name="citation_and_acknowledgement"></a>
## :black_nib: Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
    @InProceedings{huang2023vbench,
        title={{VBench}: Comprehensive Benchmark Suite for Video Generative Models},
        author={Huang, Ziqi and He, Yinan and Yu, Jiashuo and Zhang, Fan and Si, Chenyang and Jiang, Yuming and Zhang, Yuanhan and Wu, Tianxing and Jin, Qingyang and Chanpaisit, Nattapol and Wang, Yaohui and Chen, Xinyuan and Wang, Limin and Lin, Dahua and Qiao, Yu and Liu, Ziwei},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        year={2024}
    }
   ```


## :hearts: Acknowledgement

#### :muscle: VBench Contributors
Order is based on the time joining the project: 
> [Ziqi Huang](https://ziqihuangg.github.io/), [Yinan He](https://github.com/yinanhe), [Jiashuo Yu](https://scholar.google.com/citations?user=iH0Aq0YAAAAJ&hl=zh-CN), [Fan Zhang](https://github.com/zhangfan-p), [Nattapol Chanpaisit](https://nattapolchan.github.io/me), [Xiaojie Xu](https://github.com/xjxu21), [Qianli Ma](https://github.com/MqLeet), [Ziyue Dong](https://github.com/DZY-irene).

#### :hugs: Open-Sourced Repositories
This project wouldn't be possible without the following open-sourced repositories:
[AMT](https://github.com/MCG-NKU/AMT/), [UMT](https://github.com/OpenGVLab/unmasked_teacher), [RAM](https://github.com/xinyu1205/recognize-anything), [CLIP](https://github.com/openai/CLIP), [RAFT](https://github.com/princeton-vl/RAFT), [GRiT](https://github.com/JialianW/GRiT), [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch/), [ViCLIP](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid), and [LAION Aesthetic Predictor](https://github.com/LAION-AI/aesthetic-predictor).

## Related Links

We are putting together [Awesome-Evaluation-of-Visual-Generation](https://github.com/ziqihuangg/Awesome-Evaluation-of-Visual-Generation), which collects works for evaluating visual generation.
