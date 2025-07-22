![vbench_logo](https://raw.githubusercontent.com/Vchitect/VBench/master/asset/vbench_logo_github_20240605.jpg)

**How to Reach Us:**
- Code Issues: Please open an [issue](https://github.com/Vchitect/VBench/issues) in our GitHub repository for any problems or bugs.
- Evaluation Requests: To submit your sampled videos for evaluation, please complete this [Google Form](https://forms.gle/rjH6hmAHpZhRGdkv5).
- General Inquiries: **Check our [FAQ](https://github.com/Vchitect/VBench/blob/master/VBench-2.0/README-FAQ.md)** for common questions. For other questions, contact Dian Zheng at zhengd35 [at] mail2 [at] sysu [at] edu [at] cn or Ziqi Huang at ZIQI002 [at] e [dot] ntu [dot] edu [dot] sg. 

[![VBench-2.0 Report (arXiv)](https://img.shields.io/badge/VBench-2.0%20Report-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2503.21755)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Leaderboard-blue)](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)
[![Project Page](https://img.shields.io/badge/Project-Page-green?logo=googlechrome&logoColor=green)](https://vchitect.github.io/VBench-2.0-project/)
[![Video](https://img.shields.io/badge/YouTube-Video-c4302b?logo=youtube&logoColor=red)](https://www.youtube.com/watch?v=kJrzKy9tgAc)
[![VBench-2.0 Arena (View Generated Videos Here!)](https://img.shields.io/badge/%F0%9F%A4%97%20VBench2.0%20Arena-blue)](https://huggingface.co/spaces/Vchitect/VBench2.0_Video_Arena)
[![Dataset Download](https://img.shields.io/badge/Dataset-Download-red?logo=googlechrome&logoColor=red)](https://drive.google.com/drive/folders/1on66fnZ8atRoLDimcAXMxSwRxqN8_0yS?usp=sharing)

This repository contains the implementation of the following paper.

> **VBench-2.0: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness**<br>
> [Dian Zheng](https://zhengdian1.github.io/)<sup>∗</sup>, [Ziqi Huang](https://ziqihuangg.github.io/)<sup>∗</sup>, [Hongbo Liu](https://github.com/Alexios-hub), [Kai Zou](https://github.com/Jacky-hate), [Yinan He](https://github.com/yinanhe), [Fan Zhang](https://github.com/zhangfan-p), [Yuanhan Zhang](https://zhangyuanhan-ai.github.io/),  [Jingwen He](https://scholar.google.com/citations?user=GUxrycUAAAAJ&hl=zh-CN), [Wei-Shi Zheng](https://www.isee-ai.cn/~zhwshi/)<sup>+</sup>, [Yu Qiao](http://mmlab.siat.ac.cn/yuqiao/index.html)<sup>+</sup>, [Ziwei Liu](https://liuziwei7.github.io/)<sup>+</sup><br>


### Table of Contents
- [Updates](#updates)
- [Overview](#overview)
- [Evaluation Results](#evaluation_results)
- [Video Generation Models Info](https://github.com/Vchitect/VBench/tree/master/VBench-2.0/sampled_videos#what-are-the-details-of-the-video-generation-models)
- [Installation](#installation)
- [Pretrained Models](#pretrained_models)
- [Usage](#usage)
- [Prompt Suite](#prompt_suite)
- [Sampled Videos](#sampled_videos)
- [Citation and Acknowledgement](#citation_and_acknowledgement)

<a name="updates"></a>
## :fire: Updates
- [05/2025] We support **evaluating customized videos**! See [here](https://github.com/Vchitect/VBench/tree/master/VBench-2.0#new-evaluating-single-dimension-of-your-own-videos) for instructions.
- [04/2025] **[Human Anomaly Detection for AIGC Videos](https://github.com/Vchitect/VBench/tree/master/VBench-2.0/vbench2/third_party/ViTDetector):** We release the pipeline for evaluating human anatomical quality in AIGC videos, including a manually human anomaly dataset on real and AIGC videos, and the training pipeline for anomaly detection.
- [03/2025] :fire: **Major Update! We released [VBench-2.0](https://github.com/Vchitect/VBench/tree/master/VBench-2.0)!** :fire: Video generative models have progressed from achieving *superficial faithfulness* in fundamental technical aspects such as pixel fidelity and basic prompt adherence, to addressing more complex challenges associated with *intrinsic faithfulness*, including commonsense reasoning, physics-based realism, human motion, and creative composition. While VBench primarily assessed early-stage technical quality, VBench-2.0 expands the benchmarking framework to evaluate these advanced capabilities, ensuring a more comprehensive assessment of next-generation models.

<a name="overview"></a>
## :mega: Overview
![overall_structure](./asset/fig_paper_teaser.jpg)
Overview of VBench-2.0. (a) Scope of VBench-2.0. Video generative models have progressed from achieving superficial faithfulness in fundamental technical aspects such as pixel fidelity and basic prompt adherence, to addressing more complex challenges associated with intrinsic faithfulness, including commonsense reasoning, physics-based realism, human motion, and creative composition. While VBench primarily assessed early-stage technical quality, VBench-2.0 expands the benchmarking framework to evaluate these advanced capabilities, ensuring a more comprehensive assessment of next-generation models. (b) Evaluation Dimension of VBench-2.0. VBench-2.0 introduces a structured evaluation suite comprising five broad categories and 18 fine-grained capability dimensions.

<a name="evaluation_results"></a>
## :mortar_board: Evaluation Results

<p align="center">
  <img src="./asset/all_dimension.jpg" width="65%"/>
</p>
We visualize the evaluation results of the 6 recent top-performing video generative models across 18 VBench-2.0 dimensions.

<a name="installation"></a>
## :hammer: Installation
### Install with pip 
```
conda create -n vbench2 python=3.10 -y
conda activate vbench2
conda install psutil
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118 
#ensure your cuda version is 11.8, and ensure the gcc in a high version (we use 11.2)
python -m pip install ninja
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.7.2.post1
pip install -r requirement.txt
pip install retinaface_pytorch==0.0.8 --no-deps
#ensure you have set up the ffmpeg with the version 4.x
cd vbench2/third_party/Instance_detector
pip install -e .
cd ../../..
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.4/index.html --no-cache-dir
```
There will be an error about the mmcv version exceeds 2.1.0, users could directly change the `mmcv_maximum_version` to `2.3.0` in `Your_PATH_to_Anaconda/env/vbench2/lib/python3.10/site-packages/mmdet/__init__.py` and `Your_PATH_to_Anaconda/env/vbench2/lib/python3.10/site-packages/mmyolo/__init__.py`

<a name="pretrained_models"></a>
## Pretrained Models
[Optional] Please download the pre-trained weights according to the guidance in the `download.sh` file for each model in the `pretrained` folder to `~/.cache/vbench2`.
*Note that for `LLaVA-Video-7B-Qwen2` and `Qwen2.5-7B-Instruct`, we highly recommend the users to download the model before evaluation based on the `download.sh` in `pretrained`*


<a name="usage"></a>
## Usage
Use VBench-2.0 to evaluate videos, and video generative models.
- A Side Note: VBench-2.0 is designed for evaluating different models on a standard benchmark. Therefore, by default, we enforce evaluation on the **standard VBench-2.0 prompt lists** to ensure **fair comparisons** among different video generation models. That's also why we give warnings when a required video is not found. This is done via defining the set of prompts in [VBench2_full_info.json](https://github.com/Vchitect/VBench/blob/master/VBench-2.0/vbench2/VBench2_full_info.json). 

### Evaluating All of the Dimensions on the Standard Prompt Suite of VBench-2.0

Evaluate 18 dimensions sequentially on a single GPU (not recommended):
```bash
bash evaluate.sh --max_parallel_tasks 1
```
Evaluate 18 dimensions on different GPUs, one dimension per GPU. You can set the total number of GPUs available for evaluation (do not exceed 8):
```bash
bash evaluate.sh --max_parallel_tasks 8
```

### **[New]** Evaluating Single Dimension of Your Own Videos
We support evaluating any video. Simply provide the path to the video file. There is no requirement on the videos' names except `Diversity` dimension.
- Note: We support customized videos / prompts for the following dimensions: `'Human_Anatomy', 'Human_Identity', 'Human_Clothes', 'Diversity', 'Multi-View_Consistency'`
- Note: `Diversity` requires at least 20 videos for one prompt to robustly test the diversity of your model and the naming format should be `prompt-index.mp4` where index ranges from 0 to 19
- Note: `Diversity` and `Multi-View_Consistency` should be placed in separate folders while `'Human_Anatomy', 'Human_Identity', 'Human_Clothes'` can share a folder (save your time). 

To evaluate videos with customized input prompt, run our script with `--mode=custom_input`:
```bash
python evaluate.py \
    --dimension $DIMENSION \
    --videos_path /path/to/folder_or_video/ \
    --output_path "evaluation_results/$DIMENSION" \
    --mode=custom_input
```
For example:
```bash
python evaluate.py \
    --dimension "Human_Anatomy" \
    --videos_path "YOUR_VIDEO_PATH" \
    --output_path "evaluation_results/Human_Anatomy" \
    --mode=custom_input
```

### Evaluating Single Dimension on the Standard Prompt Suite of VBench-2.0

<!-- ##### command line 
```bash
    vbench2 evaluate --videos_path $VIDEO_PATH --dimension $DIMENSION
```
For example:
```bash
    vbench2 evaluate --videos_path "sampled_videos/HunyuanVideo/Human_Interaction" --dimension "Human_Interaction"
``` -->
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

### Use Human Anomaly Detector of VBench-2.0

Detecting human anomalies is crucial for video generation, as it can be used for evaluation or as feedback for fine-tuning. We fully release the training data (including unlabeled raw videos and labeled images), along with the training and inference code of it. See [Human_Anomaly](https://github.com/Vchitect/VBench/blob/master/VBench-2.0/vbench2/third_party/ViTDetector) for details.

### Submit to Leaderboard
We have provided scripts for calculating the `Total Score`, `Creativity Score`, `Commonsense Score`, `Controllability Score`, `Human Fidelity Score` and `Physics Score` in the Leaderboard. You can run them locally to obtain the aggregate scores or as a final check before submitting to the Leaderboard.

```bash
# Pack the evaluation results into a zip file.
cd evaluation_results
zip -r ../evaluation_results.zip .

# [Optional] get the total score of your submission file.
python scripts/cal_final_score.py --zip_file {path_to_evaluation_results.zip} --model_name {your_model_name}
```

You can submit the json file to [HuggingFace](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)

### How to Calculate Total Score

To calculate the **Total Score**, we follow these steps:

1. **Creativity Score**:  
   The `Creativity Score` is the average of the following dimensions:  
   **Diversity**, and **Composition**.

2. **Commonsense Score**:  
   The `Commonsense Score` is the average of the following dimensions:  
   **Motion Rationality**, and **Instance Preservation**.

3. **Controllability Score**:  
   The `Controllability Score` is the average of the following dimensions:  
   **Dynamic Spatial Relationship**, **Dynamic Attribute**, **Motion Order Understanding**, **Human Interaction**, **Complex Landscape**, **Complex Plot**, and **Camera Motion**.

4. **Human Fidelity Score**:  
   The `Human Fidelity Score` is the average of the following dimensions:  
   **Human Anatomy**, **Human Identity**, and **Human Clothes**.

5. **Physics Score**:  
   The `Physics Score` is the average of the following dimensions:  
   **Mechanics**, **Thermotics**, **Material**, and **Multi-View Consistency**.

6. **Average Calculation**:  
   The **Total Score** is the average of the `Creativity Score`, `Commonsense Score`, `Controllability Score`, `Human Fidelity Score` and `Physics Score`:
    ```bash
    Total Score = 0.2 * (Creativity Score + Commonsense Score + Controllability Score + Human Fidelity Score + Physics Score)
    ```

<a name="prompt_suite"></a>
## :bookmark_tabs: Prompt Suite

We provide prompt lists are at `prompts/`. 

Check out [details of prompt suites](https://github.com/Vchitect/VBench/tree/master/VBench-2.0/prompts), and instructions for [**how to sample videos for evaluation**](https://github.com/Vchitect/VBench/tree/master/VBench-2.0/prompts#how-to-sample-videos-for-evaluation).


<a name="sampled_videos"></a>
## :bookmark_tabs: Sampled Videos

[![Dataset Download](https://img.shields.io/badge/Dataset-Download-red?logo=googlechrome&logoColor=red)](https://drive.google.com/drive/folders/19nXdrNFw-PxxYJcd9HyWmbLVe5Agmlc2)

To facilitate future research and to ensure full transparency, we release all the videos we sampled and used for VBench-2.0 evaluation. You can download them on [Google Drive](https://drive.google.com/drive/folders/19nXdrNFw-PxxYJcd9HyWmbLVe5Agmlc2).

See detailed explanations of the sampled videos [here](https://github.com/Vchitect/VBench/tree/master/VBench-2.0/sampled_videos).

We also provide detailed setting for the models under evaluation [here](https://github.com/Vchitect/VBench/tree/master/VBench-2.0/sampled_videos#what-are-the-details-of-the-video-generation-models).


<a name="citation_and_acknowledgement"></a>
## :black_nib: Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
    @article{zheng2025vbench2,
        title={{VBench-2.0}: Advancing Video Generation Benchmark Suite for Intrinsic Faithfulness},
        author={Zheng, Dian and Huang, Ziqi and Liu, Hongbo and Zou, Kai and He, Yinan and Zhang, Fan and Zhang, Yuanhan and He, Jingwen and Zheng, Wei-Shi and Qiao, Yu and Liu, Ziwei},
        journal={arXiv preprint arXiv:2503.21755},
        year={2025}
    }

    @InProceedings{huang2023vbench,
        title={{VBench}: Comprehensive Benchmark Suite for Video Generative Models},
        author={Huang, Ziqi and He, Yinan and Yu, Jiashuo and Zhang, Fan and Si, Chenyang and Jiang, Yuming and Zhang, Yuanhan and Wu, Tianxing and Jin, Qingyang and Chanpaisit, Nattapol and Wang, Yaohui and Chen, Xinyuan and Wang, Limin and Lin, Dahua and Qiao, Yu and Liu, Ziwei},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        year={2024}
    }

    @article{huang2024vbench++,
        title={{VBench++}: Comprehensive and Versatile Benchmark Suite for Video Generative Models},
        author={Huang, Ziqi and Zhang, Fan and Xu, Xiaojie and He, Yinan and Yu, Jiashuo and Dong, Ziyue and Ma, Qianli and Chanpaisit, Nattapol and Si, Chenyang and Jiang, Yuming and Wang, Yaohui and Chen, Xinyuan and Chen, Ying-Cong and Wang, Limin and Lin, Dahua and Qiao, Yu and Liu, Ziwei},
        journal={arXiv preprint arXiv:2411.13503},
        year={2024}
    }

   ```


## :hearts: Acknowledgement

**VBench-2.0** is currently maintained by [Dian Zheng](https://zhengdian1.github.io/) and [Ziqi Huang](https://ziqihuangg.github.io/).


#### :hugs: Open-Sourced Repositories
This project wouldn't be possible without the following open-sourced repositories:
[LLaVA-Video](https://github.com/LLaVA-VL/LLaVA-NeXT), [Qwen2.5](https://github.com/QwenLM/Qwen2.5), [CoTracker](https://github.com/facebookresearch/co-tracker), [Insightface](https://github.com/deepinsight/insightface), [CLIP](https://github.com/openai/CLIP), [RAFT](https://github.com/princeton-vl/RAFT), [YOLO_World](https://github.com/AILab-CVC/YOLO-World).

## Related Links

We are putting together [Awesome-Evaluation-of-Visual-Generation](https://github.com/ziqihuangg/Awesome-Evaluation-of-Visual-Generation), which collects works for evaluating visual generation.
