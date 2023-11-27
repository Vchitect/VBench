# VBench

## Installation

1. Clone Repo

   ```bash
   git clone https://github.com/Vchitect/VBench
   cd VBench
   ```

2. Create Conda Environment and Install Dependencies
    ```
    conda env create -f vbench_env.yml
    conda activate vbench
    ```

## Pre-Trained Models
[Optional] Please download the pre-trained weights according to the guidance in the `model_path.txt` file for each model in the `pretrain` folder.

## Prompt Suite

We provide prompt lists are at `prompts/`, see [instructions](https://github.com/Vchitect/VBench/tree/main/prompts) for details.

## Evaluation Method Suite

To perform evaluation, run this:
```
import torch
from vbench import VBench

device = torch.device("cuda")
output_path = './evaluation_results/'
full_json_dir = './VBench_full_info.json'
videos_path = "{your_video_dir}" # change to folder that contains the sampled videos
my_VBench = VBench(device, full_json_dir, output_path)
my_VBench.evaluate(
    videos_path = videos_path,
    name = 'test',
    dimension_list = {list_of_dimension}, # change to the list of dimension, e.g. ['human_action','scene']
    local=False, # Whether to use local checkpoints. If true, vbench will load model weights locally.
)
```

List of dimensions supported:
```
['subject_consistency', 'background_consistency', 'temporal_flickering', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', "imaging_quality', 'object_class', 'multiple_objects', 'human_action', 'color', 'spatial_relationship', 'scene', 'temporal_style', 'appearance_style', 'overall_consistency']
```

## Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
    @article{huang2023bench,
        title={{VBench}: Comprehensive Benchmark Suite for Video Generative Models},
        author={Huang, Ziqi and He, Yinan and Yu, Jiashuo and Zhang, Fan and Si, Chenyang and Jiang, Yuming and Zhang, Yuanhan and Wu, Tianxing and Jin, Qingyang and Chanpaisit, Nattapol and Wang, Yaohui and Chen, Xinyuan and Wang, Limin and Lin, Dahua and Qiao, Yu and Liu, Ziwei}
        journal={arXiv preprint}
        year={2023}
    }
   ```


## Acknowledgement

The codebase is maintained by [Ziqi Huang](https://ziqihuangg.github.io/), [Yinan He](https://github.com/yinanhe), [Jiashuo Yu](https://scholar.google.com/citations?user=iH0Aq0YAAAAJ&hl=zh-CN), and [Fan Zhang](https://github.com/zhangfan-p).

This project is built using the following open-sourced repositories:
- [AMT](https://github.com/MCG-NKU/AMT/)
- [UMT](https://github.com/OpenGVLab/unmasked_teacher)
- [CLIP](https://github.com/openai/CLIP)
- [RAFT](https://github.com/princeton-vl/RAFT)
- [GRiT](https://github.com/JialianW/GRiT)
- [MUSIQ](https://github.com/chaofengc/IQA-PyTorch/)
- [ViCLIP](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid)
- [LAION Aesthetic Predictor](https://github.com/LAION-AI/aesthetic-predictor)
