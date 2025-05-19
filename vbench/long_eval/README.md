# VBench-Long (Beta Version, May 2024)

VBench now supports evaluating **long** video generative models.

## 1. Video Splitting
We split the long video into video clips in two steps

### :hammer: Setup Repository and Environment
```bash
git clone https://github.com/Vchitect/VBench.git

# create conda environment, following instructions in VBench README
pip install -r VBench/requirements.txt
pip install VBench

# install PySceneDetect
pip install scenedetect[opencv] --upgrade
pip install ffmpeg
```
### 1.1 Bypass Scene Cuts
First, we use PySceneDetect to split a long video into multiple semantically consistent short clips and save these clips. After this step, each split video clip ideally contains no scene cuts.


For example
```python
from vbench2_beta_long.utils import split_video_into_scenes
split_video_into_scenes(video_path, output_dir, threshold)
```

### 1.2 Create Slow-Fast Branches

Next, we split the videos from the previous step into shorter fixed-length clips to enable the slow-fast evaluation introduced in the next section.  Since some evaluation dimensions use models trained on longer video clips, such as UMT and ViCLIP, for `human_action` and `overall_consistency`, we established different fixed-length durations for different dimensions. These durations can be found in `vbench2_beta_long/configs/clip_length_mix.yaml`.


Usage:
```python
from vbench2_beta_long.utils import split_video_into_clips
split_video_into_clips(video_path, base_output_dir, duration, fps)
```


**Note: The two video splitting steps have been integrated into `VBench-Long` for automatic execution, so users do not need to manually perform this processing in advance.**

## 2. Slow-Fast Approach to Evaluate Temporal Consistency
<!-- Considering the characteristics of the consistency dimensions such as `subject_consistency` and `background_consistency`, it is clearly unreasonable to evaluate consistency dimensions only in fixed-length short video clips. Therefore, we introduce Slow-Fast Evaluation Method.  -->
Previously, VBench evaluated temporal consistency primarily by calculating the consistency between adjacent video frames. However, for longer videos, it is also crucial to consider the long-range consistency of background scenes and foreground subjects. To address this, we have adopted a slow-fast approach for evaluating temporal consistency.
- **Slow Branch**: This high-frame-rate branch includes every frame in the short video clip. The slow branch evaluation follows VBench's short video evaluation approach.
- **Fast branch**: This low-frame-rate branch extracts the first frame of each very short video clip from the same long video. We then evaluate the long-range consistency using a new set of feature extractors that emphasize high-level visual similarity over lower-level details.

<!-- Specifically, we first evaluate the consistency dimensions' score within each clip, then calculate the consistency dimensions' score between clips. Finally, we weight and combine the two scores to obtain the final consistency dimension score. -->


## 3. Usage

### 3.1 Evaluation on the Standard Prompt Suite of VBench

You can use the command below to evaluate long videos sampled based on the standard prompt of VBench:
```bash []
python vbench2_beta_long/eval_long.py \
    --videos_path $videos_path \
    --dimension $dimension \ 
    --mode 'long_vbench_standard' \
    --dev_flag \
```
For dimension `temporal_flickering`, **static filter** should be implemented before evaluaing videos. We ensembled static filter function into preprocess for **VBench-Long**, and you can use flag `static_filter_flag` to execute static filter, such as:
```bash []
python vbench2_beta_long/eval_long.py \
    --videos_path $videos_path \
    --dimension 'temporal_flickering' \ 
    --mode 'long_vbench_standard' \
    --dev_flag \
    --static_filter_flag
```

### 3.2 Evaluation on Your Own Videos

For long video evaluation, we support customized videos / prompts for the following dimensions: `subject_consistency`, `background_consistency`, `motion_smoothness`, `dynamic_degree`, `aesthetic_quality`, `imaging_quality`

```bash []
python vbench2_beta_long/eval_long.py \
    --videos_path $videos_path \
    --dimension $dimension \ 
    --mode 'long_custom_input' \
    --dev_flag
```

### 3.3 Automatic Evaluation Script
We provide the [evaluate_long.sh](https://github.com/Vchitect/VBench/blob/master/vbench2_beta_long/evaluate_long.sh) script for automating the evaluation across all dimensions. To use the script, simply provide the path to your videos in the following command and run it:
```
sh vbench2_beta_long/evaluate_long.sh $VIDEOS_PATH
```

### 3.4 Example of Evaluating OpenSoraPlan
We have provided scripts to download OpenSoraPlanv1.1 samples, and the corresponding evaluation scripts.
```bash []
# download sampled videos of OpenSoraPlan
sh scripts/download_OpenSoraPlan.sh

# evaluate OpenSoraPlan
sh scripts/evaluate_OpenSoraPlan.sh
```


## :black_nib: Citation

   If you find VBench-Long useful for your work, please consider citing our paper and repo:

   ```bibtex
    @InProceedings{huang2023vbench,
        title={{VBench}: Comprehensive Benchmark Suite for Video Generative Models},
        author={Huang, Ziqi and He, Yinan and Yu, Jiashuo and Zhang, Fan and Si, Chenyang and Jiang, Yuming and Zhang, Yuanhan and Wu, Tianxing and Jin, Qingyang and Chanpaisit, Nattapol and Wang, Yaohui and Chen, Xinyuan and Wang, Limin and Lin, Dahua and Qiao, Yu and Liu, Ziwei},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        year={2024}
    }

    @article{huang2024vbench++,
        title={VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models},
        author={Huang, Ziqi and Zhang, Fan and Xu, Xiaojie and He, Yinan and Yu, Jiashuo and Dong, Ziyue and Ma, Qianli and Chanpaisit, Nattapol and Si, Chenyang and Jiang, Yuming and Wang, Yaohui and Chen, Xinyuan and Chen, Ying-Cong and Wang, Limin and Lin, Dahua and Qiao, Yu and Liu, Ziwei},
        journal={arXiv preprint arXiv:2411.13503},
        year={2024}
    }

    @article{huang2023vbenchgithub,
        author = {VBench Contributors},
        title = {VBench},
        year = {2023},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/Vchitect/VBench}},
    }    
   ```


## :hearts: Acknowledgement

**VBench-Long** is currently maintained by [Ziqi Huang](https://ziqihuangg.github.io/) and [Qianli Ma](https://github.com/MqLeet).

In addition to the open-sourced repositories used in VBench, we also made use of [PySceneDetect](https://github.com/Breakthrough/PySceneDetect), [DINOv2](https://github.com/facebookresearch/dinov2), [DreamSim](https://github.com/ssundaram21/dreamsim).
