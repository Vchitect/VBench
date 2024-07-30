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

## 3. Static Filter
For dimension `temporal_flickering`, **static filter** shoulde be implemented before evaluaing videos. You can run this to filter static videos:
```python []
# This only filter out static videos whose prompt matches the prompt in the temporal_flickering.
python static_filter.py --videos_path $VIDEOS_PATH
```

We ensembled static filter function into preprocess for **VBench-Long**, and a flag `static_filter_flag` was designed to decide wheter to execute static filter or not. 

Static filter will be executed if set `static_filter_flag` flag, for example:
```bash []
python vbench2_beta_long/eval_long.py \
    --videos_path $videos_path \
    --dimension $dimension \ 
    --mode 'long_vbench_standard' \
    --dev_flag \
    --static_filter_flag \
```

If you have filtered videos manually, you can unset the `static_filter_flag` flag, for example
```bash []
python vbench2_beta_long/eval_long.py \
    --videos_path $videos_path \
    --dimension $dimension \ 
    --mode 'long_vbench_standard' \
    --dev_flag \
```

## 4. Usage

### 4.1 Evaluation on the Standard Prompt Suite of VBench

```python
from vbench2_beta_long import VBenchLong
my_VBench = VBenchLong(device, <path/to/VBench_full_info.json>, <path/to/save/dir>)
my_VBench.evaluate(
    videos_path = <video_path>,
    name = <name>,
    dimension_list = [<dimension>, <dimension>, ...],
    mode = 'long_vbench_standard',
)
```

For example:
```python
from vbench2_beta_long import VBenchLong
my_VBench = VBenchLong(device, "vbench/VBench_full_info.json", "evaluation_results")
my_VBench.evaluate(
    videos_path = 'sampled_videos/latte/subject_consistency',
    name ='results_latte_subject_consistency',
    dimension_list = ["subject_consistency"],
    mode = 'long_vbench_standard',
)
```

### 4.2 Evaluation on Your Own Videos

For long video evaluation, we support customized videos / prompts for the following dimensions: `subject_consistency`, `background_consistency`, `motion_smoothness`, `dynamic_degree`, `aesthetic_quality`, `imaging_quality`

```python
from vbench2_beta_long import VBenchLong
my_VBench = VBenchLong(device, <path/to/VBench_full_info.json>, <path/to/save/dir>)
my_VBench.evaluate(
    videos_path = </path/to/folder_or_video/>,
    name = <name>,
    mode = 'long_custom_input',
)
```

### 4.3 Example of Evaluating OpenSoraPlan
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
