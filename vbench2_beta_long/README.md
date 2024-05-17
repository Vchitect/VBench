# VBench-Long (Beta Version, May 2024)

VBench now supports evaluating Long Video generation models.

## Video Splitting
This section includes dividing a long video into multiple semantically consistent short clips and dividing a long video into multiple fixed-length short clips.

### :hammer: Setup Repository and Enviroment
```bash
git clone https://github.com/Vchitect/VBench.git

# create conda environment, following instrcutions in VBench README
pip install -r VBench/requirements.txt
pip install VBench

# install PySceneDetect
pip install scenedetect[opencv] --upgrade
pip install ffmpeg
```
### Splitting Long Video into Semantically Consistent Short Vidoes
Here we offer a function using PySceneDetect to split a long video into multiple semantics-consistent short videos and save these short videos.

For example
```python
from vbench2_beta_long.utils import split_video_into_scenes
split_video_into_scenes(video_path, output_dir, threshold)
```

### Splitting Long Video into Fixed-length Clips

In order to adapt to the settings in VBench, we split long videos into short videos of fixed lengths. Considering that some models are trained on different duration videos, such as UMT and ViCLIP, used for `temporal_style` and `overall_consistency`, we established different splitting strategies, which can be found in `vbench2_beta_long/configs`.


For example
```python
from vbench2_beta_long.utils import split_video_into_clips
split_video_into_clips(video_path, base_output_dir, duration, fps)
```


* Note: We have integrated the code for Semantically Consistent Clips Splitting and Fixed-length Clips splitting into the preprocessing process of `VBench-Long`, so users do not need to perform this processing in advance.


## Usage

### Evaluation on the Standard Prompt Suite of VBench

python
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

### Evaluation on Your Own Videos

* Note: We support customized videos / prompts for the following dimensions: 'subject_consistency', 'background_consistency', 'motion_smoothness', 'dynamic_degree', 'aesthetic_quality', 'imaging_quality'

python
```python
    from vbench2_beta_long import VBenchLong
    my_VBench = VBenchLong(device, <path/to/VBench_full_info.json>, <path/to/save/dir>)
    my_VBench.evaluate(
        videos_path = </path/to/folder_or_video/>,
        name = <name>,
        mode = 'long_custom_input',
    )
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

We made use of [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) and [DINO](https://github.com/facebookresearch/dino)
