# VBench-I2V (Beta Version, Mar 2024)

VBench now supports a benchmark suite for evaluating Image-to-Video (I2V) generation models.

## :fire: Highlights
- Image Suite.
- Evaluation Dimension Suite for I2V. *E.g.*, the control of camera motion given an input image.

## :bookmark_tabs: I2V Image Suite
We provide a suite of input images to benchmark the Image-to-Video (I2V) task.
You can access our image suite on [Google Drive](https://drive.google.com/drive/folders/1fdOZKQ7HWZtgutCKKA7CMzOhMFUGv4Zx?usp=sharing). 

Alternatively, you can use the following script to automatically obtain our image suite.

- First install `gdown`,
    ```
    pip install gdown
    ```
- Then run this script to download the image suite.
    ```
    sh vbench2_beta_i2v/download_data.sh
    ```

**Main philosophy behind our Image Suite**:

1. *Adaptive resolution and aspect ratio*.
Since different Image-to-Video (I2V) models have different default resolutions for the input images, we believe it's only fair to compare models when each model is evaluated on its default / best resolution. To this end, we have also introduced a pipeline to **obtain images in different resolutions and aspect ratios while preserving their main content**. More details will be released.
2. *Diverse and fair content for both foreground and background*.
We ensure that the image content is diverse, in terms of several aspects: scene category, object type, fairness of human-centric images, etc. More statistics will be released.
3. *Text prompts paired with input images*.
For each input image, we carefully designed text prompt via a series of captioning techniques. Detailed pipeline will be released.


## Dimension Suite

### Video-Image Alignment | Subject Consistency
- This dimension evaluates the alignment between the subject in the input image and the subject in the resulting video. We make use of [DINO](https://github.com/facebookresearch/dino) features, with carefully designed order-statistics schemes.
### Video-Image Alignment | Background Consistency
- This dimension assesses the coherence between the background scene in the input image and the generated video. We make use of [DINO](https://github.com/facebookresearch/dino) features, with carefully designed order-statistics schemes.
### Video-Text Alignment | Camera Motion
- This dimension assesses whether the generated video adheres to the camera control instructions specified in the prompt. We make use of [Co-Tracker](https://github.com/facebookresearch/co-tracker), with carefully designed rules to predict the camera motion type.



## Video Data
To prepare the sampled videos for evaluation:
- For each image-prompt pair, sample 5 videos.
- **Random Seed**: At the beginning of sampling, set the random seed. For some models, the random seed is independently and randomly drawn for each video sample, and this is also acceptable, but it would be the best to record the random seed of every video being sampled. We need to ensure: (1) The random seeds are random, and not cherry picked. (2) The sampling process is reproducible, so that the evaluation results are reproducible.
- Name the videos in the form of `$prompt-$index.mp4`, `$index` takes value of `0, 1, 2, 3, 4`. For example:
    ```                   
    ├── A teddy bear is climbing over a wooden fence.-0.mp4                                       
    ├── A teddy bear is climbing over a wooden fence.-1.mp4                                       
    ├── A teddy bear is climbing over a wooden fence.-2.mp4                                       
    ├── A teddy bear is climbing over a wooden fence.-3.mp4                                       
    ├── A teddy bear is climbing over a wooden fence.-4.mp4                                       
    ├── A person is whisking eggs, and the egg whites and yolks are gently streaming out-0.mp4                                                                      
    ├── A person is whisking eggs, and the egg whites and yolks are gently streaming out-1.mp4                                                                      
    ├── A person is whisking eggs, and the egg whites and yolks are gently streaming out-2.mp4                                                                      
    ├── A person is whisking eggs, and the egg whites and yolks are gently streaming out-3.mp4                                                                      
    ├── A person is whisking eggs, and the egg whites and yolks are gently streaming out-4.mp4 
    ......
    ```

### Pseudo-Code for Sampling
- If you want to evaluate certain dimensions, below are the pseudo-code for sampling.
    ```
    dimension_list = ["i2v_subject", "i2v_background", "camera_motion"]

    for dimension in dimension_list:

        # set random seed
        if args.seed:
            torch.manual_seed(args.seed)    
        
        # prepare inputs

        image_folder = "./vbench2_beta_i2v/data/crop/{resolution} # resolution = 1-1/8-5/7-4/16-9
        info_list = json.load(open("./vbench2_beta_i2v/vbench2_i2v_full_info.json", "r"))
        inputs = [(os.path.join(image_folder, info["image_name"]), info["prompt_en"]) for info in info_list if dimension in info["dimension"]]
        
        for image_path, prompt in inputs:

            # sample 5 videos for each prompt
            for index in range(5):

                # perform sampling
                video = sample_func(image_path, prompt, index)    
                cur_save_path = f'{args.save_path}/{prompt}-{index}.mp4'
                torchvision.io.write_video(cur_save_path, video, fps=fps, video_codec='h264', options={'crf': '10'})
    ```

## Usage

We have introduced three dimensions for the image-to-video task, namely: `i2v_subject`, `i2v_background`, and `camera_motion`. 

#### python
```
from vbench2_beta_i2v import VBenchI2V
my_VBench = VBenchI2V("cuda", <path/to/vbench2_i2v_full_info.json>, <path/to/save/dir>)
my_VBench.evaluate(
    videos_path = <video_path>,
    name = <name>,
    dimension_list = [<dimension>, <dimension>, ...],
    resolution = <resolution>
)
```
The `resolution` parameter specifies the image resolution. You can select the suitable ratio according to the video resolution, with options including 1:1, 8:5, 7:4, and 16:9.

For example: 
```
from vbench2_beta_i2v import VBenchI2V
my_VBench = VBenchI2V("cuda", "vbench2_beta_i2v/vbench2_i2v_full_info.json", "evaluation_results")
my_VBench.evaluate(
    videos_path = "sampled_videos",
    name = "i2v_subject",
    dimension_list = ["i2v_subject"],
    resolution = "1-1"
)
```


## :black_nib: Citation

   If you find VBench-I2V useful for your work, please consider citing our paper and repo:

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

**VBench-I2V** is currently maintained by [Ziqi Huang](https://ziqihuangg.github.io/) and [Fan Zhang](https://github.com/zhangfan-p).

We made use of [DINO](https://github.com/facebookresearch/dino) and [Co-Tracker](https://github.com/facebookresearch/co-tracker).
