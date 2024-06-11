# Competitions

We have two tracks for video generation: (1) short videos, and (2) long videos.

More information on the competition will be announced at the official competition site.



<!-- We provide 20 stories for long video generation. Take one story for example:
```
"summary": "A panda wakes up in the morning and goes to school.",
"storyline": [
    "The panda stirs awake, nestled in a mound of blankets.",
    "The panda brushes its teeth diligently, a minty freshness lingering in the air.",
    "Sunlight bathes the breakfast table as the panda enjoys bamboo shoots and bread.",
    "Laden with its backpack, the panda heads out the door eagerly.",
    "Snowflakes dance around the panda as it walks to school.",
    "Inside the classroom, the panda listens attentively to the teacher."
]
```
- The `summary` is an overall description of the story.
- The `storyline` is a detailed and step-by-step description of the story.

There are several ways to generate a long video, depending on the capability of your model.
1. Take the `summary` as input text prompt, and generate a long video directly. This is useful when your text encoder or video generation has limited capacity in understanding long input text prompts.
2. Take the `storyline` list as input in a step-by-step manner, sample each item in the list separately, and concatenate your sampled videos into a long video. This might produce better results compared to directly using `summary`. This is useful when your model has limited capacity in handling long prompts, and your model is targeted for short video generation instead of long video generation.
3. Take the `storyline` list as input in one go, concatenate the list of sentences in the storyline to produce a long sentence, and feed this long sentence into your model to generate a long video in one go. This is useful when you model has good long prompt understanding capabilities, and is able to directly generate long videos.


TODO
- [ ] Add sampling and formatting requirements
- [ ] Add video (FPS, resolution, duration) requirements
- [ ] Add evaluation pipeline for long videos -->


## Submission Requirement

### 1. Prompt List


Provide one generated video based on each text prompt.


**Short Videos**: Sample videos from `competitions/short_prompt_list.txt` 
**Long Videos**: Sample videos from `competitions/long_prompt_list.txt`



### 2. Video Requirement


The table below is the sampling requirement.
| Video Type | Prompt Count |Resolution | Duration | Frame Rate | Frame Count |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Short Videos |  200 | No Limit | 1.6-4.0s | 8-24 FPS | 16-96 |
| Long Videos |  40 | No Limit | 10.0-40.0s | 8-24 FPS | - |

Note:
- For short videos, must satisfy the requirements on `Duration`, `Frame Rate`, and `Frame Count` at the same time.
- For long videos, there is no additional requirement on `Frame Count`, as long as the sampled videos satisfy the requirements on `Duration` and `Frame Rate` at the same time.



### 3. File Structure Requirements
Please organize generated videos according to the following requirements. 
- **PNG requirement**: All submissions must be in `png` format. During submission, the teams are required to indicate the **Frame Rate** in terms of `FPS`.
- **File structure**: The `png` frames of the same video should be saved in the same sub-folder. One sub-folder for one sampled video.
- **Frame name**: Name the `png` frames according to the frame order, in 5 digits, zero-filled. The first frame would is `00000.png`.
- **Folder name**: For each prompt, use the **prompt index** followed by **the first three words of the prompt**.
    - **prompt index**: 4 digits, zero-filled, start with `0001`.
    - **the first three words of the prompt**: case sensitive, and replace space ` ` with underscore `_`.

Your submission folder should look like this:
```
YOUR-TEAM-NAME_videos.zip
├── Short_Videos
│   ├── 0001_Close_up_of     # folder name corresponds to each prompt
│   │   ├── 00000.png
│   │   │── 00001.png
│   │   │── ...
│   ├── 0002_Turtle_swimming_in     
│   │   ├── 00000.png
│   │   │── 00001.png
│   │   │── ...
│   ├── ...
│   ├── 0200_cruise_ship_in
│   │   ├── 00000.png
│   │   │── 00001.png
│   │   │── ...
├── Long_Videos
│   ├── 0001_A_stylish_woman     
│   │   ├── 00000.png
│   │   │── 00001.png
│   │   │── ...
│   ├── 0002_Several_giant_wooly
│   │   ├── 00000.png
│   │   │── 00001.png
│   │   │── ...
│   ├── ...
│   ├── 0040_The_Glenfinnan_Viaduct
│   │   ├── 00000.png
│   │   │── 00001.png
│   │   │── ...
```



### Pseudo-Code for Sampling


This is for your reference only. You can choose to save `MP4` first before converting to `PNG`.

#### 1. Sampling and save to MP4

```
type_list = ['short_prompt_list', 'long_prompt_list']

for prompt_type in type_list:

    # set random seed
    if args.seed:
        torch.manual_seed(args.seed)    
    
    # read prompt list
    with open(f'./{prompt_type}.txt', 'r') as f:
        prompt_list = f.readlines()
    prompt_list = [prompt.strip() for prompt in prompt_list]
    
    for index, prompt in enumerate(prompt_list):

        # perform sampling
        video = sample_func(prompt)
        cur_save_path = f'{args.save_path}/{video-name}.mp4'
        torchvision.io.write_video(cur_save_path, video, fps=fps)
```

#### 2. Convert MP4 to PNG
**ffmpeg**
```
ffmpeg -i input_video.mp4 -start_number 0 ./%05d.png
```
**PIL**
```
from PIL import Image
import cv2

video = cv2.VideoCapture('input_video.mp4')
frame_count = 0

while True:
    ret, frame = video.read()
    if not ret:
        break
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img.save(f"{frame_count:05d}.png")
    frame_count += 1

video.release()
```




## Evaluation Metrics
Both short and long videos will be automatically evaluated in terms of 3 aspects.
| Evaluation Aspects | Automatic Evaluation | Human Evaluation |
| ----- | -----| -----|
|`temporal_quality`| - VBench dimensions: `subject_consistency`, `background_consistency`, `motion_smoothness`, `dynamic_degree`, | Temporal consistency between frames, motion quality, motion strengths |
|`frame_wise_quality`| - VBench dimensions: `aesthetic_quality`, `imaging_quality`, | The quality of individual video frames |
|`text_alignment`| - VBench dimension: `overall_consistency`. and `CLIP Score` | Alignment between generated videos and text prompts |

Additionally, long videos will be evaluated on `subject_consistency` as a stand-alone aspect via `Human Evaluation.`



## Automatic Evaluation


### 1.Install Environment via Anaconda
```
conda create -n vbench-competition python=3.9
conda activate vbench-competition
pip install -r competitions/requirements.txt

# install PySceneDetect
pip install scenedetect[opencv] --upgrade
pip install ffmpeg
```

### 2.Evaluation Scripts

We support two forms of input, one is to directly input the path containing all mp4 format videos (using the `--video_path flag`). The other is to input the root directory of the videos saved in png format and the corresponding frame rate of the videos, and we will automatically convert the videos to the specified format and then evaluate them (using the `--submission_path` flag and `--frame_rate` flag).

Additionally, you need to use the `--prompt_file` flag to specify the category that the videos you want to evaluate belong to, whether it's short_video or long_video.

You can use the `--dimension` flag to specify one or more dimensions to be evaluated, and the `--output_path` flag to specify the path to store the results.

To perform evaluation, use the following script:
```
python run_eval.py --video_path <videos_path> --output_path <output_path>  --prompt_file <prompt_file> --dimension <dim1>
```
or
```
python run_eval.py --submission_path <submission_path> --frame_rate <fps> --output_path <output_path>  --prompt_file <prompt_file> --dimension <dim1> <dim2> <dim3>
```

For example:

```
python run_eval.py --submission_path ./Short_Videos --frame_rate 8 --output_path ./eval_results  --prompt_file ./short_prompt_list.txt --dimension temporal_quality frame_wise_quality text_alignment
```
The structure of `submission_path` should like this:
```
├── Short_Videos
│   ├── 0001_Close_up_of     # folder name corresponds to each prompt
│   │   ├── 00000.png
│   │   │── 00001.png
│   │   │── ...
│   ├── 0002_Turtle_swimming_in     
│   │   ├── 00000.png
│   │   │── 00001.png
│   │   │── ...
│   ├── ...
│   ├── 0200_cruise_ship_in
│   │   ├── 00000.png
│   │   │── 00001.png
│   │   │── ...
```

### 3.Evaluation Results

The evaluation results will be saved in a JSON file in the following format, which includes the results of the **target evaluation dimensions** as well as the results of their **corresponding sub-dimensions**.

```json
{
    "temporal_quality": [   # result of temporal_quality dimension
        0.8530498955750241,
        {
            "subject_consistency": [
                0.9986579449971517,
                [
                    ...
                    {
                        "video_path": "./evaluated_videos/0002_Turtle_swimming_in.mp4",
                        "video_results": 14.991820216178894
                    },
                    ...
                ]
            ],
            "background_consistency": [
                0.9924527994791666,
                [
                    ...
                    {
                        "video_path": "./evaluated_videos/0002_Turtle_swimming_in.mp4",
                        "video_results": 0.9943684895833333
                    },
                    ...
                ]
            ],
            "motion_smoothness": [
                0.9945638900362661,
                [
                    ...
                    {
                        "video_path": "./evaluated_videos/0002_Turtle_swimming_in.mp4",
                        "video_results": 0.9937908449241242
                    },
                    ...
                ]
            ],
            "dynamic_degree": [
                0.0,
                [
                    ...
                    {
                        "video_path": "./evaluated_videos/0002_Turtle_swimming_in.mp4",
                        "video_results": false
                    },
                    ...
                ]
            ]
        }
    ],
    "frame_wise_quality": [   # result of frame_wise_quality dimension
        0.6555798406600952,
        {
            "aesthetic_quality": [
                0.5653051853179931,
                [
                    ...
                    {
                        "video_path": "./evaluated_videos/0002_Turtle_swimming_in.mp4",
                        "video_results": 0.709746241569519
                    },
                    ...
                ]
            ],
            "imaging_quality": [
                0.7458544960021973,
                [
                    ...
                    {
                        "video_path": "./evaluated_videos/0002_Turtle_swimming_in.mp4",
                        "video_results": 66.22727489471436
                    },
                    ...
                ]
            ]
        }
    ],
    "text_alignment": [   # result of text_alignment dimension
        0.2807383455336094,
        {
            "overall_consistency": [
                0.24468591958284377,
                [
                    ...
                    {
                        "video_path": "./evaluated_videos/0002_Turtle_swimming_in.mp4",
                        "video_results": 0.2743408977985382
                    },
                    ...
                ]
            ],
            "clip_score": [
                0.316790771484375,
                [
                    ...
                    {
                        "video_path": "./evaluated_videos/0002_Turtle_swimming_in.mp4",
                        "video_results": 0.31317138671875
                    },
                    ...
                ]
            ]
        }
    ]
}
```

## :black_nib: Citation

   If you use VBench for evaluating your models, please consider citing our paper or repo:

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