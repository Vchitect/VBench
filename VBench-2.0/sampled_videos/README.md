# Sampled Videos

[![Dataset Download](https://img.shields.io/badge/Dataset-Download-red?logo=googlechrome&logoColor=red)](https://drive.google.com/drive/folders/19nXdrNFw-PxxYJcd9HyWmbLVe5Agmlc2)

To facilitate future research and to ensure full transparency, we release all the videos we sampled and used for VBench-2.0 evaluation. You can download them on [Google Drive](https://drive.google.com/drive/folders/19nXdrNFw-PxxYJcd9HyWmbLVe5Agmlc2).

## What Videos Do We Provide?
- **6 T2V Models**:
    - including [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), [Wanx](https://github.com/Wan-Video/Wan2.1), [CogVideo](https://github.com/THUDM/CogVideo), [StepVideo](https://github.com/stepfun-ai/Step-Video-T2V), Sora, Kling. More details of models are provided below.
- **Suite of Videos for each Model**: 
    - *Per Dimension*: The sampled videos for each ability dimension evaluated by VBench-2.0. The per-dimension prompts are available under [`prompts/prompt`](https://github.com/Vchitect/VBench/tree/master/VBench-2.0/prompts/prompt), and we also provide a combined list of all the dimensions' prompts at [`prompts/VBench2_full_text.txt`](https://github.com/Vchitect/VBench/blob/master/VBench-2.0/prompts/VBench2_full_text.txt).
  
What's the potential usage of these videos:
- Further labeling on video quality
- For Instruction Tuning, using our videos and our human preference labels

Below is the folder structure of different models' sampled videos:
```
t2v_sampled_videos
│── CogVideo+sample.zip
│── HunyuanVideo+sample.zip
│── Sora-480p+sample.zip
│── Kling-1.6+sample.zip
│── StepVideo+sample.zip
│── Wanx+sample.zip
│── Vidu_Q1+sample.zip

```
## How to Download the Videos?
You can utilize **gdown** to download from [Google Drive](https://drive.google.com/drive/folders/19nXdrNFw-PxxYJcd9HyWmbLVe5Agmlc2). Below is an example:
- First, install `gdown`:
```
pip install gdown
```
- Then, download zip file using `gdown`:
```
gdown --id <file_id> --output <output_filename>

# Example for HunyuanVideo
gdown --folder https://drive.google.com/drive/folders/1OmL7HComkNWqhV6Yr9CA8YufAAErK4gu
```
- Then, follow the instruction in README.md.

## What are the Details of the Video Generation Models?
We list the setting for sampling videos from these models.
| Model | Evaluation Party | Release Time | Resolution | FPS | Frame Count | Video Length | Checkpoint | Code Commit ID | Video Format | Sampled Videos (Dimension) |                             Other Settings       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                        |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |------------------------------|
| [`CogVideoX1.5-5B (10s Diffusers prompt-optimized)`](https://github.com/THUDM/CogVideo) | VBench Team | 2024-11-08 | 1360x768 | 16 | 161 | 10s | [link](https://huggingface.co/THUDM/CogVideoX1.5-5B/tree/main) |  [Commit ID](https://github.com/THUDM/CogVideo/tree/68d93ce8fc030f260e4a75eadfc318ed002eccce) | MP4 | [Google Drive](https://drive.google.com/drive/folders/1Cy_M_hTmkWI6cLyBMRx7Y7mhSdNyKlxe?usp=drive_link) | <small>applied [augmented prompts](https://github.com/Vchitect/VBench/blob/master/VBench-2.0/prompts/prompt_aug/VBench2_full_text_aug.txt)</small>|
| [`HunyuanVideo (Open-Source Version)`](https://github.com/Tencent/HunyuanVideo) | VBench Team | 2024-12-03 | 1280x720 | 24 | 129 | 5.3s | [link](https://huggingface.co/tencent/HunyuanVideo/tree/main) |  [Commit ID](https://github.com/Tencent/HunyuanVideo/tree/3579dbc7862b01106029a16f2172eec85629cce5) | MP4 | [Google Drive](https://drive.google.com/drive/folders/1OmL7HComkNWqhV6Yr9CA8YufAAErK4gu?usp=drive_link) | <small>applied [augmented prompts](https://github.com/Vchitect/VBench/blob/master/VBench-2.0/prompts/prompt_aug/VBench2_full_text_aug.txt)</small>|
| [`Sora`](https://sora.com/library) | VBench Team | 2025-01-14 | 854x480 | 30 | 150 | 5.0s | - | - | MP4 | [Google Drive](https://drive.google.com/drive/folders/1yK3xiD7HhpGjMKMFddcgouaV1bymSqSw?usp=sharing) | - |
| [`Kling 1.6`](https://sora.com/library) | VBench Team | 2025-12-09 | 1280x720 | 24 | 241 | 10.0s | - | - | MP4 | [Google Drive](https://drive.google.com/drive/folders/1VhlbBXQ_P9unUkxP8xsOhCAf3_9Iu1pn?usp=sharing) | <small>applied [augmented prompts](https://github.com/Vchitect/VBench/blob/master/VBench-2.0/prompts/prompt_aug/VBench2_full_text_aug.txt)</small> |
| [`Step-Video-T2V`](https://github.com/stepfun-ai/Step-Video-T2V) | VBench Team | 2025-03-13 | 992x544 | 25 | 200 | 8s | [link](https://huggingface.co/stepfun-ai/stepvideo-t2v/tree/main) | [Commit ID](https://github.com/stepfun-ai/Step-Video-T2V/tree/d3ca3d68513bf18d75ff50ff3452c8c8407f924f) | MP4 |[Google Drive](https://drive.google.com/drive/folders/1BeQ1iGspQ3bSCW9VLaeXyuOKjwtJSLTF?usp=drive_link) | - |
| [`Wan2.1-T2V-14B`](https://github.com/Wan-Video/Wan2.1/tree/main) | VBench Team | 2025-03-20 | 1280x720 | 16 | 81 | 5s | [link](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/tree/main) | [Commit ID](https://github.com/Wan-Video/Wan2.1/tree/b58b7c573776b76b6fe8d36086590e033173f9b1) | MP4 | [Google Drive](https://drive.google.com/drive/folders/1ZLK7Naq9jH0f9cbnpvo_s5IOpJzQn-w3?usp=drive_link) | <small>applied [Prompt Rewrite](https://github.com/Wan-Video/Wan2.1?tab=readme-ov-file#2-using-prompt-extension) provided by Wan, [augmented prompts](https://github.com/Vchitect/VBench/blob/master/VBench-2.0/prompts/prompt_aug/Wanx_full_text_aug.txt)  |
| [`Vidu Q1 (2025-04-17)`](https://www.vidu.studio/) | Shengshu Team | 2025-04-21 | 1280x720 | 24 | 125 | 5.2s | - | - | MP4 | [Google Drive](https://drive.google.com/drive/folders/1ZP6kgldD0akcqfu_IANdZIiNJGAFiGet?usp=drive_link) | - |

## How are Files Structured in Google Drive?

### 1. Sub-Folder Organization

For these models, the zip contains 18 subfolders corresponding to videos sampled for evaluating different dimensions. 

For `HunyuanVideo, CogVideo, Sora, Kling, StepVideo, Wanx`, we provide their single-stage outputs.

We take `HunyuanVideo` as an example:

```
    - HunyuanVideo
        - Camera_Motion   
            - Garden, zoom in.-0.mp4
            - Garden, zoom in.-1.mp4
            - ...
        - Complex_Landscape
            - The camera starts at the top of the forest, where thick morning mist drifts between the treetops, with sunlight filtering through the leaves and casting dappled spots of light, the-0.mp4
            - The camera starts at the top of the forest, where thick morning mist drifts between the treetops, with sunlight filtering through the leaves and casting dappled spots of light, the-1.mp4
            - ...
        - Complex_Plot
            - The race began, and the first runner quickly took off, leading the other teams. Everyone was focused on his performance. During the baton handoff of the first runner, Team A made a-0.mp4
            - The race began, and the first runner quickly took off, leading the other teams. Everyone was focused on his performance. During the baton handoff of the first runner, Team A made a-1.mp4
            - ...
        - Composition
            - A lion with the wings of an eagle, soaring through the sky with majestic ease.-0.mp4
            - A lion with the wings of an eagle, soaring through the sky with majestic ease.-1.mp4
            - ...
        - Diversity
            - A wooden toy is placed gently on the surface of a small bowl of water.-0.mp4
            - A wooden toy is placed gently on the surface of a small bowl of water.-1.mp4
            - ...
        - Dynamic_Attribute
            - The leaves gradually change from red to green.-0.mp4
            - The leaves gradually change from red to green.-1.mp4
            - ...
        - Dynamic_Spatial_Relationship
            - A dog is on the left of a table, then the dog runs to the front of the table.-0.mp4
            - A dog is on the left of a table, then the dog runs to the front of the table.-1.mp4
            - ...
        - Human_Anatomy
            - A man is doing yoga.-0.mp4
            - A man is doing yoga.-1.mp4
            - ...
        - Human_Clothes
            - A man is running.-0.mp4
            - A man is running.-1.mp4
            - ...
        - Human_Identity
            - A man is dancing.-0.mp4
            - A man is dancing.-1.mp4
            - ...
        - Human_Interaction
            - One person hands a cup of water to another.-0.mp4
            - One person hands a cup of water to another.-1.mp4
            - ...

```

## Human Preference Labels

Available for download at [Google Drive](https://drive.google.com/drive/folders/1vQ8cYL_3uB-34GUj6s5Igo3I3XEhCSWd?usp=sharing).

Each dimension contains an annotation file, each of which contains a list, and the list contains manually preferred annotation results of videos generated by different prompts. The evaluation process involves comparing videos from different models and, based on human annotations, determining which video best matches the prompt for the corresponding dimension.

### Data Structure

JSON data is composed of multiple objects, each representing an evaluation instance. Each instance contains the following key-value pairs:

`prompt_en`: The text prompt for generating the desired video content.

`videos`: This section contains the urls to videos from different models.

`human_anno`: This section represents human annotation, which is composed of a nested dictionary. The outer keys represent the model names (e.g., "HunyuanVideo", "CogVideo"), and the inner keys represent the other model names. The corresponding values within these nested dictionaries represent the human-assigned scores for the relative quality of each model's video compared to the other model's video.

For example, `human_anno["HunyuanVideo"]["CogVideo"] = 0` indicates that humans judged the CogVideo video to be better than the HunyuanVideo video for the given prompt.

`human_anno["HunyuanVideo"]["CogVideo"] = 1` indicates that humans judged the HunyuanVideo video to be better than the CogVideo video.

`human_anno["HunyuanVideo"]["CogVideo"] = 0.5` indicates that humans judged the HunyuanVideo video and the CogVideo video to be of equal quality.
