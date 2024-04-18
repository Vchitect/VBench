# Sampled Video
We provide videos sampled from the current evaluation of 8 T2V models, including [lavie](https://github.com/Vchitect/LaVie), [modelscope](https://modelscope.cn/models/iic/text-to-video-synthesis/summary), [cogvideo](https://github.com/THUDM/CogVideo), [videocrafter-0.9](https://github.com/AILab-CVC/VideoCrafter/tree/30048d49873cbcd21077a001e6a3232e0909d254), [videocrafter-1](https://github.com/AILab-CVC/VideoCrafter), [show-1](https://github.com/showlab/Show-1), pika, gen-2. You can access them on [Google Drive](https://drive.google.com/drive/folders/13pH95aUN-hVgybUZJBx1e_08R6xhZs5X).

You can utilize **gdown** to download from Google Drive. Below is an example:
- First, install `gdown` by running following command:
```
pip install gdown
```
- Then, download file use below script:
```
gdown --id <file_id> --output <output_filename>

# Example for videocrafter-1
gdown --id 1FCRj48-Yv7LM7XGgfDCvIo7Kb9EId5KX --output videocrafter-1.tar.gz
```

## Model Setting
The table below is the setting information for sampling videos using different models.
| Model | Release Time | Resolution | FPS | Frame Count | Video Length | Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [`LaVie`](https://github.com/Vchitect/LaVie) | 2023-09-26 | 512x512 | 8 | 16 | 2s | - |
| [`ModelScope`](https://modelscope.cn/models/iic/text-to-video-synthesis/summary) | 2023-08-12 | 256x256 | 8 | 16 | 2s | [link](https://modelscope.cn/models/iic/text-to-video-synthesis/files) |
| [`CogVideo`](https://github.com/THUDM/CogVideo) | 2022-05-29 | 480x480 | 10 | 33 | 3.3s | [link](https://github.com/THUDM/CogVideo?tab=readme-ov-file#download) |
| [`VideoCrafter-0.9`](https://github.com/AILab-CVC/VideoCrafter/tree/30048d49873cbcd21077a001e6a3232e0909d254) | 2023-04-05 | 256x256 | 8 | 16 | 2s | [link](https://huggingface.co/VideoCrafter/t2v-version-1-1/blob/main/models/base_t2v/model_rm_wtm.ckpt) |
| [`VideoCrafter-1.0`](https://github.com/AILab-CVC/VideoCrafter) | 2023-10-30 | 1024x576 | 10 | 16 | 1.6s | [link](https://huggingface.co/VideoCrafter/Text2Video-1024/blob/main/model.ckpt) |
| [`Show-1`](https://github.com/showlab/Show-1) | 2023-09-27 | 576x320 | 8 | 29 | 3.6s | [link](https://huggingface.co/showlab/show-1-sr2#:~:text=git%20lfs%20install%0A%0A%23%20base%0Agit%20clone%20https%3A//huggingface.co/showlab/show%2D1%2Dbase%0A%23%20interp%0Agit%20clone%20https%3A//huggingface.co/showlab/show%2D1%2Dinterpolation%0A%23%20sr1%0Agit%20clone%20https%3A//huggingface.co/showlab/show%2D1%2Dsr1%0A%23%20sr2%0Agit%20clone%20https%3A//huggingface.co/showlab/show%2D1%2Dsr2) |
| `Gen-2` | 2023-12-14 | 1408x768 | 24 | 96 | 4.0s | - |
| `Pika` | 2023-12-14 | 1088x640 | 24 | 72 | 3.0s | - |
## Folder Structure for Sampled Videos

Below is the folder structure of different models' sampled videos.

**The folders `LaVie`, `ModelScope`, `CogVideo`, `VideoCrafter-0.9`, `Show-1`, and `VideoCrafter-1` contain videos for "all_dimension" and "all_category," and are respectively divided into 11 subfolders and 8 subfolders, each corresponding to a specific dimension or category.**

*The file structure of `lavie`, `modelscope`, `cogvideo`, and `videocrafter-0.9` is as follows. Videos for models such as `lavie`, `modelscope`, and `videocrafter-0.9` are in mp4 format, while the videos for `cogvideo` are in gif format.*
```
- per_dimension
    - lavie # or modelscope, cogvideo, videocrafter-0.9
        - appearance_style   
            - The bund Shanghai, Van Gogh style-0.mp4
            - The bund Shanghai, Van Gogh style-1.mp4
            - ...
        - human_action
            - A person is finger snapping-0.mp4
            - A person is finger snapping-1.mp4
            - ...
        - object_class
            - a dining table-0.mp4
            - a dining table-1.mp4
            - ...
        - scene
            - restaurant-0.mp4
            - restaurant-1.mp4
            - ...
        - subject_consistency
            - a giraffe taking a peaceful walk-0.mp4
            - a giraffe taking a peaceful walk-1.mp4
            - ...
        - temporal_style
            - The bund Shanghai, zoom in-0.mp4
            - The bund Shanghai, zoom in-1.mp4
            - ...
        - color
            - a blue clock-0.mp4
            - a blue clock-1.mp4
            - ...
        - multiple_objects
            - a fire hydrant and a stop sign-0.mp4
            - a fire hydrant and a stop sign-1.mp4
            - ...
        - overall_consistency
            - Yellow flowers swing in the wind-0.mp4
            - Yellow flowers swing in the wind-1.mp4
            - ...
        - spatial_relationship
            - a frisbee on the left of a sports ball, front view-0.mp4
            - a frisbee on the left of a sports ball, front view-1.mp4
            - ...
        - temporal_flickering
            - static view on a desert scene with an oasis, palm trees, and a clear, calm pool of water-0.mp4
            - static view on a desert scene with an oasis, palm trees, and a clear, calm pool of water-1.mp4
            - ...
- per_category
    - lavie # or modelscope, cogvideo, videocrafter-0.9
        - animal  
            - wild rabbit in a green meadow-0.mp4
            - wild rabbit in a green meadow-1.mp4
            - ...
        - architecture
            - water tower on the desert-0.mp4
            - water tower on the desert-1.mp4
            - ...
        - food
            - waffles with whipped cream and fruit-0.mp4
            - waffles with whipped cream and fruit-1.mp4
            - ...
        - human
            - young dancer practicing at home-0.mp4
            - young dancer practicing at home-1.mp4
            - ...
        - lifestyle
            - the interior design of a shopping mall-0.mp4
            - the interior design of a shopping mall-1.mp4
            - ...
        - plant
            - coconut tree near sea under blue sky-0.mp4
            - coconut tree near sea under blue sky-1.mp4
            - ...
        - scenery
            - waterfalls in between mountain-0.mp4
            - waterfalls in between mountain-1.mp4
            - ...
        - vehicles
            - video of yacht sailing in the ocean-0.mp4
            - video of yacht sailing in the ocean-1.mp4
            - ...
```

*Under each dimension or category in `show-1`, there are two folders corresponding to the last two stages of show-1 generated videos, namely super1 and super2. The leaderboard contains the evaluation results for the final stage, super2.*
```
- per_dimension
    - show-1
        - appearance_style/{super1/super2}       # Optional subfolders super1 and super2
            - The bund Shanghai, Van Gogh style-0.mp4
            - The bund Shanghai, Van Gogh style-1.mp4
            - ...
        - human_action/{super1/super2}
            - A person is finger snapping-0.mp4
            - A person is finger snapping-1.mp4
            - ...
        - object_class/{super1/super2}
            - a dining table-0.mp4
            - a dining table-1.mp4
            - ...
        - scene/{super1/super2}
            - restaurant-0.mp4
            - restaurant-1.mp4
            - ...
        - subject_consistency/{super1/super2}
            - a giraffe taking a peaceful walk-0.mp4
            - a giraffe taking a peaceful walk-1.mp4
            - ...
        - temporal_style/{super1/super2}
            - The bund Shanghai, zoom in-0.mp4
            - The bund Shanghai, zoom in-1.mp4
            - ...
        - color/{super1/super2}
            - a blue clock-0.mp4
            - a blue clock-1.mp4
            - ...
        - multiple_objects/{super1/super2}
            - a fire hydrant and a stop sign-0.mp4
            - a fire hydrant and a stop sign-1.mp4
            - ...
        - overall_consistency/{super1/super2}
            - Yellow flowers swing in the wind-0.mp4
            - Yellow flowers swing in the wind-1.mp4
            - ...
        - spatial_relationship/{super1/super2}
            - a frisbee on the left of a sports ball, front view-0.mp4
            - a frisbee on the left of a sports ball, front view-1.mp4
            - ...
        - temporal_flickering/{super1/super2}
            - static view on a desert scene with an oasis, palm trees, and a clear, calm pool of water-0.mp4
            - static view on a desert scene with an oasis, palm trees, and a clear, calm pool of water-1.mp4
            - ...
- per_category
    - show-1
        - animal/{super1/super2}
            - wild rabbit in a green meadow-0.mp4
            - wild rabbit in a green meadow-1.mp4
            - ...
        - architecture/{super1/super2}
            - water tower on the desert-0.mp4
            - water tower on the desert-1.mp4
            - ...
        - food/{super1/super2}
            - waffles with whipped cream and fruit-0.mp4
            - waffles with whipped cream and fruit-1.mp4
            - ...
        - human/{super1/super2}
            - young dancer practicing at home-0.mp4
            - young dancer practicing at home-1.mp4
            - ...
        - lifestyle/{super1/super2}
            - the interior design of a shopping mall-0.mp4
            - the interior design of a shopping mall-1.mp4
            - ...
        - plant/{super1/super2}
            - coconut tree near sea under blue sky-0.mp4
            - coconut tree near sea under blue sky-1.mp4
            - ...
        - scenery/{super1/super2}
            - waterfalls in between mountain-0.mp4
            - waterfalls in between mountain-1.mp4
            - ...
        - vehicles/{super1/super2}
            - video of yacht sailing in the ocean-0.mp4
            - video of yacht sailing in the ocean-1.mp4
            - ...
```
*Under each dimension or category in `videocrafter-1`, there are two folders corresponding to the two resolution options for videocrafter-1 generated videos, namely 1024x576 and 512x320. The leaderboard contains the evaluation results for the 1024x576 resolution.*

```
- per_dimension
    - videocrafter-1
        - appearance_style/{1024x576/512x320}       # Optional subfolders 1024x576 and 512x320
            - The bund Shanghai, Van Gogh style-0.mp4
            - The bund Shanghai, Van Gogh style-1.mp4
            - ...
        - human_action/{1024x576/512x320}
            - A person is finger snapping-0.mp4
            - A person is finger snapping-1.mp4
            - ...
        - object_class/{1024x576/512x320}
            - a dining table-0.mp4
            - a dining table-1.mp4
            - ...
        - scene/{1024x576/512x320}
            - restaurant-0.mp4
            - restaurant-1.mp4
            - ...
        - subject_consistency/{1024x576/512x320}
            - a giraffe taking a peaceful walk-0.mp4
            - a giraffe taking a peaceful walk-1.mp4
            - ...
        - temporal_style/{1024x576/512x320}
            - The bund Shanghai, zoom in-0.mp4
            - The bund Shanghai, zoom in-1.mp4
            - ...
        - color/{1024x576/512x320}
            - a blue clock-0.mp4
            - a blue clock-1.mp4
            - ...
        - multiple_objects/{1024x576/512x320}
            - a fire hydrant and a stop sign-0.mp4
            - a fire hydrant and a stop sign-1.mp4
            - ...
        - overall_consistency/{1024x576/512x320}
            - Yellow flowers swing in the wind-0.mp4
            - Yellow flowers swing in the wind-1.mp4
            - ...
        - spatial_relationship/{1024x576/512x320}
            - a frisbee on the left of a sports ball, front view-0.mp4
            - a frisbee on the left of a sports ball, front view-1.mp4
            - ...
        - temporal_flickering/{1024x576/512x320}
            - static view on a desert scene with an oasis, palm trees, and a clear, calm pool of water-0.mp4
            - static view on a desert scene with an oasis, palm trees, and a clear, calm pool of water-1.mp4
            - ...
- per_category
    - videocrafter-1
        - animal/{1024x576/512x320}
            - wild rabbit in a green meadow-0.mp4
            - wild rabbit in a green meadow-1.mp4
            - ...
        - architecture/{1024x576/512x320}
            - water tower on the desert-0.mp4
            - water tower on the desert-1.mp4
            - ...
        - food/{1024x576/512x320}
            - waffles with whipped cream and fruit-0.mp4
            - waffles with whipped cream and fruit-1.mp4
            - ...
        - human/{1024x576/512x320}
            - young dancer practicing at home-0.mp4
            - young dancer practicing at home-1.mp4
            - ...
        - lifestyle/{1024x576/512x320}
            - the interior design of a shopping mall-0.mp4
            - the interior design of a shopping mall-1.mp4
            - ...
        - plant/{1024x576/512x320}
            - coconut tree near sea under blue sky-0.mp4
            - coconut tree near sea under blue sky-1.mp4
            - ...
        - scenery/{1024x576/512x320}
            - waterfalls in between mountain-0.mp4
            - waterfalls in between mountain-1.mp4
            - ...
        - vehicles/{1024x576/512x320}
            - video of yacht sailing in the ocean-0.mp4
            - video of yacht sailing in the ocean-1.mp4
            - ...
```

**`Gen-2` and `Pika` also include videos for "all_dimension" and "all_category", with each folder containing all the respective videos.**
```
- per_dimension
    - gen-2
        - all_dimension
            - Yellow flowers swing in the wind-0.mp4
            - Yellow flowers swing in the wind-1.mp4
            - ...
    - pika
        - all_dimension
            - Yellow flowers swing in the wind-0.mp4
            - Yellow flowers swing in the wind-1.mp4
            - ...
- per_category
    - gen-2
        - all_category
            - young people celebrating new year at the office-0.mp4
            - young people celebrating new year at the office-1.mp4
            - ...
    - pika
        - all_category
            - young people celebrating new year at the office-0.mp4
            - young people celebrating new year at the office-1.mp4
            - ...
```