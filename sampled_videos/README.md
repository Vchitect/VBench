# Sampled Videos

[![Dataset Download](https://img.shields.io/badge/Dataset-Download-red?logo=googlechrome&logoColor=red)](https://drive.google.com/drive/folders/13pH95aUN-hVgybUZJBx1e_08R6xhZs5X)

To facilitate future research and to ensure full transparency, we release all the videos we sampled and used for VBench evaluation. You can download them on [Google Drive](https://drive.google.com/drive/folders/13pH95aUN-hVgybUZJBx1e_08R6xhZs5X).

## What Videos Do We Provide?
- **8 T2V Models**:
    - including [lavie](https://github.com/Vchitect/LaVie), [modelscope](https://modelscope.cn/models/iic/text-to-video-synthesis/summary), [cogvideo](https://github.com/THUDM/CogVideo), [videocrafter-0.9](https://github.com/AILab-CVC/VideoCrafter/tree/30048d49873cbcd21077a001e6a3232e0909d254), [videocrafter-1](https://github.com/AILab-CVC/VideoCrafter), [show-1](https://github.com/showlab/Show-1), pika, gen-2. More details of models are provided below.
- **2 Suites of Videos for each Model**: 
    - *Per Dimension*: The sampled videos for each ability dimension evaluated by VBench. The per-dimension prompts are available under [`prompts/prompts_per_dimension`](https://github.com/Vchitect/VBench/tree/master/prompts/prompts_per_dimension), and we also provide a combined list of all the dimensions' prompts at [`prompts/all_dimension.txt`](https://github.com/Vchitect/VBench/blob/master/prompts/all_dimension.txt).
    - *Per Category*: The sampled videos for each ability dimension evaluated by VBench. The per-dimension prompts are available under [`prompts/prompts_per_category`](https://github.com/Vchitect/VBench/tree/master/prompts/prompts_per_category), and we also provide a combined list of all the dimensions' prompts at [`prompts/all_category.txt`](https://github.com/Vchitect/VBench/blob/master/prompts/all_category.txt).

What's the potential usage of these videos:
- Further labeling on video quality
- For Instruction Tuning, using our videos and our human preference labels

Below is the folder structure of different models' sampled videos:
```
t2v_sampled_videos
├── per_dimension
│   ├── cogvideo.zip
│   ├── gen-2-all-dimension.tar.gz
│   ├── lavie.zip
│   ├── modelscope.zip
│   ├── opensora.tar
│   ├── pika-all-dimension.zip
│   ├── show-1.tar.gz
│   ├── videocrafter-1.tar.gz
│   ├── videocrafter-2.tar
│   └── videocrafter-09.zip
└── per_category
    ├── cogvideo.zip
    ├── gen-2-all-category.tar.gz
    ├── lavie.zip
    ├── modelscope.zip
    ├── pika-all-category.zip
    ├── show-1.tar.gz
    ├── videocrafter-0.9.zip
    └── videocrafter-1.zip
```
## How to Download the Videos?
You can utilize **gdown** to download from [Google Drive](https://drive.google.com/drive/folders/13pH95aUN-hVgybUZJBx1e_08R6xhZs5X). Below is an example:
- First, install `gdown`:
```
pip install gdown
```
- Then, download zip file using `gdown`:
```
gdown --id <file_id> --output <output_filename>

# Example for videocrafter-1
gdown --id 1FCRj48-Yv7LM7XGgfDCvIo7Kb9EId5KX --output videocrafter-1.tar.gz
```

## What are the Details of the Video Generation Models?
We list the setting for sampling videos from these models.
| Model | Evaluation Party | Release Time | Resolution | FPS | Frame Count | Video Length | Checkpoint | Code Commit ID | Video Format | Sampled Videos (Dimension) | Sampled Videos (Category) |                             Other Settings       &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                        |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |------------------------------|
| [`LaVie`](https://github.com/Vchitect/LaVie) | VBench Team | 2023-09-26 | 512x512 | 8 | 16 | 2.0s | - | - | MP4 | [Google Drive](https://drive.google.com/file/d/1hviZzsInIgJA96ppVj4B2DHhTZWeM4nc/view?usp=drive_link) | [Google Drive](https://drive.google.com/file/d/1aZFhwi6y3LLYyIt5wh2i53Bdg2Rjrn90/view?usp=drive_link) |
| [`LaVie-Interpolation`](https://github.com/Vchitect/LaVie) | VBench Team | 2023-09-26 | 512x512 | 24 | 61 | 2.5s | [link](https://github.com/Vchitect/LaVie?tab=readme-ov-file#download-pre-trained-models) | - | MP4 | [Google Drive](https://drive.google.com/file/d/1Tbw6FBYp_VxeFGoChebFhBr7ewSc9uFv/view?usp=sharing) | - |
| [`ModelScope`](https://modelscope.cn/models/iic/text-to-video-synthesis/summary) | VBench Team | 2023-08-12 | 256x256 | 8 | 16 | 2.0s | [link](https://modelscope.cn/models/iic/text-to-video-synthesis/files) | - | MP4 | [Google Drive](https://drive.google.com/file/d/1UH2-lALFShjBywyImjDPPHTpE43eoMQE/view?usp=drive_link) | [Google Drive](https://drive.google.com/file/d/1gwLdeEnXsb0Aq1y5x18vfArZVp11W8tp/view?usp=drive_link) |
| [`CogVideo`](https://github.com/THUDM/CogVideo) | VBench Team | 2022-05-29 | 480x480 | 10 | 33 | 3.3s | [link](https://github.com/THUDM/CogVideo?tab=readme-ov-file#download) | - | GIF | [Google Drive](https://drive.google.com/file/d/1-oAHf6inm4CFeldKktWerXkjwQ_q26Ic/view?usp=drive_link) | [Google Drive](https://drive.google.com/file/d/1tRPwqlxgcpLp96yDyYIuSed-S18VCyft/view?usp=drive_link) |
| [`VideoCrafter-0.9`](https://github.com/AILab-CVC/VideoCrafter/tree/30048d49873cbcd21077a001e6a3232e0909d254) | VBench Team | 2023-04-05 | 256x256 | 8 | 16 | 2.0s | [link](https://huggingface.co/VideoCrafter/t2v-version-1-1/blob/main/models/base_t2v/model_rm_wtm.ckpt) | [Commit ID](https://github.com/AILab-CVC/VideoCrafter/tree/30048d49873cbcd21077a001e6a3232e0909d254) | MP4 | [Google Drive](https://drive.google.com/file/d/1VoNPAttMFOV_6FIYCGW4fzFE9m18Ry22/view?usp=drive_link) | [Google Drive](https://drive.google.com/file/d/1xVbd-Guzt-3VXAlwNCU4UQYJqJGojHdL/view?usp=drive_link) |
| [`VideoCrafter-1.0`](https://github.com/AILab-CVC/VideoCrafter) | VBench Team |  2023-10-30 |1024x576 | 10 | 16 | 1.6s | [link](https://huggingface.co/VideoCrafter/Text2Video-1024/blob/main/model.ckpt) | [Commit ID](https://github.com/AILab-CVC/VideoCrafter/tree/dab05359fd0d232ccab8bc4e782501ef62a73ab9) | MP4 | [Google Drive](https://drive.google.com/file/d/1FCRj48-Yv7LM7XGgfDCvIo7Kb9EId5KX/view?usp=drive_link) | [Google Drive](https://drive.google.com/file/d/12OYfhGfwODNGLUe9Ur4Fn2GNnHFh55_F/view?usp=drive_link) |
| [`Show-1`](https://github.com/showlab/Show-1) | VBench Team | 2023-09-27 | 576x320 | 8 | 29 | 3.6s | [link](https://huggingface.co/showlab/show-1-sr2#:~:text=git%20lfs%20install%0A%0A%23%20base%0Agit%20clone%20https%3A//huggingface.co/showlab/show%2D1%2Dbase%0A%23%20interp%0Agit%20clone%20https%3A//huggingface.co/showlab/show%2D1%2Dinterpolation%0A%23%20sr1%0Agit%20clone%20https%3A//huggingface.co/showlab/show%2D1%2Dsr1%0A%23%20sr2%0Agit%20clone%20https%3A//huggingface.co/showlab/show%2D1%2Dsr2) | [Commit ID](https://github.com/showlab/Show-1/tree/da9b24b47fbe21daabf44dba20158951defa7831) | MP4 | [Google Drive](https://drive.google.com/file/d/1QOInCcCI04LQ38BiY0o4oLehAFQfiVh2/view?usp=drive_link) | [Google Drive](https://drive.google.com/file/d/1CDjGAyEjEmOpIXuZb-HoCff3QNNXQyxo/view?usp=drive_link) |
| [`Gen-2`](https://runwayml.com/ai-tools/gen-2/) | VBench Team | 2023-06-07 | 1408x768 | 24 | 96 | 4.0s | - | - | MP4 | [Google Drive](https://drive.google.com/file/d/1tPL_PMmnBM4518UNiu52nhQCbUmF0A8q/view?usp=drive_link) | [Google Drive](https://drive.google.com/file/d/1jW_04y7SLLNyo3DKIOrsS68t3IglbBoX/view?usp=drive_link) |
| [`Pika`](https://discord.com/invite/pika) | VBench Team | 2023-06-29 | 1088x640 | 24 | 72 | 3.0s | - | - | MP4 | [Google Drive](https://drive.google.com/file/d/1G2VVD5ArLxYtKeAVdANnxNNAPlP2bbZO/view?usp=drive_link) | [Google Drive](https://drive.google.com/file/d/1t8d7GbZ6IB1on11FkvjhejiqwQRd-Er1/view?usp=drive_link) |
| [`Open-Sora`](https://github.com/hpcaitech/Open-Sora) | VBench Team | 2024-03-18 | 512x512 | 8 | 16 | 2.0s | [link](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x512x512.pth ) |  [Commit ID](https://github.com/hpcaitech/Open-Sora/tree/a5afed2fc3f7d14f6f2d1ea81dd90cb8fff92d93) | MP4 | [Google Drive](https://drive.google.com/file/d/1LCyTaVT_N_sM3HkSF1lPIPC0w80fqkEe/view?usp=sharing) | - |
| [`VideoCrafter-2.0`](https://github.com/AILab-CVC/VideoCrafter) | VBench Team | 2024-01-18 | 320x512 | 10 | 16 | 1.6s | [link](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt) | [Commit ID](https://github.com/AILab-CVC/VideoCrafter/tree/89c201c52933f5f3db7cebd46320c002dd434c0e) | MP4 | [Google Drive](https://drive.google.com/file/d/17podJKS0tbfUS8dVAPNyDv4vYo4dIDqL/view?usp=sharing) | - |
| [`T2V-Turbo (VC2)`](https://github.com/Ji4chenLi/t2v-turbo) | T2V-Turbo Team | 2024-05-29 | 320x512 | 16 | 16 | 1.0s | [link](https://huggingface.co/jiachenli-ucsb/T2V-Turbo-VC2/blob/main/unet_lora.pt) | [Commit ID](https://github.com/Ji4chenLi/t2v-turbo/tree/de442b4d71c620eefa1c296682ebd135bb587ec7) | MP4 | - | - | <small>`unet_lora.pt` is used to turn VideoCrafter-2.0 to `T2V-Turbo (VC2)`</small> |
| [`AnimateDiff-V1`](https://github.com/guoyww/animatediff/) | VBench Team |  2023-07-18 | 512x512 | 8 | 16 | 2.0s | [T2I backbone SD1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [Motion Module](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15.ckpt), [LoRA(Realistic Vision 2.0)](https://civitai.com/models/4201?modelVersionId=130072) | [Commit ID](https://github.com/guoyww/AnimateDiff/tree/cf80ddeb47b69cf0b16f225800de081d486d7f21) | MP4 | [Google Drive](https://drive.google.com/file/d/1S8ObJUtq0ETbR9sJ6nhnPCsQfr26JM9D/view?usp=drive_link) | - | <details><summary>Negative Prompt</summary><small>We apply the same negative prompt during sampling for all videos: ```semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck```</small></details> |
| [`AnimateDiff-V2`](https://github.com/guoyww/animatediff/) | VBench Team |  2023-09-10 | 512x512 | 8 | 16 | 2.0s | [T2I backbone SD1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [Motion Module](https://huggingface.co/guoyww/animatediff/blob/main/mm_sd_v15_v2.ckpt), [LoRA](https://civitai.com/models/4201?modelVersionId=130072) | [Commit ID](https://github.com/guoyww/AnimateDiff/tree/cf80ddeb47b69cf0b16f225800de081d486d7f21) | MP4 | [Google Drive](https://drive.google.com/file/d/1a9dPyArEWt61NS3E2VDws8wMAXI-MX04/view?usp=sharing) | - | <details><summary>Negative Prompt</summary><small>We apply the same negative prompt during sampling for all videos: ```semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck```</small></details> |
| [`Latte-1`](https://github.com/Vchitect/Latte) | VBench Team | 2024-05-23 | 512x512 | 8 | 16 | 2.0s | [link](https://huggingface.co/maxin-cn/Latte-1/tree/main) |  [Commit ID](https://github.com/Vchitect/Latte/tree/5f0fbed8bfa112cdc979450dded03243faee025f) | MP4 | [Google Drive](https://drive.google.com/file/d/1plPbWcX2UGX0eA3S1BwFFtPtk0sv7JUf/view?usp=drive_link) | - |
| [`OpenSora V1.2 (2s)`](https://github.com/hpcaitech/Open-Sora) | OpenSora Team | 2024-06-28 | 854×480 | 24 | 51 | 2s | [link](https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3) | - | MP4 | [link](https://hpcaitech.github.io/Open-Sora/) | -| <small>eval results & info provided by OpenSora Team</small> |
| [`HiGen`](https://github.com/ali-vilab/VGen) | VBench Team | 2024-03-08 | 448x256 | 8 | 32 | 4.0s | [link](https://modelscope.cn/models/iic/HiGen) |  [Commit ID](https://github.com/ali-vilab/VGen/tree/7ad0f25df50b2c35d7eb95cbabdf772c5b9761c8) | MP4 | [Google Drive](https://drive.google.com/file/d/1Y1rgLfXe5bC8IJwU5RErbOlCO4OyPqFU/view?usp=drive_link) | - |
| [`TF-T2V`](https://github.com/ali-vilab/VGen) | VBench Team | 2024-04-03 | 448x256 | 8 | 32 | 4.0s | [link](https://modelscope.cn/models/iic/tf-t2v/files) |  [Commit ID](https://github.com/ali-vilab/VGen/tree/7ad0f25df50b2c35d7eb95cbabdf772c5b9761c8) | MP4 | [Google Drive](https://drive.google.com/file/d/125O9CIZrFcgFGwBHzXhEGn5RysSLNbcv/view?usp=drive_link) | - |
| [`AnimateLCM`](https://github.com/G-U-N/AnimateLCM) | VBench Team | 2024-02-26 | 512x512 | 8 | 16 | 2.0s | [link](https://huggingface.co/wangfuyun/AnimateLCM/tree/main) |  [Commit ID](https://github.com/G-U-N/AnimateLCM/tree/f65d2fdd00f0a3ba45eaaa9bbc8751bf1018786d) | MP4 | [Google Drive](https://drive.google.com/file/d/101RjKgdAaLOHgk9kxleCCjdW8Wh3ccM0/view?usp=drive_link) | - | <details><summary>Negative Prompt</summary><small>We apply the same negative prompt during sampling for all videos: ```bad quality, worse quality, low resolution```</small></details>
| [`InstructVideo(ModelScope)`](https://instructvideo.github.io/) | VBench Team | 2024-06-17 | 256x256 | 8 | 16 | 2.0s | [link](https://modelscope.cn/models/iic/InstructVideo/files) |  [Commit ID](https://github.com/ali-vilab/VGen/tree/aca9a5d3168b07492b440c97404cbbd8f743a412) | MP4 | [Google Drive](https://drive.google.com/file/d/1OiDttO6_xEqHweyjPmiq1JiibZfmIKSw/view?usp=drive_link) | - |
| [`OpenSora V1.1`](https://github.com/hpcaitech/Open-Sora) | VBench Team | 2024-04-25 | 424x240 | 8 | 64 | 8.0s | [link](https://github.com/hpcaitech/Open-Sora?tab=readme-ov-file#open-sora-11-model-weights:~:text=Open%2DSora%201.1%20Model%20Weights) |  [Commit ID](https://github.com/hpcaitech/Open-Sora/commit/ea41df3d6cc5f389b6824572854d97fa9f7779c3) | MP4 | [Google Drive](https://drive.google.com/file/d/1mGxjDIf7IT_mNibG8Nmg3E1WcXYRVDoo/view?usp=drive_link) | - |
| [`OpenSoraPlan V1.1`](https://github.com/PKU-YuanGroup/Open-Sora-Plan) | VBench Team | 2024-05-27 | 512x512 | 24 | 221 | 9.2s | [link](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.1.0/tree/main) |  [Commit ID](https://github.com/PKU-YuanGroup/Open-Sora-Plan/commit/b08681f697658c81361e1ec6c07fba55c79bb4bd) | MP4 | [Google Drive](https://drive.google.com/file/d/1zsg-HPiqYZJoryTXw6cG_9deBlIbZN87/view?usp=drive_link) | - |
| [`Mira`](https://github.com/mira-space/Mira) | VBench Team | 2024-04-01 | 384x240 | 6 | 60 | 10.0s | [link](https://github.com/mira-space/Mira) |  [Commit ID](https://github.com/mira-space/Mira/commit/12f8458f082405839a73c867016d60ee40b4f514) | MP4 | [Google Drive](https://drive.google.com/file/d/1lx0evF0HN0jY3FQ41RhQL9UJbOa-gve6/view?usp=drive_link) | - |
| [`Pika 1.0`](https://pika.art/home) | VBench Team | 2023-12-28 | 1280x720 | 24 | 72 | 3.0s | - | - | MP4 | [Google Drive](https://drive.google.com/file/d/1FH157pt3KSy7O9HqSF6V9_CxMXijbAXK/view?usp=drive_link) | [Google Drive](https://drive.google.com/file/d/1akR-7EhrNOpkWd-5pRR4StMKjj7bOrQJ/view?usp=drive_link) |
| [`Gen-3`](https://runwayml.com/ai-tools/gen-3-alpha/) | VBench Team | 2024-06-17 | 1280x768 | 24 | 256 | 10.7s | - | - | MP4 | [Google Drive](https://drive.google.com/drive/u/0/folders/1AFV48EOAXydz2ZB-q2ml7b0ojYrAsl-K) | [Google Drive](https://drive.google.com/drive/u/1/folders/1JhWU509RSif78q4lwSrfKVoy9m7_ZoDp) |
| [`Kling`](https://klingai.kuaishou.com/) | VBench Team | 2024-06-06 | 1280x720 | 30 | 153 | 5.1s | - | - | MP4 | [Google Drive](https://drive.google.com/drive/u/1/folders/1g5Y9j2gb9I5FUg4Ql28jSCCPH-Pv-HDg) | - | <small>high-performance mode (lower sampling cost), not high-quality mode (better quality)</small> |
| [`Data-Juicer (T2V-Turbo)`](https://modelscope.cn/models/Data-Juicer/Data-Juicer-T2V) | Data-Juicer Team | 2024-07-23 | 320x512 | 8 | 16 | 2.0s | - | - | MP4 | - | - | <small>from Data-Juicer Team: based on T2V-Turbo, with Data-Juicer's data and loss enhancement</small>  |
| [`LaVie-2`](https://github.com/Vchitect/LaVie) | LaVie-2 Team | - | 512x512 | 8 | 16 | 2.0s | - | - | MP4 | - | - | <small>info provided by LaVie-2 Team</small>
| [`CogVideoX-2B (SAT, prompt-optimized)`](https://github.com/THUDM/CogVideo) | VBench Team | 2024-08-06 | 720x480 | 8 | 49 | 6.1s | [link](https://github.com/THUDM/CogVideo/tree/1c2e487820e35ac7f53d2634b69d48c1811f236c/sat) |  [Commit ID](https://github.com/THUDM/CogVideo/tree/1c2e487820e35ac7f53d2634b69d48c1811f236c) | MP4 | [Google Drive](https://drive.google.com/file/d/1zuQ47Uvze4157o4YMta0Zqz9G8TdHcXZ/view?usp=drive_link) | [Google Drive](https://drive.google.com/file/d/1hOYWwDSw409AaGC2eUYNPnb4vXyUwXMO/view?usp=drive_link) | <small>Prompt Optimization is used, see [here](https://github.com/Vchitect/VBench/tree/master/prompts/gpt_enhanced_prompts)</small> |
| [`OpenSora V1.2 (8s)`](https://github.com/hpcaitech/Open-Sora) | VBench Team | 2024-06-17 | 1280x720 | 24 | 204 | 8.5s | [link](https://github.com/hpcaitech/Open-Sora?tab=readme-ov-file#open-sora-12-model-weights) |  [Commit ID](https://github.com/hpcaitech/Open-Sora/tree/476b6dc79720e5d9ddfb3cd589680b2308871926) | MP4 | [Google Drive](https://drive.google.com/file/d/1NMIKt3v01xsR6C32hTMt9ZRmRtSHo-uz/view?usp=drive_link) | [Google Drive](https://drive.google.com/file/d/1hBdK9q6rVr-bUIsm_Gmdvo7698EG0oCA/view?usp=drive_link) |
| [`CogVideoX-5B (SAT, prompt-optimized)`](https://github.com/THUDM/CogVideo) | VBench Team | 2024-08-27 | 720x480 | 8 | 49 | 6.1s | [link](https://github.com/THUDM/CogVideo/tree/f37582f80121059cae8e68f50e089396463a0e85/sat) |  [Commit ID](https://github.com/THUDM/CogVideo/tree/f37582f80121059cae8e68f50e089396463a0e85) | MP4 | [Google Drive](https://drive.google.com/file/d/11KOdeSHxjRqplpy3sYNsQtJyYm416sTk/view?usp=drive_link) | - | <small>Prompt Optimization is used, see [here](https://github.com/Vchitect/VBench/tree/master/prompts/gpt_enhanced_prompts)</small> |

## How are Files Structured in Google Drive?


### 1. Sub-Folder Organization

For these models, 
- (1) The `per_dimension` zip contains 11 subfolders corresponding to videos sampled for evaluating different dimensions. 
- (1) The `per_category` zip contains 8 subfolders corresponding to videos sampled for evaluating different content categories. 


#### 1.1. Single-Stage Outputs 

For `LaVie, ModelScope, CogVideo, VideoCrafter-0.9, Open-Sora, VideoCrafter-2.0, AnimateDiff-V2`, we provide their single-stage outputs.

We take `LaVie` as an example:

```
- per_dimension
    - lavie
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

#### 1.2. Multi-Stage Outputs (Show-1)

For `show-1`, there are two folders corresponding to the last two stages of show-1 generated videos, namely `super1` and `super2`. The leaderboard results correspond to evaluation on the final stage, namely `super2`.

```
- per_dimension
    - show-1
        - appearance_style/{super1/super2}       # subfolder super1 or super2
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
#### 1.3. Multi-Resolution Outputs (VideoCrafter-1)

Under each dimension or category in `videocrafter-1`, there are two folders corresponding to the two resolution options for videocrafter-1 generated videos, namely 1024x576 and 512x320. The leaderboard currently contains the evaluation results for the 1024x576 resolution.

```
- per_dimension
    - videocrafter-1
        - appearance_style/{1024x576/512x320}       # subfolder 1024x576 or 512x320
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

### 2. Single-Folder Organization (Gen-2, Pika)

`Gen-2` and `Pika` also include videos for "all_dimension" and "all_category", but we haven't divide the videos into subfolders according to specific dimensions or categories yet.
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

## Human Preference Labels

Available for download at [Google Drive](https://drive.google.com/drive/folders/1jYAybu2BazShGV-DLityFi4j7BjTE-my?usp=sharing).

Each dimension contains an annotation file, each of which contains a list, and the list contains manually preferred annotation results of videos generated by different prompts. The evaluation process involves comparing videos from different models and, based on human annotations, determining which video best matches the prompt for the corresponding dimension.

### Data Structure

JSON data is composed of multiple objects, each representing an evaluation instance. Each instance contains the following key-value pairs:

`prompt_en`: The text prompt for generating the desired video content.

`style_en`/`color_en`/`object_en` ..: Dimension-related information.

`question_en`: The question asked to the human annotators / VLM.

`videos`: This section contains the urls to videos from different models.

`human_anno`: This section represents human annotation, which is composed of a nested dictionary. The outer keys represent the model names (e.g., "modelscope", "lavie"), and the inner keys represent the other model names. The corresponding values within these nested dictionaries represent the human-assigned scores for the relative quality of each model's video compared to the other model's video.

For example, `human_anno["modelscope"]["lavie"] = 0` indicates that humans judged the Lavie video to be better than the Modelscope video for the given prompt and style.

`human_anno["modelscope"]["videocraft"] = 1` indicates that humans judged the Modelscope video to be better than the Videocraft video.

`human_anno["cogvideo"]["videocraft"] = 0.5` indicates that humans judged the Cogvideo video and the Videocraft video to be of equal quality.
