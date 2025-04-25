# Data Augmentation Script

This document describes how to run the Wan2.1 augmentation script.


aug.py - This file utilizes the QwenPromptExpander class from the official repository at [generate.py](https://github.com/Wan-Video/Wan2.1/blob/main/generate.py)

## Usage

1. git clone https://github.com/Wan-Video/Wan2.1
2. Place [all_dimension.txt](https://github.com/Vchitect/VBench/blob/master/prompts/all_dimension.txt) and aug.py in Wan2.1/
3. run bash
    ```bash
    python aug.py \
        --input ./all_dimension.txt \
        --output ./all_dimension_aug_wanx.txt \
        --model_name Qwen/Qwen2.5-3B-Instruct
    ```

    This command will:
    - Take input from `./all_dimension.txt`
    - Write augmented output to `./all_dimension_aug_wanx.txt`
    - Use the Qwen2.5-3B-Instruct model for augmentation