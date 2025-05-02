# Data Augmentation Script

This document describes how to run the Wan2.1 official augmentation script.

## Usage

Replace the content of Wan2.1/wan/utils/prompt_extend.py with that from prompt_extend_fix_seed.py. Then run the following command:

```bash
python aug.py \
    --input ./all_dimension.txt \
    --output ./all_dimension_aug_wanx.txt \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --seed 42
```

This command will:
- Take input from `./all_dimension.txt`
- Write augmented output to `./all_dimension_aug_wanx.txt`
- Use the Qwen2.5-3B-Instruct model for augmentation