

# VBench Prompt Optimization

## Introduction

We follow [CogVideoX](https://github.com/THUDM/CogVideo?tab=readme-ov-file#prompt-optimization), and use GPT-4o to enhance VBench prompts, making them longer and more descriptive without altering their original meaning. This enhancement is achieved by adapting a [script](https://github.com/THUDM/CogVideo/blob/1c2e487820e35ac7f53d2634b69d48c1811f236c/inference/convert_demo.py) from CogVideoX, and it requires OpenAI API keys to call GPT-4o.


The enhanced prompts are available in the `.txt` files within the current folder, with filenames concatenated with `_longer` to indicate the optimized versions.



## Apply Prompt Optimization to VBench Prompts

Simply run this script:

   ```
    sh convert_vbench_prompt.sh
   ```

Some explanations:
1. **Configure API Key and Proxy:**
    Set your OpenAI API key and, if necessary, configure a proxy server.

   ```bash
   API_KEY="your-openai-api-key"
   HTTP_PROXY="http://your-proxy-server:port/"
   HTTPS_PROXY="http://your-proxy-server:port/"
    ```
2. **Set Input File Paths:**

   ```
   INPUT_FILE_CATEGORY="/path/to/your/category/files/"
   INPUT_FILE_DIMENSION="/path/to/your/dimension/files/"
    ```
    For example, in VBench, these two paths are `prompts/prompts_per_category/` and `prompts/prompts_per_dimension/`.
3. **Adjust Retry Times (optional):**
   You can set the number of retry attempts for the script. The default is one retry.
   ```
   RETRY_TIMES=1
    ```




## Sampling Videos Using Optimized Prompts for VBench Evaluation

When sampling videos with the new prompts for VBench evaluation, ensure that the video filenames follow the original VBench prompt format. This allows you to run the evaluation code properly. That is, sample using the optimized prompts, but save videos using the old original prompts as file names.


**Sample Specific Dimensions**

 ```python
 dimension_list = ['object_class', 'overall_consistency']

 for dimension in dimension_list:
    if args.seed:
        torch.manual_seed(args.seed)    

    longer_file_path = f'./prompts/prompts_per_dimension/{dimension}_longer.txt'
    with open(longer_file_path, 'r') as f:
        longer_prompt_list = [prompt.strip() for prompt in f.readlines()]

    original_file_path = f'./prompts/prompts_per_dimension/{dimension}.txt'
    with open(original_file_path, 'r') as f:
        original_prompt_list = [prompt.strip() for prompt in f.readlines()]

    for i, prompt in enumerate(longer_prompt_list):

        original_prompt = original_prompt_list[i]
        samples_per_prompt = 40 if dimension=="temporal_flickering" else 5

        for ind in range(samples_per_prompt):
            print(f"Sampling {prompt} ...")

            video = sample_func(prompt, ind)
            save_path = f'{savedir}/{original_prompt}-{ind}.mp4'
            torchvision.io.write_video(save_path, video, fps=8)

 ```
 **Sample All Dimensions**
 ```python
if args.seed:
    torch.manual_seed(args.seed)    

longer_file_path = f'./prompts/all_dimension_longer.txt'
with open(longer_file_path, 'r') as f:
    longer_prompt_list = [prompt.strip() for prompt in f.readlines()]

original_file_path = f'./prompts/all_dimension.txt'
with open(original_file_path, 'r') as f:
    original_prompt_list = [prompt.strip() for prompt in f.readlines()]

for i, prompt in enumerate(longer_prompt_list):

    original_prompt = original_prompt_list[i]
    samples_per_prompt = 5

    for ind in range(samples_per_prompt):
        print(f"Sampling {prompt} ...")

        video = sample_func(prompt, ind)
        save_path = f'{savedir}/{original_prompt}-{ind}.mp4'
        torchvision.io.write_video(save_path, video, fps=8)
 ```

