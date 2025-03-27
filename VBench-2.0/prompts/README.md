# :bookmark_tabs: Prompt Suite

We design compact yet representative prompts in terms of the evaluation dimensions.


## Prompts per Dimension
`prompts/prompt`: For each VBench-2.0 evaluation dimension, we carefully designed a set of around 70 prompts as the test cases.
We provide a combined list `prompts/VBench2_full_text.txt`, which combines all the prompts under `prompts/prompt`.

## Augmented Prompts per Dimension
`prompts/prompt_aug`: The augmented prompt we used in the paper for `HunyuanVideo`, `CogVideoX-1.5` and `Kling 1.6`.

## Metadata
`prompts/metainfo`: metadata for some prompt lists, such as the `camera_motion` and `extensive_description` labels for prompts that need to be semantically parsed.


# How to Sample Videos for Evaluation

We specify how to sample from `Prompt` for VBench-2.0 evaluation. 
#### Please make sure to use a different `random seed` for sampling each video to ensure diversity in the sampled content.
#### Please make sure to name the video based on our given prompts in `prompts/prompt` or `prompts/VBench2_full_text.txt` whatever you use the prompt refiner.
#### We offer two methods to sample the the video, dimension-based or full_prompt_list-based. See the pseudo-codes below for the difference.
## Sample Some Dimensions

### Pseudo-Code for Sampling
- If you only want to evaluate certain dimensions or all dimension without post-processing, below are the pseudo-code for sampling.
    ```python
    dimension_list = ['Camera_Motion', 'Complex_Plot']

    for dimension in dimension_list:

        # set random seed
        if args.seed:
            torch.manual_seed(args.seed)    
        
        # read prompt list
        with open(f'./prompts/prompt/{dimension}.txt', 'r') as f:
            prompt_list = f.readlines()
        prompt_list = [prompt.strip() for prompt in prompt_list]
        
        for prompt in prompt_list:

            # sample 20 videos for `Diversity` and 3 videos for others
            if dimension=='Diversity':
                iter=20
            else:
                iter=3
            for index in range(iter):

                # perform sampling
                video = sample_func(prompt, index)    
                # Note that VBench-2.0 contains prompts which exceed the max length of naming, so we use the first 180 characters of the prompt as the name.
                # If users use the prompt refiner, please still name the video by our given prompts.
                # If users use this type to sample the video, ensuring that the video is saved based on the dimension as the prompts between dimensions have overlap.
                os.makedirs(f'{args.save_path}/{dimension}', exist_ok=True)
                cur_save_path = f'{args.save_path}/{dimension}/{prompt[:180]}-{index}.mp4'
                # The fps depends on your model.
                torchvision.io.write_video(cur_save_path, video, fps=8)
    ```

### Further Explanations

To sample videos for VBench-2.0 evaluation:
- Sample videos from all the `txt` files in `prompts/prompt`. 
- For each prompt, sample 3 videos. However, for the `Diversity` dimension, sample 20 videos to ensure the number for diversity evaluation.
- **Random Seed**: At the beginning of sampling from each `txt` file, set the random seed. For some models, the random seed is independently and randomly drawn for each video sample, and this is also acceptable, but it would be the best to record the random seed of every video being sampled. We need to ensure: (1) The random seeds are random, and not cherry picked. (2) The sampling process is reproducible, so that the evaluation results are reproducible.
- Name the videos in the form of `$prompt[:180]-$index.mp4`, `$index` takes value of `0, 1, 2`, `0, 1, ..., 19` for `Diversity`. For example:
    ```   
    ├── Camera_Motion                
        ├── Garden, zoom in.-0.mp4                                       
        ├── Garden, zoom in.-1.mp4                                       
        ├── Garden, zoom in.-2.mp4  
    ├── Diversity                                     
        ├── A man is playing basketball.-0.mp4                                                                      
        ├── A man is playing basketball.-1.mp4                                                                      
        ......                                                                     
        ├── A man is playing basketball.-19.mp4 
    ......
    ```
## Evaluate All Dimensions

- If you want to evaluate all the dimensions with the `VBench2_full_text.txt`, below are the pseudo-code for sampling.
    ```python
    # set random seed
    if args.seed:
        torch.manual_seed(args.seed)    
    
    # read prompt list
    with open(f'./prompts/VBench2_full_text.txt', 'r') as f:
        prompt_list = f.readlines()
    prompt_list = [prompt.strip() for prompt in prompt_list]
    
    for idx, prompt in enumerate(prompt_list):

        # sample 20 videos for `Diversity` and 3 videos for others, in `VBench2_full_text.txt`, we put the prompts of `Diversity` in the beginning.
        if idx<10:
            iter=20
        else:
            iter=3
        for index in range(iter):

            # perform sampling
            video = sample_func(prompt, index)    
            # Note that VBench-2.0 contains prompts which exceed the max length of naming, so we use the first 180 characters of the prompt as the name.
            # If you use the prompt refiner, please still name the video by our given prompts.
            cur_save_path = f'{args.save_path}/{prompt[:180]}-{index}.mp4'
            # The fps depends on your model.
            torchvision.io.write_video(cur_save_path, video, fps=8)
    
    # `divide_video.py`, modify the `source_video_file` into yours, the `target_video_file` is set to `vbench2_videos` as default.
    # users could also directly send the whole video files to us and we will do the remaining process.
    divide_video(args.save_path) 
    ```