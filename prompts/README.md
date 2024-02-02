# :bookmark_tabs: Prompt Suite

We design compact yet representative prompts in terms of both the evaluation dimensions and the content categories.


## Prompts per Dimension
`prompts/prompts_per_dimension`: For each VBench evaluation dimension, we carefully designed a set of around 100 prompts as the test cases.
We provide a combined list `prompts/all_dimension.txt`, which combines all the prompts under `prompts/prompts_per_dimension`.

## Prompts per Category
`prompts/prompts_per_category`: 100 prompts for each of the 8 content categories: `Animal`, `Architecture`, `Food`, `Human`, `Lifestyle`, `Plant`, `Scenery`, `Vehicles`.
We provide a combined list `prompts/all_category.txt`, which combines all the prompts under `prompts/prompts_per_category`.

## Metadata
`prompts/metadata`: metadata for some prompt lists, such as the `color` and `object_class` labels for prompts that need to be semantically parsed.


# How to Sample Videos for Evaluation

We specify how to sample from `Prompts per Category` for VBench evaluation, and that for `Prompts per Category` can be carried out similarly. 


## Evaluate Some Dimensions

### Pseudo-Code for Sampling
- If you only want to evaluate certain dimensions, below are the pseudo-code for sampling.
    ```
    dimension_list = ['object_class', 'overall_consistency']

    for dimension in dimension_list:

        # set random seed
        if args.seed:
            torch.manual_seed(args.seed)    
        
        # read prompt list
        with open(f'./prompts/prompts_per_dimension/{dimension}.txt', 'r') as f:
            prompt_list = f.readlines()
        prompt_list = [prompt.strip() for prompt in prompt_list]
        
        for prompt in prompt_list:

            # sample 5 videos for each prompt
            for index in range(5):

                # perform sampling
                video = sample_func(prompt, index)    
                cur_save_path = f'{args.save_path}/{prompt}-{index}.mp4'
                torchvision.io.write_video(cur_save_path, video, fps=8)
    ```

### Further Explanations

To sample videos for VBench evaluation:
- Sample videos from all the `txt` files in `prompts/prompts_per_dimension`. 
- For each prompt, sample 5 videos.
- **Random Seed**: At the beginning of sampling from each `txt` file, set the random seed. For some models, the random seed is independently and randomly drawn for each video sample, and this is also acceptable, but it would be the best to record the random seed of every video being sampled. We need to ensure: (1) The random seeds are random, and not cherry picked. (2) The sampling process is reproducible, so that the evaluation results are reproducible.
- Name the videos in the form of `$prompt-$index.mp4`, `$index` takes value of `0, 1, 2, 3, 4`. For example:
    ```                   
    ├── A 3D model of a 1800s victorian house.-0.mp4                                       
    ├── A 3D model of a 1800s victorian house.-1.mp4                                       
    ├── A 3D model of a 1800s victorian house.-2.mp4                                       
    ├── A 3D model of a 1800s victorian house.-3.mp4                                       
    ├── A 3D model of a 1800s victorian house.-4.mp4                                       
    ├── A beautiful coastal beach in spring, waves lapping on sand by Hokusai, in the style of Ukiyo-0.mp4                                                                      
    ├── A beautiful coastal beach in spring, waves lapping on sand by Hokusai, in the style of Ukiyo-1.mp4                                                                      
    ├── A beautiful coastal beach in spring, waves lapping on sand by Hokusai, in the style of Ukiyo-2.mp4                                                                      
    ├── A beautiful coastal beach in spring, waves lapping on sand by Hokusai, in the style of Ukiyo-3.mp4                                                                      
    ├── A beautiful coastal beach in spring, waves lapping on sand by Hokusai, in the style of Ukiyo-4.mp4 
    ......
    ```
## Evaluate All Dimensions

- If you want to evaluate all the dimensions, below are the pseudo-code for sampling.
    ```
    # set random seed
    if args.seed:
        torch.manual_seed(args.seed)    
    
    # read prompt list
    with open(f'./prompts/all_dimension.txt', 'r') as f:
        prompt_list = f.readlines()
    prompt_list = [prompt.strip() for prompt in prompt_list]
    
    for prompt in prompt_list:

        # sample 5 videos for each prompt
        for index in range(5):

            # perform sampling
            video = sample_func(prompt, index)    
            cur_save_path = f'{args.save_path}/{prompt}-{index}.mp4'
            torchvision.io.write_video(cur_save_path, video, fps=8)
    ```

