# :bookmark_tabs: Prompt Suite

We design compact yet representative prompts in terms of both the evaluation dimensions and the content categories.


## Prompts per Dimension
`prompts/prompts_per_dimension`: For each VBench evaluation dimension, we carefully designed a set of around 100 prompts as the test cases.


### Pseudo-Code for Video Sampling:
```
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
            video_, video_path = sample_per_video(dimension, prompt, idx)    
            
            # save sampled video to f'{base_path}/{model_name}/{dimension_name}/{prompt}-{index}.mp4'
            torchvision.io.write_video(video_path, video_, fps=8)
```

### Further Explanations
To sample videos for VBench evaluation:
- Sample videos from all the `txt` files in `prompts/prompts_per_dimension`. 
- For each prompt, sample 5 videos.
- At the beginning of sampling from each `txt` file, set the random seed.
- Save the videos in the form of `$base_path/$model_name/$dimension_name/$prompt-$index.mp4`, `$index` takes value of `0, 1, 2, 3, 4`. For example:
    ```
    vbench_videos/lavie/overall_consistency                    
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

## Prompts per Category
`prompts/prompts_per_category`: 100 prompts for each of the 8 content categories: `Animal`, `Architecture`, `Food`, `Human`, `Lifestyle`, `Plant`, `Scenery`, `Vehicles`.

## Metadata
`prompts/metadata`: metadata for some prompt lists, such as the `color` and `object_class` labels for prompts that need to be semantically parsed.
