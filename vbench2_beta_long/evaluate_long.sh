#!/bin/bash

# Define the model list
models=("Sora")

# Define the dimension list
dimensions=("subject_consistency" "background_consistency"  "motion_smoothness" "dynamic_degree" "aesthetic_quality" "imaging_quality")

# Corresponding folder names

# Base path for videos
base_path='./long_videos/' # TODO: change to local path
output_path="evaluation_results/${model}"

# Loop over each model
for model in "${models[@]}"; do
    # Loop over each dimension
    for i in "${!dimensions[@]}"; do
        # Get the dimension and corresponding folder
        dimension=${dimensions[i]}
        

        # Construct the video path
        videos_path="${base_path}${model}"
        echo "$dimension $videos_path"

        # Run the evaluation script
        python eval_long.py --videos_path $videos_path --dimension $dimension --output_path $output_path --mode 'long_custom_input' --use_semantic_splitting
    done
done
