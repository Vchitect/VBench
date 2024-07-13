#!/bin/bash

# Define the dimension list
dimensions=(
    "subject_consistency" 
    "background_consistency" 
    "aesthetic_quality" 
    "imaging_quality" 
    "object_class" 
    "multiple_objects" "color" 
    "spatial_relationship" 
    "scene" 
    "temporal_style" 
    "overall_consistency" 
    "human_action" 
    "temporal_flickering" 
    "motion_smoothness" 
    "dynamic_degree" 
    "appearance_style")

# Corresponding folder names
folders=("subject_consistency" "scene" "overall_consistency" "overall_consistency" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "subject_consistency" "subject_consistency" "appearance_style")

# Base path for videos
base_path='./sampled_videos/videocrafter-1' # TODO: change to local path

DEFAULT_GPUS=8
# get number of gpus
REAL_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# min of REAL_GPUS and DEFAULT_GPUS
GPUS=$((REAL_GPUS < DEFAULT_GPUS ? REAL_GPUS : DEFAULT_GPUS))

echo "Using $GPUS GPUs"

# Loop over each dimension
for i in "${!dimensions[@]}"; do
    # Get the dimension and corresponding folder
    dimension=${dimensions[i]}
    folder=${folders[i]}

    # Construct the video path
    videos_path="${base_path}/${folder}/1024x576"
    echo "$dimension $videos_path"

    # Run the evaluation script
    torchrun --nproc_per_node=${GPUS} --standalone \
        evaluate.py --videos_path $videos_path --dimension $dimension
done
