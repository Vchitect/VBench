#!/bin/bash

# Define the dimension list
dimensions=("subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "motion_smoothness" "dynamic_degree" "appearance_style")

# Corresponding folder names
folders=("subject_consistency" "scene" "overall_consistency" "overall_consistency" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "subject_consistency" "subject_consistency" "appearance_style")



# Base path for videos
base_path='./sampled_videos/OpenSoraPlanv1-1'  # TODO: change to local path

# Loop over each dimension
for i in "${!dimensions[@]}"; do
    # Get the dimension and corresponding folder
    dimension=${dimensions[i]}
    folder=${folders[i]}

    # Construct the video path
    videos_path="${base_path}/${folder}"
    echo "$dimension $videos_path"

    # Run the evaluation script
    if [ "$dimension" == "temporal_flickering" ]; then
        python vbench2_beta_long/eval_long.py --videos_path $videos_path --dimension $dimension --mode 'long_vbench_standard' --dev_flag --static_filter_flag
    else
        python vbench2_beta_long/eval_long.py --videos_path $videos_path --dimension $dimension --mode 'long_vbench_standard' --dev_flag
    fi
    
done
