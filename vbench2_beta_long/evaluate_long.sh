#!/bin/bash

base_path="$1"

# Define the dimension list
dimensions=("subject_consistency" "background_consistency" "aesthetic_quality" "imaging_quality" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "motion_smoothness" "dynamic_degree" "appearance_style")

# Corresponding folder names
folders=("subject_consistency" "scene" "overall_consistency" "overall_consistency" "object_class" "multiple_objects" "color" "spatial_relationship" "scene" "temporal_style" "overall_consistency" "human_action" "temporal_flickering" "subject_consistency" "subject_consistency" "appearance_style")

# Check if the necessary subdirectories exist in the base path
subdirs_found=false

for folder in "${folders[@]}"; do
    if [ -d "$base_path/$folder" ]; then
        subdirs_found=true
        break
    fi
done

# If subdirectories are found, evaluate them, otherwise use base_path
if [ "$subdirs_found" = true ]; then
    # Loop over each dimension and corresponding folder
    for i in "${!dimensions[@]}"; do
        dimension=${dimensions[i]}
        folder=${folders[i]}

        videos_path="${base_path}/${folder}"
        echo "Evaluating '$dimension' in $videos_path"

        # Check if the dimension is 'temporal_flickering' and add the static filter flag
        if [ "$dimension" == "temporal_flickering" ]; then
            python vbench2_beta_long/eval_long.py --videos_path $videos_path --dimension $dimension --mode 'long_vbench_standard' --dev_flag --static_filter_flag
        else
            python vbench2_beta_long/eval_long.py --videos_path $videos_path --dimension $dimension --mode 'long_vbench_standard' --dev_flag
        fi
    done
else
    # If no subdirectories are found, set videos_path to base_path
    videos_path="$base_path"
    echo "No subdirectories found. Using base path $videos_path for evaluation."

    # Run the evaluation
    for i in "${!dimensions[@]}"; do
        dimension=${dimensions[i]}
        echo "Evaluating '$dimension' in $videos_path"

        # Check if the dimension is 'temporal_flickering' and add the static filter flag
        if [ "$dimension" == "temporal_flickering" ]; then
            python vbench2_beta_long/eval_long.py --videos_path $videos_path --dimension $dimension --mode 'long_vbench_standard' --dev_flag --static_filter_flag
        else
            python vbench2_beta_long/eval_long.py --videos_path $videos_path --dimension $dimension --mode 'long_vbench_standard' --dev_flag
        fi
    done
fi