# Define the dimension list
dimensions=("Human_Anatomy" "Human_Identity" "Human_Clothes" "Diversity" "Composition" "Dynamic_Spatial_Relationship" "Dynamic_Attribute" "Motion_Order_Understanding" "Human_Interaction" "Complex_Landscape" "Complex_Plot" "Camera_Motion" "Motion_Rationality" "Instance_Preservation" "Mechanics" "Thermotics" "Material" "Multi-View_Consistency")

# Corresponding folder names
folders=("Human_Anatomy" "Human_Identity" "Human_Clothes" "Diversity" "Composition" "Dynamic_Spatial_Relationship" "Dynamic_Attribute" "Motion_Order_Understanding" "Human_Interaction" "Complex_Landscape" "Complex_Plot" "Camera_Motion" "Motion_Rationality" "Instance_Preservation" "Mechanics" "Thermotics" "Material" "Multi-View_Consistency")

# Number of tasks to run in parallel
max_parallel_tasks=9
model="vbench2_videos"

# Split dimensions into batches of size `max_parallel_tasks`
total_dimensions=${#dimensions[@]}
for ((start=0; start<total_dimensions; start+=max_parallel_tasks)); do
    end=$((start + max_parallel_tasks))
    if [ $end -gt $total_dimensions ]; then
        end=$total_dimensions
    fi

    # Get the current batch of dimensions and folders
    current_dimensions=("${dimensions[@]:$start:$max_parallel_tasks}")
    current_folders=("${folders[@]:$start:$max_parallel_tasks}")

    # Run tasks in parallel for the current batch
    for i in "${!current_dimensions[@]}"; do
        dimension=${current_dimensions[i]}
        folder=${current_folders[i]}

        videos_path="${model}/${folder}"

        echo "Evaluating '$dimension' in $videos_path"
        suffix="evaluation_results"
        postfix=""
        output_path="${suffix}/${dimension}${postfix}/${model}"

        # Run the evaluation script in parallel with srun
        python evaluate.py --videos_path $videos_path --dimension $dimension --output_path $output_path &
        sleep 1
    done

    # Wait for all tasks in the current batch to finish
    wait
    echo "Batch complete: Dimensions ${current_dimensions[@]}"
done