# Define the dimension list
dimensions=("Human_Anatomy" "Human_Identity" "Human_Clothes" "Diversity" "Composition" "Dynamic_Spatial_Relationship" "Dynamic_Attribute" "Motion_Order_Understanding" "Human_Interaction" "Complex_Landscape" "Complex_Plot" "Camera_Motion" "Motion_Rationality" "Mechanics" "Thermotics" "Material" "Multi-View_Consistency" "Instance_Preservation")

# Corresponding folder names
folders=("Human_Anatomy" "Human_Identity" "Human_Clothes" "Diversity" "Composition" "Dynamic_Spatial_Relationship" "Dynamic_Attribute" "Motion_Order_Understanding" "Human_Interaction" "Complex_Landscape" "Complex_Plot" "Camera_Motion" "Motion_Rationality" "Mechanics" "Thermotics" "Material" "Multi-View_Consistency" "Instance_Preservation")

# Number of tasks to run in parallel
max_parallel_tasks=8
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
        output_path="${suffix}/${dimension}/${model}"

        # if excute the code directly by python
        gpu_id=$(( (start + i) % max_parallel_tasks ))
        echo "Running '$dimension' on GPU $gpu_id"
        CUDA_VISIBLE_DEVICES=$gpu_id python evaluate.py --videos_path $videos_path --dimension $dimension --output_path $output_path &

        # elif excute the code by slurm
        # srun -p YOUR_PARTITION --gres=gpu:1 python evaluate.py --videos_path $videos_path --dimension $dimension --output_path $output_path &

        sleep 1
    done

    # Wait for all tasks in the current batch to finish
    wait
    echo "Batch complete: Dimensions ${current_dimensions[@]}"
done