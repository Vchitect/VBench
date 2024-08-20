API_KEY="your-openai-api-key"
HTTP_PROXY="http://your-proxy-server:port/"
HTTPS_PROXY="http://your-proxy-server:port/"

INPUT_FILE_CATEGORY="prompts/prompts_per_category/"
INPUT_FILE_DIMENSION="prompts/prompts_per_dimension/"

RETRY_TIMES=1

export OPENAI_API_KEY="$API_KEY"
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"

dimension_list=("subject_consistency" "temporal_flickering" "object_class" 
                "multiple_objects" "human_action" "color" 
                "spatial_relationship" "scene" "temporal_style" 
                "appearance_style" "overall_consistency")
category_list=("animal" "architecture" "food" "human" "lifestyle" "plant" "scenery" "vehicles")

for dimension in "${dimension_list[@]}"
do
    echo "Processing dimension: $dimension"
    
    temp_input_file="${INPUT_FILE_DIMENSION}${dimension}.txt"
    temp_output_file="${INPUT_FILE_DIMENSION}${dimension}_longer.txt"

    python convert_demo_vbench.py --input_file "$temp_input_file" --output_file "$temp_output_file" --retry_times "$RETRY_TIMES"
done

for category in "${category_list[@]}"
do
    echo "Processing category: $category"
    
    temp_input_file="${INPUT_FILE_CATEGORY}${category}.txt"
    temp_output_file="${INPUT_FILE_CATEGORY}${category}_longer.txt"

    python convert_demo_vbench.py --input_file "$temp_input_file" --output_file "$temp_output_file" --retry_times "$RETRY_TIMES"
done