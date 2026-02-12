import json
import os

# Script to check issue with missing sampled videos


# --- CONFIGURATION ---
INPUT_JSON = '../vbench2_beta_i2v/vbench2_i2v_full_info.json'          # original JSON file - for I2V
# INPUT_JSON = '../VBench/VBench_full_info.json'    # original JSON file - for T2V
VIDEO_DIR = 'path/to/your/sampled_videos'            # Folder containing the mp4 files
OUTPUT_JSON = 'missing_entries.json' # File to save the missing data to
# ---------------------

def check_num_videos():
    if 'i2v' in INPUT_JSON:
        expected_count = 5590
    else:
        expected_count = 4720

    # Check number of files under VIDEO DIR == num_videos
    # Check if directory exists
    if not os.path.exists(VIDEO_DIR):
        print(f"Error: Directory '{VIDEO_DIR}' not found.")
        return False

    files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith('.mp4')]
    actual_count = len(files)

    # 4. Compare and Report
    print(f"Expected videos: {expected_count}")
    print(f"Found videos:    {actual_count}")

    if actual_count == expected_count:
        print("check_num_videos: SUCCESS. Count matches.")
        return True
    else:
        diff = expected_count - actual_count
        if diff > 0:
            print(f"check_num_videos: FAIL. Missing {diff} videos.")
        else:
            print(f"check_num_videos: FAIL. Found {abs(diff)} extra videos.")
        return False



def filter_missing_videos():
    # 1. Load the original data
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found.")
        return

    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    missing_entries = []
    
    print(f"Scanning {len(data)} entries...")

    # 2. Check each entry
    for entry in data:
        # adjust key if your json uses something other than 'image_name'
        image_name = entry.get('image_name', '') 
        
        if not image_name:
            continue

        # Convert 'xxx.jpg' -> 'xxx' -> 'xxx-0.mp4'
        base_name = os.path.splitext(image_name)[0]
        expected_video_name = f"{base_name}-0.mp4"
        expected_video_path = os.path.join(VIDEO_DIR, expected_video_name)
        
        # 3. If video file is NOT found, add the whole entry to our list
        if not os.path.exists(expected_video_path):
            missing_entries.append(entry)

    # 4. Save the missing entries to a new file
    if missing_entries:
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(missing_entries, f, indent=4)
        
        print(f"\n Done! Found {len(missing_entries)} missing videos.")
        print(f" Data for missing entries saved to: {OUTPUT_JSON}")
    else:
        print("\n All videos exist! No missing entries found.")

if __name__ == "__main__":

    # Comment out for your own purpose

    # Function to check if the correct number of videos are sampled
    check_num_videos()

    # Function to check which videos are not missing
    # Checking based on the naming of 'xxx-0.mp4', you might want to change this
    filter_missing_videos()