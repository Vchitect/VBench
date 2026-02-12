import json
import os

# Script to check if the image suite prepared is consistent with the json, and wont stop the sampling unexpectedly

# --- CONFIGURATION ---
INPUT_JSON = '../vbench2_beta_i2v/vbench2_i2v_full_info.json'          # Your original JSON file
IMAGE_SUITE_DIR = 'path/to/your/downloaded_image_suite'            # Folder containing the mp4 files
OUTPUT_JSON = 'inconsistent_image_suite.json'  # File to save the missing/ inconsistent data to
# ---------------------

def filter_missing_image():
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
        expected_video_name = image_name
        expected_video_path = os.path.join(IMAGE_SUITE_DIR, expected_video_name)
        
        
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
    filter_missing_image()